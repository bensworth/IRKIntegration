#include "IRK.hpp"
#include "bljacobi.hpp"
#include "3dg_wrapper.hpp"
#include "general/binaryio.hpp"
#include "fem/picojson.h"

#include <memory>

struct DGIRKOperator : IRKOperator
{
   // using PrecKey = std::tuple<int,double,double>;
   using PrecKey = int;

   DGWrapper &dg;
   DGMatrix M;
   std::map<int,DGPreconditioner*> prec;
   std::vector<DGMatrix *> Js;
   DGPreconditioner *current_prec;
   mutable bool jacobian_updated;
   Vector z;

   DGIRKOperator(DGWrapper &dg_, IRKOperator::ExplicitGradients jac_type)
      : IRKOperator(MPI_COMM_WORLD, dg_.Size(), 0.0, IMPLICIT, jac_type),
        dg(dg_),
        current_prec(nullptr),
        jacobian_updated(true)
   {
      dg.MassMatrix(M);
      z.SetSize(dg.Size());
   }

   /// Apply action of M*du/dt, y <- L(x,y)
   virtual void ExplicitMult(const Vector &x, Vector &y) const override
   {
      dg.Assemble(x, y);
   }

   virtual void SetExplicitGradient(const Vector &u,
                                    double dt,
                                    const BlockVector &x,
                                    const Vector &c) override
   {
      for (DGMatrix *J : Js) { delete J; }
      Js.resize(1);
      Js[0] = new DGMatrix;
      add(u, dt, x.GetBlock(c.Size()-1), z);
      dg.AssembleJacobian(z, *Js[0]);
      jacobian_updated = true;
   }

   virtual void SetExplicitGradients(const Vector &u,
                                     double dt,
                                     const BlockVector &x,
                                     const Vector &c) override
   {
      int s = c.Size();
      for (DGMatrix *J : Js) { delete J; }
      Js.resize(s);
      for (int i=0; i<s; ++i)
      {
         Js[i] = new DGMatrix;
         add(u, dt, x.GetBlock(i), z);
         dg.AssembleJacobian(z, *Js[i]);
      }
      jacobian_updated = true;
   }

    virtual void ExplicitGradientMult(const Vector &x, Vector &y) const override
    {
      ExplicitGradientMult(0, x, y);
    }

    virtual void ExplicitGradientMult(int i, const Vector &x, Vector &y) const override
    {
       Js[i]->Mult(x, y);
    }

   /** Apply action mass matrix, y = M*x.
       If not re-implemented, this method simply generates an error. */
   virtual void ImplicitMult(const Vector &x, Vector &y) const override
   {
      dg.ApplyMass(x, y);
   }

   /// Precondition (\gamma*M - dt*L')
   virtual void ImplicitPrec(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != nullptr, "");
      current_prec->Mult(x, y);
   }

   virtual void ImplicitPrec(int index, const Vector &x, Vector &y) const override
   {
      prec.at(index)->Mult(x, y);
   }

   /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L')
           + index: index of system to solve, [0,s_eff)
           + dt:    time step size
           + type:  eigenvalue type, 1 = real, 2 = complex pair
       These additional parameters are to provide ways to track when
       (\gamma*M - dt*L') must be reconstructed or not to minimize setup. */
   virtual void SetPreconditioner(int index, double dt, double gamma, int type) override
   {
      if (jacobian_updated)
      {
         jacobian_updated = false;
         for (auto &p : prec)
         {
            delete p.second;
         }
         prec.clear();
      }

      PrecKey key = index;
      bool key_exists = prec.find(key) != prec.end();
      if (!key_exists)
      {
         prec[key] = new DGPreconditioner(gamma, M, dt, *Js[0], dg.BlockSize());
      }
      current_prec = prec[key];
   }

   virtual void SetPreconditioner(int index, double dt, double gamma, Vector weights) override
   {
      if (jacobian_updated)
      {
         jacobian_updated = false;
         for (auto &p : prec)
         {
            delete p.second;
         }
         prec.clear();
      }

      PrecKey key = index;
      bool key_exists = prec.find(key) != prec.end();
      if (!key_exists)
      {
         DGMatrix B;
         B.A = Js[0]->A;
         B.A *= weights[0];
         for (int i=1; i<weights.Size(); ++i)
         {
            if (weights[i] != 0.0)
            {
               B.A.Add(weights[i], Js[i]->A);
            }
         }
         prec[key] = new DGPreconditioner(gamma, M, dt, B, dg.BlockSize());
      }
      current_prec = prec[key];
   }

   ~DGIRKOperator()
   {
      for (DGMatrix *J : Js)
      {
         delete J;
      }
   }
};

// Solve via Newton's method:
// k = f(x + dt*k)
// f = M^{-1} r
// equivalently,
// M k = r(x + dt*k)
// equivalently,
// F(k) = M k - r(x + dt*x) = 0
// DGNewtonOperator returns M k - r(x + dt * k)
// GetGradient returns M - dt*J
struct DGNewtonOperator : Operator
{
   DGWrapper &dg;
   double dt;

   Vector u0;
   mutable Vector z;

   DGMatrix M;
   mutable SparseMatrix A;

   DGNewtonOperator(DGWrapper &dg_) : Operator(dg_.Size()), dg(dg_)
   {
      dg.MassMatrix(M);
      z.SetSize(height);
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      // z = u0 + dt*x
      add(u0, dt, x, z);
      dg.Assemble(z, y);
      y *= -1.0;
      dg.ApplyMass(x, z);
      // y = M k - r(u0 + dt*k)
      y += z;
   }

   virtual Operator &GetGradient(const Vector &x) const override
   {
      add(u0, dt, x, z);
      DGMatrix J;
      dg.AssembleJacobian(z, J);
      std::unique_ptr<SparseMatrix> A_tmp(Add(1.0, M.A, -dt, J.A));
      A.Swap(*A_tmp);
      return A;
   }
};

class FE_Evolution : public TimeDependentOperator
{
private:
   DGWrapper &dg;
   DGNewtonOperator dg_newton_op;
   NewtonSolver newton;
   GMRESSolver gmres;
   BlockILU prec;
   Vector rhs;
   mutable Vector z;

public:
   FE_Evolution(DGWrapper &dg_)
      : TimeDependentOperator(dg_.Size()),
        dg(dg_),
        dg_newton_op(dg_),
        prec(dg_.BlockSize())
   {
      gmres.SetRelTol(1e-5);
      gmres.SetAbsTol(1e-10);
      gmres.SetMaxIter(100);
      gmres.SetKDim(100);
      gmres.SetPrintLevel(2);
      gmres.SetPreconditioner(prec);
      newton.SetSolver(gmres);
      newton.SetOperator(dg_newton_op);
      newton.SetPrintLevel(1);
      newton.SetRelTol(1e-6);
      // Newton will treat an empty vector as zero for rhs
      rhs.Destroy();
   }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      z.SetSize(x.Size());
      dg.Assemble(x, z);
      dg.ApplyMassInverse(z, y);
   }

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      dg_newton_op.dt = dt;
      dg_newton_op.u0 = x;
      // k = 0.0;
      k = x;
      newton.Mult(rhs, k);
      k -= x;
   }
};

void ReadVector(std::ifstream &f, Vector &b)
{
   int n = bin_io::read<int>(f);
   b.SetSize(n);
   f.read((char *)b.GetData(), sizeof(double)*n);
}

void WriteVector(std::ofstream &f, Vector &b)
{
   int n = b.Size();
   bin_io::write(f, n);
   f.write((const char *)b.GetData(), sizeof(double)*n);
}

struct Params
{
   int problem=0;
   int nsteps;
   double tf=1e-1, dt=1e-2;
   bool use_irk=true, compute_exact=false;
   RKData::Type irk_method;
   IRKOperator::ExplicitGradients jac_type=IRKOperator::APPROXIMATE;
   IRK::JacSparsity solver_sparsity=IRK::DENSE;
   IRK::JacSparsity prec_sparsity=IRK::DIAGONAL;
};

void WriteSolveStats(const std::string &prefix, IRK &irk, const Params &p)
{
   std::map<int,std::string> irk_names;

   irk_names[-13] = "asdirk3";
   irk_names[-14] = "asdirk4";

   irk_names[01] = "dirk1";
   irk_names[02] = "dirk2";
   irk_names[03] = "dirk3";
   irk_names[04] = "dirk4";

   irk_names[12] = "gauss2";
   irk_names[14] = "gauss4";
   irk_names[16] = "gauss6";
   irk_names[18] = "gauss8";
   irk_names[110] = "gauss10";

   irk_names[23] = "radau3";
   irk_names[25] = "radau5";
   irk_names[27] = "radau7";
   irk_names[29] = "radau9";

   irk_names[32] = "lobatto2";
   irk_names[34] = "lobatto4";
   irk_names[36] = "lobatto6";
   irk_names[38] = "lobatto8";

   int avg_newton_iter;
   std::vector<int> avg_krylov_iter, system_size;
   std::vector<double> eig_ratio;
   irk.GetSolveStats(avg_newton_iter, avg_krylov_iter, system_size, eig_ratio);

   std::string jac_type = (p.jac_type == 0) ? "approx" : "exact";
   double solver_sparsity = double(p.solver_sparsity);
   double prec_sparsity = double(p.prec_sparsity);

   std::map<std::string,picojson::value> dict;
   dict.emplace("irk_method", picojson::value(irk_names[p.irk_method]));
   dict.emplace("dt", picojson::value(p.dt));
   dict.emplace("num_systems", picojson::value(double(avg_krylov_iter.size())));
   dict.emplace("avg_newton_iter", picojson::value(double(avg_newton_iter)));
   dict.emplace("jac_type", picojson::value(jac_type));
   dict.emplace("solver_sparsity", picojson::value(solver_sparsity));
   dict.emplace("prec_sparsity", picojson::value(prec_sparsity));

   std::vector<picojson::value> sys_info;
   for (int i=0; i<avg_krylov_iter.size(); ++i)
   {
      std::map<std::string,picojson::value> sys;
      sys.emplace("avg_krylov_iter", picojson::value(double(avg_krylov_iter[i])/p.nsteps/avg_newton_iter));
      sys.emplace("system_size", picojson::value(double(system_size[i])));
      sys.emplace("eig_ratio", picojson::value(eig_ratio[i]));
      sys_info.emplace_back(sys);
   }
   dict.emplace("sys_info", picojson::value(sys_info));

   std::ofstream f("results/" + prefix + "_" + irk_names[p.irk_method] + ".json");
   f << picojson::value(dict) << std::endl;
}

void RunCase(Params &p, Vector &u)
{
   double t = 0.0;
   double tf = p.tf;
   double dt = p.dt;

   std::cout << "Using dt = " << dt << std::endl;

   DGWrapper dg_wrapper;
   dg_wrapper.Init(p.problem);

   DGIRKOperator dg_irk(dg_wrapper, p.jac_type);
   FE_Evolution evol(dg_wrapper);

   std::unique_ptr<ODESolver> ode;
   std::unique_ptr<RKData> tableau;

   if (p.use_irk && !p.compute_exact)
   {
      tableau.reset(new RKData(p.irk_method));
      IRK *irk = new IRK(&dg_irk, *tableau);

      IRK::KrylovParams krylov_params;
      krylov_params.reltol = 1e-5;
      krylov_params.printlevel = 2;
      krylov_params.kdim = 100;

      IRK::NewtonParams newton_params;
      newton_params.reltol = 1e-9;
      newton_params.abstol = 1e-9;
      newton_params.jac_update_rate = 1;
      newton_params.maxiter = 25;
      newton_params.jac_solver_sparsity = p.solver_sparsity;
      newton_params.jac_prec_sparsity = p.prec_sparsity;

      irk->SetKrylovParams(krylov_params);
      irk->SetNewtonParams(newton_params);
      irk->Init(dg_irk);

      ode.reset(irk);
   }
   else
   {
      if (p.compute_exact) { ode.reset(new RK4Solver); }
      // else { ode.reset(new SDIRK33Solver); }
      else { ode.reset(new BackwardEulerSolver); }
      evol.SetTime(t);
      ode->Init(evol);
   }

   dg_wrapper.InitialCondition(u);

   p.nsteps = 0;
   bool done = false;
   while (!done)
   {
      double dt_real = min(dt, tf - t);
      std::cout << " >>> Step " << ++p.nsteps << " <<<\n";
      ode->Step(u, t, dt_real);
      done = (t >= tf - 1e-8*dt);
   }

   IRK *irk = dynamic_cast<IRK*>(ode.get());
   if (irk)
   {
      std::string prefix = (p.problem == 0) ? "ev" : "naca";
      WriteSolveStats(prefix, *irk, p);
   }
}

double RunEulerVortex(Params p)
{
   Vector u;
   p.problem = 0;
   RunCase(p, u);

   if (p.compute_exact) {
      std::ofstream f("data/u_exact.dat");
      WriteVector(f, u);
      return 0.0;
   }
   else
   {
      /*Vector u_ex;
      std::ifstream f("data/u_exact.dat");
      ReadVector(f, u_ex);
      u_ex -= u;
      printf("Error: %8.6e\n", u_ex.Normlinf());
      return u_ex.Normlinf();*/
      return 0.0;
   }
}

void RunNACA(Params p)
{
   Vector u;
   p.problem = 1;
   RunCase(p, u);
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   double dt = 1e-2;
   double tf = 0.1;
   int irk_method = 16;
   int jac_type;
   int solver_sparsity=2, prec_sparsity=1;

   OptionsParser args(argc, argv);
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&irk_method, "-i", "--irk", "ID of IRK method.");
   args.AddOption(&jac_type, "-j", "--jacobian", "Jacobian type (0: approximate, 1: exact)");
   args.AddOption(&solver_sparsity, "-s", "--solver-sparsity", "Solver sparsity (0: lumped, 1: diagonal, 2: dense)");
   args.AddOption(&prec_sparsity, "-p", "--prec-sparsity", "Preconditioner sparsity (0: lumped, 1: diagonal, 2: dense)");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      return 1;
   }
   args.PrintOptions(std::cout);

   Params p;
   p.dt = dt;
   p.tf = tf;
   p.compute_exact = false;
   p.irk_method = RKData::Type(irk_method);
   p.use_irk = true;
   p.jac_type = (jac_type == 0) ? IRKOperator::APPROXIMATE : IRKOperator::EXACT;
   p.solver_sparsity = IRK::JacSparsity(solver_sparsity);
   p.prec_sparsity = IRK::JacSparsity(prec_sparsity);

   // RunEulerVortex(p);
   RunNACA(p);

   MPI_Finalize();
   return 0;
}
