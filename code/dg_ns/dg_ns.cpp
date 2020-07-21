#include "IRK.hpp"
#include "bljacobi.hpp"
#include "3dg_wrapper.hpp"
#include "general/binaryio.hpp"

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

   DGIRKOperator(DGWrapper &dg_, IRKOperator::ExplicitGradients grad_type)
      : IRKOperator(MPI_COMM_WORLD, dg_.Size(), 0.0, IMPLICIT, grad_type),
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
   mutable DGMatrix A;

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
      y += z;
   }

   virtual Operator &GetGradient(const Vector &x) const override
   {
      add(u0, dt, x, z);
      DGMatrix J;
      dg.AssembleJacobian(z, J);
      std::unique_ptr<SparseMatrix> A_tmp(Add(1.0, M.A, -dt, J.A));
      A.A.Swap(*A_tmp);
      return A.A;
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
      k = 0.0;
      newton.Mult(rhs, k);
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
   int problem;
   double tf, dt;
   bool use_irk, compute_exact;
};

double RunCase(Params &p)
{
   double t = 0.0;

   double tf = p.tf;
   double dt = p.dt;
   bool use_irk = p.use_irk;
   bool compute_exact = p.compute_exact;

   std::cout << "Using dt = " << dt << std::endl;

   DGWrapper dg_wrapper;
   dg_wrapper.Init(p.problem);

   // DGIRKOperator dg_irk(dg_wrapper, IRKOperator::APPROXIMATE);
   DGIRKOperator dg_irk(dg_wrapper, IRKOperator::EXACT);
   FE_Evolution evol(dg_wrapper);

   std::unique_ptr<ODESolver> ode;
   std::unique_ptr<RKData> tableau;

   if (use_irk && !compute_exact)
   {
      // RKData::Type irk_type = RKData::RadauIIA7;
      RKData::Type irk_type = RKData::Gauss4;
      tableau.reset(new RKData(irk_type));
      IRK *irk = new IRK(&dg_irk, *tableau);

      IRK::KrylovParams krylov_params;
      krylov_params.reltol = 1e-5;
      krylov_params.printlevel = 2;
      krylov_params.kdim = 100;

      IRK::NewtonParams newton_params;
      newton_params.reltol = 1e-9;
      newton_params.abstol = 1e-9;
      newton_params.jac_update_rate = 1;

      irk->SetKrylovParams(krylov_params);
      irk->SetNewtonParams(newton_params);
      irk->Init(dg_irk);

      ode.reset(irk);
   }
   else
   {
      if (compute_exact) { ode.reset(new RK4Solver); }
      else { ode.reset(new SDIRK33Solver); }
      evol.SetTime(t);
      ode->Init(evol);
   }

   Vector u;
   dg_wrapper.InitialCondition(u);

   int step = 1;
   bool done = false;
   while (!done)
   {
      double dt_real = min(dt, tf - t);
      std::cout << " >>> Step " << step++ << " <<<\n";
      ode->Step(u, t, dt_real);
      done = (t >= tf - 1e-8*dt);
   }

   if (compute_exact) {
      std::ofstream f("data/u_exact.dat");
      WriteVector(f, u);
      return 0.0;
   }
   else
   {
      // Vector u_ex;
      // std::ifstream f("data/u_exact.dat");
      // ReadVector(f, u_ex);
      // u_ex -= u;
      // printf("Error: %8.6e\n", u_ex.Normlinf());
      // return u_ex.Normlinf();
   }
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   Params p;
   // p.dt = 0.16;
   p.dt = 1e-3;
   p.tf = p.dt;
   p.compute_exact = false;
   p.problem = 1;

   int nruns = 1;

   ofstream f("out.txt");
   f << "dt    irk    dirk" << std::endl;
   f << std::scientific;
   f.precision(8);

   for (int i=0; i<nruns; ++i)
   {
      double irk_er=0.0, dirk_er=0.0;
      p.use_irk = true;
      irk_er = RunCase(p);
      p.use_irk = false;
      dirk_er = RunCase(p);
      f << p.dt << "    " << irk_er << "    " << dirk_er << std::endl;
      p.dt *= 0.5;
   }

   MPI_Finalize();
   return 0;
}
