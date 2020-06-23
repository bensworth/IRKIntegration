#include "IRK.hpp"
#include "bljacobi.hpp"
#include "3dg_wrapper.hpp"

struct DGIRKOperator : IRKOperator
{
   DGWrapper &dg;
   DGMatrix M;
   mutable DGMatrix J;
   std::map<std::pair<double,double>,DGPreconditioner*> prec;
   DGPreconditioner *current_prec;
   mutable bool jacobian_updated;

   DGIRKOperator(DGWrapper &dg_)
      : IRKOperator(MPI_COMM_WORLD, dg_.Size(), 0.0, IMPLICIT),
        dg(dg_),
        current_prec(nullptr),
        jacobian_updated(true)
   {
      dg.MassMatrix(M);
   }

   /// Apply action of M*du/dt, y <- L(x,y)
   virtual void ExplicitMult(const Vector &x, Vector &y) const override
   {
      dg.Assemble(x, y);
   }

   /// Gradient of L(u, t) w.r.t u evaluated at x
   virtual Operator &GetExplicitGradient(const Vector &x) const override
   {
      dg.AssembleJacobian(x, J);
      jacobian_updated = true;
      return J;
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

   /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L')
           + index: index of system to solve, [0,s_eff)
           + dt:    time step size
           + type:  eigenvalue type, 1 = real, 2 = complex pair
       These additional parameters are to provide ways to track when
       (\gamma*M - dt*L') must be reconstructed or not to minimize setup. */
   virtual void SetSystem(int index, double dt, double gamma, int type) override
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

      std::pair<double,double> key = std::make_pair(dt, gamma);
      bool key_exists = prec.find(key) != prec.end();
      if (!key_exists)
      {
         prec[key] = new DGPreconditioner(gamma, M, dt, J, dg.BlockSize());
      }

      current_prec = prec[key];
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
      gmres.SetPrintLevel(1);
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
      dg.Assemble(x, y);
   }

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      dg_newton_op.dt = dt;
      dg_newton_op.u0 = x;
      k = 0.0;
      newton.Mult(rhs, k);
   }
};

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);

   double t = 0.0;
   double tf = 0.1;
   double dt = 1e-3;
   bool use_irk = false;

   OptionsParser args(argc, argv);
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&use_irk, "-i", "--irk", "-no-i", "--no-irk",
                  "Use IRK solver.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(std::cout);
      MPI_Finalize();
      return 1;
   }

   std::cout << "Using dt = " << dt << std::endl;

   DGWrapper dg_wrapper;

   DGIRKOperator dg_irk(dg_wrapper);
   FE_Evolution evol(dg_wrapper);

   std::unique_ptr<ODESolver> ode;

   if (use_irk)
   {
      RKData::Type irk_type = RKData::RadauIIA3;
      // RKData::Type irk_type = RKData::LSDIRK3;
      IRK *irk = new IRK(&dg_irk, irk_type);

      IRK::KrylovParams krylov_params;
      krylov_params.reltol = 1e-5;
      krylov_params.printlevel = 2;
      krylov_params.kdim = 100;

      IRK::NewtonParams newton_params;

      irk->SetKrylovParams(krylov_params);
      irk->SetNewtonParams(newton_params);
      irk->Init(dg_irk);

      ode.reset(irk);
   }
   else
   {
      ode.reset(new SDIRK33Solver);
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

   MPI_Finalize();

   return 0;
}
