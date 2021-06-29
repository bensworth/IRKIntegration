#include <iomanip>
#include "mfem.hpp"
#include "IRK.hpp"

int io_digits = 16;
using namespace mfem;

// Solving
//    y_1' = -4 y_1 - 1 y_2   +   1 y_1 - 2y_2   +  (1 + t/2 + t^2/6 + t^3/24 + t^4/120)
//    y_2' = -2 y_1 - 3 y_2   +   1 y_1 - 1y_2   +  (1 + t/6 + t^2/24 + t^3/120 + t^4/720)
// with initial conditions
//    y_1(0) = 1, y_2(0) = 2

// Initial condition *solution depends on this*!
void get_ic(Vector &u)
{
   u(0) = 1.0;
   u(1) = 2.0;
}

void get_sol(Vector &sol, double t)
{
   sol.SetSize(2);
   double s13 = std::sqrt(13);
   double expm13t = std::exp(-s13*t);
   double exp13t = std::exp(s13*t);
   double exp13t_2 = std::exp(0.5*(-7+s13)*t);
   double exp313t_2 = std::exp(0.5*(-7+3*s13)*t);
   double ft1 = (285332+580752*t + 226314*t*t + 44226*t*t*t + 15309*t*t*t*t);
   double ft2 = (-1095724-55692*t + 8100*t*t - 972*t*t*t + 2187*t*t*t*t);
   sol(0) = expm13t * ( exp13t_2*(28850822. + 22837054.*s13) + \
      exp313t_2*(28850822. - 22837054.*s13) + 13*exp13t*ft1 ) / 61410960.;
   sol(1) = expm13t * ( exp13t_2*(54288754. + 8614646.*s13) + 
      exp313t_2*(54288754. - 8614646.*s13) - 13*exp13t*ft2 ) / 61410960.;
}

// Provides the time-dependent RHS of the ODEs after spatially discretizing the
//     PDE,
//         du/dt = L*u + s(t) == - A*u + D*u + s(t).
//     where:
//         A: The advection discretization,
//         D: The diffusion discretization,
//         s: The solution-independent source term discretization.
struct ODE_Op : IRKOperator
{
   DenseMatrix Ae, Ai, Ai_inv;
   bool imp_forcing;
   mutable double dt_;
   std::map<int, DenseMatrix*> prec;
   std::map<std::pair<double,double>, int> prec_index;
   mutable DenseMatrix *current_prec;

public:
   ODE_Op(bool imp_forcing_)
      : IRKOperator(MPI_COMM_WORLD, true, 2),
        imp_forcing(imp_forcing_),
        dt_(-1)
   {
      // Explicit part of operator (imaginary-eigenvalues)
      Ae.SetSize(2);
      Ae(0,0) = 1;
      Ae(0,1) = -2;
      Ae(1,0) = 1;
      Ae(1,1) = -1;
      // Implicit part of operator (real-eigenvalues)
      Ai.SetSize(2);
      Ai(0,0) = -4;
      Ai(0,1) = -1;
      Ai(1,0) = -2;
      Ai(1,1) = -3;
      // Invese implicit part of operator (real-eigenvalues)
      Ai_inv.SetSize(2);
   }

   void SetImplicit()
   {
      Ai += Ae;
      Ae = 0.0;
      imp_forcing = true;
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- (L y - f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      std::cout << "Why is Mult() being called?? This is probably bad.\n\n";
   }

   void ExplicitMult(const Vector &u, Vector &du_dt) const override
   {
      du_dt.SetSize(u.Size());
      Ae.Mult(u, du_dt);
      if (!imp_forcing) {
         this->AddForcing(du_dt, this->GetTime(), 1.0);
      }
   }

   void ImplicitMult(const Vector &u, Vector &du_dt) const override
   {
      du_dt.SetSize(u.Size());
      Ai.Mult(u, du_dt);
      if (imp_forcing) {
         this->AddForcing(du_dt, this->GetTime(), 1.0);
      }
   }

   void ExplicitGradientMult(const Vector &u, Vector &du_dt) const override
   {
      du_dt.SetSize(u.Size());
      Ai.Mult(u, du_dt);
   }

   // Apply y = (I - dt*Ai)^{-1}x
   void ImpInvMult(double dt, const Vector &x, Vector &y)
   {
      Ai_inv = Ai;
      Ai_inv *= -dt;
      Ai_inv(0,0) += 1;
      Ai_inv(1,1) += 1;
      Ai_inv.Invert();
      Ai_inv.Mult(x,y);
   }

   void AddForcing(Vector &rhs, double t, double c0) const
   {
      // Add forcing function and BCs
      double temp = c0*(1. + t/2. + t*t/6. + t*t*t/24. + t*t*t*t/120.);
      rhs(0) += temp;
      temp = c0*(1. + t/6. + t*t/24. + t*t*t/120. + t*t*t*t/720.);
      rhs(1) += temp;
   }

   void AddImplicitForcing(Vector &rhs, double t, double r, double z)
   {
      // Add forcing function and BCs
      if (imp_forcing) {
         double ti = t + r*z;
         this->AddForcing(rhs, ti, r);
         // this->AddForcing(rhs, this->GetTime(), r);
      }
   }

   // Apply action of the mass matrix
   virtual void MassMult(const Vector &x, Vector &y) const override
   {
      y = x;
   }

   // Apply action of the mass matrix
   virtual void MassInv(const Vector &x, Vector &y) const override
   {
      y = x;
   }

   ///Solve k_i = N_I(x_i + dt*k_i).
   // For linear N_I(u) = (Lu + f),
   //    <--> (I - dt*L)k_i = Lx_i + f
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
   {
      Vector rhs(x.Size());
      Ai.Mult(x, rhs);
      if (imp_forcing) {
         this->AddForcing(rhs, this->GetTime(), 1.0);
      }
      this->ImpInvMult(dt, rhs, k);
   }

   /// Solve x - dtf(x, t) = b
   // This is for other formulation of RK in terms of solution rather than stages
   virtual void ImplicitSolve2(const double dt, const Vector &b, Vector &x) override
   {
      Vector rhs(b);
      if (imp_forcing) {
         this->AddForcing(rhs, this->GetTime(), dt);
      }
      this->ImpInvMult(dt, rhs, x);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      current_prec->Mult(x, y);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(int index, const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      prec.at(index)->Mult(x, y);
   }

   /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L)
           + index -> index of system to solve, [0,s_eff)
           + dt    -> time step size
           + type  -> eigenvalue type, 1 = real, 2 = complex pair
        These additional parameters are to provide ways to track when
        (\gamma*M - dt*L) must be reconstructed or not to minimize setup. */
   virtual void SetPreconditioner(int index, double dt, double gamma, int type) override
   {
      std::pair<double,double> key = std::make_pair(dt, gamma);
      bool key_exists = prec_index.find(key) != prec_index.end();
      if (!key_exists)
      {
         prec_index[key] = index;
         prec[index] = new DenseMatrix(Ai);
         (*prec[index]) *= -dt;

         (*prec[index])(0,0) += gamma;
         (*prec[index])(1,1) += gamma;
         prec[index]->Invert();
      }
      current_prec = prec[index];
   }

   ~ODE_Op()
   {
      for (auto &P : prec)
      {
         delete P.second;
      }
   }
};


int solve_ode(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool root = (myid == 0);

   double dt = 1e-2;
   double tf = 0.1;
   int use_irk = 111;
   bool imp_force = true;
   int iters = 5;

   OptionsParser args(argc, argv);
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&use_irk, "-irk", "--irk", "Use IRK solver (provide id; if 0, Euler-IMEX).");
   args.AddOption(&iters, "-i", "--num-iters", "Number applications of iterator, default 1.");
   args.AddOption(&imp_force, "-if", "--imp-forcing","-ef", "--exp-forcing", 
      "Implicit or explicit forcing.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
      return 1;
   }
   if (root) { args.PrintOptions(std::cout); }

   // Define ODE operator
   ODE_Op oper(imp_force);
   Vector u(2);
   get_ic(u);

   // Time integration scheme
   double t = 0.0;
   std::unique_ptr<ODESolver> ode;
   RKData* coeffs = NULL;
   IRK *irk = NULL;
   if (use_irk > 3)
   {
      coeffs = new RKData(static_cast<RKData::Type>(use_irk));
      KrylovParams krylov_params;
      krylov_params.printlevel = 0;
      krylov_params.kdim = 50;
      krylov_params.maxiter = 100;
      krylov_params.reltol = 1e-12;
      krylov_params.solver = KrylovMethod::GMRES;

      if (use_irk < 100) {
         oper.SetImplicit();
         irk = new IRK(&oper, *coeffs);
      }
      else {
         irk = new PolyIMEX(&oper, *coeffs, true, iters);
      }
      irk->SetKrylovParams(krylov_params);
      irk->Init(oper);
      ode.reset(irk);
   }
   else
   {
      IMEXEuler *imex = new IMEXEuler();
      imex->Init(oper);
      ode.reset(imex);
   }

   bool done = false;
   while (!done)
   {
      ode->Step(u, t, dt);
      done = (t >= tf - 1e-8*dt);
   }
   std::cout << "\n";

   // Compute error to exact solution
   Vector sol(2);
   get_sol(sol, t);
   double err = std::sqrt( (u(0)-sol(0))*(u(0)-sol(0)) + (u(1)-sol(1))*(u(1)-sol(1)) );
   if (myid == 0) {
      std::cout << std::setprecision(io_digits) << "u-final = (" << u(0) << ", " << u(1) << ").\n";
      std::cout << std::setprecision(io_digits) << "exact   = (" << sol(0) << ", " << sol(1) << ").\n";
      std::cout << std::setprecision(io_digits) << "t-final " << t << "\nl2 " << err << "\n\n";
   }

   return 0;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int retval = solve_ode(argc, argv);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return retval;
}
