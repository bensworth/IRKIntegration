#include <iomanip>
#include "mfem.hpp"
#include "IRK.hpp"

int io_digits = 16;
using namespace mfem;

// Solving y' = \lambda y + i\eta y
// --> y = k0 * exp(\lambda t + i*\eta t)
//       = k0 * e^(\lambda t) * ( cos(\eta t) + i*sin(\eta t) )

// Complex multuplication, (a+ib)(c+id) = (ac - bd) + i(ad + bc)
void complex_mult(const Vector &a, const Vector &b, Vector & c)
{
   c(0) = a(0)*b(0) - a(1)*b(1);
   c(1) = a(0)*b(1) + a(1)*b(0);
}

// Initial condition
void get_ic(Vector &u)
{
   u(0) = 1.0;
   u(1) = 0.0;
   // u(0) = 0.9753140268839142;
   // u(1) = 0.004123261137038915;
}

void get_sol(Vector &sol, double eta_real, double eta_imag, double t)
{
   Vector ic(2);
   get_ic(ic);
   Vector temp_sol(2);
   temp_sol(0) = std::exp(eta_real*t)*std::cos(eta_imag*t);
   temp_sol(1) = std::exp(eta_real*t)*std::sin(eta_imag*t);
   sol.SetSize(2);
   complex_mult(ic, temp_sol, sol);
}

struct ComplexNumber
{
public:
   ComplexNumber(double real_, double imag_) : imag(imag_), real(real_) { };

   void Mult(const Vector &x, Vector &y)
   {
      if (x.Size() != 2) mfem_error("Vector must be size 2!\n");
      y.SetSize(2);

      y(0) = x(0)*real - x(1)*imag;
      y(1) = x(0)*imag + x(1)*real;
   }
   // Apply inverse of shift of number (this = z), y = x/(1 - dt*z)
   // Note (a+ib)/(c+id) = (ac+bd)/(c^2+d^2) + i(bc - ad)/(c^2+d^2);
   void ShiftInvMult(double dt, const Vector &x, Vector &y)
   {
      double r0 = 1 - dt*real;
      double i0 = -dt*imag;
      double denom = r0*r0 + i0*i0;
   
      y(0) = (x(0)*r0 + x(1)*i0) / denom;
      y(1) = (x(1)*r0 - x(0)*i0) / denom;
   }

   // Apply inverse of number (this = z), y = x/z
   // Note (a+ib)/(c+id) = (ac+bd)/(c^2+d^2) + i(bc - ad)/(c^2+d^2);
   void InvMult(const Vector &x, Vector &y)
   {
      double denom = real*real + imag*imag;
      y(0) = (x(0)*real + x(1)*imag) / denom;
      y(1) = (x(1)*real - x(0)*imag) / denom;
   }

   void Add(double c_real, double c_imag)
   {
      real += c_real;
      imag += c_imag;
   }

   void Add(ComplexNumber z)
   {
      real += z.Real();
      imag += z.Imag();
   }

   void Zero() { real = 0.0; imag = 0.0; };

   void RealScale(double c)
   {
      real *= c;
      imag *= c;
   }

   void Print()
   {
      std::cout << real << " + " << imag << "i\n";
   }

   double Real() { return real; };
   double Imag() { return imag; };

private:
   double real;
   double imag;
};


// Provides the time-dependent RHS of the ODEs after spatially discretizing the
//     PDE,
//         du/dt = L*u + s(t) == - A*u + D*u + s(t).
//     where:
//         A: The advection discretization,
//         D: The diffusion discretization,
//         s: The solution-independent source term discretization.
struct ODE_Op : IRKOperator
{
   ComplexNumber *Ai, *Ae;
   double Ai_r, Ai_i, Ae_r, Ae_i;
   bool imp_forcing;
   mutable double dt_;
   std::map<std::pair<double,double>, int> prec_index;
   std::map<int, ComplexNumber*> prec;
   mutable ComplexNumber *current_prec;
public:
   ODE_Op(double Ai_r_, double Ai_i_, double Ae_r_, double Ae_i_)
      : IRKOperator(MPI_COMM_WORLD, true, 2),
      // : IRKOperator(MPI_COMM_WORLD, false, 2, 0.0, EXPLICIT,
         // IRKOperator::ExplicitGradients::APPROXIMATE ),
        Ai_r(Ai_r_),
        Ai_i(Ai_i_),
        Ae_r(Ae_r_),
        Ae_i(Ae_i_),
        imp_forcing(false),
        dt_(-1)
   {
      // Implicit part of operator (real-valued)
      Ai = new ComplexNumber(Ai_r, Ai_i);
      // Explicit part of operator (real-valued)
      Ae = new ComplexNumber(Ae_r, Ae_i);
   }

   void SetImplicit()
   {
      Ai->Add(*Ae);
      Ae->Zero();
      imp_forcing = true;
   }

   virtual void SetExplicitGradient(const Vector &u, double dt, 
                                  const BlockVector &x, const Vector &c)
   { };

   virtual void SetExplicitGradient(double r, const BlockVector &x, const Vector &z)
   { };

   // Compute the right-hand side of the ODE system.
   // du_dt <- (L y - f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      std::cout << "Why is Mult() being called?? This is probably bad.\n\n";
   }

   void ExplicitMult(const Vector &u, Vector &du_dt) const override
   {
      Ae->Mult(u, du_dt);
   }

   void ImplicitMult(const Vector &u, Vector &du_dt) const override
   {
      Ai->Mult(u, du_dt);
   }

   void ExplicitGradientMult(const Vector &u, Vector &du_dt) const override
   {
      Ai->Mult(u, du_dt);
   }

   void AddImplicitForcing(Vector &rhs, double t, double r, double z)
   {
      // Add forcing function and BCs
      // if (imp_forcing) {
      //    double ti = t + r*z;
      //    rhs.Add(-r, *B);
      // }
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

   /// Solve k_i = N_I(x_i + dt*k_i).
   // For linear N_I(u) = (Lu + f),
   //    <--> (I - dt*L)k_i = Lx_i + f
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
   {
      Vector rhs(x.Size());
      Ai->Mult(x, rhs);
      Ai->ShiftInvMult(dt, rhs, k);
   }

   /// Solve x - dtf(x, t) = b
   // This is for other formulation of RK in terms of solution rather than stages
   virtual void ImplicitSolve2(const double dt, const Vector &b, Vector &x) override
   {
      Vector rhs(b);
      AddImplicitForcing(rhs, this->GetTime(), dt, 0);
      Ai->ShiftInvMult(dt, rhs, x);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      current_prec->InvMult(x, y);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(int index, const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      prec.at(index)->InvMult(x, y);
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
         prec[index] = new ComplexNumber(Ai->Real(), Ai->Imag());
         prec[index]->RealScale(-dt);
         prec[index]->Add(gamma, 0);
      }
      current_prec = prec[index];
   }

   ~ODE_Op()
   {
      delete Ai;
      delete Ae;
   }
};


int solve_ode(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool root = (myid == 0);

   double eta_imp_real = -0.5;
   double eta_imp_imag = 0.0;
   double eta_exp_real = 0.0;
   double eta_exp_imag = 0.5;
   double dt = 1e-2;
   double tf = 0.1;
   int use_irk = 111;
   bool compute_err = true;
   bool full_imp = false;
   int iters = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&eta_imp_real, "-ir", "--implicit-real", "Implicit real coefficient.");
   args.AddOption(&eta_imp_imag, "-ii", "--implicit-imag", "Implicit imaginary coefficient.");
   args.AddOption(&eta_exp_real, "-er", "--explicit-real", "Explicit real coefficient (only support 0 now).");
   args.AddOption(&eta_exp_imag, "-ei", "--explicit-imag", "Explicit imaginary coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&use_irk, "-irk", "--irk", "Use IRK solver (provide id; if 0, Euler-IMEX).");
   args.AddOption(&iters, "-i", "--num-iters", "Number applications of iterator, default 1.");
   args.AddOption(&full_imp, "-imp", "--full-imp", "-imex","--imex", \
         "Treat ODE fully implicitly."); 

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
      return 1;
   }
   if (root) { args.PrintOptions(std::cout); }

   if (full_imp) {
      eta_imp_real += eta_exp_real;
      eta_exp_real = 0.0;
      eta_imp_imag += eta_exp_imag;
      eta_exp_imag = 0.0;
   }

   // Define ODE operator
   ODE_Op oper(eta_imp_real, eta_imp_imag, eta_exp_real, eta_exp_imag);
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
         // irk = new PolyIMEX(&oper, *coeffs, false, iters); //DEBUG (use Newton)
      }
      irk->SetKrylovParams(krylov_params);

      NewtonParams NEWTON;
      NEWTON.printlevel = 1;
      NEWTON.abstol = 1e-3;
      irk->SetNewtonParams(NEWTON);

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
   
   // std::cout << "u = " << u(0) << " + i" << u(1) << "\n";

   // Compute error to exact solution
   if (compute_err)
   {
      Vector sol(2);
      get_sol(sol, eta_imp_real+eta_exp_real, eta_imp_imag+eta_exp_imag, t);
      double err = std::sqrt( (u(0)-sol(0))*(u(0)-sol(0)) + (u(1)-sol(1))*(u(1)-sol(1)) );
      if (myid == 0) {
         std::cout << std::setprecision(io_digits) << "u-final = (" << u(0) << ", " << u(1) << ").\n";
         std::cout << std::setprecision(io_digits) << "exact   = (" << sol(0) << ", " << sol(1) << ").\n";
         std::cout << std::setprecision(io_digits) << "t-final " << t << "\nl2 " << err << "\n\n";
      }
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
