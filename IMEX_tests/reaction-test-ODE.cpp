#include "mfem.hpp"

using namespace mfem;

#define forcing 0

bool root;
double eta = 0.5;    // reaction coefficient, eta

// Initial condition for manufactured solution with solution at
// t=0 given by sin(2pi*x(1-y))sin(2pi*y(1-x))
double ic_fn(double x, double y)
{
   return sin(2.0*M_PI*x*(1.0-y))*sin(2.0*M_PI*(1.0-x)*y);
}

// Exact solution (x,y,t)
// - With forcing, we solve u_t = eta*u(1-u) + f with f designed
// such that this equation is satisfed for solution
//    u* = sin(2*pi*x(1-y)(1+2t))sin(2*pi*y(1-x)(1+2t))
// - For no forcing, we solve the ODE at each spatial point
// u_t = eta*u(1-u), subject to u(0) = ic_fn(x,y) := u0. This is
// a separable equation with solution
//    u0*e^(Ct) / (1 + u0(e^(Ct)-1))
double sol_fn(double x, double y, double t)
{
#if forcing
   return sin(2.0*M_PI*x*(1.0-y)*(1.0+2.0*t))*sin(2.0*M_PI*y*(1.0-x)*(1.0+2.0*t));
#else
   double u0 = ic_fn(x, y); 
   return u0*std::exp(eta*t) / ( 1.0 + u0*(std::exp(eta*t) - 1.0) );
#endif
}

// Forcing function f for u_t = eta*u(1-u) + f.
// with solution u* = sin(2*pi*x(1-y)(1+2t))sin(2*pi*y(1-x)(1+2t))
double force_fn(double x, double y, double t)
{
#if forcing
   double s = sol_fn(x, y, t);
   double v = -eta*s*(1.0-s) +
      (4*M_PI*(1-x)*y*cos(2*M_PI*(1+2*t)*(1-x)*y)*sin(2*M_PI*(1+2*t)*x*(1-y))
      + 4*M_PI*x*(1-y)*cos(2*M_PI*(1+2*t)*x*(1-y))*sin(2*M_PI*(1+2*t)*(1-x)*y) );
   return v;
#else
   return 0.0;
#endif
}

double compute_error(const Vector &sol, std::vector<double> &xx,
   std::vector<double> &yy, double t, double dx)
{
   double err = 0.0;
   for (int i=0; i<sol.Size(); i++) {
      double temp = sol_fn(xx[i], yy[i], t);
      temp -= sol(i);
      temp *= temp;
      err += temp;
   }
   // err *= (dx*dx);
   return err;
}

// Return if error < 1 to note when divergence happens
bool simulate(ODESolver *ode, Vector &u, std::vector<double> &xx,
   std::vector<double> &yy, double tf, double dt, double dx)
{
   bool done = false;
   MPI_Barrier(MPI_COMM_WORLD);
   StopWatch timer;
   timer.Start();
   double t = 0;

   while (!done)
   {
      ode->Step(u, t, dt);
      done = (t >= tf - 1e-8*dt);
   }
   timer.Stop();

   // Compute error to exact solution
   double err = compute_error(u, xx, yy, t, dx);
   if (root) {
      std::cout << "l2(t) " << err << "\n";
   }

   if (err > 1) return false;
   else return true;
}


struct Reaction : TimeDependentOperator
{
   std::vector<double> &xx;
   std::vector<double> &yy;
public:
   Reaction(std::vector<double> &xx_, std::vector<double> &yy_)
      : TimeDependentOperator(xx_.size(), 0.0),
        xx(xx_), yy(yy_)
   {

   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- (N(u) + f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      // Set forcing function
      for (int i=0; i<xx.size(); i++) {
         du_dt(i) = force_fn(xx[i], yy[i], this->GetTime());
         du_dt(i) += eta*u(i)*(1.0 - u(i));
      }
   }

   ~Reaction()
   {

   }
};


int run_test(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   root = (myid == 0);

   std::string mesh_file = MFEM_SOURCE_DIR + std::string("/data/inline-quad.mesh");
   int ser_ref_levels = 3;
   int par_ref_levels = 0;
   double tf = 3;
   double dt = 1e-2;
   int ode_solver_type = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&eta, "-eta", "--eta-const", "Reaction coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&ode_solver_type, "-ode", "--ode", "Use ODE solver id; default forward Euler).");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
      return 1;
   }
   if (root) { args.PrintOptions(std::cout); }

   Mesh *serial_mesh = new Mesh(mesh_file.c_str(), 1, 1);
   int dim = serial_mesh->Dimension();
   if (ser_ref_levels < 0)
   {
      ser_ref_levels = (int)floor(log(50000./serial_mesh->GetNE())/log(2.)/dim);
   }
   for (int l = 0; l < ser_ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, *serial_mesh);
   delete serial_mesh;
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // Print mesh size
   double hmin, hmax, kmin, kmax;
   mesh.GetCharacteristics (hmin, hmax, kmin, kmax);
   if (root)
   {
      std::cout << "hmin = " << hmin << ", hmax = " << hmax << "\n";
   }

   int numv = mesh.GetNV();
   std::vector<double> xx(numv);
   std::vector<double> yy(numv);
   Vector sol(numv);
   for (int i=0; i<numv; i++) {
      double *temp;
      temp = mesh.GetVertex(i);
      xx[i] = temp[0];
      yy[i] = temp[1];
      // Set initial condition
      sol(i) = ic_fn(xx[i],yy[i]);
   }

   // Compute error to exact solution
   double err = compute_error(sol, xx, yy, 0, hmin);
   if (err > 1e-15) {
      if (myid == 0) std::cout << "Warning! t = " << 0 << ", error = " << err << "\n";
   }

   Reaction oper(xx, yy);
   
   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      // Explicit methods
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         if (root)
         {
            std::cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         return 3;
   }

   oper.SetTime(0.0);
   ode_solver->Init(oper);
   simulate(ode_solver, sol, xx, yy, tf, dt, hmin);

   // for (int i=0; i<numv; i++) {
   //    std::cout << "("<<xx[i]<<","<<yy[i]<<") = " << sol(i) << "\n";
   // }

   delete ode_solver;
   return 0;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int retval = run_test(argc, argv);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return retval;
}
