#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <cmath>

#include "as.hpp"
#include "IRK.hpp"

using namespace std;
using namespace mfem;

double ic_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
   double r2 = pow(x-0.25,2) + pow(y-0.25,2);
   return 0.25 * sin(8*M_PI*x)*sin(8*M_PI*y);
}

double f_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
   return cos(4*M_PI*x)*cos(4*M_PI*y);
}

struct BackwardEulerPreconditioner
{
   ConstantCoefficient mass_coeff, diff_coeff;
   HighOrderASPreconditioner prec;

   BackwardEulerPreconditioner(ParFiniteElementSpace &fes,
                               double gamma,
                               double dt_times_eps,
                               Array<int> &ess_bdr)
   : mass_coeff(gamma),
     diff_coeff(-dt_times_eps),
     prec(fes, mass_coeff, diff_coeff, ess_bdr, 1)
   { }

   void Mult(const Vector &b, Vector &x)
   {
      prec.Mult(b, x);
   }
};

struct HeatIRKOperator : IRKOperator
{
   ParFiniteElementSpace &fes;
   ConstantCoefficient diff_coeff;
   ParBilinearForm a, m;
   Array<int> ess_bdr, ess_tdof_list;
   OperatorHandle A, M;
   std::unique_ptr<OperatorJacobiSmoother> Minv;
   std::map<std::pair<double,double>,BackwardEulerPreconditioner*> prec;
   BackwardEulerPreconditioner *current_prec;
   IntegrationRules irs;

   Vector B;
   mutable Vector z;

public:
   HeatIRKOperator(ParFiniteElementSpace &fes_, double eps)
      : IRKOperator(fes_.GetComm(), fes_.GetTrueVSize(), 0.0, IMPLICIT),
        fes(fes_),
        diff_coeff(-eps),
        a(&fes),
        m(&fes),
        irs(0, Quadrature1D::GaussLobatto)
   {
      a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.Assemble();

      const Mesh &mesh = *fes.GetMesh();
      if (mesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh.bdr_attributes.Max());
         ess_bdr = 1;
         fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
      }
      a.FormSystemMatrix(ess_tdof_list, A);

      const FiniteElement *fe = fes.GetFE(0);
      int order = fe->GetOrder();
      Geometry::Type geom = fe->GetGeomType();
      const IntegrationRule &ir = irs.Get(geom, 2*(order-1));

      m.AddDomainIntegrator(new MassIntegrator(&ir));
      m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      m.Assemble();
      m.FormSystemMatrix(ess_tdof_list, M);

      Minv.reset(new OperatorJacobiSmoother(m, ess_tdof_list));

      ParLinearForm b(&fes);
      FunctionCoefficient f_coeff(f_fn);
      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
      b.Assemble();
      B.SetSize(fes.GetTrueVSize());
      b.ParallelAssemble(B);

      current_prec = nullptr;
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- M^{-1} L y
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      z.SetSize(u.Size());
      A->Mult(u, z);
      z += B;
      Minv->Mult(z, du_dt);
   }

   virtual void ApplyL(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      A->Mult(x, y);
   }

   // void ExplicitMult(const Vector &u, Vector &du_dt) const override
   // {
   //    A->Mult(u, du_dt);
   // }

   // Apply action of the mass matrix
   virtual void ImplicitMult(const Vector &x, Vector &y) const override
   {
      M->Mult(x, y);
   }

   virtual void ApplyMInv(const Vector &x, Vector &y) const override
   {
      Minv->Mult(x, y);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   inline void ImplicitPrec(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      current_prec->Mult(x, y);
   }

   /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L)
           + index -> index of system to solve, [0,s_eff)
           + dt    -> time step size
           + type  -> eigenvalue type, 1 = real, 2 = complex pair
        These additional parameters are to provide ways to track when
        (\gamma*M - dt*L) must be reconstructed or not to minimize setup. */
   virtual void SetSystem(int index, double dt, double gamma, int type) override
   {
      std::pair<double,double> key = std::make_pair(dt, gamma);
      bool key_exists = prec.find(key) != prec.end();
      if (!key_exists)
      {
         prec[key] = new BackwardEulerPreconditioner(
            fes, gamma, dt*diff_coeff.constant, ess_bdr);
      }
      current_prec = prec[key];
   }

   ~HeatIRKOperator()
   {
      for (auto &P : prec)
      {
         delete P.second;
      }
   }
};

double rhs(const Vector &xpt)
{
   int dim = xpt.Size();
   double x = xpt[0];
   double y = (dim >= 2) ? xpt[1] : 0.0;
   double z1 = ((dim >= 3) ? xpt[2] : 0.0) + 1.0;
   return sin(x)*cos(y)*z1*z1;
}

int run_heat(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   bool root = (myid == 0);

   const char *mesh_file = MFEM_DIR "data/inline-quad.mesh";
   int ser_ref_levels = 3;
   int par_ref_levels = 2;
   int order = 1;
   double eps = 1e-2;
   double dt = 1e-3;
   double tf = 0.1;
   int irk_method = 16;
   bool visualization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&eps, "-e", "--epsilon", "Diffusion coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&irk_method, "-i", "--irk", "ID of IRK method.");
   args.AddOption(&visualization, "-v", "--visualization", "-nov", "--no-visualization",
                  "Use IRK solver.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
      return 1;
   }
   if (root) { args.PrintOptions(std::cout); }

   Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
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
   mesh.GetCharacteristics(hmin, hmax, kmin, kmax);
   if (root)
   {
      std::cout << "dt = " << dt << ", hmin = " << hmin << ", hmax = " << hmax << "\n";
      std::cout << "time order = " << irk_method%10 << ", space order = " << order << "\n";
      std::cout << "time acc. = " << std::pow(dt,(irk_method%10))
                << ", space acc. = " << std::pow(hmax,order) << "\n";
   }

   H1_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParGridFunction u(&fes);
   FunctionCoefficient ic_coeff(ic_fn);
   u.ProjectCoefficient(ic_coeff);

   ParaViewDataCollection dc("Heat", &mesh);
   if (visualization) {
      dc.SetPrefixPath("ParaView");
      dc.RegisterField("u", &u);
      dc.SetLevelsOfDetail(order);
      dc.SetDataFormat(VTKFormat::BINARY);
      dc.SetHighOrderOutput(true);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   HeatIRKOperator heat(fes, eps);

   double t = 0.0;

   std::unique_ptr<ODESolver> ode;

   RKData::Type irk_type = RKData::Type(irk_method);
   // Can adjust rel tol, abs tol, maxiter, kdim, etc.
   IRK::KrylovParams krylov_params;
   krylov_params.printlevel = 2;
   krylov_params.kdim = 50;
   krylov_params.maxiter = 200;
   krylov_params.solver = IRK::KrylovMethod::CG;
   // Build IRK object using spatial discretization
   IRK *irk = new IRK(&heat, irk_type);
   // Set GMRES settings
   irk->SetKrylovParams(krylov_params);
   // Initialize solver
   irk->Init(heat);
   ode.reset(irk);

   int vis_steps = 100;
   double t_vis = t;
   double vis_int = (tf-t)/double(vis_steps);

   // Vector sizes
   if (root)
   {
      std::cout << "Number of unknowns/proc: " << fes.GetVSize() << std::endl;
      std::cout << "Total number of unknowns: " << fes.GlobalVSize() << std::endl;
   }

   bool done = false;
   while (!done)
   {
      double dt_real = min(dt, tf - t);
      ode->Step(u, t, dt_real);
      done = (t >= tf - 1e-8*dt);
      if (t - t_vis > vis_int || done)
      {
         if (root) { printf("t = %4.3f\n", t); }
         if (visualization)
         {
            t_vis = t;
            dc.SetCycle(dc.GetCycle()+1);
            dc.SetTime(t);
            dc.Save();
         }
      }
   }

   return 0;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int retval = run_heat(argc, argv);
   MPI_Finalize();
   return retval;
}
