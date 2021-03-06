#include "mfem.hpp"
#include "fem/picojson.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cmath>

#include "as.hpp"
#include "IRK.hpp"
#include "mesh_util.hpp"

using namespace std;
using namespace mfem;

int n_applications = 0;

double u_ex_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   double r2 = pow(x-0.25,2) + pow(y-0.25,2);
   // return sin(2*M_PI*x)*cos(2*M_PI*y)*exp(-t);
   return sin(2*M_PI*x)*cos(2*M_PI*y)*(2 + sin(t));
}

double f_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   // return 8*M_PI*M_PI*sin(2*M_PI*x)*cos(2*M_PI*y);
   // return exp(-t)*(8*M_PI*M_PI - 1)*sin(2*M_PI*x)*cos(2*M_PI*y);
   return cos(2*M_PI*y)*(cos(t) + 8*pow(M_PI,2)*(2 + sin(t)))*sin(2*M_PI*x);
}

struct BackwardEulerPreconditioner : Solver
{
   ConstantCoefficient mass_coeff, diff_coeff;
   HighOrderASPreconditioner prec;

   BackwardEulerPreconditioner(ParFiniteElementSpace &fes,
                               double gamma,
                               double dt_times_eps,
                               Array<int> &ess_bdr)
   : Solver(fes.GetTrueVSize()),
     mass_coeff(gamma),
     diff_coeff(-dt_times_eps),
     prec(fes, mass_coeff, diff_coeff, ess_bdr, 1)
   { }

   void Mult(const Vector &b, Vector &x) const
   {
      prec.Mult(b, x);
      ++n_applications;
   }

   void SetOperator(const Operator &op) { }
};

struct HeatOperator : Operator
{
   OperatorHandle &A, &M;
   double gamma, alpha;
   mutable Vector z;
   HeatOperator(OperatorHandle &A_, OperatorHandle &M_) : A(A_), M(M_) { }
   virtual void Mult(const Vector &x, Vector &y) const override
   {
      z.SetSize(x.Size());
      A->Mult(x, y);
      y *= -alpha;
      M->Mult(x, z);
      z *= gamma;
      y += z;
   }
   void SetSize(int n)
   {
      width = n;
      height = n;
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

   FunctionCoefficient f_coeff;
   ParLinearForm b;
   Vector B;

   HeatOperator oper;

   bool use_inner_cg;
   CGSolver inner_cg;

   mutable Vector z;

public:
   HeatIRKOperator(
      ParFiniteElementSpace &fes_,
      double eps,
      bool use_inner_cg_ = false)
   : IRKOperator(fes_.GetComm(), fes_.GetTrueVSize(), 0.0, IMPLICIT),
      fes(fes_),
      diff_coeff(-eps),
      a(&fes),
      m(&fes),
      irs(0, Quadrature1D::GaussLobatto),
      oper(A, M),
      f_coeff(f_fn),
      b(&fes),
      use_inner_cg(use_inner_cg_)
   {
      a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
      a.Assemble();

      const Mesh &mesh = *fes.GetMesh();
      if (mesh.bdr_attributes.Size())
      {
         ess_bdr.SetSize(mesh.bdr_attributes.Max());
         // ess_bdr = 1;
         ess_bdr = 0;
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

      b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
      b.Assemble();
      B.SetSize(fes.GetTrueVSize());
      b.ParallelAssemble(B);

      oper.SetSize(A->Width());

      current_prec = nullptr;

      inner_cg.SetRelTol(1e-3);
      inner_cg.SetAbsTol(1e-3);
      inner_cg.SetMaxIter(20);
      inner_cg.SetPrintLevel(0);
      inner_cg.SetOperator(oper);
   }

   virtual void SetTime(const double t_) override
   {
      t = t_;
      f_coeff.SetTime(t);
      b.Assemble();
      b.ParallelAssemble(B);
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
      if (use_inner_cg)
      {
         inner_cg.Mult(x, y);
      }
      else
      {
         current_prec->Mult(x, y);
      }
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

      oper.gamma = gamma;
      oper.alpha = dt;
      inner_cg.SetPreconditioner(*current_prec);
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
   int ref_levels = 3;
   int order = 1;
   double eps = 1.0;
   double dt = 1e-2;
   double tf = 0.1;
   int irk_method = 16;
   int mag_prec = 0;
   bool visualization = true;
   bool use_inner_cg = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine-serial",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&eps, "-e", "--epsilon", "Diffusion coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&irk_method, "-i", "--irk", "ID of IRK method.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                  "Enable ParaView visualization");
   args.AddOption(&use_inner_cg, "-cg", "--use-cg", "-no-cg", "--dont-use-cg",
                  "Use inner CG iteration");
   args.AddOption(&mag_prec, "-mag", "--mag-prec",
                  "0 -> gamma = eta, 1 -> gamma = sqrt(eta^2+beta^2).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
      return 1;
   }
   if (root) { args.PrintOptions(std::cout); }

   // Mesh *serial_mesh = new Mesh(mesh_file, 1, 1);
   Mesh *serial_mesh = new Mesh(3, 3, Element::QUADRILATERAL, false, 1.0, 1.0, false);
   int dim = serial_mesh->Dimension();
   if (ref_levels < 0)
   {
      ref_levels = (int)floor(log(50000./serial_mesh->GetNE())/log(2.)/dim);
   }
   for (int l = 0; l < ref_levels; l++)
   {
      serial_mesh->UniformRefinement();
   }

   Mesh *periodic_mesh;
   {
      Vector tx(2), ty(2);
      tx = 0.0;
      ty = 0.0;
      tx[0] = 1.0;
      ty[1] = 1.0;
      std::vector<Vector> translations = {tx, ty};
      Array<int> v2v;
      IdentifyPeriodicMeshVertices(*serial_mesh,
                                   translations,
                                   v2v,
                                   0);
      periodic_mesh = MakePeriodicMesh(serial_mesh, v2v, 0);
      periodic_mesh->RemoveInternalBoundaries();
      delete serial_mesh;
   }

   ParMesh mesh(MPI_COMM_WORLD, *periodic_mesh);
   delete periodic_mesh;

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

   Vector u;
   ParGridFunction u_gf(&fes);
   FunctionCoefficient u_ex_coeff(u_ex_fn);
   u_gf.ProjectCoefficient(u_ex_coeff);
   u_gf.GetTrueDofs(u);

   ParaViewDataCollection dc("Heat", &mesh);
   if (visualization)
   {
      dc.SetPrefixPath("ParaView");
      dc.RegisterField("u", &u_gf);
      dc.SetLevelsOfDetail(order);
      dc.SetDataFormat(VTKFormat::BINARY);
      dc.SetHighOrderOutput(true);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   HeatIRKOperator heat(fes, eps, use_inner_cg);

   double t = 0.0;

   RKData::Type irk_type = RKData::Type(irk_method);
   // Can adjust rel tol, abs tol, maxiter, kdim, etc.
   IRK::KrylovParams krylov_params;
   krylov_params.printlevel = 1;
   krylov_params.kdim = 50;
   krylov_params.maxiter = 2000;
   krylov_params.reltol = 1e-16;
   krylov_params.abstol = 1e-16;
   // krylov_params.abstol = 0.0;
   // If we use an inner CG iteration, have to use outer FGRMRES (technically
   // FCG would work, but don't have MFEM implementation). If not, we can use
   // CG as outer iteration.
   krylov_params.solver = (use_inner_cg) ? IRK::KrylovMethod::FGMRES : IRK::KrylovMethod::CG;
   // Build IRK object using spatial discretization
   IRK irk(&heat, irk_type, mag_prec);
   // Set GMRES settings
   irk.SetKrylovParams(krylov_params);
   // Initialize solver
   irk.Init(heat);

   int vis_steps = 100;
   double t_vis = t;
   double vis_int = (tf-t)/double(vis_steps);

   // Vector sizes
   if (root)
   {
      std::cout << "Number of unknowns/proc: " << fes.GetVSize() << std::endl;
      std::cout << "Total number of unknowns: " << fes.GlobalVSize() << std::endl;
   }

   int nsteps = 0;
   // bool done = false;
   while (t < tf - 1e-8*dt)
   {
      ++nsteps;
      double dt_real = min(dt, tf - t);
      irk.Step(u, t, dt_real);
      bool done = (t >= tf - 1e-8*dt);
      if (t - t_vis > vis_int || done)
      {
         if (root) { printf("t = %20.16f\n", t); }
         if (visualization)
         {
            u_gf.SetFromTrueDofs(u);
            t_vis = t;
            dc.SetCycle(dc.GetCycle()+1);
            dc.SetTime(t);
            dc.Save();
         }
      }
   }

   u_ex_coeff.SetTime(t);
   double err = u_gf.ComputeL2Error(u_ex_coeff);

   ParGridFunction u_ex(&fes);
   u_ex.ProjectCoefficient(u_ex_coeff);
   u_gf -= u_ex;

   dc.SetCycle(dc.GetCycle()+1);
   dc.SetTime(dc.GetTime()+1);
   dc.Save();

   u_gf.ProjectCoefficient(u_ex_coeff);

   dc.SetCycle(dc.GetCycle()+1);
   dc.SetTime(dc.GetTime()+1);
   dc.Save();

   if (root)
   {
      printf("L2 error: %20.16e\n", err);

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

      std::vector<int> avg_iter, type;
      std::vector<double> eig_ratio;
      irk.GetSolveStats(avg_iter, type, eig_ratio);

      std::map<std::string,picojson::value> dict;
      dict.emplace("irk_method", irk_names[irk_method]);
      dict.emplace("ref", double(ref_levels));
      dict.emplace("dt", dt);
      dict.emplace("order", double(order));
      dict.emplace("mag_prec", double(mag_prec));
      dict.emplace("num_systems", double(avg_iter.size()));
      dict.emplace("inner_cg", use_inner_cg);
      dict.emplace("total_prec_applications", double(n_applications));
      dict.emplace("error", err);

      std::vector<picojson::value> sys_info;
      for (int i=0; i<avg_iter.size(); ++i)
      {
         std::map<std::string,picojson::value> sys;
         sys.emplace("avg_iter", double(avg_iter[i])/nsteps);
         sys.emplace("degree", double(type[i]));
         sys.emplace("eig_ratio", eig_ratio[i]);
         sys_info.emplace_back(sys);
      }
      dict.emplace("sys_info", sys_info);

      std::ofstream f("results/" + irk_names[irk_method] + ".json");
      f << picojson::value(dict);
      f << std::endl;

      std::cout << "TOTAL PREC APPLICATIONS: " << n_applications << '\n';
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
