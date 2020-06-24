#include "mfem.hpp"
#include "IRK.hpp"

using namespace mfem;

double ic_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
   double r2 = pow(x-0.25,2) + pow(y-0.25,2);
   return exp(-100*r2);
}

void v_fn(const Vector &xvec, Vector &v)
{
   double x = xvec[0];
   double y = xvec[1];
   v(0) = 2*M_PI*(0.5 - y);
   v(1) = 2*M_PI*(x - 0.5);
   // v = 1;
}

class DGMassMatrix
{
   mutable Array<double> M, Minv;
   mutable Array<int> ipiv;
   Array<int> vec_offsets, M_offsets;
   int nel;
public:
   DGMassMatrix(ParFiniteElementSpace &fes)
   {
      // Precompute local mass matrix inverses
      nel = fes.GetNE();
      M_offsets.SetSize(nel+1);
      vec_offsets.SetSize(nel+1);
      vec_offsets[0] = 0;
      M_offsets[0] = 0;
      for (int i=0; i<nel; ++i)
      {
         int dof = fes.GetFE(i)->GetDof();
         vec_offsets[i+1] = vec_offsets[i] + dof;
         M_offsets[i+1] = M_offsets[i] + dof*dof;
      }
      // The final "offset" is the total size of all the blocks
      M.SetSize(M_offsets[nel]);
      Minv.SetSize(M_offsets[nel]);
      ipiv.SetSize(vec_offsets[nel]);

      // Assemble the local mass matrices and compute LU factorization
      MassIntegrator mi;
      for (int i=0; i<nel; ++i)
      {
         int offset = M_offsets[i];
         const FiniteElement *fe = fes.GetFE(i);
         ElementTransformation *tr = fes.GetElementTransformation(i);
         IsoparametricTransformation iso_tr;
         int dof = fe->GetDof();
         DenseMatrix Me(&M[offset], dof, dof);
         mi.AssembleElementMatrix(*fe, *tr, Me);
         std::copy(&M[offset], M + M_offsets[i+1], &Minv[offset]);
         LUFactors lu(&Minv[offset], &ipiv[vec_offsets[i]]);
         lu.Factor(dof);
      }
   }

   // y = Mx
   void Mult(const Vector &x, Vector &y) const
   {
      for (int i=0; i<nel; ++i)
      {
         int dof = vec_offsets[i+1] - vec_offsets[i];
         DenseMatrix Me(&M[M_offsets[i]], dof, dof);
         Me.Mult(&x[vec_offsets[i]], &y[vec_offsets[i]]);
      }
   }

   // Solve Mx = b, overwrite b with solution x
   void Solve(Vector &b) const
   {
      for (int i=0; i<nel; ++i)
      {
         int v_offset = vec_offsets[i];
         int dof = vec_offsets[i+1] - vec_offsets[i];
         LUFactors lu(&Minv[M_offsets[i]], &ipiv[v_offset]);
         lu.Solve(dof, 1, &b[v_offset]);
      }
   }

   // Solve Mx = b
   void Solve(const Vector &b, Vector &x) const
   {
      x = b;
      Solve(x);
   }
};

struct BackwardEulerPreconditioner : Solver
{
   HypreParMatrix B;
   BlockILU prec;
public:
   BackwardEulerPreconditioner(ParFiniteElementSpace &fes, double gamma,
                               HypreParMatrix &M, double dt, HypreParMatrix &A)
      : Solver(fes.GetTrueVSize()), prec(fes.GetFE(0)->GetDof()),
        B(A)
   {
      // Set B = gamma*M - dt*A
      B *= -dt;
      SparseMatrix B_diag, M_diag;
      B.GetDiag(B_diag);
      M.GetDiag(M_diag);
      // M is block diagonal, so suffices to only add the processor-local part
      B_diag.Add(gamma, M_diag);

      prec.SetOperator(B);
   }
   void Mult(const Vector &x, Vector &y) const
   {
      prec.Mult(x, y);
   }
   void SetOperator(const Operator &op) { }
};

// Provides the time-dependent RHS of the ODEs after spatially discretizing the
//     PDE,
//         du/dt = L*u + s(t) == - A*u + D*u + s(t).
//     where:
//         A: The advection discretization,
//         D: The diffusion discretization,
//         s: The solution-independent source term discretization.
struct DGAdvDiff : IRKOperator
{
   ParFiniteElementSpace &fes;
   VectorFunctionCoefficient v_coeff;
   ConstantCoefficient diff_coeff;
   ParBilinearForm a, m;
   DGMassMatrix mass;
   std::map<std::pair<double,double>,BackwardEulerPreconditioner*> prec;
   BackwardEulerPreconditioner *current_prec;
   HypreParMatrix *A_mat, *M_mat;
public:
   DGAdvDiff(ParFiniteElementSpace &fes_, double eps)
      : IRKOperator(fes_.GetComm(), fes_.GetTrueVSize(), 0.0, IMPLICIT),
        fes(fes_),
        v_coeff(fes.GetMesh()->Dimension(), v_fn),
        diff_coeff(-eps),
        a(&fes),
        m(&fes),
        mass(fes)
   {
      const int order = fes.GetOrder(0);
      const double sigma = -1.0;
      const double kappa = (order+1)*(order+1);
      // Set up convection-diffusion bilinear form
      // Convection
      a.AddDomainIntegrator(new ConvectionIntegrator(v_coeff, -1.0));
      a.AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(v_coeff, 1.0, -0.5)));
      a.AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(v_coeff, 1.0, -0.5)));
      // Diffusion
      a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma,
                                                            kappa));
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));
      a.Assemble(0);
      a.Finalize(0);
      A_mat = a.ParallelAssemble();

      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble(0);
      m.Finalize(0);
      M_mat = m.ParallelAssemble();

      current_prec = nullptr;
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- M^{-1} L y
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      A_mat->Mult(u, du_dt);
      mass.Solve(du_dt);
   }

   virtual void ApplyL(const Vector &x, Vector &y) const override
   {
      y.SetSize(x.Size());
      A_mat->Mult(x, y);
   }

   void ExplicitMult(const Vector &u, Vector &du_dt) const override
   {
      A_mat->Mult(u, du_dt);
   }

   // Apply action of the mass matrix
   virtual void ImplicitMult(const Vector &x, Vector &y) const override
   {
      mass.Mult(x, y);
   }

   virtual void ApplyMInv(const Vector &x, Vector &y) const override
   {
      mass.Solve(x, y);
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
         prec[key] = new BackwardEulerPreconditioner(fes, gamma, *M_mat, dt, *A_mat);
      }
      current_prec = prec[key];
   }

   ~DGAdvDiff()
   {
      for (auto &P : prec)
      {
         delete P.second;
      }
      delete A_mat;
      delete M_mat;
   }
};

class DG_Solver : public Solver
{
private:
   DGAdvDiff &dg;
   mutable GMRESSolver linear_solver;
public:
   DG_Solver(DGAdvDiff &dg_)
      : dg(dg_),
        linear_solver(dg.fes.GetComm())
   {
      linear_solver.iterative_mode = false;
      linear_solver.SetRelTol(1e-10);
      linear_solver.SetAbsTol(1e-10);
      linear_solver.SetMaxIter(100);
      linear_solver.SetKDim(100);
      linear_solver.SetPrintLevel(2);
   }

   void SetTimeStep(double dt)
   {
      dg.SetSystem(0, dt, 1.0, 0);
      linear_solver.SetPreconditioner(*dg.current_prec);
      linear_solver.SetOperator(dg.current_prec->B);
   }

   void SetOperator(const Operator &op) { }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      linear_solver.Mult(x, y);
   }
};

class FE_Evolution : public TimeDependentOperator
{
private:
   DGAdvDiff &dg;
   DG_Solver dg_solver;
   mutable Vector z;

public:
   FE_Evolution(DGAdvDiff &dg_)
      : TimeDependentOperator(dg_.Height()),
        dg(dg_),
        dg_solver(dg)
   { }

   virtual void Mult(const Vector &x, Vector &y) const
   {
      dg.Mult(x, y);
   }

   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      z.SetSize(x.Size());
      dg.ExplicitMult(x, z);
      dg_solver.SetTimeStep(dt);
      dg_solver.Mult(z, k);
   }
};

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   bool root = (myid == 0);

   static const double sigma = -1.0;

   const char *mesh_file = MFEM_DIR "data/inline-quad.mesh";
   int ref_levels = -1;
   int order = 1;
   double kappa = -1.0;
   double eps = 1e-2;
   double dt = 1e-3;
   double tf = 0.1;
   bool use_irk = true;
   // bool use_ilu = true;
   int nsubiter = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly, -1 for auto.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eps, "-e", "--epsilon", "Diffusion coefficient.");
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
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
   }
   if (root) { args.PrintOptions(std::cout); }

   Mesh serial_mesh(mesh_file, 1, 1);
   int dim = serial_mesh.Dimension();
   if (ref_levels < 0)
   {
      ref_levels = (int)floor(log(50000./serial_mesh.GetNE())/log(2.)/dim);
   }
   for (int l = 0; l < ref_levels; l++)
   {
      serial_mesh.UniformRefinement();
   }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);
   if (root)
   {
      std::cout << "Number of unknowns: " << fes.GetVSize() << std::endl;
   }

   ParGridFunction u(&fes);
   FunctionCoefficient ic_coeff(ic_fn);
   u.ProjectCoefficient(ic_coeff);

   ParaViewDataCollection dc("DGAdvDiff", &mesh);
   dc.SetPrefixPath("ParaView");
   dc.RegisterField("u", &u);
   dc.SetLevelsOfDetail(order);
   dc.SetDataFormat(VTKFormat::BINARY);
   dc.SetHighOrderOutput(true);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   DGAdvDiff dg(fes, eps);

   double t = 0.0;

   std::unique_ptr<ODESolver> ode;

   if (use_irk)
   {
      RKData::Type irk_type = RKData::RadauIIA9; // RKData::LSDIRK3;
      // RKData::Type irk_type = RKData::LSDIRK1;
      // Can adjust rel tol, abs tol, maxiter, kdim, etc.
      IRK::KrylovParams krylov_params;
      krylov_params.printlevel = 2;
      krylov_params.kdim = 100;
      // Build IRK object using spatial discretization
      IRK *irk = new IRK(&dg, irk_type);
      // Set GMRES settings
      irk->SetKrylovParams(krylov_params);
      // Initialize solver
      irk->Init(dg);
      ode.reset(irk);
   }
   else
   {
      // std::unique_ptr<ODESolver> ode(new RK4Solver);
      ode.reset(new SDIRK33Solver);
      // ode.reset(new BackwardEulerSolver);
      FE_Evolution evol(dg);
      evol.SetTime(t);
      ode->Init(evol);
   }

   int vis_steps = 100;
   double t_vis = t;
   double vis_int = (tf-t)/double(vis_steps);

   bool done = false;
   while (!done)
   {
      double dt_real = min(dt, tf - t);
      ode->Step(u, t, dt_real);
      done = (t >= tf - 1e-8*dt);
      if (t - t_vis > vis_int || done)
      {
         if (root) { printf("t = %4.3f\n", t); }
         t_vis = t;
         dc.SetCycle(dc.GetCycle()+1);
         dc.SetTime(t);
         dc.Save();
      }
   }

   MPI_Finalize();
   return 0;
}
