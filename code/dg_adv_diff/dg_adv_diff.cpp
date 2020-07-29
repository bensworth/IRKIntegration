#include "mfem.hpp"
#include "IRK.hpp"

using namespace mfem;

double eps = 1e-2;


// COMMENTS
//    + Using multiple AIR iterations helpful here
//    srun -n48 ./dg_adv_diff  -r 6 -i 18 -o 2 -dt 0.005 -air 1 -tf 0.05 --> 86,70 iters
//    srun -n48 ./dg_adv_diff  -r 6 -i 18 -o 2 -dt 0.005 -air 2 -tf 0.05 --> 30,17 iters
//    srun -n48 ./dg_adv_diff  -r 6 -i 18 -o 2 -dt 0.005 -air 3 -tf 0.05 --> 23,11 iters

// Matlab plot of velocity field
// [xs,ys] = meshgrid(0:0.05:1, 0:0.05:1);
// quiver(xs,ys,sin(ys*4*pi),cos(xs*2*pi))

double ic_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
#if 0
   double r0 = pow(x-0.25,2) + pow(y-0.25,2);
   double r1 = pow(x-0.75,2) + pow(y-0.25,2);
   double r2 = pow(x-0.25,2) + pow(y-0.75,2);
   double r3 = pow(x-0.75,2) + pow(y-0.75,2);
   return exp(-100*r0) + exp(-100*r1) + exp(-100*r2) + exp(-100*r3);
#else
   double r0 = pow(x-0.25,2) + pow(y-0.25,2);
   double r1 = pow(x-0.5,2) + pow(y-0.75,2);
   double r2 = pow(x-0.75,2) + pow(y-0.5,2);
   return exp(-1*r0) + exp(-1*r1) + exp(-1*r2);
#endif
}

void v_fn(const Vector &xvec, Vector &v)
{
   double x = xvec[0];
   double y = xvec[1];
   // v(0) = 2*M_PI*(0.5 - y);
   // v(1) = 2*M_PI*(x - 0.5);
   v(0) = sin(y*4*M_PI);
   v(1) = cos(x*2*M_PI);
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
   HypreParMatrix B_s;
   HypreBoomerAMG *AMG_solver;
   HypreGMRES *GMRES_solver;
   BlockILU *prec;
   int blocksize;
   bool use_gmres;
public:
   BackwardEulerPreconditioner(ParFiniteElementSpace &fes, double gamma,
                               HypreParMatrix &M, double dt, HypreParMatrix &A)
      : Solver(fes.GetTrueVSize()), B(A), AMG_solver(NULL), use_gmres(false)
   {
      // Set B = gamma*M - dt*A
      B *= -dt;
      SparseMatrix B_diag, M_diag;
      B.GetDiag(B_diag);
      M.GetDiag(M_diag);
      // M is block diagonal, so suffices to only add the processor-local part
      B_diag.Add(gamma, M_diag);
      prec = new BlockILU(B);
   }
   BackwardEulerPreconditioner(ParFiniteElementSpace &fes, double gamma,
                               HypreParMatrix &M, double dt, HypreParMatrix &A,
                               int AMG_iter, bool use_gmres_)
      : Solver(fes.GetTrueVSize()), B(A), prec(NULL), use_gmres(use_gmres_)
   {
      // Set B = gamma*M - dt*A
      B *= -dt;
      SparseMatrix B_diag, M_diag;
      B.GetDiag(B_diag);
      M.GetDiag(M_diag);
      // M is block diagonal, so suffices to only add the processor-local part
      B_diag.Add(gamma, M_diag);

      int order = fes.GetOrder(0);
      blocksize = (order+1)*(order+1);

      // BUild AMG solver
      if (blocksize > 0) {
         BlockInverseScale(&B, &B_s, NULL, NULL, blocksize, 0);
         AMG_solver = new HypreBoomerAMG(B_s);
      }
      else {
         AMG_solver = new HypreBoomerAMG(B);
      }
      AMG_solver->SetMaxLevels(50); 
      if (eps > 1e-4) {
         AMG_solver->SetLAIROptions(1.5, "", "FA", 0.1, 0.01, 0.0,
                                    0, 3, 0.0, 6, -1, 1);
      }
      else {
         AMG_solver->SetLAIROptions(1.5, "", "FA", 0.1, 0.01, 0.0,
                                    100, 3, 0.0, 6, -1, 1);
         // AMG_solver->SetLAIROptions(1.5, "", "FFC", 0.1, 0.01, 0.0,
         //                           100, 3, 0.0, 6, -1, 1);
      }
      AMG_solver->SetTol(1e-12);
      if (use_gmres) {
         if (blocksize > 0) GMRES_solver = new HypreGMRES(B_s);
         else GMRES_solver = new HypreGMRES(B);
         GMRES_solver->SetMaxIter(AMG_iter);
         GMRES_solver->SetTol(1e-12);
         GMRES_solver->SetPreconditioner(*AMG_solver);
         AMG_solver->SetMaxIter(1);
      }
      else {
         GMRES_solver = NULL;
         AMG_solver->SetTol(1e-12);
         AMG_solver->SetMaxIter(AMG_iter);
      }
      AMG_solver->SetPrintLevel(0);
   }
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      if (AMG_solver) {
         if (blocksize > 0) {
            HypreParVector b_s;
            BlockInverseScale(&B, NULL, &x, &b_s, blocksize, 2);
            if (use_gmres) GMRES_solver->Mult(b_s, y);
            else AMG_solver->Mult(b_s, y);
         }
         else {
            if (use_gmres) GMRES_solver->Mult(x, y);
            else AMG_solver->Mult(x, y);
         }
      }
      else {
         prec->Mult(x, y);
      }
   }
   void SetOperator(const Operator &op) { }
   
   ~BackwardEulerPreconditioner()
   {
      if (AMG_solver) delete AMG_solver;
      if (GMRES_solver) delete GMRES_solver;
      if (prec) delete prec;
   }
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
   int use_AIR;
   bool use_gmres;
public:
   DGAdvDiff(ParFiniteElementSpace &fes_, int use_AIR_= 1,
      bool use_gmres_ = false)
      : IRKOperator(fes_.GetComm(), fes_.GetTrueVSize(), 0.0, IMPLICIT),
        fes(fes_),
        v_coeff(fes.GetMesh()->Dimension(), v_fn),
        diff_coeff(-eps),
        a(&fes),
        m(&fes),
        mass(fes),
        use_AIR(use_AIR_),
        use_gmres(use_gmres_)
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
         if (use_AIR > 0) {
            prec[key] = new BackwardEulerPreconditioner(fes, gamma, *M_mat, dt,
                                                        *A_mat, use_AIR, use_gmres);
         }
         else {
            prec[key] = new BackwardEulerPreconditioner(fes, gamma, *M_mat, dt, *A_mat);
         }
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
   int ser_ref_levels = 3;
   int par_ref_levels = 2;
   int order = 1;
   double kappa = -1.0;
   double dt = 1e-3;
   double tf = 0.1;
   int use_irk = 16;
   int use_AIR = 1;
   // bool use_ilu = true;
   int nsubiter = 1;
   bool visualization = false;
   int vis_steps = 200;
   int use_gmres = false;
   int maxiter = 250;
   int mag_prec = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&eps, "-e", "--epsilon", "Diffusion coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&use_irk, "-i", "--irk", "Use IRK solver (provide id; if 0, no IRK).");
   args.AddOption(&use_AIR, "-air", "--use-air", "Use AIR: > 0 implies # AIR iterations, 0 = Block ILU.");
   args.AddOption(&visualization, "-v", "--visualization", "-nov", "--no-visualization",
                  "Use IRK solver.");
   args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "-1=FGMRES/fixed-point, 0=GMRES/fixed-point, 1=FGMRES/GMRES.");
   args.AddOption(&mag_prec, "-mag", "--mag-prec",
                  "0 -> gamma = eta, 1 -> gamma = sqrt(eta^2+beta^2).");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
      MPI_Finalize();
      return 1;
   }
   if (kappa < 0)
   {
      kappa = (order+1)*(order+1);
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
   mesh.GetCharacteristics (hmin, hmax, kmin, kmax);
   if (root)
   {
      std::cout << "dt = " << dt << ", hmin = " << hmin << ", hmax = " << hmax << "\n";
      std::cout << "time order = " << use_irk%10 << ", space order = " << order << "\n";
      std::cout << "time acc. = " << std::pow(dt,(use_irk%10))
                << ", space acc. = " << std::pow(hmax,order) << "\n";
   }

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParGridFunction u(&fes);
   FunctionCoefficient ic_coeff(ic_fn);
   u.ProjectCoefficient(ic_coeff);

   ParaViewDataCollection dc("DGAdvDiff", &mesh);
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

   bool temp_dg;
   if (use_gmres==1) {
      temp_dg = true;
   }
   else {
      temp_dg = false;
   }
   DGAdvDiff dg(fes, use_AIR, temp_dg);

   double t = 0.0;

   std::unique_ptr<ODESolver> ode;

   if (use_irk != 0)
   {
      RKData::Type irk_type = static_cast<RKData::Type>(use_irk); // RKData::LSDIRK3;
      // RKData::Type irk_type = RKData::LSDIRK1;
      // Can adjust rel tol, abs tol, maxiter, kdim, etc.
      IRK::KrylovParams krylov_params;
      krylov_params.printlevel = 2;
      krylov_params.kdim = 50;
      krylov_params.maxiter = maxiter;
      // krylov_params.abstol = 1e-12;
      krylov_params.reltol = 1e-12;
      if (use_gmres==-1 || use_gmres==1) krylov_params.solver = IRK::KrylovMethod::FGMRES;
      else if(use_gmres == 2) krylov_params.solver = IRK::KrylovMethod::FP;
      else krylov_params.solver = IRK::KrylovMethod::GMRES;

      // Build IRK object using spatial discretization
      IRK *irk = new IRK(&dg, irk_type, mag_prec);
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
         if (myid == 0) printf("t = %4.3f\n", t);
         if (root) { printf("t = %4.3f\n", t); }
         if (visualization) {
            t_vis = t;
            dc.SetCycle(dc.GetCycle()+1);
            dc.SetTime(t);
            dc.Save();
         }
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return 0;
}
