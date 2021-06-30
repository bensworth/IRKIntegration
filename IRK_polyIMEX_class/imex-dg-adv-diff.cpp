/* --------------------------------------------------------------- */
/* --------------------------------------------------------------- */
/* ---------------------------------------------------------------
  DEBUGGING/TODO:
  - Print matrices to file for Tommaso to test RK on. Probably print
  from other code.

  - Merge IMEX codes to one framework.



   TESTS:
   - RK: mpirun -n 4 ./imex-dg-adv-diff -dt 0.01 -tf 2 -rs 3 -o 3 -e 10 -imex 222
     Poly: mpirun -n 4 ./imex-dg-adv-diff -dt 0.2 -tf 2 -rs 3 -o 3 -e 10 -irk 123 -i 1
   - 
   --------------------------------------------------------------- */
/* --------------------------------------------------------------- */
/* --------------------------------------------------------------- */

#include "mfem.hpp"
#include "IRK.hpp"

using namespace mfem;

double eps = 1e-2;   // Diffusion coeffient

// Matlab plot of velocity field
// [xs,ys] = meshgrid(0:0.05:1, 0:0.05:1);
// quiver(xs,ys,sin(ys*4*pi),cos(xs*2*pi))

// Initial condition sin(2pix)sin(2piy) for manufactured solution
// sin(2pi(x-t))sin(2pi(y-t))
double ic_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
   return sin(2.0*M_PI*x*(1.0-y))*sin(2.0*M_PI*(1.0-x)*y);
}

// Velocity field [1,1] for manufactured solution
void v_fn(const Vector &xvec, Vector &v)
{
   v(0) = 1;
   v(1) = 1;
}

// Forcing function for u_t = e*\Delta u - [1,1]\cdot \nabla u+  f.
// for solution u* = sin(2pix(1-y)(1+2t))sin(2piy(1-x)(1+2t))
double force_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   double v = -4.0*M_PI*M_PI*eps*(2.0*t + 1.0)*(2.0*t + 1.0)*(-x*x*sin(2.0*M_PI*x*(2.0*t + 1.0)*(y - 1.0))*\
      sin(2.0*M_PI*y*(2.0*t + 1.0)*(x - 1.0)) + 2.0*x*(x - 1.0)*cos(2.0*M_PI*x*(2.0*t + 1.0)*(y - 1.0))*\
      cos(2.0*M_PI*y*(2.0*t + 1.0)*(x - 1.0)) - (x - 1.0)*(x - 1.0)*sin(2.0*M_PI*x*(2.0*t + 1.0)*(y - 1.0))*\
      sin(2.0*M_PI*y*(2.0*t + 1.0)*(x - 1.0))) - 4.0*M_PI*M_PI*eps*(2.0*t + 1.0)*(2.0*t + 1.0)*\
      (-y*y*sin(2.0*M_PI*x*(2.0*t + 1.0)*(y - 1.0))*sin(2.0*M_PI*y*(2.0*t + 1.0)*(x - 1.0)) +\
      2.0*y*(y - 1.0)*cos(2.0*M_PI*x*(2.0*t + 1.0)*(y - 1.0))*cos(2.0*M_PI*y*(2.0*t + 1.0)*(x - 1.0)) - \
      (y - 1.0)*(y - 1.0)*sin(2.0*M_PI*x*(2.0*t + 1.0)*(y - 1.0))*sin(2.0*M_PI*y*(2.0*t + 1.0)*(x - 1.0))) + \
      4.0*M_PI*x*(1.0 - y)*sin(2.0*M_PI*y*(1.0 - x)*(2.0*t + 1.0))*cos(2.0*M_PI*x*(1.0 - y)*(2.0*t + 1.0)) - \
      2.0*M_PI*x*(2.0*t + 1.0)*sin(2.0*M_PI*y*(1.0 - x)*(2.0*t + 1.0))*cos(2.0*M_PI*x*(1.0 - y)*(2.0*t + 1.0)) + \
      4.0*M_PI*y*(1.0 - x)*sin(2.0*M_PI*x*(1.0 - y)*(2.0*t + 1.0))*cos(2.0*M_PI*y*(1.0 - x)*(2.0*t + 1.0)) - \
      2.0*M_PI*y*(2.0*t + 1.0)*sin(2.0*M_PI*x*(1.0 - y)*(2.0*t + 1.0))*cos(2.0*M_PI*y*(1.0 - x)*(2.0*t + 1.0)) + \
      2.0*M_PI*(1.0 - x)*(2.0*t + 1.0)*sin(2.0*M_PI*x*(1.0 - y)*(2.0*t + 1.0))*cos(2.0*M_PI*y*(1.0 - x)*(2.0*t + 1.0)) + \
      2.0*M_PI*(1.0 - y)*(2.0*t + 1.0)*sin(2.0*M_PI*y*(1.0 - x)*(2.0*t + 1.0))*cos(2.0*M_PI*x*(1.0 - y)*(2.0*t + 1.0));
   return -v;
}

// Exact solution (x,y,t)
double sol_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   return sin(2.0*M_PI*x*(1.0-y)*(1.0+2.0*t))*sin(2.0*M_PI*y*(1.0-x)*(1.0+2.0*t));
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
      : Solver(fes.GetTrueVSize()), B(A), AMG_solver(NULL), use_gmres(false),
      GMRES_solver(NULL)
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
      : Solver(fes.GetTrueVSize()), B(A), prec(NULL), use_gmres(use_gmres_),
      GMRES_solver(NULL)
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

      // Build AMG solver
      AMG_solver = new HypreBoomerAMG(B);
      // AMG_solver->SetAdvectiveOptions(1.5, "", "FA");    // DEBUG
      AMG_solver->SetMaxLevels(50); 
      AMG_solver->SetStrengthThresh(0.2);
      AMG_solver->SetCoarsening(6);
      AMG_solver->SetRelaxType(3);
      AMG_solver->SetInterpolation(0);
      AMG_solver->SetTol(0);
      AMG_solver->SetMaxIter(1);
      AMG_solver->SetPrintLevel(0);
      if (AMG_iter > 1) {
         use_gmres = true;
         if (blocksize > 0) GMRES_solver = new HypreGMRES(B_s);
         else GMRES_solver = new HypreGMRES(B);
         GMRES_solver->SetMaxIter(AMG_iter);
         GMRES_solver->SetTol(0);
         GMRES_solver->SetPreconditioner(*AMG_solver);
      }
   } 
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      if (AMG_solver) {
         if (use_gmres) GMRES_solver->Mult(x, y);
         else AMG_solver->Mult(x, y);
      }
      else {
         prec->Mult(x, y);
      }
   }
   void Solve(const Vector &x, Vector &y)
   {
      y = 0.0;
      GMRES_solver = new HypreGMRES(B);
      GMRES_solver->SetMaxIter(250);
      GMRES_solver->SetTol(1e-12);
      GMRES_solver->SetPreconditioner(*AMG_solver);
      GMRES_solver->Mult(x, y);
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
   ParBilinearForm a_imp, a_exp, m;
   mutable FunctionCoefficient forcing_coeff;
   mutable ParLinearForm b_imp;
   mutable ParLinearForm b_exp;
   DGMassMatrix mass;
   std::map<int, BackwardEulerPreconditioner*> prec;
   std::map<std::pair<double,double>, int> prec_index;
   mutable BackwardEulerPreconditioner *current_prec;
   HypreParMatrix *A_imp, *M_mat, *A_exp;
   int use_AMG;
   bool use_gmres, imp_forcing;
   mutable double t_exp, t_imp, dt_;
   bool fully_implicit;
public:
   DGAdvDiff(ParFiniteElementSpace &fes_, int use_AMG_= 1,
      bool use_gmres_ = false, bool imp_forcing_=true)
      : IRKOperator(fes_.GetComm(), true, fes_.GetTrueVSize(), 0.0, IMPLICIT),
        fes(fes_),
        v_coeff(fes.GetMesh()->Dimension(), v_fn),
        diff_coeff(-eps),
        a_imp(&fes),
        a_exp(&fes),
        m(&fes),
        b_imp(&fes),
        b_exp(&fes),
        mass(fes),
        use_AMG(use_AMG_),
        use_gmres(use_gmres_),
        forcing_coeff(force_fn),
        t_exp(-1),
        t_imp(-1),
        imp_forcing(imp_forcing_),
        fully_implicit(false),
        dt_(-1)
   {
      const int order = fes.GetOrder(0);
      const double sigma = -1.0;
      const double kappa = (order+1)*(order+1);
      // Set up diffusion bilinear form
      a_imp.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      a_imp.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma,
                                                            kappa));
      a_imp.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));
      a_imp.Assemble(0);
      a_imp.Finalize(0);
      A_imp = a_imp.ParallelAssemble();

      // Set up advection bilinear form
      a_exp.AddDomainIntegrator(new ConvectionIntegrator(v_coeff, -1.0));
      a_exp.AddInteriorFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(v_coeff, 1.0, -0.5)));
      a_exp.AddBdrFaceIntegrator(
         new TransposeIntegrator(new DGTraceIntegrator(v_coeff, 1.0, -0.5)));
      a_exp.Assemble(0);
      a_exp.Finalize(0);
      A_exp = a_exp.ParallelAssemble();

      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble(0);
      m.Finalize(0);
      M_mat = m.ParallelAssemble();

      // BCs and forcing function
      // b_imp.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(BC_coeff, diff_coeff, sigma, kappa));
      // b_exp.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(BC_coeff,v_coeff, 1.0));
      if (imp_forcing) b_imp.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
      else b_exp.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));

      current_prec = nullptr;
   }

   void SaveMats()
   {
      M_mat->Print("M_mat.mm");
      A_imp->Print("A_imp.mm");
      A_exp->Print("A_exp.mm");
   }

   void SetImplicit()
   {
      fully_implicit = true;
      imp_forcing = true;
      (*A_imp) += (*A_exp);
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- M^{-1} (L y - f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      std::cout << "Why is Mult() being called?? This is probably bad.\n\n";
   }

   void ExplicitMult(const Vector &u, Vector &du_dt) const override
   {
      if (fully_implicit) {
         du_dt = 0.0;
      }
      else {
         A_exp->Mult(u, du_dt);

         // Add forcing function and BCs
         // TODO : could probably check whether this time needs to be
         //    set or not, but need to coordinate between exp and imp.
         // BC_coeff.SetTime(this->GetTime());
         if (!imp_forcing) {
            forcing_coeff.SetTime(this->GetTime());
            b_exp.Assemble();
            t_exp = this->GetTime();
            HypreParVector *B = new HypreParVector(*A_exp);
            B = b_exp.ParallelAssemble();
            du_dt -= *B;
            delete B;
         }
      }
   }

   void ImplicitMult(const Vector &u, Vector &du_dt) const override
   {
      A_imp->Mult(u, du_dt);

      if (imp_forcing) {
         forcing_coeff.SetTime(this->GetTime());
         b_exp.Assemble();
         t_exp = this->GetTime();
         HypreParVector *B = new HypreParVector(*A_exp);
         B = b_exp.ParallelAssemble();
         du_dt -= *B;
         delete B;
      }
   }

   // ImplicitMult without the forcing function
   void ExplicitGradientMult(const Vector &u, Vector &du_dt) const override
   {
      A_imp->Mult(u, du_dt);
   }


   void AddForcing(Vector &rhs, double t, double r, double z)
   {
      // Add forcing function and BCs
      double ti = t + r*z;
      forcing_coeff.SetTime(ti);
      HypreParVector *B = new HypreParVector(*A_imp);
      b_imp.Assemble();
      B = b_imp.ParallelAssemble();
      rhs.Add(-r, *B);
   }



   void AddImplicitForcing(Vector &rhs, double t, double r, double z)
   {
      // Add forcing function and BCs
      if (imp_forcing) {
         double ti = t + r*z;
         forcing_coeff.SetTime(ti);
         HypreParVector *B = new HypreParVector(*A_imp);
         b_imp.Assemble();
         B = b_imp.ParallelAssemble();
         rhs.Add(-r, *B);
      }
   }

   // Apply action of the mass matrix
   virtual void MassMult(const Vector &x, Vector &y) const override
   {
      mass.Mult(x, y);
   }

   // Apply action of the mass matrix
   virtual void MassInv(const Vector &x, Vector &y) const override
   {
      mass.Solve(x, y);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      HYPRE_ClearAllErrors();
      current_prec->Mult(x, y);
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(int index, const Vector &x, Vector &y) const override
   {
      MFEM_VERIFY(current_prec != NULL, "Must call SetSystem before ImplicitPrec");
      HYPRE_ClearAllErrors();
      prec.at(index)->Mult(x, y);
   }

   /// Solve M*x - dtf(x, t) = b
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
   {
      if (current_prec == NULL || dt != dt_)
      {
         delete current_prec;
         current_prec = new BackwardEulerPreconditioner(fes, 1.0, *M_mat, dt,
                                                     *A_imp, use_AMG, use_gmres);
         dt_ = dt;
      }
      Vector rhs(x.Size());
      A_imp->Mult(x, rhs);
      // AddImplicitForcing(rhs, this->GetTime(), 1.0, 0);
      current_prec->Solve(rhs, k);
   }

   /// Solve M*x - dtf(x, t) = b
   // This is for other formulation of RK in terms of solution rather than stages
   virtual void ImplicitSolve2(const double dt, const Vector &b, Vector &x) override
   {
      if (current_prec == NULL || dt != dt_)
      {
         delete current_prec;
         current_prec = new BackwardEulerPreconditioner(fes, 1.0, *M_mat, dt,
                                                     *A_imp, use_AMG, use_gmres);
         dt_ = dt;
      }
      Vector rhs(b);
      rhs = b; // This shouldnt be necessary
      if (imp_forcing) AddForcing(rhs, this->GetTime(), dt, 0);
      current_prec->Solve(rhs, x);
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
         if (use_AMG > 0) {
            prec[index] = new BackwardEulerPreconditioner(fes, gamma, *M_mat, dt,
                                                        *A_imp, use_AMG, use_gmres);
         }
         else {
            prec[index] = new BackwardEulerPreconditioner(fes, gamma, *M_mat, dt, *A_imp);
         }
      }
      current_prec = prec[index];
   }

   ~DGAdvDiff()
   {
      for (auto &P : prec)
      {
         delete P.second;
      }
      delete A_imp;
      delete A_exp;
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
      linear_solver.SetRelTol(1e-12);
      linear_solver.SetAbsTol(1e-10);
      linear_solver.SetMaxIter(100);
      linear_solver.SetKDim(100);
      linear_solver.SetPrintLevel(2);
   }

   void SetTimeStep(double dt)
   {
      dg.SetPreconditioner(0, dt, 1.0, 0);
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

   // Solve the equation:
   //    u_t = M^{-1}(Ku + b),
   // by solving associated linear system
   //    (M - dt*K) d = K*u + b
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k)
   {
      z.SetSize(x.Size());
      dg.SetTime(this->GetTime());
      dg.ExplicitMult(x, z);
      dg_solver.SetTimeStep(dt);
      dg_solver.Mult(z, k);
   }
};

int run_adv_diff(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   bool root = (myid == 0);

   static const double sigma = -1.0;

   // const char *mesh_file = MFEM_DIR + "data/inline-quad.mesh";
   const char *mesh_file = "/Users/southworth/Software/mfem/data/inline-quad.mesh";
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 1;
   double kappa = -1.0;
   double dt = 1e-2;
   double tf = 0.1;
   int use_irk = 123;
   int use_AMG = 1;
   int nsubiter = 1;
   bool visualization = false;
   int vis_steps = 5;
   int use_gmres = false;
   int maxiter = 500;
   int mag_prec = 1;
   bool compute_err = true;
   int iters = 1;
   bool full_imp = false;
   bool imp_force = true;
   bool save_mats = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&eps, "-e", "--epsilon", "Diffusion coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&use_irk, "-irk", "--irk", "Use IRK solver (provide id; if 0, no IRK).");
   args.AddOption(&use_AMG, "-amg", "--use-amg", "Use AMG: > 0 implies # AMG iterations, 0 = Block ILU.");
   args.AddOption(&visualization, "-v", "--visualization", "-nov", "--no-visualization",
                  "Use visualization.");
   args.AddOption(&save_mats, "-save", "--save-mats", "-nosave", "--no-save-mats",
                  "Save matrices to file.");
   args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "-1=FGMRES/fixed-point, 0=GMRES/fixed-point, 1=FGMRES/GMRES.");
   args.AddOption(&iters, "-i", "--num-iters",
                  "Number applications of iterator, default 1.");
   args.AddOption(&full_imp, "-imp", "--full-imp", "-imex","--imex", \
         "Treat ODE fully implicitly.");
   args.AddOption(&imp_force, "-if", "--imp-forcing","-ef", "--exp-forcing", 
      "Implicit or explicit forcing.");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0) args.PrintUsage(std::cout);
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

   // Compute error to exact solution
   if (compute_err)
   {
      double err = u.ComputeL2Error(ic_coeff);
        if (myid == 0) std::cout << "t = " << 0 << ", error = " << err << ", norm = " << u.Norml2() << "\n";
   }

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
   double t = 0.0;
   DGAdvDiff dg(fes, use_AMG, temp_dg, imp_force);
   FE_Evolution *evol = NULL;
   IRK *irk = NULL;
   RKData* coeffs = NULL;
   std::unique_ptr<ODESolver> ode;

   // Fully implicit propagation
   if (full_imp) dg.SetImplicit();

   if (use_irk > 3)
   {
      coeffs = new RKData(static_cast<RKData::Type>(use_irk));
      KrylovParams krylov_params;
      krylov_params.printlevel = 0;
      krylov_params.kdim = 50;
      krylov_params.maxiter = maxiter;
      krylov_params.reltol = 1e-12;
      if (use_gmres==-1 || use_gmres==1) krylov_params.solver = KrylovMethod::FGMRES;
      else if(use_gmres == 2) krylov_params.solver = KrylovMethod::FP;
      else krylov_params.solver = KrylovMethod::GMRES;

      if (use_irk < 100) {
         dg.SetImplicit();
         irk = new IRK(&dg, *coeffs);
      }
      else {
         irk = new PolyIMEX(&dg, *coeffs, true, iters);
      }
      irk->SetKrylovParams(krylov_params);

      // NewtonParams NEWTON;
      // NEWTON.printlevel = 1;
      // NEWTON.abstol = 1e-4;
      // irk->SetNewtonParams(NEWTON);

      irk->Init(dg);
      ode.reset(irk);
   }
   else
   {
      IMEXEuler *imex = new IMEXEuler();
      imex->Init(dg);
      ode.reset(imex);
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
   StopWatch timer;
   timer.Start();
   while (!done)
   {
      ode->Step(u, t, dt);
      done = (t >= tf - 1e-8*dt);
      if (t - t_vis > vis_int || done)
      {
         // if (root) { printf("t = %4.3f\n", t); }
         if (visualization) {
            t_vis = t;
            dc.SetCycle(dc.GetCycle()+1);
            dc.SetTime(t);
            dc.Save();
         }
      }
   }
   timer.Stop();

   // Compute error to exact solution
   if (compute_err)
   {
      ParGridFunction u_gf(&fes, u);         
      FunctionCoefficient u_ex_coeff(sol_fn);
      u_ex_coeff.SetTime(t);

      double err = u_gf.ComputeL2Error(u_ex_coeff);
      if (myid == 0) std::cout << "t-final " << t << "\nl2(t) " << err <<
         "\nruntime " << timer.RealTime() << "\n\n";
   }

   if (save_mats) {
      dg.SaveMats();
   }

   // if (evol) delete evol;
   // if (irk) delete irk;
   // if (coeffs) delete coeffs;

   return 0;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int retval = run_adv_diff(argc, argv);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return retval;
}
