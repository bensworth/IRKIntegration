#include "mfem.hpp"
#include "IRK.hpp"

using namespace mfem;

bool root;
double eps = 1e-2;   // Diffusion coeffient
double react = 1e-3;   // Reaction coeffient - NOTE : assume this to be positive
double C0 = 10;   // Constant in analytical solution
int which_reaction = 1;
int kdim_ = 10;

// Initial condition for manufactured solution
// 1 / (exp(C0*(x+y-t)) + 1)
double ic_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
   return 1.0 / (std::exp(C0*(x + y)) + 1);
}

// Velocity field [1,1] for manufactured solution
void v_fn(const Vector &xvec, Vector &v)
{
   v(0) = 1;
   v(1) = 1;
}

// Forcing function for u_t = e*\Delta u - [1,1]\cdot \nabla u - f.
// Note, sigsn are set up so assuming forcing function is -. Easiest this way.
double force_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   if (which_reaction == 1) {
      return -( 2*C0*C0*eps*(1-exp(C0*(x+y-t)))*exp(C0*(x+y-t)) - C0*(exp(C0*(x+y-t))+1)*exp(C0*(x+y-t)) -
      (-react)*std::pow(exp(C0*(x+y-t))+1,2) ) / std::pow((exp(C0*(x+y-t)) + 1),3);
   }
   else if (which_reaction == 2) {
      return -( 2*C0*C0*eps*(1-exp(C0*(x+y-t)))*exp(C0*(x+y-t)) -
         (C0*exp(C0*(x+y-t))-react)*(exp(C0*(x+y-t))+1) ) / std::pow((exp(C0*(x+y-t)) + 1),3);
   }
   else if (which_reaction == 3) {
      return -( 2*C0*C0*eps*(1-exp(C0*(x+y-t))) - C0*(exp(C0*(x+y-t))+1) - 
         0.5*(-react)*(1-exp(C0*(x+y-t))) ) * exp(C0*(x+y-t)) / std::pow((exp(C0*(x+y-t)) + 1),3);
   }
}

// Boundary condition function (and exact solution)
double BC_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   return 1.0 / (std::exp(C0*(x + y - t)) + 1);
}

// Exact solution (x,y,t)
double sol_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   return 1.0 / (std::exp(C0*(x + y - t)) + 1);
}

// Nonlinear integrator for reaction term eta*f(u)
class NonlinearReaction : public NonlinearFormIntegrator
{
public:

   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
     const FiniteElement &test_fe, ElementTransformation &Trans)
   {
      int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
      order *= 3;
      return IntRules.Get(trial_fe.GetGeomType(), order);
   }

   virtual void AssembleElementVector (const FiniteElement &el,
      ElementTransformation &Tr, const Vector &elfun, Vector &elvect)
   {
      int nd = el.GetDof();
      Vector shape(nd);
      elvect.SetSize(nd);

      const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el, el, Tr);

      elvect = 0.0;
      for (int i = 0; i < ir->GetNPoints(); i++)
      {
         const IntegrationPoint &ip = ir->IntPoint(i);
         Tr.SetIntPoint(&ip);
         double w = Tr.Weight() * ip.weight;

         el.CalcShape(ip, shape);
         double el_value = shape*elfun;
         // Take inner product with jth test function, eta=reaction coeff
         for (int j=0; j<nd; j++)
         {
            if (which_reaction == 1)         // -eta*u
            {
               elvect[j] += w*shape[j]*(-react)*el_value;
            }
            else if (which_reaction == 2)    // -eta*u^2
            {
               elvect[j] += w*shape[j]*(-react)*el_value*el_value;
            }
            else if (which_reaction == 3)    // -eta*u*(1-u)*(u-0.5)
            {
               elvect[j] += w*shape[j]*(-react)*el_value*(1 - el_value)*(el_value-0.5);
            }
         }
      }
   }
};

class DGMassMatrix
{
   mutable Array<double> M, Minv;
   mutable Array<int> iM_PIv;
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
      iM_PIv.SetSize(vec_offsets[nel]);

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
         LUFactors lu(&Minv[offset], &iM_PIv[vec_offsets[i]]);
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
         LUFactors lu(&Minv[M_offsets[i]], &iM_PIv[v_offset]);
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
   HypreGMRES *GMRES_solver, *GMRES_prec;
   BlockILU *prec;
   int blocksize;
   bool use_gmres;
public:
   BackwardEulerPreconditioner(ParFiniteElementSpace &fes, double gamma,
                               HypreParMatrix &M, double dt, HypreParMatrix &A)
      : Solver(fes.GetTrueVSize()), B(A), AMG_solver(nullptr), use_gmres(false),
      GMRES_solver(nullptr), GMRES_prec(nullptr), prec(nullptr)
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
      : Solver(fes.GetTrueVSize()), B(A), prec(nullptr), use_gmres(use_gmres_),
      GMRES_solver(nullptr), GMRES_prec(nullptr), AMG_solver(nullptr)
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
      AMG_solver->SetMaxLevels(50); 
      AMG_solver->SetTol(0);
      AMG_solver->SetMaxIter(1);
      AMG_solver->SetPrintLevel(0);
#if 1
      AMG_solver->SetStrengthThresh(0.2);
      AMG_solver->SetRelaxType(3);
      // AMG_solver->SetCoarsening(6);
      // AMG_solver->SetInterpolation(0);
#else 
      AMG_solver->SetRelaxType(3);
      AMG_solver->SetAdvectiveOptions();
#endif
      if (AMG_iter > 1) {
         use_gmres = true;
         // if (blocksize > 0) GMRES_prec = new HypreGMRES(B_s);
         // else GMRES_prec = new HypreGMRES(B);
         GMRES_prec = new HypreGMRES(B);
         GMRES_prec->SetMaxIter(AMG_iter);
         GMRES_prec->SetTol(0);
         GMRES_prec->SetPreconditioner(*AMG_solver);
      }
   } 
   void Mult(const Vector &x, Vector &y) const
   {
      y = 0.0;
      if (AMG_solver) {
         if (use_gmres) GMRES_prec->Mult(x, y);
         else AMG_solver->Mult(x, y);
      }
      else {
         prec->Mult(x, y);
      }
   }
   void Solve(const Vector &x, Vector &y)
   {
      y = 0.0;
      if (!GMRES_solver) {
         GMRES_solver = new HypreGMRES(B);
         GMRES_solver->SetPrintLevel(0);
         GMRES_solver->SetMaxIter(250);
         GMRES_solver->SetTol(1e-12);
         GMRES_solver->SetPreconditioner(*AMG_solver);
         GMRES_solver->SetKDim(kdim_);
      }
      GMRES_solver->Mult(x, y);
   }
   void SetOperator(const Operator &op) { }

   ~BackwardEulerPreconditioner()
   {
      if (AMG_solver) delete AMG_solver;
      if (GMRES_solver) delete GMRES_solver;
      if (GMRES_prec) delete GMRES_prec;
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
   mutable ParBilinearForm a_exp;
   ParBilinearForm a_imp, m;
   mutable FunctionCoefficient forcing_coeff;
   mutable FunctionCoefficient BC_coeff;
   mutable ParLinearForm lform;
   mutable ParGridFunction reaction;
   DGMassMatrix mass;
   std::map<int, BackwardEulerPreconditioner*> prec;
   std::map<std::pair<double,double>, int> prec_index;
   mutable BackwardEulerPreconditioner *current_prec;
   HypreParMatrix *A_imp, *M_mat;
   mutable HypreParMatrix *A_exp;
   int use_AMG;
   bool use_gmres;
   mutable double t_exp, t_imp, dt_;
   bool fully_implicit;
public:
   DGAdvDiff(ParFiniteElementSpace &fes_, int use_AMG_= 1,
      bool use_gmres_ = false, bool fully_implicit_=false)
      : IRKOperator(fes_.GetComm(), true, fes_.GetTrueVSize(), 0.0, IMPLICIT),
        fes(fes_),
        v_coeff(fes.GetMesh()->Dimension(), v_fn),
        diff_coeff(-eps),
        a_imp(&fes),
        a_exp(&fes),
        m(&fes),
        lform(&fes),
        BC_coeff(BC_fn),
        reaction(&fes),
        mass(fes),
        use_AMG(use_AMG_),
        use_gmres(use_gmres_),
        forcing_coeff(force_fn),
        t_exp(-1),
        t_imp(-1),
        fully_implicit(fully_implicit_),
        dt_(-1),
        A_imp(nullptr),
        A_exp(nullptr),
        M_mat(nullptr),
        current_prec(nullptr)
   {
      const int order = fes.GetOrder(0);
      const double sigma = -1.0;
      const double kappa = (order+1)*(order+1);
      const double v_alpha = -1.0;

      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble(0);
      m.Finalize(0);
      M_mat = m.ParallelAssemble();

      // Set up convection-diffusion bilinear form
      // Diffusion; diff_coeff < 0 because DiffusionIntegrator(1) = -\Delta
      a_imp.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
      a_imp.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma,
                                                            kappa));
      a_imp.AddBdrFaceIntegrator(new DGDiffusionIntegrator(diff_coeff, sigma, kappa));
      if (fully_implicit) {
         a_imp.AddDomainIntegrator(new ConvectionIntegrator(v_coeff, v_alpha));
         a_imp.AddInteriorFaceIntegrator(
            new NonconservativeDGTraceIntegrator(v_coeff, v_alpha));
         a_imp.AddBdrFaceIntegrator(
            new NonconservativeDGTraceIntegrator(v_coeff, v_alpha));
      }      
      a_imp.Assemble(0);
      a_imp.Finalize(0);
      A_imp = a_imp.ParallelAssemble();

      // Convection
      if (!fully_implicit) {
         a_exp.AddDomainIntegrator(new ConvectionIntegrator(v_coeff, v_alpha));
         a_exp.AddInteriorFaceIntegrator(
            new NonconservativeDGTraceIntegrator(v_coeff, v_alpha));
         a_exp.AddBdrFaceIntegrator(
            new NonconservativeDGTraceIntegrator(v_coeff, v_alpha));
         a_exp.Assemble(0);
         a_exp.Finalize(0);
         A_exp = a_exp.ParallelAssemble();
      }

      // BCs and forcing function
      lform.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
      lform.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(BC_coeff, diff_coeff, sigma, kappa));
      // Explicit BCs still treated as implicit forcing;
      // swap sign on v_alpha because I subtract implicit forcing. I believe
      // this is because DiffusionIntegrator = -\Delta, but we need +\Delta,
      // so for consistency I subtract rhs, which then effects adv BCs.
      lform.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(BC_coeff, v_coeff, -v_alpha));
   }

   void SaveMats()
   {
      M_mat->Print("M_mat.mm");
      A_imp->Print("A_imp.mm");
      A_exp->Print("A_exp.mm");
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- M^{-1} (L y - f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      std::cout << "Why is Mult() being called?? This is probably bad.\n\n";
   }

   void ExplicitMult(const Vector &u, Vector &du_dt) const override
   {
      du_dt.SetSize(u.Size());
      ParNonlinearForm m_nl(&fes);
      m_nl.AddDomainIntegrator(new NonlinearReaction());
      m_nl.Mult(u, du_dt);

      // Add advection term
      if (!fully_implicit) {
         Vector temp(u.Size());
         A_exp->Mult(u, temp);
         du_dt += temp;
      }
   }

   void ImplicitMult(const Vector &u, Vector &du_dt) const override
   {
      A_imp->Mult(u, du_dt);
      AddForcing(du_dt, this->GetTime(), 1.0);
   }

   // ImplicitMult without the forcing function
   void ExplicitGradientMult(const Vector &u, Vector &du_dt) const override
   {
      A_imp->Mult(u, du_dt);
   }

   void AddForcing(Vector &x, double t, double c) const
   {
      forcing_coeff.SetTime(t);
      BC_coeff.SetTime(t);
      lform.Assemble();
      HypreParVector *B;
      B = lform.ParallelAssemble();
      x.Add(-c, *B);
      delete B;
   }

   void AddImplicitForcing(Vector &rhs, double t, double r, double z) override
   {
      // Add forcing function and BCs
      double ti = t + r*z;
      forcing_coeff.SetTime(ti);
      BC_coeff.SetTime(ti);
      lform.Assemble();
      HypreParVector *B;
      B = lform.ParallelAssemble();
      rhs.Add(-r, *B);
      delete B;
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
      // A_imp->Mult(x, rhs);
      // AddImplicitForcing(rhs, this->GetTime(), 1.0, 0);
      ImplicitMult(x, rhs);
      // std::cout << "IMPLICIT SOLVE1\n";
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
      AddForcing(rhs, this->GetTime(), dt);
      // std::cout << "IMPLICIT SOLVE2\n";
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
      if (A_imp) delete A_imp;
      if (A_exp) delete A_exp;
      if (M_mat) delete M_mat;
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

   std::string mesh_file = MFEM_SOURCE_DIR + std::string("/data/inline-quad.mesh");
   // const char *mesh_file = "/Users/southworth/Software/mfem/data/inline-quad.mesh";
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
   bool star_method = true;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&react, "-eta", "--react_constant", "Reaction coefficient.");
   args.AddOption(&eps, "-eps", "--epsilon", "Diffusion coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&use_irk, "-irk", "--irk", "Use IRK solver (provide id; if 0, no IRK).");
   args.AddOption(&star_method, "-star", "--star-method", "-nostar", "--no-star-method",
                  "Bool - use IMEX-Radau* methods vs. IMEX-Radau methods.");
   args.AddOption(&use_AMG, "-amg", "--use-amg", "Use AMG: > 0 implies # AMG iterations, 0 = Block ILU.");
   args.AddOption(&visualization, "-v", "--visualization", "-nov", "--no-visualization",
                  "Use visualization.");
   args.AddOption(&save_mats, "-save", "--save-mats", "-nosave", "--no-save-mats",
                  "Save matrices to file.");
   args.AddOption(&use_gmres, "-gmres", "--use-gmres",
                  "-1=FGMRES/fixed-point, 0=GMRES/fixed-point, 1=FGMRES/GMRES.");
   args.AddOption(&iters, "-i", "--num-iters",
                  "Number applications of iterator, default 1.");
   args.AddOption(&which_reaction, "-r", "--reaction", "Reaction, options 1,2,3.");
   args.AddOption(&full_imp, "-fi", "--fully-implicit","-exp", "--exp-advection", 
      "Treat advection explicit or implicit.");

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
   int t_order;
#if 0
   if (use_irk == 123) {
      if (star_method || iters>0) t_order = 3;
      else t_order = 2;
   }
   else if (use_irk == 124) {
      if (star_method) t_order = 4+iters;
      else t_order = 3+iters;
      if (t_order > 5) t_order = 5;
   }
   else if (use_irk == 125) {
      if (star_method) t_order = 5+iters;
      else t_order = 4+iters;
      if (t_order > 7) t_order = 7;
   } 
#else
   if (use_irk == 123) {
      t_order = 3;
   }
   else if (use_irk == 124) {
      t_order = 4+iters;
   }
   else if (use_irk == 125) {
      t_order = 5+iters;
   } 
#endif
   if (root)
   {
      std::cout << "dt = " << dt << ", hmin = " << hmin << ", hmax = " << hmax << "\n";
      std::cout << "time order = " << t_order << ", space order = " << order << "\n";
      std::cout << "time acc. = " << std::pow(dt,t_order)
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
   if (use_irk < 100) {
      std::cout << "Only PolyIMEX schemes supported in this code.\n";
      return -1;
   }
   DGAdvDiff dg(fes, use_AMG, temp_dg, full_imp);

   // Set solve tolerance equal to max theoretical space-time accuracy / 10
   double tol = std::max(std::pow(dt,t_order),std::pow(hmax,order)) / 1000.0;
   RKData coeffs(static_cast<RKData::Type>(use_irk), star_method);
   KrylovParams krylov_params;
   krylov_params.printlevel = 0;
   krylov_params.kdim = kdim_;
   krylov_params.maxiter = maxiter;
   krylov_params.reltol = tol;
   krylov_params.iterative_mode = true;
   krylov_params.abstol = tol;
   if (use_gmres==-1 || use_gmres==1) krylov_params.solver = KrylovMethod::FGMRES;
   else if(use_gmres == 2) krylov_params.solver = KrylovMethod::FP;
   else krylov_params.solver = KrylovMethod::GMRES;


   PolyIMEX irk(&dg, coeffs, true, iters);
   irk.SetKrylovParams(krylov_params);
   // NewtonParams NEWTON;
   // NEWTON.printlevel = 1;
   // NEWTON.abstol = 1e-4;
   // irk.SetNewtonParams(NEWTON);
   irk.Init(dg);

   double t_vis = t;
   double vis_int = (tf-t)/double(vis_steps);

   // Vector sizes
   if (root)
   {
      std::cout << "Number of unknowns/proc: " << fes.GetVSize() << std::endl;
      std::cout << "Total number of unknowns: " << fes.GlobalVSize() << std::endl;
   
      coeffs.expA0.Print();
   }

   bool done = false;
   StopWatch timer;
   timer.Start();
   while (!done)
   {
      irk.Step(u, t, dt);
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

   return 0;
}

int main(int argc, char **argv)
{
   MPI_Init(&argc, &argv);
   int retval = run_adv_diff(argc, argv);
   // MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return retval;
}
