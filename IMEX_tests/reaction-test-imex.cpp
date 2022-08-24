/* --------------------------------------------------------------- */
/* --------------------------------------------------------------- */
/* ---------------------------------------------------------------
  DEBUGGING/TODO:
  - Print matrices to file for Tommaso to test RK on. Probably print
  from other code.

  - Merge IMEX codes to one framework.



   TESTS:
   - RK: mM_PIrun -n 4 ./imex-dg-adv-diff -dt 0.01 -tf 2 -rs 3 -o 3 -e 10 -imex 222
     Poly: mM_PIrun -n 4 ./imex-dg-adv-diff -dt 0.2 -tf 2 -rs 3 -o 3 -e 10 -irk 123 -i 1
   -  srun -n 40 ./imex-dg-adv-diff -dt 0.025 -tf 5 -rs 4 -rp 1 -o 3 -e 1 -imex 1013
         Converge, not great accuracy
      srun -n 40 ./imex-dg-adv-diff -dt 0.025 -tf 5 -rs 4 -rp 1 -o 3 -e 1 -imex 1013 -a 2
         Converge nicely
      srun -n 40 ./imex-dg-adv-diff -dt 0.025 -tf 5 -rs 4 -rp 1 -o 3 -e 1 -imex 1013 -a 3
         Diverge
      srun -n 40 ./imex-dg-adv-diff -dt 0.025 -tf 5 -rs 4 -rp 1 -o 3 -e 1 -imex -43
         Poor accuracy
      srun -n 40 ./imex-dg-adv-diff -dt 0.025 -tf 5 -rs 4 -rp 1 -o 3 -e 1 -irk 123 -i 0
         Nice accuracy

   --------------------------------------------------------------- */
/* --------------------------------------------------------------- */
/* --------------------------------------------------------------- */

#include "mfem.hpp"
#include "IRK.hpp"
#include "IMEX_utils.hpp"

using namespace mfem;

bool root;
double diff_const = 0;   // Diffusion coeffient
double react_const = 1;    // reaction coefficient
double adv_const = 0;    // reaction coefficient

// Matlab plot of velocity field
// [xs,ys] = meshgrid(0:0.05:1, 0:0.05:1);
// quiver(xs,ys,sin(ys*4*M_PI),cos(xs*2*M_PI))

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
   v(0) = adv_const;
   v(1) = adv_const;
}

// Reaction term: \gamma * u ( 1 - u) (u - 1/2)
// Forcing function for u_t = e*\Delta u - [1,1]\cdot \nabla u+  f.
// for solution u* = sin(2*pi*x(1-y)(1+2t))sin(2*pi*y(1-x)(1+2t))
double force_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   // Definitely correct for no reaction term
   // Supposed reaction solution, does not work
   double v = 4.0*(M_PI+2.0*M_PI*t)*(M_PI+2.0*M_PI*t)*diff_const*(cos(2.0*M_PI*(1+2.0*t)*(x+(-1)*y))+((-1)+ \
       2.0*x+(-2)*x*x+2.0*y+(-2)*y*y)*cos(2.0*M_PI*(1+2.0*t)*((-1)*y+ \
       x*((-1)+2.0*y))))+4.0*M_PI*((-1)+x)*y*cos(2.0*M_PI*(1+2.0*t)*((-1)+ \
       x)*y)*sin(2.0*M_PI*(1+2.0*t)*x*((-1)+y))+4.0*M_PI*x*((-1)+y)*cos( \
       2.0*M_PI*(1+2.0*t)*x*((-1)+y))*sin(2.0*M_PI*(1+2.0*t)*((-1)+x)*y)+( \
       -1)*react_const*sin(2.0*M_PI*(1+2.0*t)*x*((-1)+y))*sin(2.0*M_PI*(1+2.0*t)*(( \
       -1)+x)*y)*(1+(-1)*sin(2.0*M_PI*(1+2.0*t)*x*((-1)+y))*sin(2.0* \
       M_PI*(1+2.0*t)*((-1)+x)*y))*((-1/2)+sin(2.0*M_PI*(1+2.0*t)*x*((-1) \
       +y))*sin(2.0*M_PI*(1+2.0*t)*((-1)+x)*y))+(-2)*M_PI*(1+2.0*t)*((-1) \
       +x+y)*(-adv_const)*sin(2.0*M_PI*(1+2.0*t)*((-1)*y+x*((-1)+2.0*y)));
   return v;
}

// Exact solution (x,y,t)
double sol_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
   return sin(2.0*M_PI*x*(1.0-y)*(1.0+2.0*t))*sin(2.0*M_PI*y*(1.0-x)*(1.0+2.0*t));
}


// Return if error < 1 to note when divergence happens
bool simulate(ODESolver *ode, ParGridFunction &u,
   ParFiniteElementSpace &fes, double tf, double dt, bool myid)
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
   ParGridFunction u_gf(&fes, u);         
   FunctionCoefficient u_ex_coeff(sol_fn);
   u_ex_coeff.SetTime(t);

   double err = u_gf.ComputeL2Error(u_ex_coeff);
   double errmax = u_gf.ComputeMaxError(u_ex_coeff);
   if (myid == 0) {
      std::cout << "l2(t) " << err << "\nmax " << errmax << "\n";
   }

   if (err > 1) return false;
   else return true;
}


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
   ParBilinearForm m;
   mutable FunctionCoefficient forcing_coeff;
   mutable ParLinearForm b_imp;
   mutable ParLinearForm b_exp;
   mutable ParGridFunction reaction;
   DGMassMatrix mass;
   HypreParMatrix *M_mat;
   bool imp_forcing;
public:
   DGAdvDiff(ParFiniteElementSpace &fes_, int use_AMG_= 1,
      bool use_gmres_ = false, bool imp_forcing_=true)
      : IRKOperator(fes_.GetComm(), true, fes_.GetTrueVSize(), 0.0, IMPLICIT),
        fes(fes_),
        v_coeff(fes.GetMesh()->Dimension(), v_fn),
        diff_coeff(-diff_const),
        m(&fes),
        b_imp(&fes),
        b_exp(&fes),
        reaction(&fes),
        mass(fes),
        forcing_coeff(force_fn),
        imp_forcing(imp_forcing_)
   {
      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble(0);
      m.Finalize(0);
      M_mat = m.ParallelAssemble();

      // BCs and forcing function
      // b_imp.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(BC_coeff, diff_coeff, sigma, kappa));
      // b_exp.AddBdrFaceIntegrator(new BoundaryFlowIntegrator(BC_coeff,v_coeff, 1.0));
      if (imp_forcing) b_imp.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
      else b_exp.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- M^{-1} (L y - f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      std::cout << "Why is Mult() being called?? This is probably bad.\n\n";
   }

   // Reaction term: \react_const * u ( 1 - u) (u - 1/2) 
   void SetReaction(const Vector &u) const
   {
      if (reaction.Size() != u.Size()) {
         mfem_error("GetReaction vector sizes do not match!\n");
      }
      for (int i=0; i<reaction.Size(); i++) {
         reaction(i) = react_const*u(i)*(1 - u(i))*(u(i) - 0.5); 
      }
   }

   void ExplicitMult(const Vector &u, Vector &du_dt) const override
   {
      // Set nonlinear reaction term, reassemble explicit bilinear form
      SetReaction(u);

#if 0
      // Add nonlinear reaction term to explicit bilinear form, assemble at time
      ParBilinearForm m_nl(&fes);
      GridFunctionCoefficient apply_r(&reaction);
      m_nl.AddDomainIntegrator(new MassIntegrator(apply_r));
      m_nl.Assemble(0);
      m_nl.Finalize(0);
      HypreParMatrix *NL = m_nl.ParallelAssemble();

      NL->Mult(u, du_dt);
      delete NL;
#else
      mass.Mult(reaction, du_dt);
#endif

      // Add forcing function and BCs
      if (!imp_forcing) {
         forcing_coeff.SetTime(this->GetTime());
         b_exp.Assemble();
         HypreParVector *B = new HypreParVector(*M_mat);
         B = b_exp.ParallelAssemble();
         du_dt += *B;
         delete B;
      }
   }

   void ImplicitMult(const Vector &u, Vector &du_dt) const override
   {
      du_dt.SetSize(u.Size());
      du_dt = 0.0;
      if (imp_forcing) {
         forcing_coeff.SetTime(this->GetTime());
         b_imp.Assemble();
         HypreParVector *B = new HypreParVector(*M_mat);
         B = b_imp.ParallelAssemble();
         du_dt += *B;
         delete B;
      }
   }

   // ImplicitMult without the forcing function
   void ExplicitGradientMult(const Vector &u, Vector &du_dt) const override
   {
      du_dt = 0.0;
   }


   void AddForcing(Vector &rhs, double t, double r, double z)
   {
      // Add forcing function and BCs
      double ti = t + r*z;
      forcing_coeff.SetTime(ti);
      HypreParVector *B = new HypreParVector(*M_mat);
      if (imp_forcing) {
         b_imp.Assemble();
         B = b_imp.ParallelAssemble();
      }
      else {
         b_exp.Assemble();
         B = b_exp.ParallelAssemble();
      }
      rhs.Add(r, *B);
   }

   void AddImplicitForcing(Vector &rhs, double t, double r, double z)
   {
      // Add forcing function and BCs
      if (imp_forcing) {
         double ti = t + r*z;
         forcing_coeff.SetTime(ti);
         HypreParVector *B = new HypreParVector(*M_mat);
         b_imp.Assemble();
         B = b_imp.ParallelAssemble();
         rhs.Add(r, *B);
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
      mfem_error("not implemented\n");
   }

   /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
   void ImplicitPrec(int index, const Vector &x, Vector &y) const override
   {
      mfem_error("not implemented\n");
   }

   /// Solve M*x - dtf(x, t) = b
   virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k) override
   {
      Vector rhs(x.Size());
      rhs = 0.0;
      AddImplicitForcing(rhs, this->GetTime(), 1.0, 0);
      mass.Solve(rhs, k);
   }

   /// Solve M*x - dtf(x, t) = b
   // This is for other formulation of RK in terms of solution rather than stages
   virtual void ImplicitSolve2(const double dt, const Vector &b, Vector &x) override
   {
      Vector rhs(b);
      if (imp_forcing) AddForcing(rhs, this->GetTime(), dt, 0);
      mass.Solve(rhs, x);
   }


   /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L)
           + index -> index of system to solve, [0,s_eff)
           + dt    -> time step size
           + type  -> eigenvalue type, 1 = real, 2 = complex pair
        These additional parameters are to provide ways to track when
        (\gamma*M - dt*L) must be reconstructed or not to minimize setup. */
   virtual void SetPreconditioner(int index, double dt, double gamma, int type) override
   {

   }

   ~DGAdvDiff()
   {
      delete M_mat;
   }
};


int run_adv_diff(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   root = (myid == 0);

   static const double sigma = -1.0;

   std::string mesh_file = MFEM_SOURCE_DIR + std::string("/data/inline-quad.mesh");
   int ser_ref_levels = 1;
   int par_ref_levels = 0;
   int order = 1;
   double kappa = -1.0;
   double tf = 0.1;
   int use_AMG = 1;
   int use_gmres = false;
   int maxiter = 500;
   bool imp_force = false;
   double dt = 1e-2;
   int imex_id = -1;
   int irk_id = -1;
   bool recompute_exp = false;
   bool interpolate = false;
   double alpha = -1;
   int iter = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&react_const, "-eta", "--react_constant", "Reaction coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&imex_id, "-imex", "--imex", "Use IMEX solver (provide id; if 0, Euler-IMEX).");
   args.AddOption(&alpha, "-a", "--alpha", "alpha for IMEX-BDF.");
   args.AddOption(&iter, "-i", "--num-iters",
               "Number applications of iterator, default 1.");

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
   if (root)
   {
      std::cout << "hmin = " << hmin << ", hmax = " << hmax <<
         ", space order = " << order << ", space acc. = "
         << std::pow(hmax,order) << "\n";
   }

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   ParFiniteElementSpace fes(&mesh, &fec);

   ParGridFunction u(&fes);
   FunctionCoefficient ic_coeff(ic_fn);
   u.ProjectCoefficient(ic_coeff);

   // Compute error to exact solution
   double err = u.ComputeL2Error(ic_coeff);
   if (myid == 0) std::cout << "t = " << 0 << ", error = " << err << ", norm = " << u.Norml2() << "\n";

   bool temp_dg;
   if (use_gmres==1) {
      temp_dg = true;
   }
   else {
      temp_dg = false;
   }
   DGAdvDiff dg(fes, use_AMG, temp_dg, imp_force);
   ODESolver *ode = NULL;
   KrylovParams krylov_params;
   krylov_params.printlevel = 0;
   krylov_params.kdim = 10;
   krylov_params.maxiter = maxiter;
   krylov_params.reltol = 1e-12;
   krylov_params.solver = KrylovMethod::GMRES;

   // Vector sizes
   if (root)
   {
      std::cout << "Number of unknowns/proc: " << fes.GetVSize() << std::endl;
      std::cout << "Total number of unknowns: " << fes.GlobalVSize() << std::endl;
   }

   if (imex_id < 1000 && (imex_id > 0 || std::abs(imex_id) > 20) ) {
      if (root) std::cout << "\n -------- RK " << imex_id << " -------- \n";
      IMEXRKData coeffs(static_cast<IMEXRKData::Type>(imex_id));
      IMEXRK imex(coeffs);
      imex.Init(dg);
      ode = &imex;
      u = 0.0;
      u.ProjectCoefficient(ic_coeff);
      simulate(ode, u, fes, tf, dt, myid);
   }
   // IMEX BDF
   else if (imex_id > 1000) {
      if (root) std::cout << "\n -------- BDF " << imex_id << "(" << alpha << ") -------- \n";
      IMEXBDF imex(static_cast<BDFData::Type>(imex_id-1000) , alpha);
      imex.Init(dg);
      ode = &imex;
      u = 0.0;
      u.ProjectCoefficient(ic_coeff);
      simulate(ode, u, fes, tf, dt, myid);
   }

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
