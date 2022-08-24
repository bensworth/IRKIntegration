#include "mfem.hpp"

using namespace mfem;

#define forcing 0

bool root;
double eta = 0.5;    // reaction coefficient, eta

// Initial condition for manufactured solution with solution at
// t=0 given by sin(2pi*x(1-y))sin(2pi*y(1-x))
double ic_fn(const Vector &xvec)
{
   double x = xvec[0];
   double y = xvec[1];
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
double sol_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
#if forcing
   return sin(2.0*M_PI*x*(1.0-y)*(1.0+2.0*t))*sin(2.0*M_PI*y*(1.0-x)*(1.0+2.0*t));
#else
   double u0 = ic_fn(xvec); 
   return u0*std::exp(eta*t) / ( 1.0 + u0*(std::exp(eta*t) - 1.0) );
#endif
}

// Forcing function f for u_t = eta*u(1-u) + f.
// with solution u* = sin(2*pi*x(1-y)(1+2t))sin(2*pi*y(1-x)(1+2t))
double force_fn(const Vector &xvec, double t)
{
   double x = xvec[0];
   double y = xvec[1];
#if forcing
   double s = sol_fn(xvec, t);
   double v = -eta*s*(1.0-s) +
      (4*M_PI*(1-x)*y*cos(2*M_PI*(1+2*t)*(1-x)*y)*sin(2*M_PI*(1+2*t)*x*(1-y))
      + 4*M_PI*x*(1-y)*cos(2*M_PI*(1+2*t)*x*(1-y))*sin(2*M_PI*(1+2*t)*(1-x)*y) );
   return v;
#else
   return 0.0;
#endif
}

// Return if error < 1 to note when divergence happens
bool simulate(ODESolver *ode, ParGridFunction &u,
   ParFiniteElementSpace &fes, double tf, double dt,
   ParaViewDataCollection &dc, int vis_steps)
{
   bool done = false;
   MPI_Barrier(MPI_COMM_WORLD);
   StopWatch timer;
   timer.Start();
   double t = 0;

   double t_vis = t;
   double vis_int = 0;
   if (vis_steps > 0) vis_int = (tf-t)/double(vis_steps);
   while (!done)
   {
      ode->Step(u, t, dt);
      done = (t >= tf - 1e-8*dt);
      if (vis_steps > 0 && (t - t_vis > vis_int || done) )
      {
         t_vis = t;
         dc.SetCycle(dc.GetCycle()+1);
         dc.SetTime(t);
         dc.Save();
      }
   }
   timer.Stop();

   // Compute error to exact solution
   ParGridFunction u_gf(&fes, u);
   FunctionCoefficient u_ex_coeff(sol_fn);
   u_ex_coeff.SetTime(t);

   double err = u_gf.ComputeL2Error(u_ex_coeff);
   double errmax = u_gf.ComputeMaxError(u_ex_coeff);
   if (root) {
      std::cout << "l2(t) " << err << "\nmax " << errmax << "\n";
   }

   if (err > 1) return false;
   else return true;
}


// Nonlinear integrator for reaction term eta*u*(1-u)
class NonlinearReaction : public NonlinearFormIntegrator
{
public:

   const IntegrationRule &GetRule(const FiniteElement &trial_fe,
     const FiniteElement &test_fe, ElementTransformation &Trans)
   {
      const int order = trial_fe.GetOrder() + test_fe.GetOrder() + Trans.OrderW();
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
            elvect[j] += w*shape[j]*eta*el_value*(1 - el_value);
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


struct Reaction : TimeDependentOperator
{
   ParFiniteElementSpace &fes;
   ParBilinearForm m;
   mutable FunctionCoefficient forcing_coeff;
   mutable ParLinearForm b_exp;
   mutable ParGridFunction reaction;
   DGMassMatrix mass;
   HypreParMatrix *M_mat;
public:
   Reaction(ParFiniteElementSpace &fes_)
      : TimeDependentOperator(fes_.GetTrueVSize()),
        fes(fes_),
        m(&fes),
        b_exp(&fes),
        reaction(&fes),
        mass(fes),
        forcing_coeff(force_fn)
   {
      m.AddDomainIntegrator(new MassIntegrator);
      m.Assemble(0);
      m.Finalize(0);
      M_mat = m.ParallelAssemble();

      // BCs and forcing function
      b_exp.AddDomainIntegrator(new DomainLFIntegrator(forcing_coeff));
   }

   // Compute the right-hand side of the ODE system.
   // du_dt <- M^{-1} (N(u) + f)
   virtual void Mult(const Vector &u, Vector &du_dt) const override
   {
      du_dt.SetSize(u.Size());
      du_dt = 0.0;

      // Add nonlinear reaction term
      ParNonlinearForm m_nl(&fes);
      m_nl.AddDomainIntegrator(new NonlinearReaction());
      m_nl.Mult(u, du_dt);

      // Add forcing function and BCs
      forcing_coeff.SetTime(this->GetTime());
      b_exp.Assemble();
      HypreParVector *B = new HypreParVector(*M_mat);
      B = b_exp.ParallelAssemble();
      (*B) += du_dt;
      mass.Solve(*B, du_dt);
      delete B;
   }

   ~Reaction()
   {
      delete M_mat;
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
   int order = 1;
   double tf = 3;
   double dt = 1e-2;
   int ode_solver_type = 1;
   int vis_steps = -1;

   OptionsParser args(argc, argv);
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the serial mesh uniformly.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the parallel mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) >= 0.");
   args.AddOption(&eta, "-eta", "--etaant", "Reaction coefficient.");
   args.AddOption(&dt, "-dt", "--time-step", "Time step.");
   args.AddOption(&tf, "-tf", "--final-time", "Final time.");
   args.AddOption(&ode_solver_type, "-ode", "--ode", "Use ODE solver id; default forward Euler).");
   args.AddOption(&vis_steps, "-v", "--vis-steps",
               "Visualize this many steps.");

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

   Reaction oper(fes);
   
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
   u.ProjectCoefficient(ic_coeff);

   ParaViewDataCollection dc("reaction", &mesh);
   if (vis_steps > 0) {
      dc.SetPrefixPath("ParaView");
      dc.RegisterField("u", &u);
      dc.SetLevelsOfDetail(order);
      dc.SetHighOrderOutput(true);
      dc.SetCycle(0);
      dc.SetTime(0.0);
      dc.Save();
   }

   simulate(ode_solver, u, fes, tf, dt, dc, vis_steps);

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
