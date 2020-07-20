#include "3dg_wrapper.hpp"
#include "dg.h"

void eulervortex(mesh &msh, darray &uev, double t, std::vector<double> pars)
{
   int    N     = 4;
   double gamma = 1.4;
   double pi    = M_PI;
   double rc    = pars[0];
   double eps   = pars[1];
   double M0    = pars[2];
   double theta = pars[3];
   double x0    = pars[4];
   double y0    = pars[5];
   double rinf  = 1;
   double uinf  = 1;
   double Einf  = 1/gamma/(M0*M0)/(gamma-1) + 0.5;
   double pinf  = (gamma-1)*(Einf-0.5);
   double ubar  = uinf*cos(theta);
   double vbar  = uinf*sin(theta);

   if (!uev || !uev.has_shape(msh.ns, N, msh.nt))
   {
      uev.realloc(msh.ns, N, msh.nt);
   }

   for (int it=0; it<msh.nt; ++it)
   {
      for (int is=0; is<msh.ns; ++is)
      {
         double x = msh.p1(is, 0, it);
         double y = msh.p1(is, 1, it);

         double f = (1 - pow((x-x0)-ubar*t,2.0)
                     - pow((y-y0)-vbar*t,2.0))/pow(rc,2.0);
         double u=uinf*(cos(theta)-eps*((y-y0)-vbar*t)/(2*pi*rc)*exp(f/2));
         double v=uinf*(sin(theta)+eps*((x-x0)-ubar*t)/(2*pi*rc)*exp(f/2));
         double r=rinf*pow(1-eps*eps*(gamma-1)*M0*M0/(8*pi*pi)*exp(f),
                           (1/(gamma-1)));
         double p=pinf*pow(1-eps*eps*(gamma-1)*M0*M0/(8*pi*pi)*exp(f),
                           (gamma/(gamma-1)));

         double ru = r*u;
         double rv = r*v;
         double rE = p/(gamma-1)+0.5*(ru*ru + rv*rv)/r;

         uev(is, 0, it) = r;
         uev(is, 1, it) = ru;
         uev(is, 2, it) = rv;
         uev(is, 3, it) = rE;
      }
   }
}

void evperiodic(mesh &msh, darray &u, double t, std::vector<double> pars,
                double X, double Y, int rep)
{
   int N = 4;
   darray u1;
   std::vector<double> pars1 = pars;
   pars1[1] = 0;

   eulervortex(msh, u, t, pars1);
   darray uinf(N);
   for (int ic=0; ic<N; ++ic)
   {
      uinf(ic) = u(0,ic,0);
   }

   u *= 0.0;

   pars1[1] = pars[1];
   for (int i=-rep; i <= rep; ++i)
   {
      for (int j=-rep; j <= rep; ++j)
      {
         pars1[4] = pars[4] - i*X;
         pars1[5] = pars[5] - j*Y;

         eulervortex(msh, u1, t, pars1);
         u += u1;
      }
   }

   for (int it=0; it<msh.nt; ++it)
      for (int is=0; is<msh.ns; ++is)
         for (int ic=0; ic<N; ++ic)
         {
            u(is, ic, it) -= (pow(2*rep+1,2)-1)*uinf(ic);
         }
}

struct DGInternal
{
   mesh msh;
   data d;
   phys p;
   appl a;

   darray u_ic;

   darray r, Ddrdu, Odrdu;
   iarray Didx, Oidx;

   int N, nt, ns, nes, nf;
   std::vector<double> params;
   double X, Y;

   void InitSizes();
   void InitEulerVortex();
   void InitNACA();
};

void DGInternal::InitSizes()
{
   appinfo app;
   a.getappinfo(msh.dim, p, app);
   N = app.N;
   nt = msh.nt;
   ns = d.ns;
   nes = d.nes;
   nf=msh.nf;

   // Allocate arrays to store Jacobian
   Ddrdu.realloc(N*ns,N*ns,nt);
   Odrdu.realloc(N*d.nes,N*ns,msh.nf,nt);
   r.realloc(ns,N,nt);
   Didx.realloc(2,(N*ns)*(N*ns)*(nt));
   Oidx.realloc(2,(N*nes)*(N*ns)*(nf)*(nt));

   // Precompute indices for sparse Jacobian matrix
   dgjacindices(msh, d, N, Didx, Oidx);
}

void DGInternal::InitEulerVortex()
{
   a = dgnavierstokes;

   int np = 1; // Hard-coded 1 MPI rank for now... can consider parallel later
   int porder = 4; // Hard-coded for now...
   std::string mshfilename = stringformat("data/ev/evmsh%dpartn%d.h5", porder, np);
   msh.readfile(mshfilename);
   if (np > 1) { msh.initialize_mpi(); }

   dginit(msh, d);

   // Set the various parameters
   double    rc = 1.5;
   double   eps = 15;
   double theta = atan2(1, 2);
   double    M0 = 0.5;
   double    x0 = 5;
   double    y0 = -2.5;

   params = {rc, eps, M0, theta, x0, y0};
   X = 40;
   Y = 30;

   double    Re = std::numeric_limits<double>::infinity();
   using dg::physinit::FarFieldQty;
   dg::physinit::navierstokes(
      msh,
   {1, 1, 1, 1},             // bndcnds
   {Re, 0.72},               // pars
   1.0,                      // far field density
   {cos(theta), sin(theta)}, // far field velocity
   FarFieldQty::Mach(M0),    // far field mach
   &p);

   // Get the initial conditions
   evperiodic(msh, u_ic, 0.0, {rc, eps, M0, theta, x0, y0}, X, Y, 1);

   // Set up the solvers
   auto linsolver = LinearSolverOptions::gmres("i");
   auto newton = NewtonOptions(linsolver);

   InitSizes();
}

void DGInternal::InitNACA()
{
   a = dgnsisentrop;

   int np = 1;
   std::string mshfilename = "data/naca/mshnacales1ref1partn" + to_string(np) + ".h5";
   msh.readfile(mshfilename);
   if (np > 1) msh.initialize_mpi();

   double Re = 40e3;
   dginit(msh, d);
   using dg::physinit::FarFieldQty;
   dg::physinit::nsisentrop(
      msh,
      {2, 1},           // bndcnds
      {Re, 0.72, 0, 1}, // pars
      1.0,              // far field density
      0.1,              // far field Mach
      {1.0, 0.0},       // far field velocity
      &p);

   dgfreestream(msh, p, u_ic);

   InitSizes();
}

DGWrapper::DGWrapper()
{
   dg = new DGInternal();
}

DGWrapper::~DGWrapper()
{
   delete dg;
}

int DGWrapper::Size() const
{
   // ns: number of shape functions per element
   // N: number of solution components (4 for 2D Euler/NS, 5 for 3D)
   // nt: number of elements (number of triangles/tets for simplex meshes)
   return dg->ns*dg->N*dg->nt;
}

int DGWrapper::BlockSize() const
{
   return dg->ns*dg->N;
}

void DGWrapper::ApplyMass(const Vector &u, Vector &Mu)
{
   Mu = u;
   darray Mu_ar(Mu.GetData(), dg->ns, dg->N, dg->nt);
   dgmass(Mu_ar, dg->msh, dg->d);
}

void DGWrapper::ApplyMassInverse(const Vector &Mu, Vector &u)
{
   u = Mu;
   darray u_ar(u.GetData(), dg->ns, dg->N, dg->nt);
   dgmassinv(u_ar, dg->msh, dg->d);
}

void DGWrapper::Assemble(const Vector &u, Vector &r)
{
   darray u_ar(u.GetData(), dg->ns, dg->N, dg->nt);
   darray r_ar(r.GetData(), dg->ns, dg->N, dg->nt);
   darray NULLARR;
   dgassemble(dg->a, u_ar, r_ar, NULLARR, NULLARR, dg->msh, dg->d, dg->p);
}

void DGWrapper::AssembleJacobian(const Vector &u, DGMatrix &J)
{
   darray u_ar(u.GetData(), dg->ns, dg->N, dg->nt);
   dgassemble(dg->a, u_ar, dg->r, dg->Ddrdu, dg->Odrdu, dg->msh, dg->d, dg->p);

   SparseMatrix A(Size(), Size());

   int nnz_D = dg->Didx.size(1);
   for (int idx=0; idx<nnz_D; ++idx)
   {
      int i = dg->Didx(0,idx);
      int j = dg->Didx(1,idx);
      A.Set(i,j,dg->Ddrdu[idx]);
   }

   int nnz_O = dg->Oidx.size(1);
   for (int idx=0; idx<nnz_O; ++idx)
   {
      int i = dg->Oidx(0,idx);
      int j = dg->Oidx(1,idx);
      A.Set(i,j,dg->Odrdu[idx]);
   }

   A.Finalize();

   J.A.Swap(A);
}

void DGWrapper::MassMatrix(DGMatrix &M)
{
   SparseMatrix A(Size(), Size());

   darray M_el(dg->ns,dg->ns);
   for (int it=0; it<dg->nt; ++it)
   {
      ::MassMatrix(M_el,it,dg->msh,dg->d);
      for (int icomp=0; icomp<dg->N; ++icomp)
      {
         int cidx = 0;
         for (int jj=0; jj<dg->ns; jj++)
         {
            for (int ii=0; ii<dg->ns; ii++)
            {
               int i = ii + dg->ns*icomp + dg->N*dg->ns*it;
               int j = jj + dg->ns*icomp + dg->N*dg->ns*it;
               A.Set(i, j, M_el[cidx]);
               ++cidx;
            }
         }
      }
   }
   A.Finalize();
   M.A.Swap(A);
}

void DGWrapper::InitialCondition(Vector &u)
{
   u.SetSize(Size());
   u = dg->u_ic;
}

void DGWrapper::Init(int problem)
{
   if (problem == 0)
   {
      dg->InitEulerVortex();
   }
   else if (problem == 1)
   {
      dg->InitNACA();
   }
   else
   {
      MFEM_ABORT("Unknown problem type.")
   }
}
