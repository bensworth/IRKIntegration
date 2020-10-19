#include "strong.hpp"

MassNI::MassNI(ParFiniteElementSpace &fes, Array<int> &ess_tdof_list)
{
   int ndof = fes.GetTrueVSize();
   height = width = ndof;

   ParGridFunction Dl{&fes};
   Dl.UseDevice(false); // Force Dl to be on host
   Dl = 0.0;
   Dl.HostReadWrite();

   // Construct collocated quadrature
   IntegrationRules rules_ni(0, Quadrature1D::GaussLobatto);
   int order = fes.GetFE(0)->GetOrder();
   int ni_order = 2*(order-1);
   const IntegrationRule &ir_ni
      = rules_ni.Get(fes.GetFE(0)->GetGeomType(), ni_order);

   const Array<int> &dof_map =
      dynamic_cast<const TensorBasisElement*>(fes.GetFE(0))->GetDofMap();

   for (int iel=0; iel<fes.GetMesh()->GetNE(); ++iel)
   {
      int ndof_el = fes.GetFE(iel)->GetDof();
      Array<int> dofs;
      fes.GetElementVDofs(iel, dofs);
      ElementTransformation *tr = fes.GetElementTransformation(iel);
      for (int icart=0; icart<ndof_el; ++icart)
      {
         int i = dof_map[icart];
         const IntegrationPoint &ip = ir_ni.IntPoint(icart);
         tr->SetIntPoint(&ip);
         double w = ip.weight*tr->Weight();
         for (int d=0; d<fes.GetVDim(); ++d)
         {
            Dl[dofs[i + d*ndof_el]] += w;
         }
      }
   }

   D.SetSize(ndof);
   Dl.ParallelAssemble(D);

   for (int i=0; i<ess_tdof_list.Size(); ++i)
   {
      D[ess_tdof_list[i]] = 1.0;
   }
}

void MassNI::Mult(const Vector &x, Vector &y) const
{
   for (int i=0; i<D.Size(); ++i)
   {
      y[i] = x[i]*D[i];
   }
}

DiagonalInverse::DiagonalInverse(Vector &D_, bool use_dev) : Solver(D_.Size())
{
   D = D_;
   Init(use_dev);
}

DiagonalInverse::DiagonalInverse(HypreParMatrix &A,
                                 bool use_dev) : Solver(A.Width())
{
   SparseMatrix Aloc;
   A.GetDiag(Aloc);
   Aloc.GetDiag(D);
   Init(use_dev);
}

void DiagonalInverse::Init(bool use_dev)
{
   D.UseDevice(use_dev);
   Dinv.SetSize(D.Size());
   Dinv.UseDevice(use_dev);

   const int sz = D.Size();
   auto d_D     = D.Read(use_dev);
   auto d_Dinv  = Dinv.ReadWrite(use_dev);

   MFEM_FORALL_SWITCH(use_dev, i, sz,
   {
      d_Dinv[i] = 1.0/d_D[i];
   });
}

void DiagonalInverse::Mult(const Vector &x, Vector &y) const
{
   bool use_dev = Dinv.UseDevice();
   const int sz = y.Size();
   auto d_x = x.Read(use_dev);
   auto d_y = y.ReadWrite(use_dev);
   auto d_Dinv = Dinv.Read(use_dev);

   MFEM_FORALL_SWITCH(use_dev, i, sz,
   {
      d_y[i] = d_Dinv[i]*d_x[i];
   });
}

void DiagonalInverse::Mult(Vector &x) const
{
   bool use_dev = Dinv.UseDevice();
   const int sz = x.Size();
   auto d_x = x.ReadWrite(use_dev);
   auto d_Dinv = Dinv.Read(use_dev);

   MFEM_FORALL_SWITCH(use_dev, i, sz,
   {
      d_x[i] *= d_Dinv[i];
   });
}

void DiagonalInverse::SetOperator(const Operator &op)
{
   MFEM_ABORT("Cannot call SetOperator");
}

ProductSolver::ProductSolver(Operator &A_, Operator &B_)
   : Solver(B_.Width()), A(A_), B(B_)
{
   z.SetSize(B.Height());
}

void ProductSolver::Mult(const Vector &x, Vector &y) const
{
   B.Mult(x, z);
   A.Mult(z, y);
}

void ProductSolver::SetOperator(const Operator &op)
{
   MFEM_ABORT("Cannot call SetOperator");
}
