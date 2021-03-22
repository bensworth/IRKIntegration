#include "dg.hpp"
#include "util.hpp"

CG2DG::CG2DG(const ParFiniteElementSpace &fes_cg,
             const ParFiniteElementSpace &fes_dg)
{
   int ndof_dg = fes_dg.GetNDofs();
   int ndof_cg = fes_cg.GetNDofs();
   SparseMatrix mat(ndof_dg, ndof_cg);

   // Assembly
   DenseMatrix elmat;
   Array<int> vdofs_dg, vdofs_cg;
   for (int iel=0; iel<fes_dg.GetNE(); ++iel)
   {
      const FiniteElement &fe_dg = *fes_dg.GetFE(iel);
      const FiniteElement &fe_cg = *fes_cg.GetFE(iel);

      int ndof_dg = fe_dg.GetDof();
      int ndof_cg = fe_cg.GetDof();

      MFEM_ASSERT(ndof_dg == ndof_cg, "DG2CG must have same number of DOFs");

      fes_dg.GetElementVDofs(iel, vdofs_dg);
      fes_cg.GetElementVDofs(iel, vdofs_cg);

      const Array<int> &dof_map =
         dynamic_cast<const TensorBasisElement&>(fe_cg).GetDofMap();

      elmat.SetSize(ndof_dg, ndof_cg);
      elmat = 0.0;
      for (int i=0; i<ndof_dg; ++i)
      {
         elmat(i,dof_map[i]) = 1.0;
      }
      int skip_zeros = 1;
      mat.SetSubMatrix(vdofs_dg, vdofs_cg, elmat, skip_zeros);
   }
   // Zero out the boundary
   Vector column_scaling(ndof_dg);
   column_scaling = 1.0;
   for (int ib=0; ib<fes_cg.GetNBE(); ++ib)
   {
      fes_cg.GetBdrElementVDofs(ib, vdofs_cg);
      for (int idx=0; idx<vdofs_cg.Size(); ++idx)
      {
         int i = vdofs_cg[idx];
         column_scaling(i) = 0.0;
      }
   }
   mat.ScaleColumns(column_scaling);
   mat.Finalize();
   C.Swap(mat);

   P = fes_cg.GetProlongationMatrix();
   z.SetSize(P->Height());
}

void CG2DG::Mult(const Vector &x, Vector &y) const
{
   if (P)
   {
      P->Mult(x, z);
      C.Mult(z, y);
   }
   else
   {
      C.Mult(x, y);
   }
}

void CG2DG::MultTranspose(const Vector &x, Vector &y) const
{
   if (P)
   {
      C.MultTranspose(x, z);
      P->MultTranspose(z, y);
   }
   else
   {
      C.MultTranspose(x, y);
   }
}

bool IsBoundaryNode(int i, int order, int dim)
{
   int n = order+1;
   if (dim == 1)
   {
      return i == 0 || i == order;
   }
   else if (dim == 2)
   {
      int ix = i%n;
      int iy = i/n;
      return (ix == 0 || ix == order || iy == 0 || iy == order);
   }
   else if (dim == 3)
   {
      int ix = i%n;
      int iy = (i/n)%n;
      int iz = i/sq(n);
      return (ix == 0 || ix == order
              || iy == 0 || iy == order
              || iz == 0 || iz == order);
   }
   else
   {
      MFEM_ABORT("NOT IMPLEMENTED");
      return false;
   }
}

void ZeroInterior(const FiniteElementSpace &fes_dg, SparseMatrix &Z_)
{
   int ndof_dg = fes_dg.GetNDofs();
   SparseMatrix Z(ndof_dg, ndof_dg);

   // Assembly
   Array<int> vdofs_dg;
   for (int iel=0; iel<fes_dg.GetNE(); ++iel)
   {
      const FiniteElement &fe_dg = *fes_dg.GetFE(iel);
      int ndof_dg = fe_dg.GetDof();
      fes_dg.GetElementVDofs(iel, vdofs_dg);
      for (int i=0; i<ndof_dg; ++i)
      {
         if (IsBoundaryNode(i, fe_dg.GetOrder(), fe_dg.GetDim()))
         {
            Z.Set(vdofs_dg[i],vdofs_dg[i], 1.0);
         }
      }
   }
   Z.Finalize();
   Z_.Swap(Z);
}

DiscontPSCPreconditioner::DiscontPSCPreconditioner(
   ParFiniteElementSpace &fes_dg,
   const Solver &cg_solver_,
   const Solver &smoother_)
   : Solver(fes_dg.GetNDofs()),
     cg_solver(cg_solver_),
     smoother(smoother_)
{
   ParMesh *mesh = fes_dg.GetParMesh();
   H1_FECollection fec_ho(fes_dg.GetOrder(0), mesh->Dimension());
   fes_cg.reset(new ParFiniteElementSpace(mesh, &fec_ho));

   C.reset(new CG2DG(*fes_cg, fes_dg));
   ZeroInterior(fes_dg, Z);

   b_cg.SetSize(fes_cg->GetTrueVSize());
   x_cg.SetSize(fes_cg->GetTrueVSize());
   x_z.SetSize(fes_dg.GetNDofs());
   b_z.SetSize(fes_dg.GetNDofs());
}

void DiscontPSCPreconditioner::Mult(const Vector &b, Vector &x) const
{
   // Zero out interior DOFs
   Z.Mult(b, b_z);
   // Smoother
   smoother.Mult(b_z, x_z);
   // Restrict residual
   C->MultTranspose(b, b_cg);
   // Approximate coarse solve
   cg_solver.Mult(b_cg, x_cg);
   // Prologate
   C->Mult(x_cg, x);
   // Add edge correction term
   x += x_z;
}

void DiscontPSCPreconditioner::SetOperator(const Operator &op)
{
   MFEM_ABORT("Cannot call SetOperator")
}
