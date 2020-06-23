#include "bljacobi.hpp"

namespace mfem
{

BlockJacobi::BlockJacobi(int block_size_)
   : block_size(block_size_)
{ }

BlockJacobi::BlockJacobi(SparseMatrix &A, int block_size_)
   : block_size(block_size_)
{
   SetOperator(A);
}

void BlockJacobi::SetOperator(const Operator &op)
{
   const SparseMatrix *A = dynamic_cast<const SparseMatrix *>(&op);

   MFEM_VERIFY(A != NULL, "Operator must be a SparseMatrix");

   width = A->Width();
   height = A->Height();

   MFEM_VERIFY(height % block_size == 0, "");

   nblockrows = height/block_size;
   DB.SetSize(block_size, block_size, nblockrows);
   DB = 0.0;
   ipiv.SetSize(block_size*nblockrows);

   const int *I = A->GetI();
   const int *J = A->GetJ();
   const double *V = A->GetData();

   for (int iblock = 0; iblock < nblockrows; ++iblock)
   {
      for (int bi = 0; bi < block_size; ++bi)
      {
         int i = iblock*block_size + bi;
         for (int k = I[i]; k < I[i + 1]; ++k)
         {
            int j = J[k];
            if (j >= iblock*block_size && j < (iblock + 1)*block_size)
            {
               int bj = j - iblock*block_size;
               DB(bi, bj, iblock) = V[k];
            }
         }
      }
      LUFactors A_ii_inv(&DB(0,0,iblock), &ipiv[iblock*block_size]);
      A_ii_inv.Factor(block_size);
   }
}

void BlockJacobi::Mult(const Vector &b, Vector &x) const
{
   x = b;

   for (int i=0; i<nblockrows; ++i)
   {
      LUFactors A_ii_inv(&DB(0,0,i), &ipiv[i*block_size]);
      A_ii_inv.Solve(block_size, 1, &x[i*block_size]);
   }
}

} // namespace mfem
