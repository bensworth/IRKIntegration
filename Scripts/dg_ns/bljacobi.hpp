#ifndef __BLJACOBI_HPP__
#define __BLJACOBI_HPP__

#include "mfem.hpp"

namespace mfem
{

struct BlockJacobi : Solver
{
   BlockJacobi(int block_size_ = 1);
   BlockJacobi(SparseMatrix &A, int block_size_ = 1);

   void SetOperator(const Operator &op) override;

   void Mult(const Vector &b, Vector &x) const override;

   int block_size, nblockrows;
   mutable Vector y;
   mutable DenseTensor DB;
   mutable Array<int> ipiv;
};

} // namespace mfem

#endif
