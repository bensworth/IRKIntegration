#ifndef __AS_ILU__
#define __AS_ILU__

#include "mfem.hpp"
#include <memory>

using namespace mfem;

void MDFOrdering(SparseMatrix &C, Array<int> &p);

class ILU0 : public Solver
{
   std::unique_ptr<SparseMatrix> LU;
   int n;
   Array<int> ID, p;
   mutable Vector y;
public:
   ILU0(SparseMatrix &A);

   virtual void SetOperator(const Operator &op);
   void Factorize(const SparseMatrix *A);
   virtual void Mult(const Vector &b, Vector &x) const;
   SparseMatrix *GetLU();
};

#endif
