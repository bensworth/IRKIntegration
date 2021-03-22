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
   ILU0(SparseMatrix &A, double shift=0.0);

   virtual void SetOperator(const Operator &op);
   void Factorize(const SparseMatrix *A, double shift);
   virtual void Mult(const Vector &b, Vector &x) const;
   SparseMatrix *GetLU();
};

#endif
