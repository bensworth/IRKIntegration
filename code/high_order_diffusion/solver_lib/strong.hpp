#ifndef __AS_STRONG__
#define __AS_STRONG__

#include "mfem.hpp"
#include "general/forall.hpp"

using namespace mfem;

struct MassNI : Operator
{
   Vector D;
   MassNI(ParFiniteElementSpace &fes, Array<int> &ess_tdof_list);
   void Mult(const Vector &x, Vector &y) const;
};

struct DiagonalInverse : Solver
{
   Vector D, Dinv;

   DiagonalInverse(Vector &D_, bool use_dev=true);
   DiagonalInverse(HypreParMatrix &A, bool use_dev=true);
   void Init(bool use_dev);
   void Mult(const Vector &x, Vector &y) const;
   void Mult(Vector &x) const;
   void SetOperator(const Operator &op);

};

struct ProductSolver : Solver
{
   Operator &A, &B;
   mutable Vector z;

   ProductSolver(Operator &A_, Operator &B_);
   void Mult(const Vector &x, Vector &y) const;
   void SetOperator(const Operator &op);
};

#endif
