#ifndef __3DG_WRAPPER_HPP__
#define __3DG_WRAPPER_HPP__

#include "mfem.hpp"

using namespace mfem;

struct DGMatrix : Operator
{
   SparseMatrix A;
   virtual void Mult(const Vector &x, Vector &y) const override
   {
      A.Mult(x, y);
   }
};

struct DGPreconditioner : Solver
{
   SparseMatrix *B;
   BlockILU prec;

   DGPreconditioner(double gamma, DGMatrix &M, double dt, DGMatrix &A,
                    int block_size)
      : B(Add(gamma, M.A, -dt, A.A)),
        prec(*B, block_size)
   { }

   ~DGPreconditioner()
   {
      delete B;
   }

   virtual void Mult(const Vector &x, Vector &y) const override
   {
      prec.Mult(x, y);
   }
   virtual void SetOperator(const Operator &op) override
   {
      MFEM_ABORT("Not implemented")
   }
};

struct DGWrapper
{
   struct DGInternal *dg;
   DGWrapper();
   ~DGWrapper();
   int Size() const;
   int BlockSize() const;
   void ApplyMass(const Vector &u, Vector &Mu);
   void ApplyMassInverse(const Vector &Mu, Vector &u);
   void Assemble(const Vector &u, Vector &r);
   void AssembleJacobian(const Vector &u, DGMatrix &J);
   void MassMatrix(DGMatrix &M);
   void ExactSolution(Vector &u, double t);
   void InitialCondition(Vector &u);
};

#endif
