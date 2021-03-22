#ifndef __AS_DG__
#define __AS_DG__

#include "mfem.hpp"
#include <memory>

using namespace mfem;

struct CG2DG : Operator
{
   const Operator *P;
   SparseMatrix C;
   mutable Vector z;
   CG2DG(const ParFiniteElementSpace &fes_cg,
         const ParFiniteElementSpace &fes_dg);
   void Mult(const Vector &x, Vector &y) const;
   void MultTranspose(const Vector &x, Vector &y) const;
};

class DiscontPSCPreconditioner : public Solver
{
protected:
   std::unique_ptr<ParFiniteElementSpace> fes_cg;
   const Solver &cg_solver;
   const Solver &smoother;
   std::unique_ptr<CG2DG> C;
   SparseMatrix Z;

   mutable Vector b_z, x_z, b_cg, x_cg;
   mutable Vector x_sm;

public:
   DiscontPSCPreconditioner(ParFiniteElementSpace &fes_dg,
                            const Solver &cg_solver_,
                            const Solver &smoother_);
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);
};

#endif
