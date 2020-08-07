#ifndef __AS_SOLVER__
#define __AS_SOLVER__

#include "mfem.hpp"
#include "patch.hpp"

using namespace mfem;

struct CoarseSolver
{
   SparseMatrix *R;
   InterpolationGridTransfer tr;
   Solver &solv;
   IterativeSolver *it_solv;
   Array<int> ess_dof;
   mutable Vector b0, x0;
   mutable ParGridFunction b0_gf, x0_gf;

   CoarseSolver(ParFiniteElementSpace &fes, ParFiniteElementSpace &fes0,
                Array<int> &ess_bdr, Solver &solv_);
   void Solve(const Vector &b, Vector &x) const;
};

class AdditiveSchwarz : public Solver
{
   SparseMatrix &A;
   PatchesAdditiveSolver patch_solv;
   CoarseSolver coarse_solv;

   const Operator *P;
   SparseMatrix *R;

   bool additive;
   double alpha;

   mutable Vector r, e;
   mutable ParGridFunction x, b;

   bool pure_neumann;
public:
   AdditiveSchwarz(ParFiniteElementSpace &fes,
                   ParFiniteElementSpace &fes0,
                   LORGhostMesh &glor,
                   Array<int> &ess_bdr_,
                   SparseMatrix &A_,
                   Solver &A0_solv,
                   bool multilevel=true,
                   bool additive_=true,
                   int npatches=1,
                   bool pure_neumann_=false);
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);
};

/// High Order preconditioner for Poisson solve based on GMG V-cycles, low-order refined meshes, and Additive Schwarz
class HighOrderASPreconditioner : public Solver
{
protected:
   int order;
   ParMesh *pmesh_ho;
   ParMesh pmesh_lor;
   LORGhostMesh glor;
   H1_FECollection fec_lor;
   ParFiniteElementSpace fespace_lor;
   ParFiniteElementSpace fespace_coarse;

   // Need to store so that we can init at a later date
   ParFiniteElementSpace &fes_ho;
   Coefficient &mass_coeff, &diff_coeff;
   Array<int> ess_bdr;
   int npatches;

   // TODO1: FormSystemMatrix currently (6/11/19) depends on bilinear forms staying in scope
   // We really shouldn't need to keep around a_ghost_lor and a_coarse ...
   std::unique_ptr<BilinearForm> a_ghost_lor;
   std::unique_ptr<ParBilinearForm> a_coarse;
   SparseMatrix A_ghost_lor;
   HypreParMatrix A0;
   HypreBoomerAMG A0_amg;
   std::unique_ptr<Solver> A0_solv;
   std::unique_ptr<Solver> as_prec;

   void InitGhostEssBdr(Array<int>& ess_bdr);
   void InitCoarseSolver();

public:
   HighOrderASPreconditioner(ParFiniteElementSpace& fes_ho,
                             Coefficient& _mass_coeff,
                             Coefficient& _diff_coeff,
                             Array<int> ess_bdr,
                             int npatches=1);

   /// Precondition A\ operation
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);
};

/*
class VectorHOASPreconditioner : public Solver
{
protected:
   int vdim;
   ParFiniteElementSpace scalar_fes;
   HighOrderASPreconditioner scalar_prec;
public:
   VectorHOASPreconditioner(ParFiniteElementSpace& fes_ho,
                            Coefficient& lap_coeff,
                            const Array<int> &ess_bdr,
                            bool unsteady=false,
                            int npatches=1);
   int GetVDim() const;
   virtual void Mult(const Vector &b, Vector &x) const;
   virtual void SetOperator(const Operator &op);

   void SetParameters(double dt);
};
*/

#endif
