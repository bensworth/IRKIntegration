#ifndef __AS_PATCH__
#define __AS_PATCH__

#include "mfem.hpp"

#include "coarse.hpp"
#include "ghost.hpp"
#include "util.hpp"

using namespace mfem;

// This struct describes how a coarsened mesh is embedded in its parent mesh.
// Each macro element has some interior DOFs removed in the coarsening process
// This requires renumbering the global DOFs (keeping track so that DOFs that
// coincide are assigned the same global DOF ID)
struct MeshEmbedding
{
   struct DofInfo
   {
      int idx;
      bool interior;
      DofInfo(int idx_, bool interior_) : idx(idx_), interior(interior_) { }
   };
   // Outer index: macro element index
   // Inner index: node index
   std::vector<std::vector<DofInfo>> loc2glob;
   // Total number of (global) DOFs
   int ndofs;
   // DOFs per element (number kept after coarsening)
   int n1d;
};

// Construct a coarsened hierarchy on a patch by removing interior DOFs from
// each macro element
struct PatchCoarsening
{
   LocalCoarsening1D local_coarsening;
   int nlevels;
   std::vector<SparseMatrix*> P;
   std::vector<MeshEmbedding> embed;
   Array<int> perm;

   PatchCoarsening(LORGhostMesh &glor, const Array<int> &els,
                   Array<int> &ess_bdr, bool multilevel);

   ~PatchCoarsening();
};

// We define a patch using a PatchInfo struct.
// els is a list of fine elements in the patch.
// els0 is a list of coarse elements. For this patch to be well-defined,
// we need all of the fine elements in els to be sub-elements of els0
struct PatchInfo
{
   Array<int> els0;

   PatchInfo(Array<int> &els0_) : els0(els0_) { }
};

PatchInfo VerticesPatch(const std::vector<int> &v, Table &v2e);
PatchInfo VertexPatch(int iv, Table &v2e);

// Multilevel solver for solving the local problem corresponding to a patch
struct PatchSolver
{
   PatchCoarsening coarsening;

   int nlevels;
   std::vector<SparseMatrix*> Ps, As;
   std::vector<Solver*> Ss;
   std::vector<Vector*> bs, xs, rs;
   Vector x_final, b_init;

   PatchSolver(const PatchInfo &info, LORGhostMesh &glor, SparseMatrix &A,
               Array<int> &ess_bdr, bool multilevel);
   void SolveAdd(const Vector &b_in, Vector &x_out);
   ~PatchSolver();
};

// Additive method for solving all of the patches
struct PatchesAdditiveSolver
{
   LORGhostMesh *glor;
   std::vector<PatchSolver *> solv;
   mutable Vector x_g, b_g;

   void Init(LORGhostMesh &glor_, SparseMatrix &A, Array<int> &ess_bdr,
             bool multilevel=true, int npatches=1);
   void Solve(const Vector &b, Vector &x) const;
   PatchInfo VerticesPatch(const std::vector<int> &v, Table &v2e);
   PatchInfo VertexPatch(int iv, Table &v2e);
   ~PatchesAdditiveSolver();
};

#endif
