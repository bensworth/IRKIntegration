#ifndef __AS_GHOST_
#define __AS_GHOST_

#include "mfem.hpp"
#include <memory>

using namespace mfem;

struct LORGhostMesh;
class GhostMesh
{
   friend struct LORGhostMesh;
   ParMesh &pmesh;
   std::unique_ptr<Mesh> mesh;
   int my_rank;
   MPI_Comm comm;
   std::vector<bool> vertex_ownership;
   std::map<int,std::vector<int>> borrowers;
   std::map<int,std::vector<int>> owners;
public:
   GhostMesh(ParMesh &pmesh_);
   bool IsVertexOwned(int iv) const;
};

struct LORGhostMesh
{
   GhostMesh gmesh;
   Mesh &mesh0;
   Mesh mesh_lor;
   H1_FECollection fec;
   FiniteElementSpace fes;
   int nref;
   Table ho2lor;

   // Interpolation between non-ghost and ghost meshes
   Array<int> perm;

   // Communication structures
   std::map<int,Array<int>> borrower_dofs;
   std::map<int,Array<int>> owner_dofs;

   std::vector<MPI_Request> borrower_reqs, owner_reqs;
   std::vector<MPI_Status> borrower_stat, owner_stat;
   std::map<int,Vector> borrower_msg, owner_msg;

   LORGhostMesh(ParMesh &pmesh, int ref_factor, int ref_type);
   void PopulateOwnedDofs(const Vector &x_in, Vector &x_g);
   void ExtractOwnedDofs(const Vector &x_g, Vector &x_out);
   void PopulateGhostLayer(Vector &x);
   void AddToOwners(Vector &x, double alpha=1.0);

   void Populate(const Vector &x_in, Vector &x_g);
};

#endif
