#include "ghost.hpp"
#include "util.hpp"
#include <mpi.h>
#include <fstream>

struct GhostElement
{
   int v[8];
   int attr;
   int owner_rank;
   std::vector<double> nodes;
};

struct GhostBE
{
   int attr;
   int v[4];
};

struct ElementOwnership
{
   int owner_rank;
   std::vector<int> borrowers;
};

struct GhostMeshGen : Mesh
{
   static Geometry::Type GetElemType(int dim)
   {
      switch (dim)
      {
         case 1: return Geometry::Type::SEGMENT;
         case 2: return Geometry::Type::SQUARE;
         case 3: return Geometry::Type::CUBE;
         default:
            MFEM_ABORT("Dimensions 1 through 3 supported.")
            return Geometry::Type::INVALID;
      }
   }

   static Geometry::Type GetBdrType(int dim)
   {
      switch (dim)
      {
         case 1: return Geometry::Type::POINT;
         case 2: return Geometry::Type::SEGMENT;
         case 3: return Geometry::Type::SQUARE;
         default:
            MFEM_ABORT("Dimensions 1 through 3 supported.")
            return Geometry::Type::INVALID;
      }
   }

   GhostMeshGen(int dim, int sdim, int nv, double *v, int nel, int *el,
                int *el_attr, int nbe, int *be, int *b_attr, int attr)
      : Mesh(v, nv, el, GetElemType(dim), el_attr, nel, be, GetBdrType(dim),
             b_attr, nbe, dim, sdim)
   {
      // Fix the boundary elements
      // Find all faces that do not belong to two elements
      int nfaces = GetNumFaces();
      std::vector<int> bdr;
      std::map<int,int> face2bdr;
      for (int iface=0; iface<nfaces; ++iface)
      {
         int iel1, iel2;
         GetFaceElements(iface, &iel1, &iel2);
         if (iel2 < 0)
         {
            bdr.push_back(iface);
            face2bdr[iface] = bdr.size() - 1;
         }
      }
      int nbdr = bdr.size();

      // Mark those faces that already have a boundary element
      std::vector<int> face2be(nbdr, -1);
      for (int ib=0; ib<nbe; ++ib)
      {
         int iface = GetBdrElementEdgeIndex(ib);
         face2be[face2bdr[iface]] = ib;
      }

      // Create a list of new boundary elements
      Array<Element*> new_boundary(nbdr);
      Array<int> new_be2face(nbdr);
      for (int i=0; i<nbdr; ++i)
      {
         int be = face2be[i];
         if (be >= 0)
         {
            // If the boundary element already exists, reuse it
            new_boundary[i] = boundary[be];
         }
         else
         {
            // Otherwise create a new one
            if (dim == 1)
            {
               new_boundary[i] = new Point(&bdr[i], attr);
            }
            else
            {
               new_boundary[i] = faces[bdr[i]]->Duplicate(this);
            }
            new_boundary[i]->SetAttribute(attr);
         }
         new_be2face[i] = bdr[i];
      }

      mfem::Swap(boundary, new_boundary);
      NumOfBdrElements = nbdr;
      FinalizeTopology(false);
   }
};

GhostMesh::GhostMesh(ParMesh &pmesh_) : pmesh(pmesh_)
{
   GroupTopology &gtopo = pmesh.gtopo;
   comm = gtopo.GetComm();
   my_rank = gtopo.MyRank();
   int ng = pmesh.GetNGroups();
   int nranks = gtopo.NRanks();
   int dim = pmesh.Dimension();
   int sdim = pmesh.SpaceDimension();
   int nv_per_el = pow(2, dim);
   int nv_per_be = pow(2, dim-1);

   // NOTE: we make the assumptation that the following (or equivalent) has
   //       been called before entry into this function
   // pmesh.SetCurvature(order, false, -1, Ordering::byNODES);

   int order = pmesh.GetNodalFESpace()->GetOrder(0);
   GridFunction *orig_nodes = pmesh.GetNodes();
   orig_nodes->HostReadWrite(); // Force grid function on host
   FiniteElementSpace *orig_fes = orig_nodes->FESpace();
   int nnodes = orig_fes->GetFE(0)->GetDof();

   // If there is only one group, then there is no communication (e.g. serial)
   // TODO IMPROVEMENT: make special case for one group more efficient?
   //                   might be too much of a special case?

   // Overview of the algorithm
   // =========================
   // Each mesh partition is extended to a (potentially) larger mesh partition
   // according to the following rule. If a vertex v is owned by the partition
   // (owned means that v belongs to a group g, and the current process is the
   // master of the group), then the partition will be extended by adding all
   // elements from all partitions which contain v.

   // Note that in order to create an extended mesh with proper connectivity,
   // we need to know if elements from potentially several different partitions
   // are connected. For this we need a global vertex identification.

   // The algorithm is as follows:
   // 1. We establish a global vertex numbering that is common across all
   //    processes.
   // 2. Within each group, all of the "borrowers" send the "owner" the number
   //    of elements containing vertices within the group.
   // 3. After the owner of the group knows how many elements it will be
   //    receiving from each of borrowing partitions, that information is sent.

   // Part 1.
   // In order to create the ghost mesh, we will need a global vertex
   // identification. Each vertex will be assigned a global vertex number,
   // which will be determined by its rank and its number local to that rank.

   // NOTE: the global vertex numbering will not be contiguous, but it can still
   // be used as a unique identifier.

   // To which group does each vertex belong
   Array<int> v2group(pmesh.GetNV());
   v2group = 0;
   for (int g=1; g<ng; ++g)
   {
      int nv = pmesh.GroupNVertices(g);
      for (int i=0; i<nv; ++i)
      {
         int iv = pmesh.GroupVertex(g, i);
         v2group[iv] = g;
      }
   }

   // Create the offsets for each rank that will be used for the global number
   int nv_local = pmesh.GetNV();
   Array<int> nv_all(nranks);
   MPI_Allgather(&nv_local, 1, MPI_INT, nv_all.GetData(), 1, MPI_INT, comm);

   // Compute the partial sums
   Array<int> v_offsets(nranks);
   v_offsets[0] = 0;
   for (int i=1; i<nranks; ++i)
   {
      v_offsets[i] = v_offsets[i-1] + nv_all[i-1];
   }
   // We use the GroupCommunicator to replace the vertex numbers with the
   // numbers local to the group master.
   // Local vertex indices
   Array<int> vlocal(pmesh.GetNV());
   for (int iv=0; iv<pmesh.GetNV(); ++iv)
   {
      vlocal[iv] = iv;
   }

   GroupCommunicator gcomm(gtopo);
   gcomm.Create(v2group);
   gcomm.Bcast(vlocal);

   // Now assign a global numbering to all of the vertices.
   Array<int> vglob(pmesh.GetNV());
   for (int iv=0; iv<pmesh.GetNV(); ++iv)
   {
      int g = v2group[iv];
      int owner = gtopo.GetNeighborRank(gtopo.GetGroupMaster(g));

      vglob[iv] = v_offsets[owner] + vlocal[iv];
   }

   // We will need to know which elements contain a given vertex.
   // Assume ownership of v2e here. Must delete later.
   Table *v2e = pmesh.GetVertexToElementTable();

   // Part 2.
   // Each partition needs to figure out how many elements and vertices it is
   // sending to its neighbors
   std::map<int,std::set<int>> els_to_send, v_to_send, be_to_send;
   std::set<int> borrowing_ranks, ranks_we_borrow_from;

   // We also need to send boundary attribute data. We do some preprocessing
   // to make this easier.
   std::map<int,std::vector<int>> el2be;
   for (int ib=0; ib<pmesh.GetNBE(); ++ib)
   {
      int iface = pmesh.GetBdrElementEdgeIndex(ib);
      int iel1, iel2;
      pmesh.GetFaceElements(iface, &iel1, &iel2);
      MFEM_ASSERT(iel2 < 0, "");
      el2be[iel1].push_back(ib);
   }

   // In the end, the group owner will borrow some of our elements. We need to
   // keep track of who is borrowing our elements so that we can send and
   // receive data from them when communicating.
   std::vector<std::set<int>> el_borrowers(pmesh.GetNE());

   // First we loop through the groups. For any group that we are not owner
   // (i.e. we are "borrowing"), we need to send our data to the group owner.
   // Here we post all the send requests.
   for (int g=1; g<ng; ++g)
   {
      if (!gtopo.IAmMaster(g))
      {
         // If the current process is not the group master, then every vertex
         // is a "borrow" vertex. The group master is the "owner" of the vertex,
         // and the owner constructs the ghost mesh with all elements containing
         // these vertices. So we need to send to the owner all of the elements
         // containing any vertex in the group.

         // The format of the message is as follows:
         // number of vertices (int)
         // vertex id and coordinates for each vertex
         //        (int + sdim*double)*nv
         // number of elements
         // element attribute (int)
         // vertex numbers for the element
         //        (int)*2^dim*nel
         // nodal values for the element
         //        (double)*sdim*nnodes
         // number of boundary elements
         // boundary attribute
         // vertex numbers of the boundary element
         //        (int)*2^(dim-1)*nbe

         int their_rank = gtopo.GetGroupMasterRank(g);
         borrowing_ranks.insert(their_rank);

         int nv_gp = pmesh.GroupNVertices(g);
         for (int i=0; i<nv_gp; ++i)
         {
            int iv = pmesh.GroupVertex(g, i);
            int nel_v = v2e->RowSize(iv);
            int *els_row = v2e->GetRow(iv);
            for (int j=0; j<nel_v; ++j)
            {
               int iel = els_row[j];
               els_to_send[their_rank].insert(iel);
               el_borrowers[iel].insert(their_rank);
            }
         }

         for (int iel : els_to_send[their_rank])
         {
            int *el_v = pmesh.GetElement(iel)->GetVertices();
            for (int i=0; i<nv_per_el; ++i)
            {
               int iv = el_v[i];
               v_to_send[their_rank].insert(iv);
            }
            for (int ib : el2be[iel])
            {
               be_to_send[their_rank].insert(ib);
            }
         }
      }
      else
      {
         // We are the group owner, so we borrow elements from all members of
         // this group for our ghost layer
         const int *gp = gtopo.GetGroup(g);
         // Skip 0 (ourselves)
         for (int i=1; i<gtopo.GetGroupSize(g); ++i)
         {
            int their_rank = gtopo.GetNeighborRank(gp[i]);
            ranks_we_borrow_from.insert(their_rank);
         }
      }
   }

   int nborrowers = borrowing_ranks.size();
   std::vector<MPI_Request> send_reqs(nborrowers);
   std::vector<MPI_Status> send_stat(nborrowers);
   std::vector<std::vector<char>> msg(nborrowers);

   int isend = 0;
   for (int their_rank : borrowing_ranks)
   {
      int nv_send = v_to_send[their_rank].size();
      int nel_send = els_to_send[their_rank].size();
      int nbe_send = be_to_send[their_rank].size();

      size_t msg_sz = (3 + (1 + nv_per_el)*nel_send)*sizeof(int)
                      + nv_send*(sizeof(int) + sdim*sizeof(double))
                      + nel_send*sdim*nnodes*sizeof(double)
                      + nbe_send*(1 + nv_per_be)*sizeof(int);

      // Now create the message
      msg[isend].reserve(msg_sz);
      // Add the vertex coordinates
      put(msg[isend], nv_send);
      for (int iv : v_to_send[their_rank])
      {
         put(msg[isend], vglob[iv]);
         double *v_coords = pmesh.GetVertex(iv);
         for (int d=0; d<sdim; ++d)
         {
            put(msg[isend], v_coords[d]);
         }
      }
      // Add the elements
      put(msg[isend], nel_send);
      for (int iel : els_to_send[their_rank])
      {
         put(msg[isend], pmesh.GetAttribute(iel));
         const Element *el = pmesh.GetElement(iel);
         int el_nv = el->GetNVertices();
         MFEM_ASSERT(el_nv == nv_per_el,
                     "Must be tensor product element");
         const int *el_v = el->GetVertices();
         for (int i=0; i<el_nv; ++i)
         {
            put(msg[isend], vglob[el_v[i]]);
         }
         Array<int> vdofs;
         orig_fes->GetElementVDofs(iel, vdofs);
         MFEM_ASSERT(vdofs.Size() == nnodes*sdim, "");
         for (int i=0; i<nnodes*sdim; ++i)
         {
            put(msg[isend], (*orig_nodes)[vdofs[i]]);
         }
      }

      // Add the boundary elements
      put(msg[isend], nbe_send);
      for (int ib : be_to_send[their_rank])
      {
         put(msg[isend], pmesh.GetBdrAttribute(ib));
         const Element *be = pmesh.GetBdrElement(ib);
         int be_nv = be->GetNVertices();
         const int *be_v = be->GetVertices();
         for (int i=0; i<be_nv; ++i)
         {
            put(msg[isend], vglob[be_v[i]]);
         }
      }

      MFEM_ASSERT(msg[isend].size() == msg_sz, "Wrong size predicted.");

      // Send the ghost layer data to the owner.
      MPI_Isend(msg[isend].data(), msg_sz, MPI_BYTE, their_rank, 1, comm,
                &send_reqs[isend]);
      ++isend;
   }
   delete v2e;

   std::vector<int> v_to_add_id;
   std::vector<Vertex> v_to_add;
   std::vector<GhostElement> els_to_add;
   std::vector<GhostBE> be_to_add;

   // Now we receive data from everyone who we borrow from.
   // Here we perform synchronous receive requests.
   for (int their_rank : ranks_we_borrow_from)
   {
      MPI_Status msg_status;
      int tag = 1;
      MPI_Probe(their_rank, tag, comm, &msg_status);
      int msg_sz = -1;
      MPI_Get_count(&msg_status, MPI_BYTE, &msg_sz);

      MPI_Status stat;
      std::vector<char> msg(msg_sz);
      MPI_Recv(msg.data(), msg_sz, MPI_BYTE, their_rank, tag, comm, &stat);

      // Now that we have received the message, we need to parse it to
      // add the desired elements
      size_t ind = 0;

      int nv_to_add;
      get(msg, nv_to_add, ind);
      for (int i=0; i<nv_to_add; ++i)
      {
         int iv;
         Vertex vert;
         get(msg, iv, ind);
         for (int d=0; d<sdim; ++d)
         {
            get(msg, vert(d), ind);
         }
         v_to_add_id.push_back(iv);
         v_to_add.push_back(vert);
      }

      int nel_to_add;
      get(msg, nel_to_add, ind);
      for (int i=0; i<nel_to_add; ++i)
      {
         int el_attr;
         get(msg, el_attr, ind);

         GhostElement el;
         el.owner_rank = their_rank;
         el.attr = el_attr;

         for (int j=0; j<nv_per_el; ++j)
         {
            get(msg, el.v[j], ind);
         }

         el.nodes.resize(nnodes*sdim);
         for (int j=0; j<nnodes*sdim; ++j)
         {
            get(msg, el.nodes[j], ind);
         }

         els_to_add.push_back(el);
      }

      int nbe_to_add;
      get(msg, nbe_to_add,ind);
      for (int i=0; i<nbe_to_add; ++i)
      {
         GhostBE be;
         get(msg, be.attr, ind);
         for (int j=0; j<nv_per_be; ++j)
         {
            get(msg, be.v[j], ind);
         }
         be_to_add.push_back(be);
      }
   }

   // All communication should be finished now because the receives are
   // synchronous. This is maybe not the most efficient way to do it, so there
   // is possible room for future improvement.
   MPI_Waitall(nborrowers, send_reqs.data(), send_stat.data());

   // Now we actually construct the ghost mesh.
   // This is done in two steps.
   // (A) Add all the vertices of the ghost mesh
   //   (A.1) Add the vertices from the local partition
   //   (A.2) Add the ghost vertices
   //   Note: we need to ensure that we don't add overlapping vertices
   // (B) Add all the elements to the ghost mesh
   //   (B.1) Add the elements from the local partition.
   //         We own elements from the local partition, and we need to keep
   //         track of who is borrowing each element for their ghost layers.
   //   (B.2) Add the ghost elements

   // Create a local numbering of all vertices (mapping from global)
   std::unordered_map<int,int> glob2ghost;
   UniqueIndexGenerator gen;
   for (int i=0; i<nv_local; ++i)
   {
      int iv_global = vglob[i];
      int iv_local = gen.Get(iv_global);
      glob2ghost[iv_global] = iv_local;
   }
   for (int iv_global : v_to_add_id)
   {
      int iv_local = gen.Get(iv_global);
      glob2ghost[iv_global] = iv_local;
   }

   int nv_combined = gen.counter;
   std::vector<double> v_combined(nv_combined*3, 0.0);
   vertex_ownership.resize(nv_combined, false);
   // Step (A.1)
   for (int i=0; i<nv_local; ++i)
   {
      int iv_global = vglob[i];
      int iv = gen.Get(iv_global);
      for (int d=0; d<sdim; ++d)
      {
         v_combined[iv*3 + d] = pmesh.GetVertex(i)[d];
      }
      // Do we own this vertex
      int g = v2group[i];
      vertex_ownership[iv] = gtopo.IAmMaster(g);
   }
   // Step (A.2)
   for (size_t i=0; i<v_to_add.size(); ++i)
   {
      int iv_global = v_to_add_id[i];
      int iv = gen.Get(iv_global);
      for (int d=0; d<sdim; ++d)
      {
         v_combined[iv*3 + d] = v_to_add[i](d);
      }
   }

   int nel_local = pmesh.GetNE();
   int nel_ghost = els_to_add.size();
   int nel_combined = nel_local + nel_ghost;
   int nbe_ghost = be_to_add.size();
   std::vector<int> el_combined(nv_per_el*(nel_local + nel_ghost));
   std::vector<int> el_attrs(nel_local + nel_ghost, 1);

   std::vector<ElementOwnership> el_owners(nel_local + nel_ghost);

   // Step (B.1)
   // Add the local elements
   for (int i=0; i<nel_local; ++i)
   {
      int *el_v = pmesh.GetElement(i)->GetVertices();
      el_owners[i].owner_rank = my_rank;
      el_owners[i].borrowers.insert(
         el_owners[i].borrowers.end(),
         el_borrowers[i].begin(),
         el_borrowers[i].end()
      );
      el_attrs[i] = pmesh.GetAttribute(i);
      for (int j=0; j<nv_per_el; ++j)
      {
         int iv_local = el_v[j];
         int iv_global = vglob[iv_local];
         int iv = glob2ghost[iv_global];
         el_combined[i*nv_per_el + j] = iv;
      }
   }
   // Step (B.2)
   // Add the ghost elements
   for (int ighost=0; ighost<nel_ghost; ++ighost)
   {
      int i = nel_local + ighost;
      el_owners[i].owner_rank = els_to_add[ighost].owner_rank;
      el_attrs[i] = els_to_add[ighost].attr;
      for (int j=0; j<nv_per_el; ++j)
      {
         int iv_global = els_to_add[ighost].v[j];
         int iv = glob2ghost[iv_global];
         el_combined[i*nv_per_el + j] = iv;
      }
   }

   // Now add the boundary elements. We need to do this explicitly in order to
   // preserve the boundary attribute numbering which is important for setting
   // the correct boundary conditions.
   std::vector<int> bdr_idx, bdr_attr;
   for (int ib=0; ib<pmesh.GetNBE(); ++ib)
   {
      Element *el = pmesh.GetBdrElement(ib);
      int attr = el->GetAttribute();
      int nv = el->GetNVertices();
      int *v = el->GetVertices();
      for (int iv=0; iv<nv; ++iv)
      {
         bdr_idx.push_back(v[iv]);
      }
      bdr_attr.push_back(attr);
   }

   // Add the boundary elements from the ghost layer
   for (int ib=0; ib<nbe_ghost; ++ib)
   {
      GhostBE &be = be_to_add[ib];
      for (int iv=0; iv<nv_per_be; ++iv)
      {
         int v = glob2ghost[be.v[iv]];
         bdr_idx.push_back(v);
      }
      bdr_attr.push_back(be.attr);
   }
   int nbdr = bdr_attr.size();

   int local_max_attr = pmesh.bdr_attributes.Size()
                      ? pmesh.bdr_attributes.Max() : 0;
   int dummy_attr;
   MPI_Allreduce(&local_max_attr, &dummy_attr, 1, MPI_INT, MPI_MAX, comm);
   ++dummy_attr;

   bool discont = false;
   if (pmesh.GetNodalFESpace())
   {
      discont = pmesh.GetNodalFESpace()->IsDGSpace();
   }

   mesh.reset(new GhostMeshGen(
                 dim, sdim, nv_combined, v_combined.data(),
                 nel_combined, el_combined.data(), el_attrs.data(),
                 nbdr, bdr_idx.data(), bdr_attr.data(), dummy_attr));

   // Set the nodes for curved meshes
   mesh->SetCurvature(order, discont, -1, Ordering::byNODES);
   GridFunction *new_nodes = mesh->GetNodes();
   // Force grid function on host
   new_nodes->HostReadWrite();
   FiniteElementSpace *new_fes = new_nodes->FESpace();

   for (int i=0; i<nel_local; ++i)
   {
      Array<int> new_vdofs, orig_vdofs;
      orig_fes->GetElementVDofs(i, orig_vdofs);
      new_fes->GetElementVDofs(i, new_vdofs);
      for (int j=0; j<nnodes*sdim; ++j)
      {
         (*new_nodes)[new_vdofs[j]] = (*orig_nodes)[orig_vdofs[j]];
      }
   }
   for (int ighost=0; ighost<nel_ghost; ++ighost)
   {
      int i = nel_local + ighost;
      Array<int> vdofs;
      new_fes->GetElementVDofs(i, vdofs);
      for (int j=0; j<nnodes*sdim; ++j)
      {
         (*new_nodes)[vdofs[j]] = els_to_add[ighost].nodes[j];
      }
   }
   mesh->Finalize(false, true);

   // There are two main operations that we want to perform with the ghost
   // mesh.
   // 1. Populate the ghost layer. This requires sending information from any
   //    element that we own that is borrowed from us to the borrowing process.
   //    We then receive messages from all of the process who we borrow from
   //    and insert the results into the correct elements.
   //
   // 2. Distribute the ghost layer information to the owners of the elements.
   //    This requires sending information from any element we borrow to that
   //    element's owner.
   //    We then receive information corresponding to any element that we own
   //    that is borrowed by another process.

   // In order to address operation 1, we split this into two parts: (1S) and
   // (1R), for sending and receiving respectively.
   // For part (1S), we need to have a list of destination processors that
   // are borrowing our elements. This is stored in the borrowers variable
   // For part (1R), we need to have a list of processes who we are borrowing
   // from. Those processes own elements in our ghost layer. This is stored in
   // the owners variable.
   for (int iel=0; iel<mesh->GetNE(); ++iel)
   {
      ElementOwnership &el_owner_info = el_owners[iel];
      for (int borrower_rank : el_owner_info.borrowers)
      {
         // Add this element to the corresponding borrower table
         // Note: the [] access operator will create an empty entry if it
         // doesn't exist already.
         borrowers[borrower_rank].push_back(iel);
      }
      if (el_owner_info.owner_rank != my_rank)
      {
         // See note above if entry does not exist in map
         owners[el_owner_info.owner_rank].push_back(iel);
      }
   }

   // DEBUG: Write out to a file for vis/analysis
   std::ostringstream fname;
   fname << "output/mesh_ghost." << std::setfill('0') << std::setw(6)
         << gtopo.MyRank();
   std::ofstream of(fname.str().c_str());
   mesh->Print(of);
}

bool GhostMesh::IsVertexOwned(int iv) const
{
   return vertex_ownership[iv];
}

LORGhostMesh::LORGhostMesh(ParMesh &pmesh, int order, int ref_type)
   : gmesh(pmesh),
     mesh0(*gmesh.mesh),
     mesh_lor(&mesh0, order, ref_type),
     fec(1, mesh_lor.Dimension()),
     fes(&mesh_lor, &fec)
{
   // DEBUG: Write out to a file for vis/analysis
   std::ostringstream fname;
   fname << "output/mesh_ghost_lor." << std::setfill('0') << std::setw(6)
         << gmesh.my_rank;
   std::ofstream of(fname.str().c_str());
   mesh_lor.Print(of);

   // Create the ho2lor map, which lists the refined elements corresponding
   // to a given coarse element
   int nel_ho = mesh0.GetNE();
   int nel_lor = mesh_lor.GetNE();
   nref = nel_lor/nel_ho;
   ho2lor.SetSize(nel_ho, nref);
   const CoarseFineTransformations &cf_tr =
      mesh_lor.GetRefinementTransforms();
   for (int ilor=0; ilor<nel_lor; ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      ho2lor.AddConnection(iho, ilor);
   }
   ho2lor.ShiftUpI();

   Mesh &orig_mesh = gmesh.pmesh;
   H1_FECollection fec_ho(order, mesh_lor.Dimension());
   FiniteElementSpace fes_ho(&orig_mesh, &fec_ho);

   const TensorBasisElement *fe_ho =
      dynamic_cast<const TensorBasisElement *>(fes_ho.GetFE(0));
   MFEM_ASSERT(fe_ho != NULL, "Must use tensor-product basis");
   const Array<int> &dof_map = fe_ho->GetDofMap();

   perm.SetSize(fes_ho.GetNDofs());
   perm = -1;
   for (int iel0=0; iel0<orig_mesh.GetNE(); ++iel0)
   {
      Array<int> dofs_ho;
      fes_ho.GetElementDofs(iel0, dofs_ho);
      int ndofs = dofs_ho.Size();
      for (int ii=0; ii<ndofs; ++ii)
      {
         int iref, lor_idx;
         GetLORIndex(ii, order, iref, lor_idx);
         int iel = ho2lor.GetRow(iel0)[iref];
         Array<int> lor_dofs;
         fes.GetElementDofs(iel, lor_dofs);

         int lor_dof = lor_dofs[lor_idx];
         int ho_dof = dofs_ho[dof_map[ii]];

         MFEM_ASSERT(perm[ho_dof] < 0 || perm[ho_dof] == lor_dof, "");
         perm[ho_dof] = lor_dof;
      }
   }

   // Figure out what data we need to send (and to whom) to populate the ghost
   // layers.
   for (auto borrower : gmesh.borrowers)
   {
      int their_rank = borrower.first;
      std::vector<int> &borrowed_els = borrower.second;
      std::set<int> inserted_dofs;
      std::vector<int> dof_vec;
      for (int iel0 : borrowed_els)
      {
         for (int iref=0; iref<nref; ++iref)
         {
            Array<int> dofs;
            int iel = ho2lor.GetRow(iel0)[iref];
            fes.GetElementDofs(iel, dofs);
            for (int i=0; i<dofs.Size(); ++i)
            {
               int idof = dofs[i];
               if (inserted_dofs.find(idof) == inserted_dofs.end())
               {
                  inserted_dofs.insert(idof);
                  dof_vec.push_back(idof);
               }
            }
         }
      }
      borrower_dofs[their_rank].SetSize(dof_vec.size());
      borrower_dofs[their_rank].Assign(dof_vec.data());
      borrower_msg[their_rank].SetSize(dof_vec.size());
   }

   // Figure out what data we need to receive (and from whom) to populate our
   // ghost layer.
   for (auto owner : gmesh.owners)
   {
      int their_rank = owner.first;
      std::vector<int> &owned_els = owner.second;
      std::set<int> inserted_dofs;
      std::vector<int> dof_vec;
      for (int iel0 : owned_els)
      {
         for (int iref=0; iref<nref; ++iref)
         {
            Array<int> dofs;
            int iel = ho2lor.GetRow(iel0)[iref];
            fes.GetElementDofs(iel, dofs);
            for (int i=0; i<dofs.Size(); ++i)
            {
               int idof = dofs[i];
               if (inserted_dofs.find(idof) == inserted_dofs.end())
               {
                  inserted_dofs.insert(idof);
                  dof_vec.push_back(idof);
               }
            }
         }
      }
      owner_dofs[their_rank].SetSize(dof_vec.size());
      owner_dofs[their_rank].Assign(dof_vec.data());
      owner_msg[their_rank].SetSize(dof_vec.size());
   }

   // Preallocate vectors for the MPI requests
   borrower_reqs.resize(borrower_dofs.size());
   borrower_stat.resize(borrower_dofs.size());
   owner_reqs.resize(owner_dofs.size());
   owner_stat.resize(owner_dofs.size());
}

void LORGhostMesh::PopulateOwnedDofs(const Vector &x_in, Vector &x_g)
{
   for (int i=0; i<x_in.Size(); ++i)
   {
      x_g[perm[i]] = x_in[i];
   }
}

void LORGhostMesh::ExtractOwnedDofs(const Vector &x_g, Vector &x_out)
{
   for (int i=0; i<x_out.Size(); ++i)
   {
      x_out[i] = x_g[perm[i]];
   }
}

void LORGhostMesh::PopulateGhostLayer(Vector &x)
{
   // First post the send requests. We need to send data to anyone who is
   // borrowing from us.
   int req_idx = 0;
   for (auto borrower : borrower_dofs)
   {
      int their_rank = borrower.first;
      Array<int> &dofs_to_send = borrower.second;
      Vector &msg = borrower_msg[their_rank];
      int msg_sz = sizeof(double)*msg.Size();

      x.GetSubVector(dofs_to_send, msg);

      MPI_Isend(msg.GetData(), msg_sz, MPI_BYTE, their_rank, 1, gmesh.comm,
                &borrower_reqs[req_idx++]);
   }

   // Then post the receive requests
   req_idx = 0;
   for (auto owner : owner_dofs)
   {
      int their_rank = owner.first;
      Vector &msg = owner_msg[their_rank];
      int msg_sz = sizeof(double)*msg.Size();

      MPI_Irecv(msg.GetData(), msg_sz, MPI_BYTE, their_rank, 1, gmesh.comm,
                &owner_reqs[req_idx++]);
   }

   MPI_Waitall(borrower_reqs.size(),borrower_reqs.data(),borrower_stat.data());
   MPI_Waitall(owner_reqs.size(),owner_reqs.data(),owner_stat.data());

   // Communication is done, now we populate the vector with messages received
   // from the element owners
   for (auto owner : owner_dofs)
   {
      int their_rank = owner.first;
      Array<int> &dofs_to_recv = owner.second;
      x.SetSubVector(dofs_to_recv, owner_msg[their_rank]);
   }
}

void LORGhostMesh::AddToOwners(Vector &x, double alpha)
{
   // First post the send requests. We need to send data to everyone who we
   // borrow from (they are the owners)
   int req_idx = 0;
   for (auto owner : owner_dofs)
   {
      int their_rank = owner.first;
      Array<int> &dofs_to_send = owner.second;
      Vector &msg = owner_msg[their_rank];
      int msg_sz = sizeof(double)*msg.Size();

      x.GetSubVector(dofs_to_send, msg);

      MPI_Isend(msg.GetData(), msg_sz, MPI_BYTE, their_rank, 1, gmesh.comm,
                &owner_reqs[req_idx++]);
   }

   // Then post the receive requests. We receive data from everyone who
   // borrows from us.
   req_idx = 0;
   for (auto borrower : borrower_dofs)
   {
      int their_rank = borrower.first;
      Vector &msg = borrower_msg[their_rank];
      int msg_sz = sizeof(double)*msg.Size();

      MPI_Irecv(msg.GetData(), msg_sz, MPI_BYTE, their_rank, 1, gmesh.comm,
                &borrower_reqs[req_idx++]);
   }

   MPI_Waitall(borrower_reqs.size(),borrower_reqs.data(),borrower_stat.data());
   MPI_Waitall(owner_reqs.size(),owner_reqs.data(),owner_stat.data());

   // Communication is done, now we add the ghost information from borrowing
   // partitions (times coefficient alpha) to our owned elements
   for (auto borrower : borrower_dofs)
   {
      int their_rank = borrower.first;
      Array<int> &dofs_to_add = borrower.second;
      x.AddElementVector(dofs_to_add, alpha, borrower_msg[their_rank]);
   }

}

void LORGhostMesh::Populate(const Vector &x_in, Vector &x_g)
{
   PopulateOwnedDofs(x_in, x_g);
   PopulateGhostLayer(x_g);
}
