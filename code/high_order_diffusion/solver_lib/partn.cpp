#include "partn.hpp"
#include "metis.h"

VertexPartitioning PartitionMeshVertices(Mesh &mesh, int &nparts)
{
   idx_t nel = mesh.GetNE();
   idx_t nv = mesh.GetNV();
   Geometry::Type geom = mesh.GetElement(0)->GetGeometryType();
   int nv_per_element = Geometry::NumVerts[geom];
   Array<idx_t> eptr(nel+1), eind(nel*nv_per_element);

   // Construct the Metis arrays for the element vertices.
   // The vertices for element i are found in the eind array, for indices in
   // between eptr[i] and (up to but not including) eptr[i+1].
   idx_t idx = 0;
   for (int i=0; i<nel; ++i)
   {
      Array<int> v;
      eptr[i] = idx;
      mesh.GetElementVertices(i, v);
      for (int j=0; j<v.Size(); ++j)
      {
         eind[int(idx++)] = v[j];
      }
   }
   eptr[int(nel)] = idx;

   Array<idx_t> npart(nv);

   nparts = (nparts > nv) ? nv : nparts;
   if (nparts > 1)
   {
      idx_t objval;
      Array<idx_t> epart(nel);
      idx_t opts[METIS_NOPTIONS];
      METIS_SetDefaultOptions(opts);

      idx_t nparts_l = nparts;
      idx_t res = METIS_PartMeshNodal(
                     &nel,
                     &nv,
                     eptr.GetData(),
                     eind.GetData(),
                     NULL, // All nodes are equally weighted
                     NULL, // All nodes have equal size,
                     &nparts_l,
                     NULL, // All partitions have equal weight
                     opts,
                     &objval,
                     epart.GetData(),
                     npart.GetData()
                  );
      MFEM_ASSERT(res == METIS_OK, "Error partitioning mesh with Metis");
   }
   else if (nparts == 1)
   {
      // One partition is trivial (but Metis freaks out)
      npart = 0;
   }
   else
   {
      // Negative number of partitions: one patch per vertex
      for (int i=0; i<nv; ++i)
      {
         npart[i] = i;
      }
      nparts = nv;
   }

   VertexPartitioning vp;

   vp.v.resize(nparts);
   for (int i=0; i<nv; ++i)
   {
      int part = npart[i];
      vp.v[part].push_back(i);
   }
   return vp;
}
