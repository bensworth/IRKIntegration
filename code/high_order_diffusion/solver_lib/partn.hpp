#ifndef __AS_PARTN__
#define __AS_PARTN__

#include "mfem.hpp"

using namespace mfem;

struct VertexPartitioning
{
   std::vector<std::vector<int>> v;
};

VertexPartitioning PartitionMeshVertices(Mesh &mesh, int &nparts);

#endif