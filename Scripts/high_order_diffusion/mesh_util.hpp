#ifndef __MESH_UTIL_HPP__
#define __MESH_UTIL_HPP__

#include "mfem.hpp"

namespace mfem
{

void IdentifyPeriodicMeshVertices(const Mesh & mesh,
                                  const std::vector<Vector> & trans_vecs,
                                  Array<int> & v2v,
                                  int logging);

Mesh *MakePeriodicMesh(Mesh * mesh, const Array<int> & v2v, int logging);

} // namespace mfem

#endif
