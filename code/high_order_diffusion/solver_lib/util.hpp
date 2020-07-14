#ifndef __AS_UTIL__
#define __AS_UTIL__

#include "mfem.hpp"
#include <unordered_map>

using namespace mfem;

template <typename T> T sq(T x) { return x*x; }

struct UniqueIndexGenerator
{
   int counter = 0;
   std::unordered_map<int,int> idx;
   int Get(int i)
   {
      std::unordered_map<int,int>::iterator f = idx.find(i);
      if (f == idx.end())
      {
         idx[i] = counter;
         return counter++;
      }
      else
      {
         return (*f).second;
      }
   }
   void Reset()
   {
      counter = 0;
      idx.clear();
   }
};

// Convert from Cartesian numbering to H1 node numbering.
// Given coordinates (x,y,z) that identify a vertex on [0,1]^d, the index i
// in Cartesian numbering is computed by x + 2*y + 4*z.
// This routine converts to MFEM's H1 tensor-product numbering.
// (See constructor for TensorBasisElement for 'derivation')
inline int Cart2Node(int cartidx)
{
   static int dof_map[8] = {0,1,3,2,4,5,7,6};
   return dof_map[cartidx];
}

// Cartesian indexing: x=0, x=1, y=0, y=1
inline int Cart2Face2D(int fid)
{
   static int map[4] = {3, 1, 0, 2};
   return map[fid];
}

// Cartesian indexing: x=0, x=1, y=0, y=1, z=0, z=1
inline int Cart2Face3D(int fid)
{
   static int map[6] = {4, 2, 1, 3, 0, 5};
   return map[fid];
}

int GetFaceIndex(int iel, int fid, Mesh &mesh);

void GetLORIndex(int ii, int order, int &iref, int &dof);

SparseMatrix *MakeZeroRowsIdentity(SparseMatrix *A);

template <class T>
void put(std::vector<char> &vec, const T &val)
{
   for (unsigned char ib=0; ib<sizeof(T); ++ib)
   {
      char byte = *(reinterpret_cast<const char*>(&val) + ib);
      vec.push_back(byte);
   }
}

template <class T>
void get(std::vector<char> &vec, T &val, size_t &ind)
{
   std::memcpy(&val, &vec[ind], sizeof(T));
   ind += sizeof(T);
}

void ParPrintMesh(const std::string &prefix, ParMesh &pmesh, int myid);
void ParPrintSoln(const std::string &prefix, ParGridFunction &x, int myid);

#endif
