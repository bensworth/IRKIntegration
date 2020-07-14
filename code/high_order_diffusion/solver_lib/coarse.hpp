#ifndef __AS_COARSE__
#define __AS_COARSE__

#include "mfem.hpp"
#include <memory>

using namespace mfem;

struct LocalCoarsening1D
{
   int nlevels;
   std::vector<SparseMatrix> P;
   std::vector<std::vector<int>> dofs_to_keep;

   void Init(int porder);
};

#endif
