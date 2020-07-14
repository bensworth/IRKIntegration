#include "util.hpp"
#include <fstream>

SparseMatrix *MakeZeroRowsIdentity(SparseMatrix *A)
{
   const double threshold = 1e-12;
   int height = A->Height();
   int *I = A->GetI();
   double *V = A->GetData();

   Vector diag(A->Height());

   for (int i = 0; i < height; i++)
   {
      double zero = 0.0;
      for (int j = I[i]; j < I[i+1]; j++)
      {
         zero += fabs(V[j]);
      }
      if (zero <= threshold)
      {
         diag[i] = 1.0;
      }
      else
      {
         diag[i] = 0.0;
      }
   }

   SparseMatrix D(diag);
   return Add(*A, D);
}

int GetFaceIndex(int iel, int fid, Mesh &mesh)
{
   int dim = mesh.Dimension();
   Array<int> f, cor;
   if (dim == 1)
   {
      mesh.GetElementVertices(iel, f);
      return f[fid];
   }
   else if (dim == 2)
   {
      mesh.GetElementEdges(iel, f, cor);
      return f[Cart2Face2D(fid)];
   }
   else if (dim == 3)
   {
      mesh.GetElementFaces(iel, f, cor);
      return f[Cart2Face3D(fid)];
   }
   MFEM_ABORT("Dimension must be 1, 2, or 3.");
   return -1;
}

void GetLORIndex(int ii, int order, int &iref, int &dof)
{
   // ii is a linear index into a high-order coarse element, ordered using
   // Cartesian numbering. The coarse element is divided up into LOR elements.
   // We want to return the element number and local DOF index in the LOR space
   // corresponding to the high-order DOF ii.
   //
   // We try to identify from which element to take the DOF.
   // If we can, we take the element so that the DOF lies at at (0,0,0)
   // vertex. This doesn't work for DOFs lying on the right, top, or back
   // faces of the macro element, so we need to adjust.
   int i = ii % (order+1);
   int j = (ii / (order+1)) % (order+1);
   int k = ii / sq(order+1);
   int cidx = 0;
   if (i == order) { --i; cidx += 1; }
   if (j == order) { --j; cidx += 2; }
   if (k == order) { --k; cidx += 4; }
   dof = Cart2Node(cidx);
   iref = k*sq(order) + j*order + i;
}

void ParPrintMesh(const std::string &prefix, ParMesh &pmesh, int myid)
{
   std::ostringstream fname;
   fname << prefix << "." << std::setfill('0') << std::setw(6) << myid;
   std::ofstream ofs(fname.str().c_str());
   pmesh.Print(ofs);
}

void ParPrintSoln(const std::string &prefix, ParGridFunction &x, int myid)
{
   std::ostringstream fname;
   fname << prefix << "." << std::setfill('0') << std::setw(6) << myid;
   std::ofstream ofs(fname.str().c_str());
   x.Save(ofs);
}
