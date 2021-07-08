#include "irk_ilu.hpp"

namespace mfem
{

ILU_IRK::ILU_IRK(RKData::Type rk_type, HypreParMatrix &M, HypreParMatrix &L_,
                 int block_size, double dt, MPI_Comm comm)
: rk_data(rk_type), L(L_), prec(block_size), gmres(comm)
{
   int s = rk_data.s;
   DenseMatrix &Ainv = rk_data.invA0;
   Array2D<HypreParMatrix*> blocks(s,s);
   for (int i=0; i<s; ++i)
   {
      for (int j=0; j<s; ++j)
      {
         if (i == j) { continue; }
         blocks(i,j) = new HypreParMatrix(M);
         (*blocks(i,j)) *= Ainv(i,j);
      }
      blocks(i,i) = Add(Ainv(i,i), M, -dt, L);
   }

   HypreParMatrix *block_matrix = HypreParMatrixFromBlocks(blocks, NULL);
   J.reset(block_matrix);

   for (int i=0; i<s; ++i)
   {
      for (int j=0; j<s; ++j)
      {
         delete blocks(i,j);
      }
   }

   int n = L.Height(); // System size
   w.SetSize(n*s);
   b.SetSize(n*s);
   Lu.SetSize(n);

   prec.SetOperator(*J);
   gmres.SetAbsTol(1e-10);
   gmres.SetRelTol(1e-10);
   gmres.SetMaxIter(1000);
   gmres.SetKDim(1000);
   gmres.SetPrintLevel(2);
   gmres.SetPreconditioner(prec);
   gmres.SetOperator(*J);
}

void ILU_IRK::Step(Vector &u, double &t, double &dt)
{
   int s = rk_data.s;
   int n = u.Size();

   // Form RHS
   L.Mult(u, Lu);
   for (int i=0; i<s; ++i)
   {
      bs.MakeRef(b, n*i, n);
      bs = Lu;
   }
   // Solve system
   gmres.Mult(b, w);
   // Update
   Vector &btAinv = rk_data.d0;
   for (int i=0; i<s; ++i)
   {
      ws.MakeRef(w, n*i, n);
      u.Add(dt*btAinv[i], ws);
   }

   t += dt;
}

}
