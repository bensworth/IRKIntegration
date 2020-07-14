#include "ilu.hpp"
#include "util.hpp"

ILU0::ILU0(SparseMatrix &A)
{
   Factorize(&A);
}

void ILU0::SetOperator(const Operator &op)
{
   const SparseMatrix *A = dynamic_cast<const SparseMatrix*>(&op);
   if (!A)
   {
      MFEM_ABORT("ILU0 must be created using a sparse matrix")
   }
   Factorize(A);
}

void ILU0::Factorize(const SparseMatrix *A)
{
   MFEM_ASSERT(A->Finalized(), "Matrix must be finalized for ILU0");

   n = A->Height();
   LU.reset(new SparseMatrix(*A));
   // LU->UseDevice(false); // This probably is unecessary
   // Note: MDFOrdering overwrites LU
   MDFOrdering(*LU, p);
   Array<int> pinv(n);
   for (int i=0; i<n; ++i)
   {
      pinv[p[i]] = i;
   }

   // Set LU = A(P,P) using the permutation generated above
   const int *IA = A->GetI();
   const int *JA = A->GetJ();
   const double *VA = A->GetData();

   int *I = LU->GetI();
   int *J = LU->GetJ();
   double *V = LU->GetData();

   I[0] = 0;
   for (int i=0; i<n; ++i)
   {
      int pi = p[i];
      int nnz_pi = IA[pi+1] - IA[pi];
      I[i+1] = I[i] + nnz_pi;
      for (int jj=0; jj<nnz_pi; ++jj)
      {
         int pj = JA[IA[pi] + jj];
         int j = pinv[pj];

         J[I[i] + jj] = j;
         V[I[i] + jj] = VA[IA[pi] + jj];
      }
   }

   // Compute factorization (overwrite LU)
   // Note: sorting the column indices is critical for the algorithm below.
   // This assumption is made several times in the code.
   LU->SortColumnIndices();

   ID.SetSize(n);
   ID[0] = 0;

   // Loop over rows
   for (int i=1; i<n; ++i)
   {
      // Find all nonzeros to the left of the diagonal in row i
      for (int kk=I[i]; kk<I[i+1]; ++kk)
      {
         int k = J[kk];
         // Make sure we're still to the left of the diagonal
         if (k >= i)
         {
            // Keep track of where the diagonal is
            if (k == i)
            {
               ID[i] = kk;
            }
            else
            {
               MFEM_ABORT("Missing diagonal entry");
            }
            break;
         }
         // L_ik = L_ik / U_
         V[kk] = V[kk]/V[ID[k]];
         // Modify everything to the right of k in row i
         for (int jj=kk+1; jj<I[i+1]; ++jj)
         {
            int j = J[jj];
            if (j <= k)
            {
               continue;
            }
            for (int ll=I[k]; ll<I[k+1]; ++ll)
            {
               int l = J[ll];
               if (l == j)
               {
                  V[jj] = V[jj] - V[kk]*V[ll];
                  break;
               }
            }
         }
      }
   }

   // Allocate work vector
   y.SetSize(n);
}

void ILU0::Mult(const Vector &b, Vector &x) const
{
   int *I = LU->GetI();
   int *J = LU->GetJ();
   double *V = LU->GetData();

   // Forward substitute to solve Ly = b
   // Implicitly, L has identity on the diagonal
   y = 0.0;
   for (int i=0; i<n; ++i)
   {
      y[i] = b[p[i]];
      for (int jj=I[i]; jj<ID[i]; ++jj)
      {
         int j = J[jj];
         y[i] -= V[jj]*y[j];
      }
   }
   // Backward substitution to solve Ux = y
   for (int i=n-1; i >= 0; --i)
   {
      x[p[i]] = y[i];
      for (int jj=ID[i]+1; jj<I[i+1]; ++jj)
      {
         int j = J[jj];
         x[p[i]] -= V[jj]*x[p[j]];
      }
      x[p[i]] /= V[ID[i]];
   }
}

SparseMatrix *ILU0::GetLU()
{
   return LU.get();
}

struct WeightMinHeap
{
   const std::vector<double> &w;
   std::vector<int> c, loc;

   WeightMinHeap(const std::vector<double> &w_) : w(w_)
   {
      c.reserve(w.size());
      loc.resize(w.size());
      for (int i=0; i<w.size(); ++i) { push(i); }
   }

   int percolate_up(int pos, double val)
   {
      for (; pos > 0 && w[c[(pos-1)/2]] > val; pos = (pos-1)/2)
      {
         c[pos] = c[(pos-1)/2];
         loc[c[(pos-1)/2]] = pos;
      }
      return pos;
   }

   int percolate_down(int pos, double val)
   {
      while (2*pos+1 < c.size())
      {
         int left = 2*pos+1;
         int right = left+1;
         int tgt;
         if (right < c.size() && w[c[right]] < w[c[left]]) { tgt = right; }
         else { tgt = left; }
         if (w[c[tgt]] < val)
         {
            c[pos] = c[tgt];
            loc[c[tgt]] = pos;
            pos = tgt;
         }
         else
         {
            break;
         }
      }
      return pos;
   }

   void push(int i)
   {
      double val = w[i];
      c.push_back(-1);
      int pos = c.size()-1;
      pos = percolate_up(pos, val);
      c[pos] = i;
      loc[i] = pos;
   }

   int pop()
   {
      int i = c[0];
      int j = c.back();
      c.pop_back();
      double val = w[j];
      int pos = 0;
      pos = percolate_down(pos, val);
      c[pos] = j;
      loc[j] = pos;
      // Mark as removed
      loc[i] = -1;
      return i;
   }

   void update(int i)
   {
      int pos = loc[i];
      double val = w[i];
      pos = percolate_up(pos, val);
      pos = percolate_down(pos, val);
      c[pos] = i;
      loc[i] = pos;
   }

   bool picked(int i)
   {
      return loc[i] < 0;
   }
};

void MDFOrdering(SparseMatrix &C, Array<int> &p)
{
   int n = C.Width();
   // Scale rows by reciprocal of diagonal and take absolute value
   Vector D;
   C.GetDiag(D);
   int *I = C.GetI();
   int *J = C.GetJ();
   double *V = C.GetData();
   for (int i=0; i<n; ++i)
   {
      for (int j=I[i]; j<I[i+1]; ++j)
      {
         V[j] = abs(V[j]/D[i]);
      }
   }

   std::vector<double> w(n, 0.0);
   // Compute the discarded-fill weights
   for (int ik=0; ik<n; ++ik)
   {
      for (int i=I[ik]; i<I[ik+1]; ++i)
      {
         int ji = J[i];
         double val;
         for (int j=I[ji]; j<I[ji+1]; ++j)
         {
            if (J[j] == ik)
            {
               val = V[j];
               break;
            }
         }
         for (int j=I[ik]; j<I[ik+1]; ++j)
         {
            int jj = J[j];
            if (ji == jj) { continue; }
            w[ik] += sq(val*V[j]);
         }
      }
      w[ik] = sqrt(w[ik]);
   }

   WeightMinHeap w_heap(w);

   // Compute ordering
   p.SetSize(n);
   for (int ii=0; ii<n; ++ii)
   {
      int pi = w_heap.pop();
      p[ii] = pi;
      w[pi] = -1;
      for (int k=I[pi]; k<I[pi+1]; ++k)
      {
         int ik = J[k];
         if (w_heap.picked(ik)) { continue; }
         // Recompute weight
         w[ik] = 0.0;
         for (int i=I[ik]; i<I[ik+1]; ++i)
         {
            int ji = J[i];
            if (w_heap.picked(ji)) { continue; }
            double val;
            for (int j=I[ji]; j<I[ji+1]; ++j)
            {
               if (J[j] == ik)
               {
                  val = V[j];
                  break;
               }
            }
            for (int j=I[ik]; j<I[ik+1]; ++j)
            {
               int jj = J[j];
               // Ignore entries we have already chosen
               if (ji == jj || w_heap.picked(jj)) { continue; }
               w[ik] += sq(val*V[j]);
            }
         }
         w[ik] = sqrt(w[ik]);
         w_heap.update(ik);
      }
   }
}
