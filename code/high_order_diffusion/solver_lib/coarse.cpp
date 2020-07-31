#include "coarse.hpp"

void LocalCoarsening1D::Init(int porder)
{
   const double *x1d_ = poly1d.ClosedPoints(porder, BasisType::GaussLobatto);
   std::vector<double> x1d(x1d_, x1d_ + porder + 1);

   nlevels = std::ceil(std::log2(double(porder+1))) - 1;

   P.resize(nlevels);

   int ndof = porder+1;
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      // Create list of points to keep
      std::vector<int> to_keep;
      // Keep the endpoints and half of the interior points
      int nkeep = 2 + (ndof-2)/2;
      double step = double(ndof+1)/nkeep;
      // Choose the points to keep
      // Note: left endpoint is automatically included (when i=0)
      for (int i=0; i<nkeep; ++i)
      {
         to_keep.push_back(std::round(i*step));
      }
      // Ensure right endpoint is included
      to_keep.back() = ndof-1;

      dofs_to_keep.push_back(to_keep);

      std::vector<double> x1d_new(nkeep);
      for (int i=0; i<nkeep; ++i)
      {
         x1d_new[i] = x1d[to_keep[i]];
      }

      // Form local interpolation operator
      // Note: SetSize will initialize the entries to zero
      SparseMatrix P_current(ndof, nkeep);
      for (int j=0; j<nkeep; ++j)
      {
         double xleft = (j > 0) ? x1d_new[j-1] : 0.0;
         double xright = (j < nkeep - 1) ? x1d_new[j+1] : 1.0;
         double x0 = x1d_new[j];
         for (int i=0; i<ndof; ++i)
         {
            double x = x1d[i];
            if (x >= xleft && x <= xright)
            {
               if (x == x0)
               {
                  P_current.Set(i,j,1.0);
               }
               else if (x < x0)
               {
                  P_current.Set(i,j,(x - xleft)/(x0 - xleft));
               }
               else
               {
                  P_current.Set(i,j,(xright - x)/(xright - x0));
               }
            }
         }
      }
      P_current.Finalize();
      P[ilevel].Swap(P_current);
      std::swap(x1d, x1d_new);
      ndof = nkeep;
   }
}
