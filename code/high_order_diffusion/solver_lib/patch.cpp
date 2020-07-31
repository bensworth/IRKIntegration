#include "patch.hpp"
#include "ilu.hpp"
#include "partn.hpp"

#include <unordered_set>

bool IsFaceInterior(int iel, int face, Mesh &mesh,
                    const std::unordered_set<int> &els)
{
   int iel1, iel2;
   int nbr;
   mesh.GetFaceElements(face, &iel1, &iel2);

   nbr = (iel1 == iel) ? iel2 : iel1;

   return !(els.find(nbr) == els.end());
}

bool IsDofInteriorToPatch(int iel, int ii, int n,
                          Mesh &mesh0, const std::unordered_set<int> &els)
{
   int dim = mesh0.Dimension();
   int i = ii%n;
   int j = (ii/n)%n;
   int k = ii/sq(n);
   int face = -1;

   if (i == 0 || i == n-1)
   {
      int fid = (i == 0) ? 0 : 1;
      face = GetFaceIndex(iel, fid, mesh0);
      bool interior = IsFaceInterior(iel, face, mesh0, els);
      // Short circuit to avoid further checks
      if (!interior) { return false; }
   }
   if (dim >= 2 && (j == 0 || j == n-1))
   {
      int fid = (j == 0) ? 2 : 3;
      face = GetFaceIndex(iel, fid, mesh0);
      bool interior = IsFaceInterior(iel, face, mesh0, els);
      if (!interior) { return false; }
   }
   if (dim >= 3 && (k == 0 || k == n-1))
   {
      int fid = (k == 0) ? 4 : 5;
      face = GetFaceIndex(iel, fid, mesh0);
      bool interior = IsFaceInterior(iel, face, mesh0, els);
      if (!interior) { return false; }
   }
   return true;
}

PatchCoarsening::PatchCoarsening(LORGhostMesh &glor, const Array<int> &els,
                                 Array<int> &ess_bdr, bool multilevel)
{
   FiniteElementSpace &fes = glor.fes;
   Mesh &mesh0 = glor.mesh0;
   Table &ho2lor = glor.ho2lor;
   int nref = glor.nref;

   int dim = fes.GetMesh()->Dimension();
   int porder = std::round(std::pow(nref, 1.0/dim));

   if (multilevel)
   {
      local_coarsening.Init(porder);
      nlevels = local_coarsening.nlevels;
   }
   else
   {
      nlevels = 0;
   }

   Array<int> is_dof_ess, is_dof_bdry, interp_global_dof;
   glor.fes.GetEssentialVDofs(ess_bdr, is_dof_ess);
   Array<int> all_ess(ess_bdr.Size());
   all_ess = 1;
   glor.fes.GetEssentialVDofs(all_ess, is_dof_bdry);

   std::unordered_set<int> els_set(els.begin(), els.end());

   // Global prolongation matrices
   P.resize(nlevels + 1);
   // Embeddings
   embed.resize(nlevels + 1);
   // First construct mapping from each coarse element to the global DOFs on
   // the LOR mesh. The DOFs on the coarse element are ordered using tensor
   // product (Cartesian) indexing
   embed[0].loc2glob.resize(els.Size());
   // Loop over every coarse element in the patch, and extract the fine DOFs
   UniqueIndexGenerator gen;
   P[0] = new SparseMatrix(fes.GetNDofs());
   int nlocaldofs = pow(porder+1, dim);

   // First figure out which DOFs we should interpolate. We don't interpolate
   // any DOF lying on an interior patch boundary or essential domain boundary.
   // Need to take care of corners, which may appear to be interior to one
   // element.
   interp_global_dof.SetSize(is_dof_ess.Size());
   interp_global_dof = 1;
   for (int iel0=0; iel0<els.Size(); ++iel0)
   {
      for (int ii=0; ii<nlocaldofs; ++ii)
      {
         int iref, lor_idx;
         GetLORIndex(ii, porder, iref, lor_idx);
         bool interior_pt = IsDofInteriorToPatch(
                               els[iel0], ii, porder+1, mesh0, els_set);
         int iel = ho2lor.GetRow(els[iel0])[iref];
         Array<int> global_dofs;
         fes.GetElementDofs(iel, global_dofs);
         int global_dof_id = global_dofs[lor_idx];
         if (!interior_pt)
         {
            if (!is_dof_bdry[global_dof_id])
            {
               interp_global_dof[global_dof_id] = 0;
            }
            else
            {
               if (is_dof_ess[global_dof_id])
               {
                  interp_global_dof[global_dof_id] = 0;
               }
            }
         }
      }
   }

   for (int iel0=0; iel0<els.Size(); ++iel0)
   {
      for (int ii=0; ii<nlocaldofs; ++ii)
      {
         int iref, lor_idx;
         GetLORIndex(ii, porder, iref, lor_idx);
         int iel = ho2lor.GetRow(els[iel0])[iref];
         Array<int> global_dofs;
         fes.GetElementDofs(iel, global_dofs);
         int global_dof_id = global_dofs[lor_idx];
         int dof_idx = gen.Get(global_dof_id);
         bool interpolate_dof = interp_global_dof[global_dof_id];
         embed[0].loc2glob[iel0].emplace_back(dof_idx, interpolate_dof);
         if (interpolate_dof)
         {
            P[0]->Set(global_dof_id, dof_idx, 1.0);
         }
      }
   }

   P[0]->SetWidth(gen.counter);
   P[0]->Finalize();
   // P[0]->UseDevice(false); // Ensure CSR operations performed on host
   embed[0].ndofs = gen.counter;
   embed[0].n1d = porder+1;

   perm.SetSize(gen.counter);
   perm = -1;
   {
      int *I = P[0]->GetI();
      int *J = P[0]->GetJ();
      for (int i=0; i<P[0]->Height(); ++i)
      {
         int iidx1 = I[i];
         int iidx2 = I[i+1];
         MFEM_ASSERT(iidx2 - iidx1 <= 1, "");
         if (iidx2 > iidx1)
         {
            int j = J[iidx1];
            perm[j] = i;
         }
      }
   }

   // Now, embed each coarsening in its parent
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      gen.Reset();
      std::vector<int> &dofs_to_keep = local_coarsening.dofs_to_keep[ilevel];
      int ndofs1d = dofs_to_keep.size();
      int n1d_prev = embed[ilevel].n1d;
      int ndofs_to_keep = pow(ndofs1d, dim);
      embed[ilevel+1].loc2glob.resize(els.Size());
      for (int iel0=0; iel0<els.Size(); ++iel0)
      {
         for (int ii=0; ii<ndofs_to_keep; ++ii)
         {
            int i = ii % ndofs1d;
            int j = (ii / ndofs1d) % ndofs1d;
            int k = ii / sq(ndofs1d);
            int i1 = dofs_to_keep[i];
            int j1 = dofs_to_keep[j];
            int k1 = dofs_to_keep[k];
            int dofidx = k1*sq(n1d_prev) + j1*n1d_prev + i1;
            int parentidx = embed[ilevel].loc2glob[iel0][dofidx].idx;
            bool interior_dof = embed[ilevel].loc2glob[iel0][dofidx].interior;
            int glob_idx = gen.Get(parentidx);
            embed[ilevel+1].loc2glob[iel0].emplace_back(glob_idx, interior_dof);
         }
      }
      embed[ilevel+1].ndofs = gen.counter;
      embed[ilevel+1].n1d = ndofs1d;
   }

   // Construct the prolongation matrix
   for (int ilevel=0; ilevel<nlevels; ++ilevel)
   {
      P[ilevel+1] = new SparseMatrix(embed[ilevel].ndofs,
                                     embed[ilevel+1].ndofs);

      int ncols1d = embed[ilevel+1].n1d;
      int nrows1d = embed[ilevel].n1d;
      int nrows = pow(nrows1d, dim);

      for (int iel0=0; iel0<els.Size(); ++iel0)
      {
         for (int ii=0; ii<nrows; ++ii)
         {
            int ix = ii % nrows1d;
            int iy = (ii / nrows1d) % nrows1d;
            int iz = ii / sq(nrows1d);
            int i = embed[ilevel].loc2glob[iel0][ii].idx;

            int *PI = local_coarsening.P[ilevel].GetI();
            int *PJ = local_coarsening.P[ilevel].GetJ();
            double *PV = local_coarsening.P[ilevel].GetData();

            for (int kx=PI[ix]; kx<PI[ix+1]; ++kx)
            {
               int jx = PJ[kx];
               double valx = PV[kx];
               if (dim >= 2)
               {
                  for (int ky=PI[iy]; ky<PI[iy+1]; ++ky)
                  {
                     int jy = PJ[ky];
                     double valy = PV[ky];
                     if (dim == 3)
                     {
                        for (int kz=PI[iz]; kz<PI[iz+1]; ++kz)
                        {
                           int jz = PJ[kz];
                           double valz = PV[kz];
                           int jj = jz*sq(ncols1d) + jy*ncols1d + jx;
                           int j = embed[ilevel+1].loc2glob[iel0][jj].idx;
                           P[ilevel+1]->Set(i, j, valx*valy*valz);
                        }
                     }
                     else
                     {
                        int jj = jy*ncols1d + jx;
                        int j = embed[ilevel+1].loc2glob[iel0][jj].idx;
                        P[ilevel+1]->Set(i, j, valx*valy);
                     }
                  }
               }
               else
               {
                  int j = embed[ilevel+1].loc2glob[iel0][jx].idx;
                  P[ilevel+1]->Set(i, j, valx);
               }
            }
         }
      }
      P[ilevel+1]->Finalize();
      // P[ilevel+1]->UseDevice(false); // Ensure CSR operations performed on host
      P[ilevel+1]->BuildTranspose();
   }
}

PatchCoarsening::~PatchCoarsening()
{
   for (size_t i=0; i<P.size(); ++i)
   {
      delete P[i];
   }
}

PatchSolver::PatchSolver(const PatchInfo &info, LORGhostMesh &glor,
                         SparseMatrix &A, Array<int> &ess_bdr, bool multilevel)
   :  coarsening(glor, info.els0, ess_bdr, multilevel)
{
   nlevels = coarsening.nlevels;
   Ps.resize(nlevels+1);
   As.resize(nlevels+1);
   Ss.resize(nlevels+1);
   for (int ilevel=0; ilevel<nlevels+1; ++ilevel)
   {
      Ps[ilevel] = coarsening.P[ilevel];
      SparseMatrix *tmp;
      if (ilevel == 0)
      {
         tmp = RAP(*Ps[ilevel], A, *Ps[ilevel]);
      }
      else
      {
         tmp = RAP(*Ps[ilevel], *As[ilevel-1], *Ps[ilevel]);
      }
      As[ilevel] = MakeZeroRowsIdentity(tmp);
      // Ensure Mult using As performed on host
      //  (TODO: I believe RAP already correctly sets this device...)
      // As[ilevel]->UseDevice(false);

      delete tmp;
      if (ilevel == nlevels)
      {
         int coarse_system_size = As[ilevel]->Width();
         int threshold = 1e3;
         if (coarse_system_size > threshold && multilevel)
         {
            std::cout << "Coarse system size: " << coarse_system_size << "\n"
                      << "Poisson preconditioner falling back to ILU(0)\n";
            Ss[ilevel] = new ILU0(*As[ilevel]);
         }
         else
         {
            // TODO: UMFPack is only removed because we didn't install SuiteSparse
            //Ss[ilevel] = new UMFPackSolver(*As[ilevel]);
            Ss[ilevel] = new ILU0(*As[ilevel]);
         }
      }
      else
      {
         //Ss[ilevel] = new GSSmoother(*As[ilevel]);
         //Ss[ilevel] = new UMFPackSolver(*As[ilevel]);
         //Ss[ilevel] = new DSmoother(*As[ilevel], 1);
         Ss[ilevel] = new ILU0(*As[ilevel]);
      }
   }

   bs.resize(nlevels+1);
   xs.resize(nlevels+1);
   rs.resize(nlevels+1);
   for (int ilevel=0; ilevel<nlevels+1; ++ilevel)
   {
      // Force work vectors to be on the host
      bs[ilevel] = new Vector(As[ilevel]->Width(), MemoryType::HOST);
      xs[ilevel] = new Vector(As[ilevel]->Width(), MemoryType::HOST);
      rs[ilevel] = new Vector(As[ilevel]->Width(), MemoryType::HOST);
      bs[ilevel]->UseDevice(false);
      xs[ilevel]->UseDevice(false);
      rs[ilevel]->UseDevice(false);
   }
   x_final.SetSize(As[0]->Width());
   b_init.SetSize(As[0]->Width());
}

void PatchSolver::SolveAdd(const Vector &b_in, Vector &x_out)
{
   // Restrict global residual to the patch
   for (int i=0; i<bs[0]->Size(); ++i)
   {
      int pi = coarsening.perm[i];
      if (pi >= 0) { b_init[i] = b_in[pi]; }
      else { b_init[i] = 0; }
   }

   int niter = 1;
   double beta = 0.5;
   x_final = 0.0;
   *bs[0] = b_init;

   for (int iiter=0; iiter<niter; ++iiter)
   {
      // Pre-smoothing
      for (int i=0; i<nlevels; ++i)
      {
         // Smooth
         Ss[i]->Mult(*bs[i], *xs[i]);
         if (beta != 1.0) { *xs[i] *= beta; }
         // Compute residual, r[i] = b[i] - A[i]*x[i]
         As[i]->Mult(*xs[i], *rs[i]);
         add(*bs[i], -1.0, *rs[i], *rs[i]);
         // Restrict to next level down in the heirarchy
         Ps[i+1]->MultTranspose(*rs[i], *bs[i+1]);
      }
      // Coarse solve
      Ss[nlevels]->Mult(*bs[nlevels], *xs[nlevels]);
      // Post-smoothing
      for (int i=nlevels; i > 0; --i)
      {
         // Prolongate error (and correct solution)
         Ps[i]->AddMult(*xs[i], *xs[i-1]);
         // Compute residual
         As[i-1]->Mult(*xs[i-1], *rs[i-1]);
         add(*bs[i-1], -1.0, *rs[i-1], *rs[i-1]);
         // Smooth
         Ss[i-1]->Mult(*rs[i-1], *bs[i-1]);
         xs[i-1]->Add(beta, *bs[i-1]);
      }
      x_final += *xs[0];
      if (iiter < niter - 1)
      {
         As[0]->Mult(x_final, *rs[0]);
         add(b_init, -1.0, *rs[0], *bs[0]);
         if (bs[0]->Normlinf() < 1e-12) { break; }
      }
   }
   // Prolongate solution from patch (+= since additive Schwarz)
   for (int i=0; i<bs[0]->Size(); ++i)
   {
      int pi = coarsening.perm[i];
      if (pi >= 0) { x_out[pi] += x_final[i]; }
   }
}

PatchSolver::~PatchSolver()
{
   for (int i=0; i<nlevels+1; ++i)
   {
      // Ps are owned by the coarsening object, which will delete them
      delete As[i];
      delete Ss[i];
      delete bs[i];
      delete xs[i];
      delete rs[i];
   }
}

PatchInfo PatchesAdditiveSolver::VerticesPatch(const std::vector<int> &v,
                                               Table &v2e)
{
   std::set<int> els0;
   for (int iv : v)
   {
      // If the vertex is owned by someone else, don't use it
      if (!glor->gmesh.IsVertexOwned(iv)) { continue; }
      // Loop over the coarse elements containing the given vertex
      Array<int> row;
      v2e.GetRow(iv, row);
      for (int j=0; j<row.Size(); ++j)
      {
         // Add the coarse element to the list
         int el_ho = row[j];
         els0.insert(el_ho);
      }
   }

   Array<int> els0_a(els0.size());
   int idx = 0;
   for (int i : els0)
   {
      els0_a[idx++] = i;
   }

   return PatchInfo(els0_a);
}

PatchInfo PatchesAdditiveSolver::VertexPatch(int iv, Table &v2e)
{
   std::vector<int> v = {iv};
   return VerticesPatch(v, v2e);
}

void PatchesAdditiveSolver::Init(
   LORGhostMesh &glor_,
   SparseMatrix &A,
   Array<int> &ess_bdr,
   bool multilevel,
   int npatches)
{
   glor = &glor_;

   b_g.SetSize(glor->fes.GetNDofs());
   b_g.UseDevice(false);
   x_g.SetSize(glor->fes.GetNDofs());
   x_g.UseDevice(false);

   // Create the patches associated with each vertex of the coarse mesh.
   // We are responsible for deleting this object.
   Table *v2e = glor->mesh0.GetVertexToElementTable();

   // PartitionMeshVertices handles when npatches > nv and npatches < 0.
   // If such a value for npatches is given, it will be overwritten with a
   // valid value.
   VertexPartitioning vp = PartitionMeshVertices(glor->mesh0, npatches);
   for (int i=0; i<npatches; ++i)
   {
      PatchInfo info = VerticesPatch(vp.v[i], *v2e);
      // Metis may give empty partitions, so ignore those
      if (info.els0.Size() > 0)
      {
         PatchSolver *psolv =
            new PatchSolver(info, *glor, A, ess_bdr, multilevel);
         solv.push_back(psolv);
      }
   }
   delete v2e;
}

void PatchesAdditiveSolver::Solve(const Vector &b, Vector &x) const
{
   glor->Populate(b, b_g);
   // Additive Schwarz
   x_g = 0.0;
   for (size_t i=0; i<solv.size(); ++i)
   {
      solv[i]->SolveAdd(b_g, x_g);
   }
   glor->AddToOwners(x_g);
   glor->ExtractOwnedDofs(x_g, x);
}

PatchesAdditiveSolver::~PatchesAdditiveSolver()
{
   for (size_t i=0; i<solv.size(); ++i)
   {
      delete solv[i];
   }
}
