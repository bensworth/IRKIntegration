#include "as.hpp"
#include <algorithm>

CoarseSolver::CoarseSolver(
   ParFiniteElementSpace &fes,
   ParFiniteElementSpace &fes0,
   Array<int> &ess_bdr,
   Solver &solv_)
   : tr(fes0, fes), solv(solv_)
{
   tr.SetOperatorType(Operator::MFEM_SPARSEMAT);
   R = (SparseMatrix *)&tr.ForwardOperator();

   Array<int> is_dof_ess0, is_dof_ess;
   fes0.GetEssentialVDofs(ess_bdr, is_dof_ess0);
   fes.GetEssentialVDofs(ess_bdr, is_dof_ess);
   fes.MarkerToList(is_dof_ess, ess_dof);

   int ndof0 = fes0.GetVSize();
   int ndof = fes.GetVSize();

   for (int i=0; i<ndof0; ++i)
   {
      if (is_dof_ess0[i])
      {
         R->EliminateCol(i, Matrix::DIAG_ZERO);
      }
   }

   for (int i=0; i<ndof; ++i)
   {
      bool is_true_dof = (fes.GetLocalTDofNumber(i) >= 0);
      if (!is_true_dof || is_dof_ess[i])
      {
         R->EliminateRow(i, Matrix::DIAG_ZERO);
      }
   }

   R->Finalize();
   // R->UseDevice(false); // Ensure Mult performed on host
   R->BuildTranspose();

   it_solv = dynamic_cast<IterativeSolver *>(&solv);
   b0.SetSize(fes0.GetTrueVSize());
   x0.SetSize(fes0.GetTrueVSize());
   b0_gf.SetSpace(&fes0);
   x0_gf.SetSpace(&fes0);
   b0_gf.UseDevice(false); // Ensure GridFunctions are only on the host
   x0_gf.UseDevice(false);
}

void CoarseSolver::Solve(const Vector &b, Vector &x) const
{
   R->MultTranspose(b, b0_gf);
   b0_gf.ParallelAssemble(b0);
   x0 = 0.0;
   solv.Mult(b0, x0);
   x0_gf.Distribute(x0);
   R->Mult(x0_gf, x);
   // Set essential DOFs (technically unnecessary to do this every interation
   // but the cost should be negligible)
   for (int i=0; i<ess_dof.Size(); ++i)
   {
      x[ess_dof[i]] = b[ess_dof[i]];
   }
}

AdditiveSchwarz::AdditiveSchwarz(
   ParFiniteElementSpace &fes,
   ParFiniteElementSpace &fes0,
   LORGhostMesh &glor,
   Array<int> &ess_bdr,
   SparseMatrix &A_,
   Solver &A0_solv,
   bool multilevel,
   bool additive_,
   int npatches,
   bool pure_neumann_)
   : Solver(fes.GetTrueVSize()),
     A(A_),
     coarse_solv(fes, fes0, ess_bdr, A0_solv),
     additive(additive_),
     pure_neumann(pure_neumann_)
{
   P = fes.GetProlongationMatrix();
   const SparseMatrix *Rtemp = fes.GetRestrictionMatrix();
   R = new SparseMatrix(*Rtemp); // Deep copy so we can avoid using the device
   // R->UseDevice(false);

   if (additive)
   {
      alpha = 1.0;
   }
   else
   {
      alpha = 0.5/(1 + pow(2, fes.GetMesh()->Dimension()));
   }

   patch_solv.Init(glor, A, ess_bdr, multilevel, npatches);

   // Set up workspace vectors
   r.SetSize(fes.GetNDofs());
   e.SetSize(fes.GetNDofs());
   x.SetSpace(&fes);
   b.SetSpace(&fes);
   x.UseDevice(false); // Ensure grid functions are only on the host
   b.UseDevice(false);
}

void Orthogonalize(Vector &v, MPI_Comm comm)
{
   double loc_sum = v.Sum();
   double global_sum = 0.0;
   int loc_size = v.Size();
   int global_size = 0;

   MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
   MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

   v -= global_sum / static_cast<double>(global_size);
}

void AdditiveSchwarz::Mult(const Vector &b_in, Vector &x_out) const
{
   if (P) { b.Distribute(b_in); } // TODO3: Ensure this is always performed on the host?
   else { b = b_in; }
   b.HostRead();
   if (additive)
   {
      patch_solv.Solve(b, x);
      if (alpha != 1.0) { x *= alpha; }
      coarse_solv.Solve(b, e);
      if (alpha != 1.0) { e *= alpha; }
      x += e;
   }
   else
   {
      patch_solv.Solve(b, x);
      x *= alpha;

      r = b;
      A.AddMult(x, r, -1.0);
      coarse_solv.Solve(r, e);
      x += e;

      r = b;
      A.AddMult(x, r, -1.0);
      patch_solv.Solve(r, e);
      e *= alpha;
      x += e;
   }
   if (R) { R->Mult(x, x_out); }
   else { x_out = x; }

   if (pure_neumann) { Orthogonalize(x_out, x.ParFESpace()->GetComm()); }
}

void AdditiveSchwarz::SetOperator(const Operator &op)
{
   MFEM_ABORT("AdditiveSchwarz cannot call SetOperator.")
}

template<typename OpT>
void ScalarAssembleDiffusion(BilinearForm &a, OpT &A,
                             FiniteElementSpace &fes,
                             Coefficient &mass_coeff,
                             Coefficient &diff_coeff,
                             const Array<int>& ess_bdr)
{
   a.AddDomainIntegrator(new MassIntegrator(mass_coeff));
   a.AddDomainIntegrator(new DiffusionIntegrator(diff_coeff));
   a.Assemble();

   Array<int> ess_tdof_list;
   fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   a.FormSystemMatrix(ess_tdof_list, A);
}


HighOrderASPreconditioner::HighOrderASPreconditioner(
   ParFiniteElementSpace& _fes_ho,
   Coefficient& _mass_coeff,
   Coefficient& _diff_coeff,
   Array<int> _ess_bdr, // intentionally copy, so that we can edit to include ghost boundary
   int _npatches)
   : Solver(_fes_ho.GetTrueVSize()),
     order(_fes_ho.GetOrder(0)),
     pmesh_ho(_fes_ho.GetParMesh()),
     pmesh_lor(pmesh_ho, order, BasisType::GaussLobatto),
     glor(*pmesh_ho, order, BasisType::GaussLobatto),
     fec_lor(1, pmesh_ho->Dimension()),
     fespace_lor(&pmesh_lor, &fec_lor),
     fespace_coarse(pmesh_ho, &fec_lor),
     fes_ho(_fes_ho),
     mass_coeff(_mass_coeff),
     diff_coeff(_diff_coeff),
     ess_bdr(_ess_bdr),
     npatches(_npatches)
{
   int pure_neumann_local = (ess_bdr.Sum() == 0);
   int pure_neumann_global;
   MPI_Allreduce(&pure_neumann_local, &pure_neumann_global,
                 1, MPI_INT, MPI_SUM, pmesh_ho->GetComm());
   bool pure_neumann = (pure_neumann_global != 0);

   // Modify essential boundaries to include ghost boundaries
   InitGhostEssBdr(ess_bdr);

   // Setup bilinear forms
   a_coarse.reset(new ParBilinearForm(&fespace_coarse));
   ScalarAssembleDiffusion(*a_coarse, A0, fespace_coarse,
      mass_coeff, diff_coeff, ess_bdr);
   a_ghost_lor.reset(new BilinearForm(&glor.fes));
   ScalarAssembleDiffusion(*a_ghost_lor, A_ghost_lor, glor.fes,
      mass_coeff, diff_coeff, ess_bdr);

   // Setup coarse solver
   InitCoarseSolver();

   // Finalize Additive Schwarz preconditioner
   as_prec.reset(new AdditiveSchwarz(fespace_lor, fespace_coarse, glor, ess_bdr,
                                     A_ghost_lor, *A0_solv, true, true, npatches,
                                     pure_neumann));
}

void HighOrderASPreconditioner::InitGhostEssBdr(Array<int>& ess_bdr)
{
   int previous_size = ess_bdr.Size();
   // Grow our internal copy of ess_bdr
   ess_bdr.SetSize(glor.mesh_lor.bdr_attributes.Max());
   // and append ghost BCs as essential (homogeneous Dirichlet)
   for (int i = previous_size; i < ess_bdr.Size(); ++i)
   {
      ess_bdr[i] = 1;
   }
}

void HighOrderASPreconditioner::InitCoarseSolver()
{
   int num_procs;
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   // Construct the coarse solver as...
   // Multiple V-cycles using AMG
   A0_amg.SetOperator(A0);
   A0_amg.SetPrintLevel(0);
   SLISolver *sli = new SLISolver(MPI_COMM_WORLD);
   sli->SetOperator(A0);
   sli->SetPreconditioner(A0_amg);
   sli->SetMaxIter(10);
   sli->SetRelTol(1e-8);
   A0_solv.reset(sli);
}

void HighOrderASPreconditioner::Mult(const Vector &b, Vector &x) const
{
   as_prec->Mult(b, x);
}

void HighOrderASPreconditioner::SetOperator(const Operator &op)
{
   MFEM_ABORT("HighOrderASPreconditioner cannot call SetOperator.");
}

/*
VectorHOASPreconditioner::VectorHOASPreconditioner(
   ParFiniteElementSpace& fes_ho,
   Coefficient& lap_coeff,
   const Array<int> &ess_bdr,
   bool unsteady,
   int npatches)
   : Solver(fes_ho.GetTrueVSize()),
     vdim(fes_ho.GetVDim()),
     scalar_fes(fes_ho.GetParMesh(), fes_ho.FEColl()),
     scalar_prec(scalar_fes, lap_coeff, ess_bdr, unsteady, npatches)
{
   MFEM_ASSERT(fes_ho.GetOrdering() == Ordering::byNODES,
               "VectorHOASPreconditioner only supports Ordering::byNODES!");
}

int VectorHOASPreconditioner::GetVDim() const
{
   return vdim;
}

void VectorHOASPreconditioner::Mult(const Vector &b, Vector &x) const
{
   int scalarSize = width/vdim;
   b.HostRead();
   x.HostReadWrite();
   Vector xScalar{nullptr, scalarSize};
   Vector bScalar{nullptr, scalarSize};
   for (int i = 0; i < vdim; ++i)
   {
      // bScalar, xScalar are scalar references within vector Vectors to avoid copies
      bScalar.SetData(b.GetData() + i*scalarSize);
      xScalar.SetData(x.GetData() + i*scalarSize);
      scalar_prec.Mult(bScalar, xScalar);
   }
}

void VectorHOASPreconditioner::SetOperator(const Operator &op)
{
   MFEM_ABORT("VectorHOASPreconditioner cannot call SetOperator.");
}

void VectorHOASPreconditioner::SetParameters(double dt)
{
   scalar_prec.SetParameters(dt);
}
*/
