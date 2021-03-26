#include "IRK.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip> 
#include <cmath> 


IRK::IRK(IRKOperator *IRKOper_, const RKData &ButcherTableau)
    : m_IRKOper(IRKOper_), m_Butcher(ButcherTableau)
    m_stageOper(NULL),
    m_newton_solver(NULL),
    m_tri_jac_solver(NULL),
    m_jac_solverSparsity(NULL),
    m_jac_precSparsity(NULL),
    m_solversInit(false),
    m_krylov2(false),
    m_comm(IRKOper_->GetComm())
{
    // Get proc IDs
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProcess);
    
    // This stream will only print from process 0
    if (m_rank > 0) mfem::out.Disable();
    
    // Setup block sizes for stage vectors
    m_stageOffsets.SetSize(m_Butcher.s + 1); 
    for (int i = 0; i <= m_Butcher.s; i++) m_stageOffsets[i] = i*m_IRKOper->Height();
    
    // Initialize stage vectors
    m_w.Update(m_stageOffsets);
    
    // Stage operator, F(w)
    m_stageOper = new IRKStageOper(m_IRKOper, m_stageOffsets, m_Butcher, IMEX); 
    
    // Setup information for recording statistics of solver
    m_avg_newton_iter = 0;
    m_avg_krylov_iter.resize(m_Butcher.s_eff, 0);
    m_system_size.resize(m_Butcher.s_eff);
    m_eig_ratio.resize(m_Butcher.s_eff);
    
    // Set sizes, and eigenvalue ratios, beta/eta
    int row = 0;
    for (int i = 0; i < m_Butcher.s_eff; i++) {
        m_system_size[i] = m_Butcher.R0_block_sizes[i];
        
        // 1x1 block has beta=0
        if (m_system_size[i] == 1) {
            m_eig_ratio[i] = 0.;
        // 2x2 block
        }
        else {
            // eta == diag entry of block, beta == sqrt(-1 * product of off diagonals)
            m_eig_ratio[i] = sqrt(-m_Butcher.R0(row+1,row)*m_Butcher.R0(row,row+1)) / m_Butcher.R0(row,row);
            row++;
        }
        row++;
    }
}

IRK::~IRK() {
    if (m_tri_jac_solver) delete m_tri_jac_solver;
    if (m_newton_solver) delete m_newton_solver;
    if (m_stageOper) delete m_stageOper;
    if (m_jac_solverSparsity) delete m_jac_solverSparsity;
    if (m_jac_precSparsity) delete m_jac_precSparsity;
}

void IRK::SetSolvers()
{    
    if (m_solversInit) return;
    m_solversInit = true;
    
    // Setup Newton solver
    m_newton_solver = new NewtonSolver(m_comm);
    m_newton_solver->iterative_mode = false;
    m_newton_solver->SetMaxIter(m_newton_params.maxiter);
    m_newton_solver->SetRelTol(m_newton_params.reltol);
    m_newton_solver->SetPrintLevel(m_newton_params.printlevel);    
    if (m_newton_params.printlevel == 2) m_newton_solver->SetPrintLevel(-1);

    // Set sparsity patterns for Jacobian and its preconditioner if using a 
    // non-Kronecker-product Jacobian
    if (m_IRKOper->GetExplicitGradientsType() == IRKOperator::ExplicitGradients::EXACT) {
        m_jac_solverSparsity = new QuasiMatrixProduct(m_Butcher.Q0);
        m_jac_precSparsity   = new QuasiMatrixProduct(m_Butcher.Q0);
        
        m_jac_solverSparsity->Sparsify(static_cast<int>(m_newton_params.jac_solver_sparsity));
        m_jac_precSparsity->Sparsify(static_cast<int>(m_newton_params.jac_prec_sparsity));
    }

    // Create Jacobian solver
    // User requested different Krylov solvers for 1x1 and 2x2 blocks
    if (m_krylov2) {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper, 
                                m_newton_params.jac_update_rate,
                                m_newton_params.gamma_idx,
                                m_krylov_params, m_krylov_params2, 
                                m_jac_solverSparsity, m_jac_precSparsity, IMEX);
    // Use same Krylov solvers for 1x1 and 2x2 blocks
    }
    else {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper, 
                                    m_newton_params.jac_update_rate,
                                    m_newton_params.gamma_idx,
                                    m_krylov_params, m_jac_solverSparsity,
                                    m_jac_precSparsity, IMEX);
    }
    m_newton_solver->SetSolver(*m_tri_jac_solver);
}

inline void IRK::SetKrylovParams(KrylovParams params1, KrylovParams params2)
{ 
    MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
    m_krylov_params  = params1;
    m_krylov_params2 = params2;
    m_krylov2 = true; // Using two Krylov solvers
}

inline void IRK::GetSolveStats(int &avg_newton_iter, 
                            vector<int> &avg_krylov_iter, 
                            vector<int> &system_size, 
                            vector<double> &eig_ratio) const
{
    avg_newton_iter = m_avg_newton_iter;
    avg_krylov_iter = m_avg_krylov_iter;
    system_size     = m_system_size;
    eig_ratio       = m_eig_ratio;
}

void IRK::Init(TimeDependentOperator &F)
{    
    ODESolver::Init(F);
    m_w.Update(m_stageOffsets, mem_type); // Stage vectors
    m_w = 0.;       // Initialize stage vectors to 0
    SetSolvers();   // Initialize solvers for computing stage vectors
}

/** Apply RK update to take x from t to t+dt,
        x = x + (dt*b0^\top \otimes I)*k 
          = x + (dt*d0^\top \otimes I)*w */
void IRK::Step(Vector &x, double &t, double &dt)
{
    // Reset iteration counter for Jacobian solve from previous Newton iteration
    m_tri_jac_solver->ResetNumIterations();
    
    // Pass current values of dt and x to stage operator
    m_stageOper->SetParameters(&x, t, dt); 
    
    // Reset operator for Newton solver
    m_newton_solver->SetOperator(*m_stageOper);
    
    // Solve for stages (Empty RHS is interpreted as zero RHS)
    m_newton_solver->Mult(Vector(), m_w);
    
    // Facilitate different printing from Newton solver.
    if (m_newton_params.printlevel == 2) {
        mfem::out   << "Newton: Number of iterations: " 
                    << m_newton_solver->GetNumIterations() 
                    << ", ||r|| = " << m_newton_solver->GetFinalNorm() 
                    << '\n';
    }
    
    // Add iteration counts from this solve to existing counts
    m_avg_newton_iter += m_newton_solver->GetNumIterations();
    vector<int> krylov_iters = m_tri_jac_solver->GetNumIterations();
    for (int system = 0; system < m_avg_krylov_iter.size(); system++) {
        m_avg_krylov_iter[system] += krylov_iters[system];
    }
    
    // Check for convergence 
    if (!m_newton_solver->GetConverged()) {
        string msg = "IRK::Step() Newton solver at t=" + to_string(t) + " not converged\n";
        mfem_error(msg.c_str());
    }

    // Update solution with weighted sum of stages, x = x + (dt*d0^\top \otimes I)*w        
    // NOTE: Stiffly accurate schemes have all but d0(s)=0 
    for (int i = 0; i < m_Butcher.s; i++) {
        if (fabs(m_Butcher.d0[i]) > 1e-15) x.Add(dt*m_Butcher.d0[i], m_w.GetBlock(i));
    }
    t += dt; // Time that current x is evaulated at
}

/// Time step 
void IRK::Run(Vector &x, double &t, double &dt, double tf) 
{    
    m_w = 0.;       // Initialize stage vectors to 0
    SetSolvers();   // Initialize solvers for computing stage vectors
    
    /* Main time-stepping loop */
    int step = 0;
    int numsteps = ceil(tf/dt);
    while (t-tf < 0) {
        step++;
        mfem::out << "Time-step " << step << " of " << numsteps << " (t=" << t << "-->t=" << t+dt << ")\n";

        // Step from t to t+dt
        Step(x, t, dt);
    }
    
    // Average out number of Newton and Krylov iters over whole of time stepping
    m_avg_newton_iter = round(m_avg_newton_iter / double(numsteps));
    for (int i = 0; i < m_avg_krylov_iter.size(); i++) {
        m_avg_krylov_iter[i] = round(m_avg_krylov_iter[i] / double(numsteps));
    }
}


void PolyIMEX::Step(Vector &x, double &t, double &dt)
{
    // Reset iteration counter for Jacobian solve from previous Newton iteration
    m_tri_jac_solver->ResetNumIterations();

    // Construct RHS from previous explicit time steps
    // TODO : Need to include (A0 x I)^{-1} here as well



    // ------------- Linearly implicit IMEX ------------- //
    if (IMEX == 1) {

        // TODO: Maybe still need to set m_tri_jac_solver as IRKStageOper?

        // Solve for solution at implicit stages
        //  ->=-> Make sure initial guess is initialized
        m_tri_jac_solver->Mult(   );

        // Apply explicit part of operator for next time step
        y1 = ExplicitMult(x);
        y2 = ExplicitMult(k1);
        ...
    }
    // ------------- Nonlinearly implicit IMEX ------------- //
    else if (IMEX == 2) {



    }
    // ------------- Fully implicit and nonlinear ------------- //
    else {
    
        // Pass current values of dt and x to stage operator
        m_stageOper->SetParameters(&x, t, dt); 
        
        // Reset operator for Newton solver
        m_newton_solver->SetOperator(*m_stageOper);
        
        // Solve for stages (Empty RHS is interpreted as zero RHS)
        m_newton_solver->Mult(Vector(), m_w);

        // Facilitate different printing from Newton solver.
        if (m_newton_params.printlevel == 2) {
            mfem::out   << "Newton: Number of iterations: " 
                        << m_newton_solver->GetNumIterations() 
                        << ", ||r|| = " << m_newton_solver->GetFinalNorm() 
                        << '\n';
        }
        
        // Add iteration counts from this solve to existing counts
        m_avg_newton_iter += m_newton_solver->GetNumIterations();
        vector<int> krylov_iters = m_tri_jac_solver->GetNumIterations();
        for (int system = 0; system < m_avg_krylov_iter.size(); system++) {
            m_avg_krylov_iter[system] += krylov_iters[system];
        }
        
        // Check for convergence 
        if (!m_newton_solver->GetConverged()) {
            string msg = "IRK::Step() Newton solver at t=" + to_string(t) + " not converged\n";
            mfem_error(msg.c_str());
        }

        // Update solution with weighted sum of stages, x = x + (dt*d0^\top \otimes I)*w        
        // NOTE: Stiffly accurate schemes have all but d0(s)=0 
        // *** (A0 x I)^{-1} is included in d0 = b(A0 x I)^{-1} ***
        for (int i = 0; i < m_Butcher.s; i++) {
            if (fabs(m_Butcher.d0[i]) > 1e-15) x.Add(dt*m_Butcher.d0[i], m_w.GetBlock(i));
        }
        t += dt; // Time that current x is evaulated at
    }
}