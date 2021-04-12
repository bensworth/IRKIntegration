#include "IRK_utils.hpp"
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
    m_stageOper = new IRKStageOper(m_IRKOper, m_stageOffsets, m_Butcher); 
    
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
                                m_jac_solverSparsity, m_jac_precSparsity);
    // Use same Krylov solvers for 1x1 and 2x2 blocks
    }
    else {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper, 
                                    m_newton_params.jac_update_rate,
                                    m_newton_params.gamma_idx,
                                    m_krylov_params, m_jac_solverSparsity,
                                    m_jac_precSparsity);
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


PolyIMEX::PolyIMEX(IRKOperator *IRKOper_, const RKData &ButcherTableau,
    bool linearly_imp_, int num_iters_)
    : initialized(false), linearly_imp(linearly_imp_)
{
    // TODO : Initialize block vectors, arrays, etc.
    


    // If linearly implicit, check that IRKOperator is also linearly implicit
    if (linearly_imp != m_IRKOper->IsLinearlyImplicit()) {
        mfem_error("PolyIMEX and IRKOperator must have same value for linearly_impl.\n");
    }

    // Check if explicit stage is first or last
    int& s = m_Butcher.s;
    if (std::abs(m_Butcher.imexA0(0,0)) < 1e-15) {
        exp_ind = 0;
    }
    else if (std::abs(m_Butcher.imexA0(s,s)) < 1e-15) {
        exp_ind = s;
    }
    else {
        mfem_error("Current implementation requires either first or last stage be explicit.\n");
    }
}

void PolyIMEX::SetSolvers()
{    
    if (m_solversInit) return;
    m_solversInit = true;
    
    // Setup Newton solver for nonlinear implicit splitting; here (unlike
    // IRK), we want to use previous solution as initial guess, so we set
    // iterative_mode = true;
    if (!linearly_imp) {
        m_newton_solver = new NewtonSolver(m_comm);
        m_newton_solver->iterative_mode = true;
        m_newton_solver->SetMaxIter(m_newton_params.maxiter);
        m_newton_solver->SetRelTol(m_newton_params.reltol);
        m_newton_solver->SetPrintLevel(m_newton_params.printlevel);    
        if (m_newton_params.printlevel == 2) m_newton_solver->SetPrintLevel(-1);
    }

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
                                m_jac_solverSparsity, m_jac_precSparsity);
    // Use same Krylov solvers for 1x1 and 2x2 blocks
    }
    else {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper, 
                                    m_newton_params.jac_update_rate,
                                    m_newton_params.gamma_idx,
                                    m_krylov_params, m_jac_solverSparsity,
                                    m_jac_precSparsity);
    }
    m_newton_solver->SetSolver(*m_tri_jac_solver);
}


void PolyIMEX::Step(Vector &x, double &t, double &dt)
{
    double r = dt / m_Butcher.alpha;

    // Pass current value of r = dt/\alpha.
    m_stageOper->SetParameters(NULL, t, r); 

    // Check if this is the first time step, if so use iterator to
    // initialize stage values at high order by applying ord+num_iters
    // times.
    if (!initialized) {
        // Initialize explicit components
        IRKOper_->ExplicitMult(x, expPart(0));
        for (i=1; i<=m_Butcher.s; i++) expPart(i) = expPart(0);

        // Perform order+num_iters applications of iterator
        for (int i=0; i<(m_Butcher.order+num_iters); i++) {
            Iterate(x, t, dt);        
        }
        initialized = true;
    }

    // Reset iteration counter for Jacobian solve from previous Newton iteration
    m_tri_jac_solver->ResetNumIterations();

    // Apply propagator
    Iterate(x, r, false);

    // Call to iterator after propagator applied
    for (int i=0; i<num_iters; i++) {
        Iterate(x, r, true);
    }

    t += dt; // Update current time
}


void PolyIMEX::FormImpRHS(Vector &x_prev, double r, bool iterator)
{
    // Coefficients to form right-hand side for iterator or propagator
    DenseMatrix *coeffs;
    if (iterator) {
        coeffs = &(m_Butcher.imex_rhs_it);
    }
    else {
        coeffs = &(m_Butcher.imex_rhs);
    }

    // Initialize right-hand side for all blocks to solution
    // at previous time step, then add explicit components
    //      rhs_i = x_n + \sum_{j=0}^s fe_j^n,
    // where fe_j^n is the explicit splitting applied to the jth
    // stage of the nth time step, where j=0 is the solution at
    // the nth time step, j=1,...,s-1 the internal stages, and
    // j=s the (n+1)st time step.
    for (int i=0; i <m_Butcher.s; i++) {
        rhs.GetBlock(i) = x_prev;
        for (int j=0; j<(m_Butcher.s+1); j++) {
            if (std::abs((*coeffs)(i,j)) > 1e-15) { 
                rhs.GetBlock(i) += r * (*coeffs)(i,j) * expPart(j);
            }
        }
    }
    coeffs = NULL;

    // If explicit stage has already been solved for, apply implicit
    // part of operator to the explicit solution and update RHS
    if (exp_ind == 0) {
        bool need_update = false;
        if (iterator) {
            coeffs = &(m_Butcher.imexA0_it);
        }
        else {
            coeffs = &(m_Butcher.imexA0);
        }
        // Check that this needs to be computed
        for (int i=1; i<(m_Butcher.s+1); i++) {
            if (std::abs((*coeffs)(i,0)) > 1e-15) {
                need_update = true;
                break;
            }
        }
        if (need_update) {
            Vector temp;
            m_IRKOper->ImplicitMult(sol_exp, temp);
            for (int i=1; i<(m_Butcher.s+1); i++) {
                if (std::abs((*coeffs)(i,0)) > 1e-15) {
                    rhs.GetBlock(i-1) += r * (*coeffs)(i,0) * temp;
                }
            }
        }
        coeffs = NULL;
    }

    // Scale by (A0 x I)^{-1}, rhs <- (A0 x I)^{-1}rhs
    KronTransform(m_Butcher.invA0, rhs);
}

void PolyIMEX::UpdateExplicitComponents()
{
    // Explicit stage is first, followed by implicit
    if (exp_ind == 0) {
        m_IRKOper->ExplicitMult(sol_exp, expPart(0));
        for (i=1; i<=m_Butcher.s; i++) {
            m_IRKOper->ExplicitMult(sol_imp(i-1), expPart(i));
        }
    }
    // Implicit stages are first, followed by explicit
    else {
        for (i=0; i<m_Butcher.s; i++) {
            m_IRKOper->ExplicitMult(sol_imp(i), expPart(i));
        }
        m_IRKOper->ExplicitMult(sol_exp, expPart(s));
    } 
}

void PolyIMEX::Iterate(Vector &x, double r, bool iterator)
{
    // If explicit stage is first, solve for explicit stage
    if (exp_ind == 0) {
        sol_exp = x;
        // Updates from previous explicit steps
        //   NOTE : for initial examples, below coefficients were all zero,
        //   but they may not be in general.
        if (iterator) {
            for (int j=0; j<=m_Butcher.s; j++) {
                if (std::abs(m_Butcher.imex_rhs_it(0,j)) > 1e-15) { 
                    sol_exp += r * (m_Butcher.imex_rhs_it(0,j)) *  expPart(j);
                }
            }
        }
        else {
            coeffs = &(m_Butcher.imex_rhs);
            for (int j=0; j<=m_Butcher.s; j++) {
                if (std::abs(m_Butcher.imex_rhs(0,j)) > 1e-15) { 
                    sol_exp += r * (m_Butcher.imex_rhs(0,j)) *  expPart(j);
                }
            }
        }
    }

    // Construct right-hand side for implicit equation 
    FormImpRHS(x, r, iterator);

    // ------------- Linearly implicit IMEX ------------- //
    if (linearly_imp) {

        // TODO : add forcing function to rhs vector??

        // Solve for solution at implicit stages
        m_tri_jac_solver->Mult(rhs, sol_imp);
    }
    // ------------- Nonlinearly implicit IMEX ------------- //
    else {
        // Reset operator for Newton solver
        m_newton_solver->SetOperator(*m_stageOper);

        // Solve for stages (Empty RHS is interpreted as zero RHS)
        m_newton_solver->Mult(rhs, sol_imp);
    
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
    }

    // Update solution value with final stage of implicit solve
    if (exp_ind == 0) {
        x = sol_imp.GetBlock(m_Butcher.s-1);
    }
    // Compute final explicit stage and update solution
    else {
        sol_exp = x;
        // Updates from previous implicit and explicit steps
        if (iterator) { // Compute for iterator
            for (int j=0; j<=m_Butcher.s; j++) {  // explicit
                if (std::abs(m_Butcher.imex_rhs_it(s,j)) > 1e-15) { 
                    sol_exp += r * (m_Butcher.imex_rhs_it(s,j)) *  expPart(j);
                }
            }
            for (int j=0; j<m_Butcher.s; j++) { // implicit
                if (std::abs(m_Butcher.imexA0_it(s,j)) > 1e-15) {
                    Vector temp;
                    m_IRKOper->ImplicitMult(sol_imp.GetBlock(j), temp);
                    sol_exp += r * (m_Butcher.imexA0_it(s,j)) *  temp;
                }
            }
        }
        else { // Compute for propagator
            coeffs = &(m_Butcher.imex_rhs);
            for (int j=0; j<=m_Butcher.s; j++) { // explicit
                if (std::abs(m_Butcher.imex_rhs(s,j)) > 1e-15) { 
                    sol_exp += r * (m_Butcher.imex_rhs(s,j)) *  expPart(j);
                }
            }
            for (int j=0; j<m_Butcher.s; j++) { // implicit
                if (std::abs(m_Butcher.imexA0(s,j)) > 1e-15) {
                    Vector temp;
                    m_IRKOper->ImplicitMult(sol_imp.GetBlock(j), temp);
                    sol_exp += r * (m_Butcher.imexA0(s,j)) *  temp;
                }
            }
        }
        x = sol_exp;
    }

    // Once all stages have been updated, update stored
    // explicit components for future time steps/iterations. 
    UpdateExplicitComponents();
}





