#include "IRK.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip> 
#include <cmath> 


IRK::IRK(IRKOperator *IRKOper_, const RKData &ButcherTableau)
    : m_IRKOper(IRKOper_),
    m_Butcher(ButcherTableau),
    m_stageOper(NULL),
    m_newton_solver(NULL),
    m_tri_jac_solver(NULL),
    m_jac_solverSparsity(NULL),
    m_jac_precSparsity(NULL),
    m_solversInit(false),
    m_krylov2(false),
    m_comm(IRKOper_->GetComm()),
    gamma_idx(1)
{
    // Get proc IDs
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProcess);
    
    // This stream will only print from process 0
    if (m_rank > 0) mfem::out.Disable();
    
    // Setup block sizes for stage vectors
    m_stageOffsets.SetSize(m_Butcher.s + 1); 
    for (int i = 0; i <= m_Butcher.s; i++) m_stageOffsets[i] = i*m_IRKOper->Height();

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
                                gamma_idx, m_krylov_params, m_krylov_params2, 
                                m_jac_solverSparsity, m_jac_precSparsity);
    // Use same Krylov solvers for 1x1 and 2x2 blocks
    }
    else {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper, 
                                m_newton_params.jac_update_rate,
                                gamma_idx, m_krylov_params, m_jac_solverSparsity,
                                m_jac_precSparsity);
    }
    m_newton_solver->SetSolver(*m_tri_jac_solver);
}

void IRK::SetKrylovParams(KrylovParams params1, KrylovParams params2)
{ 
    MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
    m_krylov_params  = params1;
    m_krylov_params2 = params2;
    m_krylov2 = true; // Using two Krylov solvers
}

void IRK::GetSolveStats(int &avg_newton_iter, 
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
    sol_imp.Update(m_stageOffsets, mem_type); // Stage vectors
    sol_imp = 0.;       // Initialize stage vectors to 0
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
    m_newton_solver->Mult(Vector(), sol_imp);
    
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
        if (fabs(m_Butcher.d0[i]) > 1e-15) x.Add(dt*m_Butcher.d0[i], sol_imp.GetBlock(i));
    }
    t += dt; // Time that current x is evaulated at
}

/// Time step 
void IRK::Run(Vector &x, double &t, double &dt, double tf) 
{    
    sol_imp = 0.;       // Initialize stage vectors to 0
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
    : IRK(IRKOper_, ButcherTableau), 
    initialized(false),
    linearly_imp(linearly_imp_), 
    num_iters(num_iters_)
{
    if (!ButcherTableau.IsIMEX()) {
        mfem_error("PolyIMEX requires IMEX butcher tableaux.\n");
    }

    // If linearly implicit, check that IRKOperator is also linearly implicit
    if (linearly_imp != m_IRKOper->IsLinearlyImplicit()) {
        mfem_error("PolyIMEX and IRKOperator must have same value for linearly_impl.\n");
    }

    // Check if explicit stage is first or last
    if (std::abs(m_Butcher.A0(0,0)) < 1e-15) {
        exp_ind = 0;
    }
    else if (std::abs(m_Butcher.A0(m_Butcher.s,m_Butcher.s)) < 1e-15) {
        exp_ind = m_Butcher.s;
    }
    else {
        mfem_error("Current implementation requires either first or last stage be explicit.\n");
    }

    // Setup block sizes for s+1 vectors
    m_stageOffsets2.SetSize(m_Butcher.s + 2); 
    for (int i = 0; i <= (m_Butcher.s+1); i++) {
        m_stageOffsets2[i] = i*m_IRKOper->Height();
    }
}

void PolyIMEX::Init(TimeDependentOperator &F)
{
    ODESolver::Init(F);
    sol_exp.SetSize(F.Width());
    sol_imp.Update(m_stageOffsets, mem_type); // Stage vectors
    sol_imp = 0.;       // Initialize stage vectors to 0
    exp_part.Update(m_stageOffsets2, mem_type); // Explicit vectors
    rhs.Update(m_stageOffsets, mem_type); // Right-hand side vectors
    SetSolvers();   // Initialize solvers for computing implicit update
}


void PolyIMEX::SetSolvers()
{   
    if (m_solversInit) return;
    m_solversInit = true;

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
                                gamma_idx, m_krylov_params, m_krylov_params2, 
                                m_jac_solverSparsity, m_jac_precSparsity);
    // Use same Krylov solvers for 1x1 and 2x2 blocks
    }
    else {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper, 
                                m_newton_params.jac_update_rate,
                                gamma_idx, m_krylov_params, m_jac_solverSparsity,
                                m_jac_precSparsity);
    }

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
        m_newton_solver->SetSolver(*m_tri_jac_solver);
    }
}


void PolyIMEX::Step(Vector &x, double &t, double &dt)
{
    double r = dt / m_Butcher.alpha;

    // Pass current value of r = dt/\alpha.
    m_stageOper->SetParameters(t, r); 

    // Check if this is the first time step, if so use iterator to
    // initialize stage values at high order by applying ord+num_iters
    // times.
    if (!initialized) {
        // Initialize explicit components
        // - NOTE: do not need to initialize sol_imp/sol_exp here
        m_IRKOper->SetTime(t + m_Butcher.z0(0)*r);
        m_IRKOper->ExplicitMult(x, exp_part.GetBlock(0));

        for (int i=1; i<=m_Butcher.s; i++) {
        
            // TODO : Do we need to set this or include explicit time-dependent forcing functions??
            // m_IRKOper->SetTime(t + m_Butcher.z0(i)*r);
            exp_part.GetBlock(i) = exp_part.GetBlock(0);
        }

        // Perform order+num_iters applications of iterator
        for (int i=0; i<(m_Butcher.order+num_iters); i++) {
            Iterate(x, t, r, true, true);        
        }
        initialized = true;
    }

    // Apply propagator
    Iterate(x, t, r, false);

    // TODO : may need to update x before propagator for some schemes, e.g., Lobatto
    // // Update solution vector after all iterators completed
    // if (exp_ind == 0) {
    //     x = sol_imp.GetBlock(m_Butcher.s-1);
    // }
    // else {
    //     x = sol_exp;
    // }

    // Call to iterator after propagator applied
    for (int i=0; i<num_iters; i++) {
        Iterate(x, t, r, true);
    }

    // Update solution vector after all iterators completed
    if (exp_ind == 0) {
        x = sol_imp.GetBlock(m_Butcher.s-1);
        // x = sol_exp;
    }
    else {
        x = sol_exp;
        // x = sol_imp.GetBlock(0);
    }
    t += dt; // Update current time
    total_steps += 1;

    /// DEBUG
    // int myid;
    // MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    // bool root = (myid == 0);
    // if (myid == 0) std::cout << "precs/step " << total_precs << ", total steps " << total_steps << "\n";
}


void PolyIMEX::FormImpRHS(Vector &x_prev, const double &t,
    const double &r, bool iterator)
{
    // Coefficients to form right-hand side for iterator or propagator
    const DenseMatrix *coeffs;
    if (iterator) {
        coeffs = &(m_Butcher.expA0_it);
    }
    else {
        coeffs = &(m_Butcher.expA0);
    }

    // Initialize right-hand side for all blocks to solution
    // at previous time step, then add explicit components
    //      rhs_i = M*x_n + \sum_{j=0}^s fe_j^n,
    // where M is mass matrix, and fe_j^n is the explicit splitting
    // applied to the jth stage of the nth time step, with j=0 the
    // solution at the nth time step, j=1,...,s-1 the internal
    // stages, and j=s the (n+1)st time step.
    //
    // NOTE: for exp_ind=0, we are only considering all rows
    // except the first row of explicit coefficients, while for
    // exp_ind=s, we are considering all but the last row.
    Vector temp(x_prev);
    m_IRKOper->MassMult(x_prev, temp);
    if (exp_ind == 0) {
        for (int i=0; i<m_Butcher.s; i++) {
            rhs.GetBlock(i) = temp;
            for (int j=0; j<=m_Butcher.s; j++) {
                if (std::abs((*coeffs)(i+1,j)) > 1e-15) { 
                    rhs.GetBlock(i).Add(r * (*coeffs)(i+1,j), exp_part.GetBlock(j));
                }
            }
        }
    }
    else {
        for (int i=0; i<m_Butcher.s; i++) {
            rhs.GetBlock(i) = temp;
            for (int j=0; j<=m_Butcher.s; j++) {
                if (std::abs((*coeffs)(i,j)) > 1e-15) { 
                    rhs.GetBlock(i).Add(r * (*coeffs)(i,j), exp_part.GetBlock(j));
                }
            }
        }
    }
    coeffs = NULL;

    // If explicit stage has already been solved for, apply implicit
    // part of operator to the explicit solution and update RHS
    if (exp_ind == 0) {
        bool need_update = false;
        if (iterator) {
            coeffs = &(m_Butcher.A0);
            // coeffs = &(m_Butcher.A0_it);     // *--- IF A0_it != A0 ---*
        }
        else {
            coeffs = &(m_Butcher.A0);
        }
        // Check that this needs to be computed
        for (int i=1; i<=m_Butcher.s; i++) {
            if (std::abs((*coeffs)(i,0)) > 1e-15) {
                need_update = true;
                break;
            }
        }
        // NOTE : we assume that if expliit stage has been calculated
        // already, it was done at the *first* quadrature point. 
        if (need_update) {

            std::cout << "Not being called I hope\n";

            m_IRKOper->SetTime(t + m_Butcher.z0(0)*r);
            m_IRKOper->ImplicitMult(sol_exp, temp);
            for (int i=1; i<=m_Butcher.s; i++) {
                if (std::abs((*coeffs)(i,0)) > 1e-15) {
                    rhs.GetBlock(i-1).Add(r * (*coeffs)(i,0), temp);
                }
            }
        }
        coeffs = NULL;
    }

    // Scale by (A0 x I)^{-1}, rhs <- (A0 x I)^{-1}rhs
    KronTransform(m_Butcher.invA0, rhs);
}

// This is called at the end of Iterate(). Regardless of whether Iterate()
// is called as a propagator or iterator, the explicit components must
// now be evaluated at the same quadrature points in time as the current
// implicit components, 
void PolyIMEX::UpdateExplicitComponents(const double &t, const double &r)
{
    // Explicit stage is first, followed by implicit
    if (exp_ind == 0) {
        m_IRKOper->SetTime(t + m_Butcher.z0(0)*r);
        m_IRKOper->ExplicitMult(sol_exp, exp_part.GetBlock(0));
        for (int i=1; i<=m_Butcher.s; i++) {
            m_IRKOper->SetTime(t + m_Butcher.z0(i)*r);
            m_IRKOper->ExplicitMult(sol_imp.GetBlock(i-1), exp_part.GetBlock(i));
        }
    }
    // Implicit stages are first, followed by explicit
    else {
        for (int i=0; i<m_Butcher.s; i++) {
            m_IRKOper->SetTime(t + m_Butcher.z0(i)*r);
            m_IRKOper->ExplicitMult(sol_imp.GetBlock(i), exp_part.GetBlock(i));
        }
        m_IRKOper->SetTime(t + m_Butcher.z0(m_Butcher.s)*r);
        m_IRKOper->ExplicitMult(sol_exp, exp_part.GetBlock(m_Butcher.s));
    } 
}

void PolyIMEX::Iterate(Vector &x, const double &t, const double &r, bool iterator, bool init)
{
    // If explicit stage is first, solve for explicit stage, 
    // u_{n+1} = u_n + dt*M^{-1}\sum_j expA0_it(0,j) exp_part(j)
    if (exp_ind == 0) {

        // Check for updates from previous explicit steps
        //   NOTE : for initial examples, below coefficients were all zero,
        //   but they may not be in general.
        bool compute = false;
        if (iterator) {
            for (int j=0; j<=m_Butcher.s; j++) {
                if (std::abs(m_Butcher.expA0_it(0,j)) > 1e-15) { 
                    compute = true;
                }
            }
        }
        else {
            for (int j=0; j<=m_Butcher.s; j++) {
                if (std::abs(m_Butcher.expA0(0,j)) > 1e-15) { 
                    compute = true;
                }
            }
        }
        // If nonzero coefficients were found above, sum vectors and
        // solve for explicit stage
        if (compute) {
            std::cout << "This shouldn't be happening.\n";
            Vector temp(x);
            temp = 0.0;
            // Updates from previous explicit steps
            //   NOTE : for initial examples, below coefficients were all zero,
            //   but they may not be in general.
            if (iterator) {
                for (int j=0; j<=m_Butcher.s; j++) {
                    if (std::abs(m_Butcher.expA0_it(0,j)) > 1e-15) { 
                        temp.Add(r * (m_Butcher.expA0_it(0,j)), exp_part.GetBlock(j));
                        compute = true;
                    }
                }
            }
            else {
                for (int j=0; j<=m_Butcher.s; j++) {
                    if (std::abs(m_Butcher.expA0(0,j)) > 1e-15) { 
                        temp.Add(r * (m_Butcher.expA0(0,j)), exp_part.GetBlock(j));
                        compute = true;
                    }
                }
            }            
            m_IRKOper->MassInv(temp, sol_exp);
            sol_exp += x;
        }
        else {
            sol_exp = x;
        }
    }

    // Construct right-hand side for implicit equation 
    FormImpRHS(x, t, r, iterator);

    // Add implicit (time-dependent) forcing functions; if this is not
    // implemented by the user, rhs is not modified. Use different time
    // points from {z} depending on whether implicit stage equations are
    // before or after the explicit stages.
    if (exp_ind == 0) {
        for (int i=1; i<=m_Butcher.s; i++) {
            m_IRKOper->AddImplicitForcing(rhs.GetBlock(i-1), t, r, m_Butcher.z0(i));
        }
    }
    else {
        for (int i=0; i<m_Butcher.s; i++) {
            m_IRKOper->AddImplicitForcing(rhs.GetBlock(i), t, r, m_Butcher.z0(i));
        }
    }

    // Reset iteration counter for Jacobian solve from previous Newton iteration
    m_tri_jac_solver->ResetNumIterations();

    // ------------- Linearly implicit IMEX ------------- //
    if (linearly_imp) {

        // Solve linear system for implicit stages
        m_tri_jac_solver->Mult(rhs, sol_imp);

        // Average Krylov iterations here
        if (!init) {
            total_precs += m_tri_jac_solver->GetNumPrecs();
            // DEBUG
            // int myid;
            // MPI_Comm_rank(MPI_COMM_WORLD, &myid);
            // bool root = (myid == 0);
            // if (iterator) {
            //     if (root) std::cout << "iterator " << m_tri_jac_solver->GetNumPrecs() << "\n";
            // }
            // else {
            //     if (root) std::cout << "propagator " << m_tri_jac_solver->GetNumPrecs() << "\n";
            // }
        }
    }
    // ------------- Nonlinearly implicit IMEX ------------- //
    else {
        // Reset operator for Newton solver
        m_newton_solver->SetOperator(*m_stageOper);

        // Solve for stages
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
            #if 0
            string msg = "IRK::Step() Newton solver at t=" + to_string(t) + " not converged\n";
            #else
            string msg = "IRK::Step() Newton solver not converged\n";
            #endif
            mfem_error(msg.c_str());
        }
    }

    // Compute final explicit stage and update solution
    if (exp_ind != 0) {
        Vector temp(x);
        temp = 0.0;
        // Updates from previous implicit and explicit steps
        if (iterator) { // Compute for iterator
            for (int j=0; j<=m_Butcher.s; j++) {  // explicit
                if (std::abs(m_Butcher.expA0_it(m_Butcher.s,j)) > 1e-15) { 
                    temp.Add(r * (m_Butcher.expA0_it(m_Butcher.s,j)), exp_part.GetBlock(j));
                }
            }
            for (int j=0; j<m_Butcher.s; j++) { // implicit
                // if (std::abs(m_Butcher.A0_it(m_Butcher.s,j)) > 1e-15) {      // *--- IF A0_it != A0 ---*
                if (std::abs(m_Butcher.A0(m_Butcher.s,j)) > 1e-15) { 
                    m_IRKOper->SetTime(t + m_Butcher.z0(j)*r);
                    m_IRKOper->ImplicitMult(sol_imp.GetBlock(j), temp);
                    // temp.Add(r * (m_Butcher.A0_it(m_Butcher.s,j)), temp);    // *--- IF A0_it != A0 ---*
                    temp.Add(r * (m_Butcher.A0(m_Butcher.s,j)), temp);

                    // TODO : can compute this indirectly via rhs and solution
                    // without using ImplicitMult

                }
            }
        }
        else { // Compute for propagator
            for (int j=0; j<=m_Butcher.s; j++) { // explicit
                if (std::abs(m_Butcher.expA0(m_Butcher.s,j)) > 1e-15) { 
                    temp.Add(r * (m_Butcher.expA0(m_Butcher.s,j)), exp_part.GetBlock(j));
                }
            }
            for (int j=0; j<m_Butcher.s; j++) { // implicit
                if (std::abs(m_Butcher.A0(m_Butcher.s,j)) > 1e-15) {
                    m_IRKOper->SetTime(t + m_Butcher.z0(j)*r);
                    m_IRKOper->ImplicitMult(sol_imp.GetBlock(j), temp);
                    temp.Add(r * (m_Butcher.A0(m_Butcher.s,j)), temp);

                    // TODO : can compute this indirectly via rhs and solution
                    // without using ImplicitMult

                }
            }
        }
        m_IRKOper->MassInv(temp, sol_exp);
        sol_exp += x;
    }

    // Once all stages have been updated, update stored
    // explicit components for future time steps/iterations. 
    UpdateExplicitComponents(t, r);
}



