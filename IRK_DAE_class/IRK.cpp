#include "IRK.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip>
#include <cmath>


/* TODO:
    -I think it might make sense to re-order Schur decompositions so that if
        there is at least one real eigenvalue, it be put at the bottom of the diagonal of R0.
        (I already have the MATLAB code to do this, but if we want to pull Schur
        decompositions from elsewhere, it might not be possible to do this, or may be too painful).
        The reason for this is that for the simplified Newton, we can evaluate the approximate Jacobian
        correctly w.r.t, the final stage, and so if this is a 1x1 system, when we do the backward
        substitution, it would be as if we've solved for this stage using the true Jacobian.
*/


/// Kronecker transform between two block vectors: y <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, const BlockVector &x, BlockVector &y)
{
    for (int block = 0; block < A.Height(); block++) {
        int j = 0;
        if (fabs(A(block,j)) > 0.) {
            y.GetBlock(block).Set(A(block,j), x.GetBlock(j));
        } else {
            y.GetBlock(block) = 0.;
        }
        for (int j = 1; j < A.Width(); j++) {
            if (fabs(A(block,j)) > 0.) y.GetBlock(block).Add(A(block,j), x.GetBlock(j));
        }
    }
};

/// Kronecker transform between two block vectors using A transpose: y <- (A^\top \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, const BlockVector &x, BlockVector &y)
{
    for (int block = 0; block < A.Width(); block++) {
        int i = 0;
        if (fabs(A(i,block)) > 0.) {
            y.GetBlock(block).Set(A(i,block), x.GetBlock(i));
        } else {
            y.GetBlock(block) = 0.;
        }
        for (int i = 1; i < A.Height(); i++) {
            if (fabs(A(i,block)) > 0.) y.GetBlock(block).Add(A(i,block), x.GetBlock(i));
        }
    }
};


/// Constructor
IRK::IRK(IRKOperator *IRKOper_, const RKData &ButcherTableau)
        : m_comm(IRKOper_->GetComm()),
        m_Butcher(ButcherTableau),
        m_IRKOper(IRKOper_),
        m_stageOper(NULL),
        m_solversInit(false),
        m_newton_solver(NULL),
        m_tri_jac_solver(NULL),
        m_jac_solverSparsity(NULL),
        m_jac_precSparsity(NULL),
        m_krylov2(false),
        m_schur_precondition(true)
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
        } else {
            // eta == diag entry of block, beta == sqrt(-1 * product of off diagonals)
            m_eig_ratio[i] = sqrt(-m_Butcher.R0(row+1,row)*m_Butcher.R0(row,row+1)) / m_Butcher.R0(row,row);
            row++;
        }
        row++;
    }
}

/// Destructor
IRK::~IRK() {
    if (m_tri_jac_solver) delete m_tri_jac_solver;
    if (m_newton_solver) delete m_newton_solver;
    if (m_stageOper) delete m_stageOper;

    if (m_jac_solverSparsity) delete m_jac_solverSparsity;
    if (m_jac_precSparsity) delete m_jac_precSparsity;
}


/// Build solvers
void IRK::SetSolvers()
{
    if (m_solversInit) return;
    m_solversInit = true;

    // Setup Newton solver
    m_newton_solver = new NewtonSolver(m_comm);
    m_newton_solver->iterative_mode = false;
    m_newton_solver->SetMaxIter(m_newton_params.maxiter);
    m_newton_solver->SetRelTol(m_newton_params.reltol);
    m_newton_solver->SetAbsTol(m_newton_params.abstol);
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
                                m_newton_params.jac_update_rate, m_newton_params.gamma_idx,
                                m_krylov_params, m_krylov_params2,
                                m_jac_solverSparsity, m_jac_precSparsity,
                                m_schur_precondition);
    // Use same Krylov solvers for 1x1 and 2x2 blocks
    } else {
        m_tri_jac_solver = new TriJacSolver(*m_stageOper,
                                    m_newton_params.jac_update_rate, m_newton_params.gamma_idx,
                                    m_krylov_params,
                                    m_jac_solverSparsity, m_jac_precSparsity,
                                    m_schur_precondition);
    }
    m_newton_solver->SetSolver(*m_tri_jac_solver);
}


/// Call base class' init and initialize remaing things here.
void IRK::Init(TimeDependentOperator &F)
{
    m_w = 0.;     // Initialize stage vectors to 0
    SetSolvers(); // Initialize solvers for computing stage vectors
    ODESolver::Init(F);
    m_w.Update(m_stageOffsets, mem_type); // Stage vectors
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
        mfem_warning(msg.c_str());
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


/// Set dimensions of data structures
void RKData::SizeData() {

    // Basic `Butcher Tableau` data
    A0.SetSize(s);
    invA0.SetSize(s);
    b0.SetSize(s);
    c0.SetSize(s);
    d0.SetSize(s);

    // NOTE:
    //     s := 2*n(cc_eig_pairs) + n(r_eigs)
    // s_eff := n(cc_eig_pairs) + n(r_eigs)
    // ==>  n(cc_eig_pairs) = s - s_eff
    // ==>  n(r_eigs) = 2*s_eff - s
    zeta.SetSize(2*s_eff-s);
    beta.SetSize(s-s_eff);
    eta.SetSize(s-s_eff);
    Q0.SetSize(s);
    R0.SetSize(s);
    R0_block_sizes.SetSize(s_eff);
}


/** Set dummy 2x2 RK data that has the specified value of beta_on_eta == beta/eta
    and real part of eigenvalue equal to eta_ */
void RKData::SetDummyData(double beta_on_eta, double eta_) {
    s = 2;      // 2 stages
    s_eff = 1;  // Have one 2x2 system

    SizeData();
    /* --- eta --- */
    eta(0) = eta_; //
    /* --- beta --- */
    beta(0) = beta_on_eta*eta(0);
    /* --- R block sizes --- */
    R0_block_sizes[0] = 2;

    double phi = 1.0; // Some non-zero constant...
    double a = eta(0);
    double b = phi;
    double c = -beta(0)*beta(0)/phi;
    double d = eta(0);

    /* --- inv(A) --- */
    invA0(0, 0) = a;
    invA0(0, 1) = b;
    invA0(1, 0) = c;
    invA0(1, 1) = d;

    /* --- A --- */
    double det = a*d - b*d;
    A0(0, 0) = d/det;
    A0(0, 1) = -b/det;
    A0(1, 0) = -c/det;
    A0(1, 1) = a/det;

    /* --- Q --- */
    // choose Q as identity, so inv(A) is equal to R.
    Q0(0, 0) = 1.0;
    Q0(0, 1) = 0.0;
    Q0(1, 0) = 0.0;
    Q0(1, 1) = 1.0;
    /* --- R --- */
    // Set R = inv(A0)
    R0(0, 0) = a;
    R0(0, 1) = b;
    R0(1, 0) = c;
    R0(1, 1) = d;

    /* --- b --- */
    // These are just non-zero dummy values.
    b0(0) = 0.5;
    b0(1) = 0.5;
    /* --- c --- */
    // c is the row sums of A
    c0(0) = A0(0, 0) + A0(0, 1);
    c0(1) = A0(1, 0) + A0(1, 1);
    /* --- d --- */
    // d = inv(A0)^T * b
    d0(0) = invA0(0, 0)*b0(0) + invA0(1, 0)*b0(1);
    d0(1) = invA0(0, 1)*b0(0) + invA0(1, 1)*b0(1);;
}


/// Set data required by solvers
void RKData::SetData() {
    switch(ID) {
        // 2-stage 3rd-order A-stable SDIRK
        case Type::ASDIRK3:
            s = 2;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.788675134594813;
            A0(0, 1) = +0.000000000000000;
            A0(1, 0) = -0.577350269189626;
            A0(1, 1) = +0.788675134594813;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.267949192431123;
            invA0(0, 1) = -0.000000000000000;
            invA0(1, 0) = +0.928203230275509;
            invA0(1, 1) = +1.267949192431123;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.788675134594813;
            c0(1) = +0.211324865405187;
            /* --- d --- */
            d0(0) = +1.098076211353316;
            d0(1) = +0.633974596215561;
            /* --- zeta --- */
            zeta(0) = +1.267949192431123;
            zeta(1) = +1.267949192431123;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +1.000000000000000;
            Q0(1, 0) = -1.000000000000000;
            Q0(1, 1) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +1.267949192431123;
            R0(0, 1) = -0.928203230275509;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +1.267949192431123;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;

        // 3-stage 4th-order A-stable SDIRK
        case Type::ASDIRK4:
            s = 3;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +1.068579021301629;
            A0(0, 1) = +0.000000000000000;
            A0(0, 2) = +0.000000000000000;
            A0(1, 0) = -0.568579021301629;
            A0(1, 1) = +1.068579021301629;
            A0(1, 2) = +0.000000000000000;
            A0(2, 0) = +2.137158042603258;
            A0(2, 1) = -3.274316085206515;
            A0(2, 2) = +1.068579021301629;
            /* --- inv(A) --- */
            invA0(0, 0) = +0.935822227524088;
            invA0(0, 1) = -0.000000000000000;
            invA0(0, 2) = +0.000000000000000;
            invA0(1, 0) = +0.497940606760015;
            invA0(1, 1) = +0.935822227524088;
            invA0(1, 2) = +0.000000000000000;
            invA0(2, 0) = -0.345865915800969;
            invA0(2, 1) = +2.867525668568206;
            invA0(2, 2) = +0.935822227524088;
            /* --- b --- */
            b0(0) = +0.128886400515720;
            b0(1) = +0.742227198968559;
            b0(2) = +0.128886400515720;
            /* --- c --- */
            c0(0) = +1.068579021301629;
            c0(1) = +0.500000000000000;
            c0(2) = -0.068579021301629;
            /* --- d --- */
            d0(0) = +0.445622407287714;
            d0(1) = +1.064177772475912;
            d0(2) = +0.120614758428183;
            /* --- zeta --- */
            zeta(0) = +0.935822227524088;
            zeta(1) = +0.935822227524088;
            zeta(2) = +0.935822227524088;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(0, 2) = -1.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            Q0(1, 2) = +0.000000000000000;
            Q0(2, 0) = -1.000000000000000;
            Q0(2, 1) = +0.000000000000000;
            Q0(2, 2) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +0.935822227524088;
            R0(0, 1) = -2.867525668568206;
            R0(0, 2) = -0.345865915800969;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +0.935822227524088;
            R0(1, 2) = -0.497940606760015;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +0.935822227524088;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 1;
            break;

        // 1-stage 1st-order L-stable SDIRK
        case Type::LSDIRK1:
            s = 1;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +1.000000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.000000000000000;
            /* --- b --- */
            b0(0) = +1.000000000000000;
            /* --- c --- */
            c0(0) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +1.000000000000000;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = -1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +1.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            break;

        // 2-stage 2nd-order L-stable SDIRK
        case Type::LSDIRK2:
            s = 2;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.292893218813453;
            A0(0, 1) = +0.000000000000000;
            A0(1, 0) = +0.414213562373095;
            A0(1, 1) = +0.292893218813453;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.414213562373094;
            invA0(0, 1) = +0.000000000000000;
            invA0(1, 0) = -4.828427124746186;
            invA0(1, 1) = +3.414213562373094;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.292893218813453;
            c0(1) = +0.707106781186547;
            /* --- d --- */
            d0(0) = -0.707106781186546;
            d0(1) = +1.707106781186547;
            /* --- zeta --- */
            zeta(0) = +3.414213562373094;
            zeta(1) = +3.414213562373094;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +1.000000000000000;
            Q0(1, 0) = -1.000000000000000;
            Q0(1, 1) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +3.414213562373094;
            R0(0, 1) = +4.828427124746187;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +3.414213562373094;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            break;

        // 3-stage 3rd-order L-stable SDIRK
        case Type::LSDIRK3:
            s = 3;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.435866521508459;
            A0(0, 1) = +0.000000000000000;
            A0(0, 2) = +0.000000000000000;
            A0(1, 0) = +0.282066739245771;
            A0(1, 1) = +0.435866521508459;
            A0(1, 2) = +0.000000000000000;
            A0(2, 0) = +1.208496649176010;
            A0(2, 1) = -0.644363170684469;
            A0(2, 2) = +0.435866521508459;
            /* --- inv(A) --- */
            invA0(0, 0) = +2.294280360279042;
            invA0(0, 1) = -0.000000000000000;
            invA0(0, 2) = -0.000000000000000;
            invA0(1, 0) = -1.484721005641544;
            invA0(1, 1) = +2.294280360279042;
            invA0(1, 2) = +0.000000000000000;
            invA0(2, 0) = -8.556127801552645;
            invA0(2, 1) = +3.391748836942547;
            invA0(2, 2) = +2.294280360279042;
            /* --- b --- */
            b0(0) = +1.208496649176010;
            b0(1) = -0.644363170684469;
            b0(2) = +0.435866521508459;
            /* --- c --- */
            c0(0) = +0.435866521508459;
            c0(1) = +0.717933260754229;
            c0(2) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +2.294280360279042;
            zeta(1) = +2.294280360279042;
            zeta(2) = +2.294280360279042;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(0, 2) = -1.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            Q0(1, 2) = +0.000000000000000;
            Q0(2, 0) = -1.000000000000000;
            Q0(2, 1) = +0.000000000000000;
            Q0(2, 2) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +2.294280360279042;
            R0(0, 1) = -3.391748836942547;
            R0(0, 2) = -8.556127801552645;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +2.294280360279042;
            R0(1, 2) = +1.484721005641544;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +2.294280360279042;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 1;
            break;

        // 5-stage 4th-order L-stable SDIRK
        case Type::LSDIRK4:
            s = 5;
            s_eff = 5;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.250000000000000;
            A0(0, 1) = +0.000000000000000;
            A0(0, 2) = +0.000000000000000;
            A0(0, 3) = +0.000000000000000;
            A0(0, 4) = +0.000000000000000;
            A0(1, 0) = +0.500000000000000;
            A0(1, 1) = +0.250000000000000;
            A0(1, 2) = +0.000000000000000;
            A0(1, 3) = +0.000000000000000;
            A0(1, 4) = +0.000000000000000;
            A0(2, 0) = +0.340000000000000;
            A0(2, 1) = -0.040000000000000;
            A0(2, 2) = +0.250000000000000;
            A0(2, 3) = +0.000000000000000;
            A0(2, 4) = +0.000000000000000;
            A0(3, 0) = +0.272794117647059;
            A0(3, 1) = -0.050367647058824;
            A0(3, 2) = +0.027573529411765;
            A0(3, 3) = +0.250000000000000;
            A0(3, 4) = +0.000000000000000;
            A0(4, 0) = +1.041666666666667;
            A0(4, 1) = -1.020833333333333;
            A0(4, 2) = +7.812500000000000;
            A0(4, 3) = -7.083333333333333;
            A0(4, 4) = +0.250000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +4.000000000000000;
            invA0(0, 1) = +0.000000000000000;
            invA0(0, 2) = +0.000000000000000;
            invA0(0, 3) = +0.000000000000000;
            invA0(0, 4) = +0.000000000000000;
            invA0(1, 0) = -7.999999999999995;
            invA0(1, 1) = +3.999999999999998;
            invA0(1, 2) = +0.000000000000001;
            invA0(1, 3) = -0.000000000000003;
            invA0(1, 4) = +0.000000000000000;
            invA0(2, 0) = -6.719999999999998;
            invA0(2, 1) = +0.639999999999999;
            invA0(2, 2) = +4.000000000000001;
            invA0(2, 3) = -0.000000000000002;
            invA0(2, 4) = -0.000000000000000;
            invA0(3, 0) = -5.235294117647047;
            invA0(3, 1) = +0.735294117647057;
            invA0(3, 2) = -0.441176470588240;
            invA0(3, 3) = +3.999999999999998;
            invA0(3, 4) = +0.000000000000000;
            invA0(4, 0) = +12.333333333333645;
            invA0(4, 1) = +17.166666666666643;
            invA0(4, 2) = -137.500000000000171;
            invA0(4, 3) = +113.333333333333329;
            invA0(4, 4) = +4.000000000000000;
            /* --- b --- */
            b0(0) = +1.041666666666667;
            b0(1) = -1.020833333333333;
            b0(2) = +7.812500000000000;
            b0(3) = -7.083333333333333;
            b0(4) = +0.250000000000000;
            /* --- c --- */
            c0(0) = +0.250000000000000;
            c0(1) = +0.750000000000000;
            c0(2) = +0.550000000000000;
            c0(3) = +0.500000000000000;
            c0(4) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = +0.000000000000000;
            d0(4) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +4.000000000000000;
            zeta(1) = +4.000000000000000;
            zeta(2) = +4.000000000000000;
            zeta(3) = +4.000000000000000;
            zeta(4) = +4.000000000000000;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = +0.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(0, 2) = +0.000000000000000;
            Q0(0, 3) = +0.000000000000000;
            Q0(0, 4) = -1.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +0.000000000000000;
            Q0(1, 2) = +0.000000000000000;
            Q0(1, 3) = +1.000000000000000;
            Q0(1, 4) = +0.000000000000000;
            Q0(2, 0) = +0.000000000000000;
            Q0(2, 1) = +0.000000000000000;
            Q0(2, 2) = -1.000000000000000;
            Q0(2, 3) = +0.000000000000000;
            Q0(2, 4) = +0.000000000000000;
            Q0(3, 0) = +0.000000000000000;
            Q0(3, 1) = +1.000000000000000;
            Q0(3, 2) = +0.000000000000000;
            Q0(3, 3) = +0.000000000000000;
            Q0(3, 4) = +0.000000000000000;
            Q0(4, 0) = -1.000000000000000;
            Q0(4, 1) = +0.000000000000000;
            Q0(4, 2) = +0.000000000000000;
            Q0(4, 3) = +0.000000000000000;
            Q0(4, 4) = +0.000000000000000;
            /* --- R --- */
            R0(0, 0) = +4.000000000000000;
            R0(0, 1) = -113.333333333333329;
            R0(0, 2) = -137.500000000000000;
            R0(0, 3) = -17.166666666666671;
            R0(0, 4) = +12.333333333333357;
            R0(1, 0) = +0.000000000000000;
            R0(1, 1) = +4.000000000000000;
            R0(1, 2) = +0.441176470588235;
            R0(1, 3) = +0.735294117647059;
            R0(1, 4) = +5.235294117647059;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +4.000000000000000;
            R0(2, 3) = -0.640000000000000;
            R0(2, 4) = -6.720000000000001;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +4.000000000000000;
            R0(3, 4) = +8.000000000000000;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = +0.000000000000000;
            R0(4, 4) = +4.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 1;
            R0_block_sizes[3] = 1;
            R0_block_sizes[4] = 1;
            break;

        // 1-stage 2nd-order Gauss--Legendre
        case Type::Gauss2:
            s = 1;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.500000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +2.000000000000000;
            /* --- b --- */
            b0(0) = +1.000000000000000;
            /* --- c --- */
            c0(0) = +0.500000000000000;
            /* --- d --- */
            d0(0) = +2.000000000000000;
            /* --- zeta --- */
            zeta(0) = +2.000000000000000;
            /* --- eta --- */
            /* --- beta --- */
            /* --- Q --- */
            Q0(0, 0) = +1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +2.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 1;
            break;

        // 2-stage 4th-order Gauss--Legendre
        case Type::Gauss4:
            s = 2;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.250000000000000;
            A0(0, 1) = -0.038675134594813;
            A0(1, 0) = +0.538675134594813;
            A0(1, 1) = +0.250000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.000000000000000;
            invA0(0, 1) = +0.464101615137754;
            invA0(1, 0) = -6.464101615137755;
            invA0(1, 1) = +3.000000000000000;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.211324865405187;
            c0(1) = +0.788675134594813;
            /* --- d --- */
            d0(0) = -1.732050807568877;
            d0(1) = +1.732050807568877;
            /* --- zeta --- */
            /* --- eta --- */
            eta(0) = +3.000000000000000;
            /* --- beta --- */
            beta(0) = +1.732050807568877;
            /* --- Q --- */
            Q0(0, 0) = +1.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +3.000000000000000;
            R0(0, 1) = +0.464101615137754;
            R0(1, 0) = -6.464101615137755;
            R0(1, 1) = +3.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            break;

        // 3-stage 6th-order Gauss--Legendre
        case Type::Gauss6:
            s = 3;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.138888888888889;
            A0(0, 1) = -0.035976667524939;
            A0(0, 2) = +0.009789444015308;
            A0(1, 0) = +0.300263194980865;
            A0(1, 1) = +0.222222222222222;
            A0(1, 2) = -0.022485417203087;
            A0(2, 0) = +0.267988333762469;
            A0(2, 1) = +0.480421111969383;
            A0(2, 2) = +0.138888888888889;
            /* --- inv(A) --- */
            invA0(0, 0) = +4.999999999999997;
            invA0(0, 1) = +1.163977794943223;
            invA0(0, 2) = -0.163977794943222;
            invA0(1, 0) = -5.727486121839513;
            invA0(1, 1) = +2.000000000000000;
            invA0(1, 2) = +0.727486121839514;
            invA0(2, 0) = +10.163977794943225;
            invA0(2, 1) = -9.163977794943223;
            invA0(2, 2) = +5.000000000000001;
            /* --- b --- */
            b0(0) = +0.277777777777778;
            b0(1) = +0.444444444444444;
            b0(2) = +0.277777777777778;
            /* --- c --- */
            c0(0) = +0.112701665379258;
            c0(1) = +0.500000000000000;
            c0(2) = +0.887298334620742;
            /* --- d --- */
            d0(0) = +1.666666666666668;
            d0(1) = -1.333333333333334;
            d0(2) = +1.666666666666667;
            /* --- zeta --- */
            zeta(0) = +4.644370709252176;
            /* --- eta --- */
            eta(0) = +3.677814645373911;
            /* --- beta --- */
            beta(0) = +3.508761919567443;
            /* --- Q --- */
            Q0(0, 0) = -0.083238672942822;
            Q0(0, 1) = -0.181066379097451;
            Q0(0, 2) = +0.979941982816971;
            Q0(1, 0) = +0.060321530401366;
            Q0(1, 1) = +0.980635889214054;
            Q0(1, 2) = +0.186318452535969;
            Q0(2, 0) = +0.994702285257631;
            Q0(2, 1) = -0.074620500841923;
            Q0(2, 2) = +0.070704628966890;
            /* --- R --- */
            R0(0, 0) = +3.677814645373911;
            R0(0, 1) = -10.983730977131060;
            R0(0, 2) = +7.822707869262245;
            R0(1, 0) = +1.120876889086220;
            R0(1, 1) = +3.677814645373911;
            R0(1, 2) = -6.654600663998786;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +4.644370709252176;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            break;

        // 4-stage 8th-order Gauss--Legendre
        case Type::Gauss8:
            s = 4;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.086963711284363;
            A0(0, 1) = -0.026604180084999;
            A0(0, 2) = +0.012627462689405;
            A0(0, 3) = -0.003555149685796;
            A0(1, 0) = +0.188118117499868;
            A0(1, 1) = +0.163036288715637;
            A0(1, 2) = -0.027880428602471;
            A0(1, 3) = +0.006735500594538;
            A0(2, 0) = +0.167191921974189;
            A0(2, 1) = +0.353953006033744;
            A0(2, 2) = +0.163036288715637;
            A0(2, 3) = -0.014190694931141;
            A0(3, 0) = +0.177482572254523;
            A0(3, 1) = +0.313445114741868;
            A0(3, 2) = +0.352676757516272;
            A0(3, 3) = +0.086963711284363;
            /* --- inv(A) --- */
            invA0(0, 0) = +7.738612787525829;
            invA0(0, 1) = +2.045089650303909;
            invA0(0, 2) = -0.437070802395799;
            invA0(0, 3) = +0.086644023503262;
            invA0(1, 0) = -7.201340999706891;
            invA0(1, 1) = +2.261387212474169;
            invA0(1, 2) = +1.448782034533682;
            invA0(1, 3) = -0.233133981212417;
            invA0(2, 0) = +6.343622218624971;
            invA0(2, 1) = -5.971556459482020;
            invA0(2, 2) = +2.261387212474169;
            invA0(2, 3) = +1.090852762294336;
            invA0(3, 0) = -15.563869598554923;
            invA0(3, 1) = +11.892783878056845;
            invA0(3, 2) = -13.500802725964952;
            invA0(3, 3) = +7.738612787525829;
            /* --- b --- */
            b0(0) = +0.173927422568727;
            b0(1) = +0.326072577431273;
            b0(2) = +0.326072577431273;
            b0(3) = +0.173927422568727;
            /* --- c --- */
            c0(0) = +0.069431844202974;
            c0(1) = +0.330009478207572;
            c0(2) = +0.669990521792428;
            c0(3) = +0.930568155797026;
            /* --- d --- */
            d0(0) = -1.640705321739257;
            d0(1) = +1.214393969798579;
            d0(2) = -1.214393969798578;
            d0(3) = +1.640705321739257;
            /* --- zeta --- */
            /* --- eta --- */
            eta(0) = +4.207578794359268;
            eta(1) = +5.792421205640723;
            /* --- beta --- */
            beta(0) = +5.314836083713508;
            beta(1) = +1.734468257869062;
            /* --- Q --- */
            Q0(0, 0) = +0.048240345831208;
            Q0(0, 1) = +0.022512328854129;
            Q0(0, 2) = -0.336698980990652;
            Q0(0, 3) = -0.940106302650665;
            Q0(1, 0) = -0.122245946830287;
            Q0(1, 1) = +0.092517479973163;
            Q0(1, 2) = +0.929016942971458;
            Q0(1, 3) = -0.336784744391653;
            Q0(2, 0) = +0.111699705302275;
            Q0(2, 1) = -0.987901978189380;
            Q0(2, 2) = +0.094333738332995;
            Q0(2, 3) = -0.051710764227736;
            Q0(3, 0) = +0.985013691962216;
            Q0(3, 1) = +0.122406668276236;
            Q0(3, 2) = +0.121088652168346;
            Q0(3, 3) = +0.010108042566567;
            /* --- R --- */
            R0(0, 0) = +4.207578794359268;
            R0(0, 1) = +14.740509080125308;
            R0(0, 2) = +14.291665730718535;
            R0(0, 3) = +9.670422106794417;
            R0(1, 0) = -1.916316624018728;
            R0(1, 1) = +4.207578794359268;
            R0(1, 2) = +9.614204997121385;
            R0(1, 3) = +5.775112524448543;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +5.792421205640723;
            R0(2, 3) = +9.181556269430962;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = -0.327654707902998;
            R0(3, 3) = +5.792421205640723;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 2;
            break;

        // 5-stage 10th-order Gauss--Legendre
        case Type::Gauss10:
            s = 5;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.059231721264047;
            A0(0, 1) = -0.019570364359076;
            A0(0, 2) = +0.011254400818643;
            A0(0, 3) = -0.005593793660812;
            A0(0, 4) = +0.001588112967866;
            A0(1, 0) = +0.128151005670045;
            A0(1, 1) = +0.119657167624842;
            A0(1, 2) = -0.024592114619642;
            A0(1, 3) = +0.010318280670683;
            A0(1, 4) = -0.002768994398770;
            A0(2, 0) = +0.113776288004225;
            A0(2, 1) = +0.260004651680642;
            A0(2, 2) = +0.142222222222222;
            A0(2, 3) = -0.020690316430958;
            A0(2, 4) = +0.004687154523870;
            A0(3, 0) = +0.121232436926864;
            A0(3, 1) = +0.228996054579000;
            A0(3, 2) = +0.309036559064087;
            A0(3, 3) = +0.119657167624842;
            A0(3, 4) = -0.009687563141951;
            A0(4, 0) = +0.116875329560229;
            A0(4, 1) = +0.244908128910495;
            A0(4, 2) = +0.273190043625802;
            A0(4, 3) = +0.258884699608759;
            A0(4, 4) = +0.059231721264047;
            /* --- inv(A) --- */
            invA0(0, 0) = +11.183300132670379;
            invA0(0, 1) = +3.131312162011809;
            invA0(0, 2) = -0.758731795980806;
            invA0(0, 3) = +0.239101223353686;
            invA0(0, 4) = -0.054314760565339;
            invA0(1, 0) = -9.447599601516151;
            invA0(1, 1) = +2.816699867329624;
            invA0(1, 2) = +2.217886332274817;
            invA0(1, 3) = -0.557122620293796;
            invA0(1, 4) = +0.118357949604667;
            invA0(2, 0) = +6.420116503559337;
            invA0(2, 1) = -6.220120454669752;
            invA0(2, 2) = +2.000000000000000;
            invA0(2, 3) = +1.865995288831779;
            invA0(2, 4) = -0.315991337721364;
            invA0(3, 0) = -8.015920784810973;
            invA0(3, 1) = +6.190522354953043;
            invA0(3, 2) = -7.393116276382382;
            invA0(3, 3) = +2.816699867329624;
            invA0(3, 4) = +1.550036766309846;
            invA0(4, 0) = +22.420915025906101;
            invA0(4, 1) = -16.193390239999243;
            invA0(4, 2) = +15.415443221569852;
            invA0(4, 3) = -19.085601178657356;
            invA0(4, 4) = +11.183300132670379;
            /* --- b --- */
            b0(0) = +0.118463442528095;
            b0(1) = +0.239314335249683;
            b0(2) = +0.284444444444444;
            b0(3) = +0.239314335249683;
            b0(4) = +0.118463442528095;
            /* --- c --- */
            c0(0) = +0.046910077030668;
            c0(1) = +0.230765344947158;
            c0(2) = +0.500000000000000;
            c0(3) = +0.769234655052841;
            c0(4) = +0.953089922969332;
            /* --- d --- */
            d0(0) = +1.627766710890126;
            d0(1) = -1.161100044223459;
            d0(2) = +1.066666666666666;
            d0(3) = -1.161100044223459;
            d0(4) = +1.627766710890126;
            /* --- zeta --- */
            zeta(0) = +7.293477190659350;
            /* --- eta --- */
            eta(0) = +4.649348606363301;
            eta(1) = +6.703912798307031;
            /* --- beta --- */
            beta(0) = +7.142045840675963;
            beta(1) = +3.485322832366363;
            /* --- Q --- */
            Q0(0, 0) = +0.052286373550732;
            Q0(0, 1) = +0.017761459146135;
            Q0(0, 2) = +0.107901918652109;
            Q0(0, 3) = -0.882638337196087;
            Q0(0, 4) = +0.454155708290488;
            Q0(1, 0) = -0.087217337438407;
            Q0(1, 1) = -0.051247596714227;
            Q0(1, 2) = -0.222975377135070;
            Q0(1, 3) = -0.469846600315604;
            Q0(1, 4) = -0.848111415584363;
            Q0(2, 0) = -0.005226019290336;
            Q0(2, 1) = +0.155957034844084;
            Q0(2, 2) = +0.954541442335525;
            Q0(2, 3) = -0.011034361806211;
            Q0(2, 4) = -0.253730111986263;
            Q0(3, 0) = +0.981199881220355;
            Q0(3, 1) = -0.167084318758466;
            Q0(3, 2) = +0.007191508152453;
            Q0(3, 3) = +0.006182333497419;
            Q0(3, 4) = -0.096123277520073;
            Q0(4, 0) = -0.163947409270801;
            Q0(4, 1) = -0.972017720058475;
            Q0(4, 2) = +0.165644421026408;
            Q0(4, 3) = +0.005810292217529;
            Q0(4, 4) = +0.028826466534928;
            /* --- R --- */
            R0(0, 0) = +4.649348606363301;
            R0(0, 1) = -2.866988395237885;
            R0(0, 2) = -12.612472186300799;
            R0(0, 3) = +5.040226491530089;
            R0(0, 4) = -10.015206098251161;
            R0(1, 0) = +17.791777209507831;
            R0(1, 1) = +4.649348606363301;
            R0(1, 2) = -19.736222014661173;
            R0(1, 3) = +10.353647543145909;
            R0(1, 4) = -18.748802010104690;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +7.293477190659350;
            R0(2, 3) = -7.436145789040616;
            R0(2, 4) = +12.655118452703233;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +6.703912798307031;
            R0(3, 4) = +1.090191404805731;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = -11.142516068523712;
            R0(4, 4) = +6.703912798307031;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 2;
            break;

        // 2-stage 3rd-order Radau IIA
        case Type::RadauIIA3:
            s = 2;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.416666666666667;
            A0(0, 1) = -0.083333333333333;
            A0(1, 0) = +0.750000000000000;
            A0(1, 1) = +0.250000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.500000000000000;
            invA0(0, 1) = +0.500000000000000;
            invA0(1, 0) = -4.500000000000000;
            invA0(1, 1) = +2.500000000000000;
            /* --- b --- */
            b0(0) = +0.750000000000000;
            b0(1) = +0.250000000000000;
            /* --- c --- */
            c0(0) = +0.333333333333333;
            c0(1) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = +1.000000000000000;
            /* --- zeta --- */
            /* --- eta --- */
            eta(0) = +2.000000000000000;
            /* --- beta --- */
            beta(0) = +1.414213562373095;
            /* --- Q --- */
            Q0(0, 0) = +0.992507556682903;
            Q0(0, 1) = +0.122183263695704;
            Q0(1, 0) = -0.122183263695704;
            Q0(1, 1) = +0.992507556682903;
            /* --- R --- */
            R0(0, 0) = +2.000000000000000;
            R0(0, 1) = +0.438447187191170;
            R0(1, 0) = -4.561552812808831;
            R0(1, 1) = +2.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            break;

        // 3-stage 5th-order Radau IIA
        case Type::RadauIIA5:
            s = 3;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.196815477223660;
            A0(0, 1) = -0.065535425850198;
            A0(0, 2) = +0.023770974348220;
            A0(1, 0) = +0.394424314739087;
            A0(1, 1) = +0.292073411665228;
            A0(1, 2) = -0.041548752125998;
            A0(2, 0) = +0.376403062700467;
            A0(2, 1) = +0.512485826188422;
            A0(2, 2) = +0.111111111111111;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.224744871391589;
            invA0(0, 1) = +1.167840084690405;
            invA0(0, 2) = -0.253197264742181;
            invA0(1, 0) = -3.567840084690406;
            invA0(1, 1) = +0.775255128608411;
            invA0(1, 2) = +1.053197264742181;
            invA0(2, 0) = +5.531972647421809;
            invA0(2, 1) = -7.531972647421810;
            invA0(2, 2) = +5.000000000000001;
            /* --- b --- */
            b0(0) = +0.376403062700467;
            b0(1) = +0.512485826188422;
            b0(2) = +0.111111111111111;
            /* --- c --- */
            c0(0) = +0.155051025721682;
            c0(1) = +0.644948974278318;
            c0(2) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +3.637834252744495;
            /* --- eta --- */
            eta(0) = +2.681082873627751;
            /* --- beta --- */
            beta(0) = +3.050430199247411;
            /* --- Q --- */
            Q0(0, 0) = +0.138665108751908;
            Q0(0, 1) = +0.046278149309489;
            Q0(0, 2) = +0.989257459163847;
            Q0(1, 0) = -0.229641242351741;
            Q0(1, 1) = -0.970178886551833;
            Q0(1, 2) = +0.077574660168093;
            Q0(2, 0) = -0.963346711950568;
            Q0(2, 1) = +0.237931210616714;
            Q0(2, 2) = +0.123902589111344;
            /* --- R --- */
            R0(0, 0) = +2.681082873627751;
            R0(0, 1) = -8.423875798085390;
            R0(0, 2) = -4.088577813059627;
            R0(1, 0) = +1.104613199852199;
            R0(1, 1) = +2.681082873627751;
            R0(1, 2) = +4.700152006343942;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +3.637834252744495;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            break;

        // 4-stage 7th-order Radau IIA
        case Type::RadauIIA7:
            s = 4;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.112999479323156;
            A0(0, 1) = -0.040309220723522;
            A0(0, 2) = +0.025802377420336;
            A0(0, 3) = -0.009904676507266;
            A0(1, 0) = +0.234383995747400;
            A0(1, 1) = +0.206892573935359;
            A0(1, 2) = -0.047857128048541;
            A0(1, 3) = +0.016047422806516;
            A0(2, 0) = +0.216681784623250;
            A0(2, 1) = +0.406123263867373;
            A0(2, 2) = +0.189036518170056;
            A0(2, 3) = -0.024182104899833;
            A0(3, 0) = +0.220462211176768;
            A0(3, 1) = +0.388193468843172;
            A0(3, 2) = +0.328844319980060;
            A0(3, 3) = +0.062500000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +5.644107875950090;
            invA0(0, 1) = +1.923507277054713;
            invA0(0, 2) = -0.585901482103817;
            invA0(0, 3) = +0.173878352574246;
            invA0(1, 0) = -5.049214638391410;
            invA0(1, 1) = +1.221100028894693;
            invA0(1, 2) = +1.754680988760837;
            invA0(1, 3) = -0.434791461212582;
            invA0(2, 0) = +3.492466158625437;
            invA0(2, 1) = -3.984517895782496;
            invA0(2, 2) = +0.634792095155218;
            invA0(2, 3) = +1.822137598434254;
            invA0(3, 0) = -6.923488256445451;
            invA0(3, 1) = +6.595237669628144;
            invA0(3, 2) = -12.171749413182692;
            invA0(3, 3) = +8.500000000000002;
            /* --- b --- */
            b0(0) = +0.220462211176768;
            b0(1) = +0.388193468843172;
            b0(2) = +0.328844319980060;
            b0(3) = +0.062500000000000;
            /* --- c --- */
            c0(0) = +0.088587959512704;
            c0(1) = +0.409466864440735;
            c0(2) = +0.787659461760847;
            c0(3) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = -0.000000000000000;
            d0(3) = +1.000000000000000;
            /* --- zeta --- */
            /* --- eta --- */
            eta(0) = +3.212806896871531;
            eta(1) = +4.787193103128468;
            /* --- beta --- */
            beta(0) = +4.773087433276639;
            beta(1) = +1.567476416895209;
            /* --- Q --- */
            Q0(0, 0) = +0.054570292994775;
            Q0(0, 1) = +0.125683294495023;
            Q0(0, 2) = +0.956633137728588;
            Q0(0, 3) = -0.257058033149907;
            Q0(1, 0) = -0.177667119738735;
            Q0(1, 1) = -0.126991262238752;
            Q0(1, 2) = +0.278165559200843;
            Q0(1, 3) = +0.935377750191457;
            Q0(2, 0) = +0.321254131229142;
            Q0(2, 1) = -0.939193753944121;
            Q0(2, 2) = +0.080747339995071;
            Q0(2, 3) = -0.090502722634621;
            Q0(3, 0) = +0.928575393198859;
            Q0(3, 1) = +0.293243962175244;
            Q0(3, 2) = -0.030932645502120;
            Q0(3, 3) = +0.225386089268147;
            /* --- R --- */
            R0(0, 0) = +3.212806896871531;
            R0(0, 1) = +12.141571045775553;
            R0(0, 2) = -3.796586217388935;
            R0(0, 3) = +8.446761444650136;
            R0(1, 0) = -1.876393389274784;
            R0(1, 1) = +3.212806896871531;
            R0(1, 2) = -2.571864526766571;
            R0(1, 3) = +7.005683821998450;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +4.787193103128468;
            R0(2, 3) = +0.344651300061515;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = -7.128893223626639;
            R0(3, 3) = +4.787193103128468;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 2;
            break;

        // 5-stage 9th-order Radau IIA
        case Type::RadauIIA9:
            s = 5;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.072998864317903;
            A0(0, 1) = -0.026735331107946;
            A0(0, 2) = +0.018676929763984;
            A0(0, 3) = -0.012879106093306;
            A0(0, 4) = +0.005042839233882;
            A0(1, 0) = +0.153775231479182;
            A0(1, 1) = +0.146214867847494;
            A0(1, 2) = -0.036444568905128;
            A0(1, 3) = +0.021233063119305;
            A0(1, 4) = -0.007935579902729;
            A0(2, 0) = +0.140063045684810;
            A0(2, 1) = +0.298967129491283;
            A0(2, 2) = +0.167585070135249;
            A0(2, 3) = -0.033969101686618;
            A0(2, 4) = +0.010944288744192;
            A0(3, 0) = +0.144894308109535;
            A0(3, 1) = +0.276500068760159;
            A0(3, 2) = +0.325797922910421;
            A0(3, 3) = +0.128756753254910;
            A0(3, 4) = -0.015708917378805;
            A0(4, 0) = +0.143713560791226;
            A0(4, 1) = +0.281356015149462;
            A0(4, 2) = +0.311826522975741;
            A0(4, 3) = +0.223103901083571;
            A0(4, 4) = +0.040000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +8.755923977938361;
            invA0(0, 1) = +2.891942615380119;
            invA0(0, 2) = -0.875186396200266;
            invA0(0, 3) = +0.399705207939967;
            invA0(0, 4) = -0.133706163849216;
            invA0(1, 0) = -7.161380720145386;
            invA0(1, 1) = +1.806077724083645;
            invA0(1, 2) = +2.363797176068608;
            invA0(1, 3) = -0.865900780283135;
            invA0(1, 4) = +0.274338077775194;
            invA0(2, 0) = +4.122165246243370;
            invA0(2, 1) = -4.496017125813395;
            invA0(2, 2) = +0.856765245397177;
            invA0(2, 3) = +2.518320949211065;
            invA0(2, 4) = -0.657062757134360;
            invA0(3, 0) = -3.878663219724007;
            invA0(3, 1) = +3.393151918064955;
            invA0(3, 2) = -5.188340906407188;
            invA0(3, 3) = +0.581233052580817;
            invA0(3, 4) = +2.809983655279712;
            invA0(4, 0) = +8.412424223594297;
            invA0(4, 1) = -6.970256116656662;
            invA0(4, 2) = +8.777114204150472;
            invA0(4, 3) = -18.219282311088101;
            invA0(4, 4) = +13.000000000000000;
            /* --- b --- */
            b0(0) = +0.143713560791226;
            b0(1) = +0.281356015149462;
            b0(2) = +0.311826522975741;
            b0(3) = +0.223103901083571;
            b0(4) = +0.040000000000000;
            /* --- c --- */
            c0(0) = +0.057104196114518;
            c0(1) = +0.276843013638124;
            c0(2) = +0.583590432368917;
            c0(3) = +0.860240135656219;
            c0(4) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = -0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = +0.000000000000000;
            d0(4) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +6.286704751729317;
            /* --- eta --- */
            eta(0) = +3.655694325463577;
            eta(1) = +5.700953298671759;
            /* --- beta --- */
            beta(0) = +6.543736899360073;
            beta(1) = +3.210265600308560;
            /* --- Q --- */
            Q0(0, 0) = +0.104376011865108;
            Q0(0, 1) = +0.002753215699337;
            Q0(0, 2) = +0.034631183267103;
            Q0(0, 3) = +0.423574187128014;
            Q0(0, 4) = +0.899157192650324;
            Q0(1, 0) = -0.216061922297273;
            Q0(1, 1) = -0.032215375782065;
            Q0(1, 2) = -0.064584428263704;
            Q0(1, 3) = -0.869905614769537;
            Q0(1, 4) = +0.437461413515194;
            Q0(2, 0) = +0.291316048441817;
            Q0(2, 1) = +0.192522319893028;
            Q0(2, 2) = +0.925248363516386;
            Q0(2, 3) = -0.148275233623860;
            Q0(2, 4) = -0.000192788196279;
            Q0(3, 0) = +0.869441565532768;
            Q0(3, 1) = -0.401017866194354;
            Q0(3, 2) = -0.220183152562227;
            Q0(3, 3) = -0.186450946621164;
            Q0(3, 4) = -0.003385106894551;
            Q0(4, 0) = -0.318793378106442;
            Q0(4, 1) = -0.895027606670579;
            Q0(4, 2) = +0.300107277334918;
            Q0(4, 3) = +0.084259296158572;
            Q0(4, 4) = -0.011504715315962;
            /* --- R --- */
            R0(0, 0) = +3.655694325463577;
            R0(0, 1) = -2.761497758898925;
            R0(0, 2) = -9.607413444485667;
            R0(0, 3) = -4.682534851529246;
            R0(0, 4) = -0.500335166972660;
            R0(1, 0) = +15.506256512451463;
            R0(1, 1) = +3.655694325463577;
            R0(1, 2) = -13.213023448159136;
            R0(1, 3) = -8.868956774490409;
            R0(1, 4) = -2.602235394882079;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +6.286704751729317;
            R0(2, 3) = +9.642681278729382;
            R0(2, 4) = +4.066371799790950;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +5.700953298671759;
            R0(3, 4) = +9.280382169459997;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = -1.110493623682757;
            R0(4, 4) = +5.700953298671759;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 2;
            break;

        // 2-stage 2nd-order Lobatto IIIC
        case Type::LobIIIC2:
            s = 2;
            s_eff = 1;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.500000000000000;
            A0(0, 1) = -0.500000000000000;
            A0(1, 0) = +0.500000000000000;
            A0(1, 1) = +0.500000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +1.000000000000000;
            invA0(0, 1) = +1.000000000000000;
            invA0(1, 0) = -1.000000000000000;
            invA0(1, 1) = +1.000000000000000;
            /* --- b --- */
            b0(0) = +0.500000000000000;
            b0(1) = +0.500000000000000;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = +1.000000000000000;
            /* --- zeta --- */
            /* --- eta --- */
            eta(0) = +1.000000000000000;
            /* --- beta --- */
            beta(0) = +1.000000000000000;
            /* --- Q --- */
            Q0(0, 0) = +1.000000000000000;
            Q0(0, 1) = +0.000000000000000;
            Q0(1, 0) = +0.000000000000000;
            Q0(1, 1) = +1.000000000000000;
            /* --- R --- */
            R0(0, 0) = +1.000000000000000;
            R0(0, 1) = +1.000000000000000;
            R0(1, 0) = -1.000000000000000;
            R0(1, 1) = +1.000000000000000;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            break;

        // 3-stage 4th-order Lobatto IIIC
        case Type::LobIIIC4:
            s = 3;
            s_eff = 2;
            SizeData();

            /* --- A --- */
            A0(0, 0) = +0.166666666666667;
            A0(0, 1) = -0.333333333333333;
            A0(0, 2) = +0.166666666666667;
            A0(1, 0) = +0.166666666666667;
            A0(1, 1) = +0.416666666666667;
            A0(1, 2) = -0.083333333333333;
            A0(2, 0) = +0.166666666666667;
            A0(2, 1) = +0.666666666666667;
            A0(2, 2) = +0.166666666666667;
            /* --- inv(A) --- */
            invA0(0, 0) = +3.000000000000000;
            invA0(0, 1) = +4.000000000000000;
            invA0(0, 2) = -1.000000000000000;
            invA0(1, 0) = -1.000000000000000;
            invA0(1, 1) = +0.000000000000000;
            invA0(1, 2) = +1.000000000000000;
            invA0(2, 0) = +1.000000000000000;
            invA0(2, 1) = -4.000000000000000;
            invA0(2, 2) = +3.000000000000000;
            /* --- b --- */
            b0(0) = +0.166666666666667;
            b0(1) = +0.666666666666667;
            b0(2) = +0.166666666666667;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +0.500000000000000;
            c0(2) = +1.000000000000000;
            /* --- d --- */
            d0(0) = +0.000000000000000;
            d0(1) = -0.000000000000000;
            d0(2) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +2.625816818958467;
            /* --- eta --- */
            eta(0) = +1.687091590520766;
            /* --- beta --- */
            beta(0) = +2.508731754924880;
            /* --- Q --- */
            Q0(0, 0) = -0.575784070910962;
            Q0(0, 1) = -0.375422043049930;
            Q0(0, 2) = +0.726313288655396;
            Q0(1, 0) = +0.270505426506658;
            Q0(1, 1) = +0.750844086099862;
            Q0(1, 2) = +0.602544581420590;
            Q0(2, 0) = +0.771556555228230;
            Q0(2, 1) = -0.543407257920869;
            Q0(2, 2) = +0.330770364638777;
            /* --- R --- */
            R0(0, 0) = +1.687091590520766;
            R0(0, 1) = -5.303878657262186;
            R0(0, 2) = -3.092458332654278;
            R0(1, 0) = +1.186628772049857;
            R0(1, 1) = +1.687091590520766;
            R0(1, 2) = -1.519873274599652;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +2.625816818958467;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            break;

        // 4-stage 6th-order Lobatto IIIC
        case Type::LobIIIC6:
            s = 4;
            s_eff = 2;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.083333333333333;
            A0(0, 1) = -0.186338998124982;
            A0(0, 2) = +0.186338998124982;
            A0(0, 3) = -0.083333333333333;
            A0(1, 0) = +0.083333333333333;
            A0(1, 1) = +0.250000000000000;
            A0(1, 2) = -0.094207930708309;
            A0(1, 3) = +0.037267799624996;
            A0(2, 0) = +0.083333333333333;
            A0(2, 1) = +0.427541264041642;
            A0(2, 2) = +0.250000000000000;
            A0(2, 3) = -0.037267799624996;
            A0(3, 0) = +0.083333333333333;
            A0(3, 1) = +0.416666666666667;
            A0(3, 2) = +0.416666666666667;
            A0(3, 3) = +0.083333333333333;
            /* --- inv(A) --- */
            invA0(0, 0) = +5.999999999999999;
            invA0(0, 1) = +8.090169943749475;
            invA0(0, 2) = -3.090169943749473;
            invA0(0, 3) = +0.999999999999999;
            invA0(1, 0) = -1.618033988749895;
            invA0(1, 1) = -0.000000000000000;
            invA0(1, 2) = +2.236067977499789;
            invA0(1, 3) = -0.618033988749895;
            invA0(2, 0) = +0.618033988749895;
            invA0(2, 1) = -2.236067977499789;
            invA0(2, 2) = -0.000000000000001;
            invA0(2, 3) = +1.618033988749895;
            invA0(3, 0) = -1.000000000000000;
            invA0(3, 1) = +3.090169943749475;
            invA0(3, 2) = -8.090169943749475;
            invA0(3, 3) = +6.000000000000000;
            /* --- b --- */
            b0(0) = +0.083333333333333;
            b0(1) = +0.416666666666667;
            b0(2) = +0.416666666666667;
            b0(3) = +0.083333333333333;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +0.276393202250021;
            c0(2) = +0.723606797749979;
            c0(3) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = +0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = +1.000000000000000;
            /* --- zeta --- */
            /* --- eta --- */
            eta(0) = +3.779019967010190;
            eta(1) = +2.220980032989806;
            /* --- beta --- */
            beta(0) = +1.380176524272838;
            beta(1) = +4.160391445506934;
            /* --- Q --- */
            Q0(0, 0) = +0.106073254812433;
            Q0(0, 1) = -0.856299374198130;
            Q0(0, 2) = -0.458004872738390;
            Q0(0, 3) = +0.213848972195978;
            Q0(1, 0) = -0.065022767538930;
            Q0(1, 1) = -0.038030973632481;
            Q0(1, 2) = +0.467333521122043;
            Q0(1, 3) = +0.880866087882725;
            Q0(2, 0) = -0.300594214556409;
            Q0(2, 1) = -0.510458447047536;
            Q0(2, 2) = +0.692545715189596;
            Q0(2, 3) = -0.411650002290251;
            Q0(3, 0) = -0.945602253852162;
            Q0(3, 1) = +0.068827324735684;
            Q0(3, 2) = -0.303663216336939;
            Q0(3, 3) = +0.094275277370812;
            /* --- R --- */
            R0(0, 0) = +3.779019967010190;
            R0(0, 1) = -5.310199803237381;
            R0(0, 2) = +5.458377049964716;
            R0(0, 3) = -4.426194015356730;
            R0(1, 0) = +0.358722328488004;
            R0(1, 1) = +3.779019967010190;
            R0(1, 2) = +1.662610499364627;
            R0(1, 3) = -7.019736960225162;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +2.220980032989806;
            R0(2, 3) = -8.208087489090918;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +2.108756394574483;
            R0(3, 3) = +2.220980032989806;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 2;
            break;

        // 5-stage 8th-order Lobatto IIIC
        case Type::LobIIIC8:
            s = 5;
            s_eff = 3;
            SizeData();
            /* --- A --- */
            A0(0, 0) = +0.050000000000000;
            A0(0, 1) = -0.116666666666667;
            A0(0, 2) = +0.133333333333333;
            A0(0, 3) = -0.116666666666667;
            A0(0, 4) = +0.050000000000000;
            A0(1, 0) = +0.050000000000000;
            A0(1, 1) = +0.161111111111111;
            A0(1, 2) = -0.069011541029643;
            A0(1, 3) = +0.052002165993115;
            A0(1, 4) = -0.021428571428571;
            A0(2, 0) = +0.050000000000000;
            A0(2, 1) = +0.281309183323043;
            A0(2, 2) = +0.202777777777778;
            A0(2, 3) = -0.052836961100821;
            A0(2, 4) = +0.018750000000000;
            A0(3, 0) = +0.050000000000000;
            A0(3, 1) = +0.270220056229107;
            A0(3, 2) = +0.367424239442342;
            A0(3, 3) = +0.161111111111111;
            A0(3, 4) = -0.021428571428571;
            A0(4, 0) = +0.050000000000000;
            A0(4, 1) = +0.272222222222222;
            A0(4, 2) = +0.355555555555556;
            A0(4, 3) = +0.272222222222222;
            A0(4, 4) = +0.050000000000000;
            /* --- inv(A) --- */
            invA0(0, 0) = +10.000000000000002;
            invA0(0, 1) = +13.513004977448478;
            invA0(0, 2) = -5.333333333333334;
            invA0(0, 3) = +2.820328355884856;
            invA0(0, 4) = -1.000000000000001;
            invA0(1, 0) = -2.481980506061966;
            invA0(1, 1) = +0.000000000000000;
            invA0(1, 2) = +3.491486243775877;
            invA0(1, 3) = -1.527525231651947;
            invA0(1, 4) = +0.518019493938034;
            invA0(2, 0) = +0.750000000000000;
            invA0(2, 1) = -2.673169155390907;
            invA0(2, 2) = +0.000000000000001;
            invA0(2, 3) = +2.673169155390906;
            invA0(2, 4) = -0.750000000000000;
            invA0(3, 0) = -0.518019493938035;
            invA0(3, 1) = +1.527525231651947;
            invA0(3, 2) = -3.491486243775877;
            invA0(3, 3) = -0.000000000000001;
            invA0(3, 4) = +2.481980506061966;
            invA0(4, 0) = +0.999999999999998;
            invA0(4, 1) = -2.820328355884851;
            invA0(4, 2) = +5.333333333333331;
            invA0(4, 3) = -13.513004977448482;
            invA0(4, 4) = +10.000000000000004;
            /* --- b --- */
            b0(0) = +0.050000000000000;
            b0(1) = +0.272222222222222;
            b0(2) = +0.355555555555556;
            b0(3) = +0.272222222222222;
            b0(4) = +0.050000000000000;
            /* --- c --- */
            c0(0) = +0.000000000000000;
            c0(1) = +0.172673164646011;
            c0(2) = +0.500000000000000;
            c0(3) = +0.827326835353989;
            c0(4) = +1.000000000000000;
            /* --- d --- */
            d0(0) = -0.000000000000000;
            d0(1) = -0.000000000000000;
            d0(2) = +0.000000000000000;
            d0(3) = -0.000000000000000;
            d0(4) = +1.000000000000000;
            /* --- zeta --- */
            zeta(0) = +5.277122810196474;
            /* --- eta --- */
            eta(0) = +2.664731518065850;
            eta(1) = +4.696707076835921;
            /* --- beta --- */
            beta(0) = +5.884022927615487;
            beta(1) = +2.908975454213622;
            /* --- Q --- */
            Q0(0, 0) = -0.571732147184072;
            Q0(0, 1) = -0.217020758096524;
            Q0(0, 2) = -0.319217568516840;
            Q0(0, 3) = +0.640115515555092;
            Q0(0, 4) = -0.338196116369386;
            Q0(1, 0) = +0.261426736644018;
            Q0(1, 1) = +0.145745772324781;
            Q0(1, 2) = +0.157982461885322;
            Q0(1, 3) = -0.130646891000507;
            Q0(1, 4) = -0.931872932768018;
            Q0(2, 0) = -0.124973465898830;
            Q0(2, 1) = -0.282976730324878;
            Q0(2, 2) = -0.710816244885500;
            Q0(2, 3) = -0.621569830676823;
            Q0(2, 4) = -0.112681029970164;
            Q0(3, 0) = -0.765238254685085;
            Q0(3, 1) = +0.325571308560971;
            Q0(3, 2) = +0.371091608018041;
            Q0(3, 3) = -0.410895349437400;
            Q0(3, 4) = -0.043240801076333;
            Q0(4, 0) = +0.059753323185469;
            Q0(4, 1) = +0.863474176534077;
            Q0(4, 2) = -0.479763452289052;
            Q0(4, 3) = +0.134162118426338;
            Q0(4, 4) = +0.051666649341120;
            /* --- R --- */
            R0(0, 0) = +2.664731518065850;
            R0(0, 1) = -3.395839948655504;
            R0(0, 2) = -5.334381337873263;
            R0(0, 3) = -5.795630734591422;
            R0(0, 4) = +9.491995902665616;
            R0(1, 0) = +10.195334979321485;
            R0(1, 1) = +2.664731518065850;
            R0(1, 2) = -13.338533879421322;
            R0(1, 3) = +2.816806937357789;
            R0(1, 4) = +5.035627803894975;
            R0(2, 0) = +0.000000000000000;
            R0(2, 1) = +0.000000000000000;
            R0(2, 2) = +5.277122810196474;
            R0(2, 3) = -3.815356369998933;
            R0(2, 4) = +1.959142180222903;
            R0(3, 0) = +0.000000000000000;
            R0(3, 1) = +0.000000000000000;
            R0(3, 2) = +0.000000000000000;
            R0(3, 3) = +4.696707076835921;
            R0(3, 4) = -10.644178015207251;
            R0(4, 0) = +0.000000000000000;
            R0(4, 1) = +0.000000000000000;
            R0(4, 2) = +0.000000000000000;
            R0(4, 3) = +0.795001566220291;
            R0(4, 4) = +4.696707076835921;
            /* --- R block sizes --- */
            R0_block_sizes[0] = 2;
            R0_block_sizes[1] = 1;
            R0_block_sizes[2] = 2;
            break;

        default:
            mfem_error("RKData:: Invalid Runge Kutta type.\n");
    }
}
