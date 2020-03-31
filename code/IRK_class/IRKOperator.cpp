#include "IRKOperator.hpp"

#if 0 Template preconditioner for AIR

/* Constructor for preconditioner */
CharPolyPrecon::CharPolyPrecon(MPI_Comm comm, double gamma, double dt, int type, SpatialDiscretization &S) 
    : Solver(S.height, false), m_type(type), m_S(S), m_precon(NULL), m_solver(NULL) {
    
    /* Build J, the operator to be inverted */
    m_J = new HypreParMatrix( *(S.m_L) ); // J <- deepcopy(L)
    *m_J *= -dt; // J <- -dt*J
    m_J->Add(gamma, *(S.m_M)); // J <- J + gamma*M

    /* Build AMG preconditioner for J */
    HypreBoomerAMG * amg = new HypreBoomerAMG(*m_J);

    //AIR_parameters AIR = {1.5, "", "FA", 100, 10, 10, 0.1, 0.05, 0.0, 1e-5};
    // AMG_parameters AIR = {1.5, "", "FA", 0.1, 0.01, 0.0, 100, 10, 0.e-4, 6}; // w/ on-proc relaxation
    AMG_parameters AIR = {1.5, "", "FFC", 0.1, 0.01, 0.0, 100, 0, 0.e-4, 6};    // w/ Jacobi relaxation
    // 
    amg->SetLAIROptions(AIR.distance, AIR.prerelax, AIR.postrelax,
                           AIR.strength_tolC, AIR.strength_tolR, 
                           AIR.filter_tolR, AIR.interp_type, 
                           AIR.relax_type, AIR.filterA_tol,
                           AIR.coarsening);

    // Krylov preconditioner is a single AMG iteration
    amg->SetPrintLevel(0);
    amg->SetTol(0.0);
    amg->SetMaxIter(1);
    m_precon = amg;
}

void IRKOperator::SetMSolver() {
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(m_M);
    M_solver.iterative_mode = false;
    M_solver.SetRelTol(1e-9);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
}

#endif



#if 0   // FOR DERIVED CLASS TO IMPLEMENT
void IRKOperator::ExplicitMult(const Vector &x, Vector &y) {
    // if (m_M_exists) {
    //     m_L->Mult(x, m_z); // z <- L * x
    //     //M_solver->Mult(z, y); // y <- M^-1 * z
    // Vector z(y.Size()); // An auxillary vector
    //     y = m_z; 
    // } else {
    //     m_L->Mult(x, y); // y <- L * x
    // }
    m_L->Mult(x, y);
}
#endif

/* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients.
Coefficients must be provided for all monomial terms (even if they're 0) and 
in increasing order (from 0th to nth) */
void IRKOperator::PolynomialMult(Vector coefficients, double alpha,
                                    const Vector &x, Vector &y) {
    int n = coefficients.Size() - 1;
    y.Set(coefficients[n], x); // y <- coefficients[n]*x
    Vector z(y.Size()); // An auxillary vector
    for (int ell = n-1; ell >= 0; ell--) {
        this->ExplicitMult(y, z); // z <- M^{-1}*L*y       
        add(coefficients[ell], x, alpha, z, y); // y <- coefficients[ell]*x + alpha*z
    } 
}



