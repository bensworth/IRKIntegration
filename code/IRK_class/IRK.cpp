#include "IRK.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip> 
#include <cmath> 


// TODO:
//  --> FIX ConstructRHS() TO USE CURRENT X AND TIME! IS THIS SAME AS EXPLICITMULT?
//  - Apply mass matrix before preconditioning (see TODO1)
// --> OR, CAN WE NOT APPLY MASS MATRIX FOR ONE TERM IN POLYNOMIAL?
//         so it actually looks like MP(M^{-1}L)?
//  - Add stiffly accurate option for Adjugate
//  - Directly construct Butcher Matrix/Vector instead of allocate using new
//  - Add other Butcher tableauxs



IRK::IRK(IRKOperator *S, IRK::Type RK_ID, MPI_Comm globComm);
        : m_S(S), m_CharPolyPrec(*S), m_CharPolyOps(), m_krylov(NULL),
        m_z(NULL), m_y(NULL), m_w(NULL)
{
    m_RK_ID = static_cast<int>(RK_ID);

    // Get proc IDs
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProcess);
    
    /* Set up basic information */
    SetButcherCoeffs();
    SetXCoeffs();
    
    // Set initial condition
    m_S->SetU0();
    
    // Initialize other vectors based on size of u
    int dimU = m_S->Height();
    m_z = new Vector(dimU);
    m_y = new Vector(dimU);
    m_w = new Vector(dimU);
    
    
    /* Set spatial discretization once at the start since it's assumed time-independent */
    m_S->SetL(0.0);
    m_S->SetM();
    
    /* --- Construct object for every factor in char. poly --- */
    m_CharPolyOps.SetSize(m_zetaSize + m_etaSize);
    // Linear factors. I.e., those of type 1
    int count = 0;
    for (int i = 0; i < m_zetaSize; i++) {
        m_CharPolyOps[count] = new CharPolyOp(dt, m_zeta(i), *m_S);
        count++;
    }
    // Quadratic factors. I.e., those of type 2
    for (int i = 0; i < m_etaSize; i++) {
        m_CharPolyOps[count] = new CharPolyOp(dt, m_eta(i), m_beta(i), *m_S);
        count++;
    }   
}

IRK::~IRK() {
    if (m_z) delete m_z;
    if (m_y) delete m_y;
    if (m_w) delete m_w;
    
    for (int i = 0; i < m_CharPolyOps.Size(); i++) {
        delete m_CharPolyOps[i];
    }
}

void IRK::SetSolve(IRK::Solve solveID, double reltol, int maxiter,
                   double abstol, int kdim, int printlevel)
{
    m_solveID = static_cast<int>(solveID);
    // CG
    if (m_solveID == 0) {
        m_krylov = new CGSolver(m_comm);
        m_krylov->SetRelTol(reltol);
        m_krylov->SetAbsTol(abstol);
        m_krylov->SetMaxIter(maxiter);
        m_krylov->SetPrintLevel(printlevel);
    }
    // MINRES
    else if (m_solveID == 1) {
        m_krylov = new MINRESSolver(m_comm);
        m_krylov->SetRelTol(reltol);
        m_krylov->SetAbsTol(abstol);
        m_krylov->SetMaxIter(maxiter);
        m_krylov->SetPrintLevel(printlevel);
    }
    // GMRES
    else if (m_solveID == 2) {
        m_krylov = new GMRESSolver(m_comm);
        m_krylov->SetRelTol(reltol);
        m_krylov->SetAbsTol(abstol);
        m_krylov->SetMaxIter(maxiter);
        m_krylov->SetPrintLevel(printlevel);
        static_cast<GMRESSolver*>(m_krylov)->SetKDim(kdim);
    }
    // BiCGStab
    else if (m_solveID == 3) { 
        m_krylov = new BiCGSTABSolver(m_comm);
        m_krylov->SetRelTol(reltol);
        m_krylov->SetAbsTol(abstol);
        m_krylov->SetMaxIter(maxiter);
        m_krylov->SetPrintLevel(printlevel);
    }
    // GGMRES
    else if (m_solveID == 4) {
        m_krylov = new FGMRESSolver(m_comm);
        m_krylov->SetRelTol(reltol);
        m_krylov->SetAbsTol(abstol);
        m_krylov->SetMaxIter(maxiter);
        m_krylov->SetPrintLevel(printlevel);
        static_cast<FGMRESSolver*>(m_krylov)->SetKDim(kdim);
    }
    else {
        MFEM_ERROR("Invalid solve type.\n");   
    }

    m_krylov->iterative_mode = false;
}


void IRK::Step(Vector &x, double &t, double &dt)
{
    // Construct default Krylov solver if pointer NULL
    if (!m_krylov) SetSolve();

    *m_y = 0.0;     // TODO : what does this do?
    ConstructRHS(x, t, dt); /* Set m_z */
    
        // TODO1 : Apply mass matrix to right-hand side here!!

    /* Sequentially invert factors in characteristic polynomial */
    for (int i = 0; i < m_zetaSize + m_etaSize; i++) {
        if (m_rank == 0) {
            std::cout << "System " << i << " of " << m_zetaSize + m_etaSize-1 <<
            ";\t type = " << m_CharPolyOps[i]->Type() << "\n \t";
        }
        
        // Ensure that correct time step is used in factored polynomial
        m_CharPolyOps[i]->Setdt(dt);

        // Set operator and preconditioner for ith polynomial term
        m_CharPolyPrec.SetType(m_CharPolyOps[i]->Type())
        m_S.SetSystem(i, t, dt, m_CharPolyOps[i]->Gamma(),
                      m_CharPolyOps[i]->Type());
        m_krylov -> SetPreconditioner(m_CharPolyPrec);
        m_krylov -> SetOperator(m_CharPolyOps[i]);

        // Use preconditioned Krylov to invert this term in polynomial 
        m_krylov -> Mult(*m_z, *m_y); // y <- char_poly_factor(i)^-1 * z
        *m_z = *m_y;
    }

    // Update solution vector with weighted sum of stage vectors        
    (m_S->m_u)->Add(dt, *m_y);
    t += dt;
}

void IRK::Run(Vector &x, double &t, double &dt, double tf) 
{
    // Construct default Krylov solver if pointer NULL
    if (!m_krylov) SetSolve();

    *m_y = 0.0;     // TODO : what does this do?

    /* Main time-stepping loop */
    // for (int step = 0; step < m_nt; step++) {
    int step = 0;
    int numsteps = int((tf-t)/dt);
    while (t < tf) {
        step++;
        if (m_rank == 0) std::cout << "RK time-step " << step << " of " << numsteps << '\n';

        ConstructRHS(x, t, dt);
        
        // TODO1 : Apply mass matrix to right-hand side here!!

        /* Sequentially invert factors in characteristic polynomial */
        for (int i = 0; i < (m_zetaSize + m_etaSize); i++) {
            if (m_rank == 0) {
                std::cout << "System " << i << " of " << m_zetaSize + m_etaSize - 1 <<
                ";\t type = " << m_CharPolyOps[i]->Type() << "\n \t";
            }

            // Ensure that correct time step is used in factored polynomial
            m_CharPolyOps[i]->Setdt(dt);

            // Set operator and preconditioner for ith polynomial term
            m_CharPolyPrec.SetType(m_CharPolyOps[i]->Type())
            m_S.SetSystem(i, t, dt, m_CharPolyOps[i]->Gamma(),
                          m_CharPolyOps[i]->Type());
            m_krylov -> SetPreconditioner(m_CharPolyPrec);
            m_krylov -> SetOperator(m_CharPolyOps[i]);

            // Use preconditioned Krylov to invert this term in polynomial 
            m_krylov -> Mult(*m_z, *m_y); // y <- char_poly_factor(i)^-1 * z
            *m_z = *m_y;
        }
    
        // Update solution vector with weighted sum of stage vectors        
        (m_S->m_u)->Add(dt, *m_y);
        t += dt;
    }
}

/* Form the RHS of the linear system for IRK integration at time t, m_z */
void IRK::ConstructRHS(const Vector &x, double t, double dt) {
    *m_z = 0.0; /* z <- 0 */
    *m_w = 0.0; /* w <- 0 */
    
    Vector temp(x), f(x);
    m_S -> ExplicitMult(x, temp); // temp <- L*u
    
    for (int i = 0; i < m_s; i++) {
        m_S->SetG(t + dt*m_c0(i)); /* Set g at time t + dt*c[i] */
        add(*(m_S -> m_g), temp, f); // f <- L*u + g
        m_S -> PolynomialMult(m_XCoeffs[i], dt, f, *m_w); /* w <- X_i(dt*M^{-1}*L) * f */
        *m_z += *m_w;
        *m_w = 0.0;
    }
}

/* Set constants from Butcher tables and associated parameters */
// enum Type { 
//   SDIRK2 = 02, SDIRK3 = 03, SDIRK4 = 04,
//   Gauss4 = 14, Gauss6 = 16, Gauss8 = 18,
//   RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27,
//   LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36
// };
void IRK::SetButcherCoeffs() {
    
    /* Data arrays for filling Butcher arrays; NOTE: DenseMatrix is stored in column major ordering */
    double * A;
    double * invA;
    double * b;
    double * c;
    double * d;
    double * zeta;
    double * beta;
    double * eta;

    // First-order L-stable SDIRK (backward Euler)
    if (m_RK_ID == 1) {
        /* --- Dimensions --- */
        m_s        = 1;
        m_zetaSize = 1;
        m_etaSize  = 0;
        SizeButcherArrays(A, invA, b, c, d, zeta, eta, beta); /* Set data arrays to correct dimensions */
        /* ---------------- */

        /* --- Tableaux constants --- */
        /* --- A --- */
        Set(A, 0, 0, +1.000000000000000);
        /* --- inv(A) --- */
        Set(invA, 0, 0, +1.000000000000000);
        /* --- b --- */
        Set(b, 0, +1.000000000000000);
        /* --- c --- */
        Set(c, 0, +1.000000000000000);
        /* --- d --- */
        Set(d, 0, +1.000000000000000);
        /* --- zeta --- */
        Set(zeta, 0, +1.000000000000000);
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */
        
    // Second-order L-stable SDIRK
    }
    else if (m_RK_ID == 2) {
        /* --- Dimensions --- */
        m_s        = 2;
        m_zetaSize = 2;
        m_etaSize  = 0;
        SizeButcherArrays(A, invA, b, c, d, zeta, eta, beta); /* Set data arrays to correct dimensions */
        /* ---------------- */

        /* --- Tableaux constants --- */
        /* --- A --- */
        Set(A, 0, 0, +0.292893218813453);
        Set(A, 0, 1, +0.000000000000000);
        Set(A, 1, 0, +0.414213562373095);
        Set(A, 1, 1, +0.292893218813453);
        /* --- inv(A) --- */
        Set(invA, 0, 0, +3.414213562373094);
        Set(invA, 0, 1, +0.000000000000000);
        Set(invA, 1, 0, -4.828427124746186);
        Set(invA, 1, 1, +3.414213562373094);
        /* --- b --- */
        Set(b, 0, +0.500000000000000);
        Set(b, 1, +0.500000000000000);
        /* --- c --- */
        Set(c, 0, +0.292893218813453);
        Set(c, 1, +0.707106781186547);
        /* --- d --- */
        Set(d, 0, -0.707106781186546);
        Set(d, 1, +1.707106781186547);
        /* --- zeta --- */
        Set(zeta, 0, +3.414213562373094);
        Set(zeta, 1, +3.414213562373094);
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */ 

    // Third-order L-stable SDIRK
    }
    else if (m_RK_ID == 3) {
        /* --- Dimensions --- */
        m_s        = 3;
        m_zetaSize = 3;
        m_etaSize  = 0;
        SizeButcherArrays(A, invA, b, c, d, zeta, eta, beta); /* Set data arrays to correct dimensions */
        /* ---------------- */

        /* --- Tableaux constants --- */
        /* --- A --- */
        Set(A, 0, 0, +0.435866521508459);
        Set(A, 0, 1, +0.000000000000000);
        Set(A, 0, 2, +0.000000000000000);
        Set(A, 1, 0, +0.282066739245771);
        Set(A, 1, 1, +0.435866521508459);
        Set(A, 1, 2, +0.000000000000000);
        Set(A, 2, 0, +1.208496649176010);
        Set(A, 2, 1, -0.644363170684469);
        Set(A, 2, 2, +0.435866521508459);
        /* --- inv(A) --- */
        Set(invA, 0, 0, +2.294280360279042);
        Set(invA, 0, 1, -0.000000000000000);
        Set(invA, 0, 2, -0.000000000000000);
        Set(invA, 1, 0, -1.484721005641544);
        Set(invA, 1, 1, +2.294280360279042);
        Set(invA, 1, 2, +0.000000000000000);
        Set(invA, 2, 0, -8.556127801552645);
        Set(invA, 2, 1, +3.391748836942547);
        Set(invA, 2, 2, +2.294280360279042);
        /* --- b --- */
        Set(b, 0, +1.208496649176010);
        Set(b, 1, -0.644363170684469);
        Set(b, 2, +0.435866521508459);
        /* --- c --- */
        Set(c, 0, +0.435866521508459);
        Set(c, 1, +0.717933260754229);
        Set(c, 2, +1.000000000000000);
        /* --- d --- */
        Set(d, 0, +0.000000000000000);
        Set(d, 1, +0.000000000000000);
        Set(d, 2, +1.000000000000000);
        /* --- zeta --- */
        Set(zeta, 0, +2.294280360279042);
        Set(zeta, 1, +2.294280360279042);
        Set(zeta, 2, +2.294280360279042);
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */
    }
    // Fourth-order L-stable SDIRK
    else if (m_RK_ID == 4) {

        // TODO

    }
    // 2-stage 4th-order Gauss--Legendre
    else if (m_RK_ID == 14) {
        /* --- Dimensions --- */
        m_s        = 2;
        m_zetaSize = 0;
        m_etaSize  = 1;
        SizeButcherArrays(A, invA, b, c, d, zeta, eta, beta); /* Set data arrays to correct dimensions */
        /* ---------------- */

        /* --- Tableaux constants --- */
        /* --- A --- */
        Set(A, 0, 0, +0.250000000000000);
        Set(A, 0, 1, -0.038675134594813);
        Set(A, 1, 0, +0.538675134594813);
        Set(A, 1, 1, +0.250000000000000);
        /* --- inv(A) --- */
        Set(invA, 0, 0, +3.000000000000000);
        Set(invA, 0, 1, +0.464101615137754);
        Set(invA, 1, 0, -6.464101615137755);
        Set(invA, 1, 1, +3.000000000000000);
        /* --- b --- */
        Set(b, 0, +0.500000000000000);
        Set(b, 1, +0.500000000000000);
        /* --- c --- */
        Set(c, 0, +0.211324865405187);
        Set(c, 1, +0.788675134594813);
        /* --- d --- */
        Set(d, 0, -1.732050807568877);
        Set(d, 1, +1.732050807568877);
        /* --- zeta --- */
        /* --- eta --- */
        Set(eta, 0, +3.000000000000000);
        /* --- beta --- */
        Set(beta, 0, +1.732050807568877);
        /* -------------------------- */
    
    }
    // 3-stage 6th-order Gauss--Legendre
    else if (m_RK_ID == 16) {
        /* --- Dimensions --- */
        m_s        = 3;
        m_zetaSize = 1;
        m_etaSize  = 1;
        SizeButcherArrays(A, invA, b, c, d, zeta, eta, beta); /* Set data arrays to correct dimensions */
        /* ---------------- */

        /* --- Tableaux constants --- */
        /* --- A --- */
        Set(A, 0, 0, +0.138888888888889);
        Set(A, 0, 1, -0.035976667524939);
        Set(A, 0, 2, +0.009789444015308);
        Set(A, 1, 0, +0.300263194980865);
        Set(A, 1, 1, +0.222222222222222);
        Set(A, 1, 2, -0.022485417203087);
        Set(A, 2, 0, +0.267988333762469);
        Set(A, 2, 1, +0.480421111969383);
        Set(A, 2, 2, +0.138888888888889);
        /* --- inv(A) --- */
        Set(invA, 0, 0, +4.999999999999997);
        Set(invA, 0, 1, +1.163977794943223);
        Set(invA, 0, 2, -0.163977794943222);
        Set(invA, 1, 0, -5.727486121839513);
        Set(invA, 1, 1, +2.000000000000000);
        Set(invA, 1, 2, +0.727486121839514);
        Set(invA, 2, 0, +10.163977794943225);
        Set(invA, 2, 1, -9.163977794943223);
        Set(invA, 2, 2, +5.000000000000001);
        /* --- b --- */
        Set(b, 0, +0.277777777777778);
        Set(b, 1, +0.444444444444444);
        Set(b, 2, +0.277777777777778);
        /* --- c --- */
        Set(c, 0, +0.112701665379258);
        Set(c, 1, +0.500000000000000);
        Set(c, 2, +0.887298334620742);
        /* --- d --- */
        Set(d, 0, +1.666666666666668);
        Set(d, 1, -1.333333333333334);
        Set(d, 2, +1.666666666666667);
        /* --- zeta --- */
        Set(zeta, 0, +4.644370709252176);
        /* --- eta --- */
        Set(eta, 0, +3.677814645373916);
        /* --- beta --- */
        Set(beta, 0, +3.508761919567443);
        /* -------------------------- */    
    }    
    // 4-stage 8th-order Gauss--Legendre
    else if (m_RK_ID == 18) {

        // TODO

    }
    // 2-stage 3rd-order Radau IIA
    else if (m_RK_ID == 23) {

        // TODO

    }
    // 3-stage 5th-order Radau IIA
    else if (m_RK_ID == 25) {

        // TODO

    }
    // 4-stage 7th-order Radau IIA
    else if (m_RK_ID == 27) {

        // TODO

    }
    // 2-stage 2nd-order Lobatto IIIC
    else if (m_RK_ID == 23) {

        // TODO

    }
    // 3-stage 4th-order Lobatto IIIC
    else if (m_RK_ID == 25) {

        // TODO

    }
    // 4-stage 6th-order Lobatto IIIC
    else if (m_RK_ID == 27) {

        // TODO

    }
    else {
        MFEM_ERROR("Invalid Runge Kutta type.\n");
    }
    
    /* Insert data into arrays TODO: WHat about  memory??? This is still ownded by the arrays... */
    m_A0    = DenseMatrix(A, m_s, m_s);
    m_invA0 = DenseMatrix(invA, m_s, m_s);
    m_b0    = Vector(b, m_s);
    m_c0    = Vector(c, m_s);
    m_d0    = Vector(d, m_s);
    m_zeta  = Vector(zeta, m_zetaSize);
    m_eta   = Vector(eta, m_etaSize);
    m_beta  = Vector(beta, m_etaSize);
    
    // m_invA0.Print();
    // m_A0.Print();
}

/* Initialize all Butcher arrays with correct sizes; all entries initialized to 
    zero so that only non-zero entries need be inserted */
void IRK::SizeButcherArrays(double * &A, double * &invA, double * &b, double * &c, double * &d, 
                            double * &zeta, double * &eta, double * &beta) {
    A    = new double[m_s*m_s]();
    invA = new double[m_s*m_s]();
    b    = new double[m_s]();
    c    = new double[m_s]();
    d    = new double[m_s]();
    zeta = new double[m_zetaSize]();
    beta = new double[m_etaSize]();
    eta  = new double[m_etaSize]();
}


/* Given the precomputed vector m_d0, and the matrix m_invA0, compute and 
    store coefficients of the polynomials X_j */
void IRK::SetXCoeffs() {
    
    // Data array for coefficients; elements in X are set in column major ordering
    double * X = new double[m_s*m_s];
    
    /* Shallow copy matrix and vector so the following formulae appear shorter */
    DenseMatrix B = m_invA0;
    Vector      d = m_d0;
    
    if (m_s == 1) {
        /* s=1: Coefficients for polynomial X_{1}(z) */
        Set(X, 0, 0, +d(0));
        
    }
    else if (m_s == 2) {
        /* s=2: Coefficients for polynomial X_{1}(z) */
        Set(X, 0, 0, +B(1,1)*d(0)-B(1,0)*d(1));
        Set(X, 1, 0, -d(0));

        /* s=2: Coefficients for polynomial X_{2}(z) */
        Set(X, 0, 1, +B(0,0)*d(1)-B(0,1)*d(0));
        Set(X, 1, 1, -d(1));

    }
    else if (m_s == 3) {    
        /* s=3: Coefficients for polynomial X_{1}(z) */
        Set(X, 0, 0, +d(2)*(B(1,0)*B(2,1)-B(1,1)*B(2,0))-d(1)*(B(1,0)*B(2,2)-B(1,2)*B(2,0))+d(0)*(B(1,1)*B(2,2)-B(1,2)*B(2,1)));
        Set(X, 1, 0, +B(1,0)*d(1)+B(2,0)*d(2)-d(0)*(B(1,1)+B(2,2)));
        Set(X, 2, 0, +d(0));

        /* s=3: Coefficients for polynomial X_{2}(z) */
        Set(X, 0, 1, +d(1)*(B(0,0)*B(2,2)-B(0,2)*B(2,0))-d(2)*(B(0,0)*B(2,1)-B(0,1)*B(2,0))-d(0)*(B(0,1)*B(2,2)-B(0,2)*B(2,1)));
        Set(X, 1, 1, +B(0,1)*d(0)+B(2,1)*d(2)-d(1)*(B(0,0)+B(2,2)));
        Set(X, 2, 1, +d(1));

        /* s=3: Coefficients for polynomial X_{3}(z) */
        Set(X, 0, 2, +d(2)*(B(0,0)*B(1,1)-B(0,1)*B(1,0))-d(1)*(B(0,0)*B(1,2)-B(0,2)*B(1,0))+d(0)*(B(0,1)*B(1,2)-B(0,2)*B(1,1)));
        Set(X, 1, 2, +B(0,2)*d(0)+B(1,2)*d(1)-d(2)*(B(0,0)+B(1,1)));
        Set(X, 2, 2, +d(2));
    
    }
    else {
        cout << "WARNING: Coefficients of polynomials {X_j} not implemented for s = " << m_s << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    // Insert data into m_XCoeffs; m_XCoeffs[i] gives the set of coefficients for X_i
    m_XCoeffs = new Vector[m_s];
    for (int i = 0; i < m_s; i++) {
        m_XCoeffs[i] = Vector(X + i*m_s, m_s); // Read in ith "column" of X
    }
}




/* Print data to file that allows one to extract the 
    relevant data for plotting, etc. from the saved solution. Pass
    in dictionary with information also to be saved to file that isn't a member 
    variable (e.g. space disc info)
*/
void IRK::SaveSolInfo(string filename, map<string, string> additionalInfo) 
{
    ofstream solinfo;
    solinfo.open(filename);
    solinfo << scientific; // This means parameters will be printed with enough significant digits
    solinfo << "nt " << m_nt << "\n";
    solinfo << "dt " << m_dt << "\n";
    
    // Time-discretization-specific information
    solinfo << "timeDisc " << m_RK_ID << "\n";
    
    // Print out contents from additionalInfo to file too
    map<string, string>::iterator it;
    for (it=additionalInfo.begin(); it!=additionalInfo.end(); it++) {
        solinfo << it->first << " " << it->second << "\n";
    }
    solinfo.close();
}
