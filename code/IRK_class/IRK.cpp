#include "IRK.hpp"

#include <iostream>
#include <fstream>
#include <map>
#include <iomanip> 
#include <cmath> 


IRK::IRK(MPI_Comm globComm, int RK_ID, SpatialDiscretization * S,
        double dt, int nt) 
        : m_RK_ID{RK_ID}, m_S(S), m_z(NULL), m_y(NULL), m_w(NULL),
        m_dt{dt}, m_nt(nt),
        m_CharPolyFactorOperators()
        {
    
    /* Set up basic information */
    SetButcherCoeffs();
    SetXCoeffs();
    
    // Set initial condition
    m_S->SetU0();
    
    // Initialize other vectors based on size of u
    int dimU = m_S->m_u->Size();
    m_z = new Vector(dimU);
    m_y = new Vector(dimU);
    m_w = new Vector(dimU);
    
    
    /* Set spatial discretization once at the start since it's assumed time-independent */
    m_S->SetL(0.0);
    
    
    /* --- Construct object for every factor in the char. poly --- */
    m_CharPolyFactorOperators.SetSize(m_zetaSize + m_etaSize);
    /* Real-valued, linear factors */
    int count = 0;
    for (int i = 0; i < m_zetaSize; i++) {
        m_CharPolyFactorOperators[count] = new CharPolyFactorOperator(dt, m_zeta(i), 0.0, 0.0, *m_S);
        count++;
    }
    /* Complex conjugate pair, quadratic factors; pass zeta == 0.0! */
    for (int i = 0; i < m_etaSize; i++) {
        m_CharPolyFactorOperators[count] = new CharPolyFactorOperator(dt, 0.0, m_eta(i), m_beta(i), *m_S);
        count++;
    }   
}



IRK::~IRK() {
    if (m_z) delete m_z;
    if (m_y) delete m_y;
    if (m_w) delete m_w;
    
    for (int i = 0; i < m_CharPolyFactorOperators.Size(); i++) {
        delete m_CharPolyFactorOperators[i];
    }
}


/* Constructor for factors in char polynomial */
CharPolyFactorOperator::CharPolyFactorOperator(double dt, double zeta, 
                                                     double eta, double beta, 
                                                     SpatialDiscretization &S) 
    : Operator(S.m_spatialDOFs), m_dt{dt}, m_zeta{zeta}, m_eta{eta}, m_beta{beta}, m_S{S}
{
    // Set coefficients that define the operator as a polynomial in L; zeta is 
    // assumed to be passed as zero for conjugate pair factors
    if (zeta != 0.0) {
        m_conjPair = false;
        m_c = Vector(2);
        m_c(0) = m_zeta;
        m_c(1) = -1.0;
    } else {
        m_conjPair = true;
        m_c = Vector(3);
        m_c(0) = m_eta*m_eta + m_beta*m_beta;
        m_c(1) = -2.0*m_eta;
        m_c(2) = 1.0;        
    }
        
    
    // Set up preconditioner TODO
    m_precon = NULL; 
    
    // TODO : Add check if L is SPD, if it is, setup CG solver instead
    
    // Set up GMRES solver
    GMRESSolver * m_gmres = new GMRESSolver();
    m_gmres->iterative_mode = false;
    m_gmres->SetRelTol(1e-12);
    m_gmres->SetAbsTol(1e-12);
    m_gmres->SetMaxIter(300);
    m_gmres->SetPrintLevel(2);
    //m_gmres->SetPreconditioner(TODO);
    m_gmres->SetOperator(*this);
    m_solver = m_gmres;
}

/* Destructor */
CharPolyFactorOperator::~CharPolyFactorOperator() {
    delete m_solver;
}


/* Primary function */
void IRK::TimeStep() {
    /* Time at the start of each time step */
    double t = 0.0;
    
    *m_y = 2.0;
    *m_z = 1.0;
    
    /* Main time-stepping loop */
    for (int step = 0; step < m_nt; step++) {
        std::cout << "RK time-step " << step+1 << " of " << m_nt << '\n';
        
        SetRHSLinearSystem(t); /* Set z */
        
        /* Sequentially invert factors in characteristic polynomial */
        for (int i = 0; i < m_zetaSize + m_etaSize; i++) {
            m_CharPolyFactorOperators[i]->m_solver->Mult(*m_z, *m_y); // y <- char_poly_factor(i)^-1 * z
        }
        
        // Update solution vector at previous time with weighted sum of stage vectors
        *(m_S->m_u) += *m_y;
        
        t += m_dt; // Time the current solution is evaluated at
    }
    
}


/* Form the RHS of the linear system at time t, m_z */
void IRK::SetRHSLinearSystem(double t) {
    *m_z = 0.0; /* z <- 0 */
    *m_w = 0.0; /* w <- 0 */
    
    for (int i = 0; i < m_s; i++) {
        m_S->SetG(t + m_dt*m_c0(i)); /* Set g at time t + dt*c[i] */
        m_S->SolDepPolyMult(m_XCoeffs[i], m_dt, *(m_S->m_g), *m_w); /* w <- P(dt*L) * g */
        *m_z += *m_w;
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


void IRK::Test() {
    
}

/* Set constants from Butcher tables and associated parameters 

TODO: Will need to have a more systematic approach for organising these...
*/
void IRK::SetButcherCoeffs() {
    
    /* Data arrays for filling Butcher arrays */
    double * A;
    double * invA;
    double * b;
    double * c;
    double * d;
    double * zeta;
    double * beta;
    double * eta;

    // Backward Euler
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
        
    // implicit 4th-order method, Hammer & Hollingsworth (A-stable)
    // note: coincides with s=2-stage, p=2s-order Gauss method
    } else if (m_RK_ID == 2) {
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
    
    /* A 6th-order Gauss--Legendre method */
    } else if (m_RK_ID == 3) {
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
        Set(A, 2, 0, +0.845995670075437);
        Set(A, 2, 1, +0.480421111969383);
        Set(A, 2, 2, +0.138888888888889);
        /* --- inv(A) --- */
        Set(invA, 0, 0, +5.523521392865545);
        Set(invA, 0, 1, +1.285851250237871);
        Set(invA, 0, 2, -0.181146971664762);
        Set(invA, 1, 0, -8.050084421979170);
        Set(invA, 1, 1, +1.459309430412914);
        Set(invA, 1, 2, +0.803657031398669);
        Set(invA, 2, 0, -5.799201641846299);
        Set(invA, 2, 1, -12.880135075166677);
        Set(invA, 2, 2, +5.523521392865548);
        /* --- b --- */
        Set(b, 0, +0.277777777777778);
        Set(b, 1, +0.444444444444444);
        Set(b, 2, +0.277777777777778);
        /* --- c --- */
        Set(c, 0, +0.112701665379258);
        Set(c, 1, +0.500000000000000);
        Set(c, 2, +0.887298334620742);
        /* --- d --- */
        Set(d, 0, -3.654393145596507);
        Set(d, 1, -2.572052426741152);
        Set(d, 2, +1.841173797621849);
        /* --- zeta --- */
        Set(zeta, 0, +4.249667103686992);
        /* --- eta --- */
        Set(eta, 0, +4.128342556228503);
        /* --- beta --- */
        Set(beta, 0, +3.761765721693538);
        /* -------------------------- */     
    }    
    
    
    /* Insert data into arrays TODO: WHat about  memory??? This is still ownded by the arrays... */
    m_A0    = DenseMatrix(A, m_s, m_s);
    m_invA0 = DenseMatrix(invA, m_s, m_s);
    m_b0    = Vector(b, m_s);
    m_c0    = Vector(c, m_s);
    m_d0    = Vector(d, m_s);
    m_zeta  = Vector(zeta, m_zetaSize);
    m_beta  = Vector(eta, m_etaSize);
    m_eta   = Vector(beta, m_etaSize);
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
    
    // Data array
    double * X = new double[m_s*m_s];
    
    if (m_s == 1) {
        /* s=1: Coefficients for polynomial X_{1}(z) */
        Set(X, 0, 0, +m_d0(0));
    } else if (m_s == 2) {
        /* s=2: Coefficients for polynomial X_{1}(z) */
        Set(X, 0, 0, +m_invA0(1,1)*m_d0(0)-m_invA0(1,0)*m_d0(1));
        Set(X, 0, 1, -m_d0(0));

        /* s=2: Coefficients for polynomial X_{2}(z) */
        Set(X, 1, 0, +m_invA0(0,0)*m_d0(1)-m_invA0(0,1)*m_d0(0));
        Set(X, 1, 1, -m_d0(1));
    
    } else if (m_s == 3) {
        /* s=3: Coefficients for polynomial X_{1}(z) */
        Set(X, 0, 0, +m_d0(2)*(m_invA0(1,0)*m_invA0(2,1)-m_invA0(1,1)*m_invA0(2,0))-m_d0(1)*(m_invA0(1,0)*m_invA0(2,2)-m_invA0(1,2)*m_invA0(2,0))+m_d0(0)*(m_invA0(1,1)*m_invA0(2,2)-m_invA0(1,2)*m_invA0(2,1)));
        Set(X, 0, 1, +m_invA0(1,0)*m_d0(1)+m_invA0(2,0)*m_d0(2)-m_d0(0)*(m_invA0(1,1)+m_invA0(2,2)));
        Set(X, 0, 2, +m_d0(0));

        /* s=3: Coefficients for polynomial X_{2}(z) */
        Set(X, 1, 0, +m_d0(1)*(m_invA0(0,0)*m_invA0(2,2)-m_invA0(0,2)*m_invA0(2,0))-m_d0(2)*(m_invA0(0,0)*m_invA0(2,1)-m_invA0(0,1)*m_invA0(2,0))-m_d0(0)*(m_invA0(0,1)*m_invA0(2,2)-m_invA0(0,2)*m_invA0(2,1)));
        Set(X, 1, 1, +m_invA0(0,1)*m_d0(0)+m_invA0(2,1)*m_d0(2)-m_d0(1)*(m_invA0(0,0)+m_invA0(2,2)));
        Set(X, 1, 2, +m_d0(1));

        /* s=3: Coefficients for polynomial X_{3}(z) */
        Set(X, 2, 0, +m_d0(2)*(m_invA0(0,0)*m_invA0(1,1)-m_invA0(0,1)*m_invA0(1,0))-m_d0(1)*(m_invA0(0,0)*m_invA0(1,2)-m_invA0(0,2)*m_invA0(1,0))+m_d0(0)*(m_invA0(0,1)*m_invA0(1,2)-m_invA0(0,2)*m_invA0(1,1)));
        Set(X, 2, 1, +m_invA0(0,2)*m_d0(0)+m_invA0(1,2)*m_d0(1)-m_d0(2)*(m_invA0(0,0)+m_invA0(1,1)));
        Set(X, 2, 2, +m_d0(2));
    
    } else {
        cout << "WARNING: Coefficients of polynomials {X_j} not implemented for s = " << m_s << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    // Insert data into m_XCoeffs; m_XCoeffs[i] gives the set of coefficients for X_i
    m_XCoeffs = new Vector[m_s];
    for (int i = 0; i < m_s; i++) {
        m_XCoeffs[i] = Vector(X + i*m_s, m_s);
    }
}


