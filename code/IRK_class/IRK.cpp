#include "IRK.hpp"


IRK::IRK(MPI_Comm globComm, int RK_ID, SpatialDiscretization * S,
        double dt, int nt) 
        : m_RK_ID{RK_ID}, m_S(S), m_z(NULL), m_y(NULL), m_w(NULL),
        m_dt{dt}, m_nt(nt) {
    
    /* Set up basic information */
    SetButcherCoeffs();
    SetXCoeffs();
    
    // Set initial condition
    m_S->SetU0();
    
    std::cout << "here 1" << '\n';
    
    // Initialize other vectors based on size of u
    int dimU = m_S->m_u->Size();
    m_z = new Vector(dimU);
    m_y = new Vector(dimU);
    m_w = new Vector(dimU);
    
    std::cout << "here 2" << '\n';
    
    
    /* Set spatial discretization once at the start since it's assumed time-independent */
    m_S->SetL(0.0);
    
    
    /* Set up classes for the eigenvalue operators we need to invert; can just do this once here since operators are time independent */
    RealEigOps.reserve(m_zetaSize);
    for (int i = 0; i < m_zetaSize; i++) {
        RealEigOps[i] = RealEigOp(dt, m_zeta(i));
        // TODO build preconditioner for  each operator...
    }
    
    
    ConjEigOps.reserve(m_etaSize);
    for (int i = 0; i < m_etaSize; i++) {
        ConjEigOps[i] = ConjEigOp(dt, m_eta(i), m_beta(i));
        // TODO build preconditioner for  each operator...
    }
    
}



IRK::~IRK() {
    if (m_z) delete m_z;
    if (m_y) delete m_y;
    if (m_w) delete m_w;
}


ConjEigOp::ConjEigOp(double dt, double eta, double beta) : m_dt{dt}, m_eta0{eta}, m_beta0{beta} {
    
}

RealEigOp::RealEigOp(double dt, double zeta) : m_dt{dt}, m_zeta0{zeta} {
    
}



/* Primary function */
void IRK::TimeStep() {
    
    
    
    // GMRESSolver gmres(MPI_COMM_WORLD);
    // gmres.SetAbsTol(0.0);
    // gmres.SetRelTol(1e-12);
    // gmres.SetMaxIter(200);
    // gmres.SetKDim(10);
    // gmres.SetPrintLevel(1);
    // gmres.SetOperator(*A);
    // gmres.SetPreconditioner(*amg);
    // gmres.Mult(*B, *X);
    
    /* Time at the start of each time step */
    double t = 0.0;
    
    *m_y = 0.0;
    
    /* Main loop */
    for (int step = 0; step < m_nt; step++) {
        
        SetRHSLinearSystem(t); /* Set z */
        
        std::cout << "RHS at time t = "  << t << '\n';
        m_z->Print();
        
        /* Solve the real-valued systems that are linear in L */
        for (int i = 0; i < m_zetaSize; i++) {
            //RealEig.m_zeta0 = m_zeta[i];
        }
        
        /* Solve the complex conjugate systems that are quadratic in L */
        for (int i = 0; i < m_etaSize; i++) {
            //ConjEig.m_eta0 = m_eta[i];
            //ConjEig.m_beta0 = m_beta[i];
        }
        
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
        PolyMult(m_XCoeffs[i], m_dt, *(m_S->m_g), *m_w); /* w <- P(dt*L) * g */
        *m_z += *m_w;
    }
}

/* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients.
Coefficients must be provided for all monomial terms (even if they're 0) and 
in increasing order (from 0th to nth) */
void IRK::PolyMult(Vector coefficients, double alpha, const Vector &x, Vector &y) {
    int n = coefficients.Size();
    y.Set(coefficients[n-1], x); // y <- = coefficients[n-1]*x
    Vector z(y.Size()); // An auxillary vector
    for (int ell = n-2; ell >= 0; ell--) {
        m_S->Mult(y, z); // z <- M^{-1}*L*y        
        add(coefficients[ell], x, alpha, z, y); // y <- coefficients[ell]*x + alpha*z
    } 
}

/* Compute the action of the quadratic polynomial in L: y <- [(eta*eta + beta*beta)*I - 2*eta*dt*L + (dt*L)*2] *x */
void ConjEigOp::Mult(const Vector &x, Vector &y) {
    Vector c(3);
    c(0) = m_eta0*m_eta0 + m_beta0*m_beta0;
    c(1) = -2.0*m_eta0;
    c(2) = 1.0;
    PolyMult(c, m_dt, x, y);
}

/* Compute the action of the linear polynomial in L: y <- [(zeta*I - dt*L)]*x */
void RealEigOp::Mult(const Vector &x, Vector &y) {
    Vector c(2);
    c(0) = m_zeta0;
    c(1) = -1.0;
    //PolyMult(c, m_dt, x, y);
}


void IRK::Test() {
    
    // for (int j = 0; j < m_s; j++) {
    //     for (int k = 0; k < m_s; k++) {
    //         cout << "x[" << j << "][" << k << "]=" << m_XCoeffs[j][k] << '\n';
    //     }
    //     cout << '\n';
    // }
    
    // for (int  i  = 0 ; i < m_s;  i++)  {
    //     std::cout << "X[] =  " << '\n';
    //     m_XCoeffs[i].Print(cout);
    // }
    // 
    // int FD_ID = 3;
    // FDadvection SpaceDisc(MPI_COMM_WORLD, true, 1, 4, 1, FD_ID); 
    // 
    // SpaceDisc.Test(0.134);
    // SpaceDisc.SaveL();
    // SpaceDisc.SaveM();
    // //SpaceDisc.PrintG();
    // SpaceDisc.SaveU();
    
    m_S->SetL(0.134);
    
    
    // m_S->SaveL();
    // m_S->SaveM();
    // //SpaceDisc.PrintG();
    // m_S->SaveU();

    

    //m_S->Mult();
    
    std::cout << "dis is  d!!!!" << '\n';
    
    m_z = new Vector(*(m_S->m_u));
    
    //Vector d = Vector(m_z.Size()); 
    PolyMult(m_XCoeffs[0], 1.0, *(m_S->m_u), *m_z);
    m_z->Print(cout);
    
    
    
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

/* Initialize all Butcher arrays with correct sizes; all entries initialized to zero so that only non-zero entries need be inserted */
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


