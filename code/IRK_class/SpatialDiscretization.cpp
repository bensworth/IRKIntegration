#include "SpatialDiscretization.hpp"


/* -------------------------------------------------------------------------- */
/* ----- Some utility functions that may be helpful for derived classes ----- */
/* -------------------------------------------------------------------------- */

/** Get parallel (square) matrix A from its local CSR data
NOTES: HypreParMatrix makes copies of the data, so it can be deleted */
void SpatialDiscretization::GetHypreParMatrixFromCSRData(MPI_Comm comm,  
                                                        int localMinRow, int localMaxRow, HYPRE_Int globalNumRows, 
                                                        int * A_rowptr, int * A_colinds, double * A_data,
                                                        HypreParMatrix * &A) 
{
    int localNumRows = localMaxRow - localMinRow + 1;
    HYPRE_Int rows[2] = {localMinRow, localMaxRow+1};
    HYPRE_Int * cols = rows;
    // TODO: Maybe lodge issue on MFEM github. It's insane that the next line of 
    // code doesn't lead to MFEM reordering the matrix so that the first entry in
    // every row is the diagonal... I.e., the MFEM constructor checks whether rows
    // and cols point to the same location rather than checking if their values are 
    // the same. Such a simple difference (i.e., the above line c.f. the line below) 
    // shouldn't lead to massively different behaviour (most HYPRE functions
    // assume matrices are ordered like this and so they don't work as expected, 
    // but do not throw an error). I think the rest of the constructor musn't work
    // as expected anyway, because even in cols != rows, it's still meant to re-order
    // so that the 1st entry is the diagonal one. I suspect the issue is that the
    // re-ordering only happens when rows == cols; surely this is not right?
    //HYPRE_Int cols[2] = {localMinRow, localMaxRow+1};
    A = new HypreParMatrix(comm, localNumRows, globalNumRows, globalNumRows, 
                            A_rowptr, A_colinds, A_data, 
                            rows, cols); 
}

/** Get parallel vector x from its local data
NOTES: HypreParVector doesn't make a copy of the data, so it cannot be deleted */
void SpatialDiscretization::GetHypreParVectorFromData(MPI_Comm comm, 
                                                     int localMinRow, int localMaxRow, HYPRE_Int globalNumRows, 
                                                     double * x_data, HypreParVector * &x)
{
    int rows[2] = {localMinRow, localMaxRow+1}; 
    x = new HypreParVector(comm, globalNumRows, x_data, rows);
}
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/* Get identity mass matrix operator */
void SpatialDiscretization::GetSpatialDiscretizationM(HypreParMatrix * &M) {
    if (m_M_exists) {
        std::cout << "WARNING: If a mass matrix exists (as indicated), the derived class must implement it!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    if (!m_L) {
        std::cout << "WARNING: Cannot get mass matrix M w/ out first getting discretization L" << '\n';
        MPI_Finalize();
        exit(1);
    }
    GetHypreParIdentityMatrix(*m_L, M);
}

/* Get identity matrix that's compatible with A */
void SpatialDiscretization::GetHypreParIdentityMatrix(const HypreParMatrix &A, HypreParMatrix * &I) 
{
    int globalNumRows = A.GetGlobalNumRows();
    int * row_starts = A.GetRowStarts();
    int localNumRows = row_starts[1] - row_starts[0];
    
    int    * I_rowptr  = new int[localNumRows+1];
    int    * I_colinds = new int[localNumRows];
    double * I_data    = new double[localNumRows];
    I_rowptr[0] = 0;
    for (int i = 0; i < localNumRows; i++) {
        I_colinds[i]   = i + row_starts[0];
        I_data[i]      = 1.0;
        I_rowptr[i+1]  = i+1;
    }
    
    I = new HypreParMatrix(A.GetComm(), 
                            localNumRows, globalNumRows, globalNumRows, 
                            I_rowptr, I_colinds, I_data, 
                            row_starts, row_starts); 
    
    /* These are copied into I, so can delete */
    delete[] I_rowptr;
    delete[] I_colinds;
    delete[] I_data;
} 

/* Constructor */
SpatialDiscretization::SpatialDiscretization(MPI_Comm spatialComm, bool M_exists) 
    : m_spatialComm{spatialComm}, m_M_exists{M_exists}, 
    m_M(NULL), m_L(NULL), m_u(NULL), m_g(NULL),
    m_t_L{0.0}, m_t_g{0.0}, m_t_u{0.0}, 
    m_useSpatialParallel(false), m_nDOFs(-1)
{
    // Get number of processes
    MPI_Comm_rank(m_spatialComm, &m_spatialRank);
    MPI_Comm_size(m_spatialComm, &m_spatialCommSize);
    
    if (m_spatialCommSize > 1) m_useSpatialParallel = true;
    
    // Set initial condition 
    //SetU0();
};


/* Destructor: Clean up memory */
SpatialDiscretization::~SpatialDiscretization() {
    if (m_M) delete m_M;
    if (m_L) delete m_L;
    if (m_u) delete m_u;
    if (m_g) delete m_g;
    //if (m_z) delete m_z;
}

/* Functions setting HYPRE matrix/vector member variables  */
void SpatialDiscretization::SetM() {
    if (!m_M) GetSpatialDiscretizationM(m_M);
}

// void SpatialDiscretization::SetMSolver() {
//     M_solver.SetPreconditioner(M_prec);
//     M_solver.SetOperator(m_M);
//     M_solver.iterative_mode = false;
//     M_solver.SetRelTol(1e-9);
//     M_solver.SetAbsTol(0.0);
//     M_solver.SetMaxIter(100);
//     M_solver.SetPrintLevel(0);
// }


/* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients.
Coefficients must be provided for all monomial terms (even if they're 0) and 
in increasing order (from 0th to nth) */
void SpatialDiscretization::SolDepPolyMult(Vector coefficients, double alpha, const Vector &x, Vector &y) {
    int n = coefficients.Size() - 1;
    y.Set(coefficients[n], x); // y <- coefficients[n]*x
    Vector z(y.Size()); // An auxillary vector
    for (int ell = n-1; ell >= 0; ell--) {
        SolDepMult(y, z); // z <- M^{-1}*L*y        
        add(coefficients[ell], x, alpha, z, y); // y <- coefficients[ell]*x + alpha*z
    } 
}


// TODO: Properly organise mass matrix...
void SpatialDiscretization::SolDepMult(const Vector &x, Vector &y) {
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


void SpatialDiscretization::SetL(double t) {
    /* Get L for the first time */
    if (!m_L) {
        m_t_L = t;
        GetSpatialDiscretizationL(t, m_L);
        
    /* Get L at a different time than what it's currently stored at */
    } else if (m_L_isTimedependent && t != m_t_L) {
        m_t_L = t;
        delete m_L;
        m_L = NULL;
        GetSpatialDiscretizationL(t, m_L);
    }
}

/* Set initial condition in solution vector */
void SpatialDiscretization::SetU0() {
    if (!m_u) {
        GetSpatialDiscretizationU0(m_u);
        m_nDOFs = m_u->Size(); // TODO : Need a  better way  of setting this...
    } else {
        std::cout << "WARNING: Initial condition cannot overwrite existing value of m_u" << '\n';
    }
}

void SpatialDiscretization::SetG(double t) {
    /* Get g for the first time */
    if (!m_g) {
        m_t_g = t;
        GetSpatialDiscretizationG(t, m_g);
        
    /* Get g at a different time than what it's currently stored at */
    } else if (m_G_isTimedependent && t != m_t_g) {
        m_t_g = t;
        delete m_g;
        m_g = NULL;
        GetSpatialDiscretizationG(t, m_g);
    }
    
    if (m_M_exists) {
        std::cout << "WARNING: Need to implement inv(M) * RHS" << '\n';
        MPI_Finalize();
        exit(1);
    }
}



// void SpatialDiscretization::Test(double t) {
//     SetU0();
//     SetG(t);
//     SetL(t);
//     SetM();
// 
// 
//     //m_z(m_u->Size());
// 
//     m_z = Vector(m_u->Size());
// 
//     std::cout << "broken1.5" << '\n';
// 
//     //Vector z = Vector(m_u->Size());
//     //m_M->Mult(*m_u, z);
//     //z.Print(cout);
//     //m_z = new HypreParVector(m_u->Size());
// 
// 
//     std::cout << "broken2" << '\n';
// 
//     Array<double> c(2); // = Array({1.0, 3.0});
//     c[0] = 1.0;
//     c[1] = -3.0;
// 
//     Vector d = Vector(m_z.Size()); 
//     PolyMult(c, 1.0, *m_u, d);
//     d.Print(cout);
// }


