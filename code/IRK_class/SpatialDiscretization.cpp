#include "SpatialDiscretization.hpp"


/* -------------------------------------------------------------------------- */
/* ----- Some utility functions that may be helpful for derived classes ----- */
/* -------------------------------------------------------------------------- */

/** Get parallel (square) matrix A from its local CSR data
NOTES: HypreParMatrix makes copies of the data, so it can be deleted */
void SpatialDisretization::GetHypreParMatrixFromCSRData(MPI_Comm comm,  
                                                        int localMinRow, int localMaxRow, int globalNumRows, 
                                                        int * A_rowptr, int * A_colinds, double * A_data,
                                                        HypreParMatrix * &A) 
{
    int localNumRows = localMaxRow - localMinRow + 1;
    int rows[2] = {localMinRow, localMaxRow+1};
    int cols[2] = {localMinRow, localMaxRow+1};
    A = new HypreParMatrix(comm, localNumRows, globalNumRows, globalNumRows, 
                            A_rowptr, A_colinds, A_data, 
                            rows, cols); 
}

/** Get parallel vector x from its local data
NOTES: HypreParVector doesn't make a copy of the data, so it cannot be deleted */
void SpatialDisretization::GetHypreParVectorFromData(MPI_Comm comm, 
                                                     int localMinRow, int localMaxRow, int globalNumRows, 
                                                     double * x_data, HypreParVector * &x)
{
    int rows[2] = {localMinRow, localMaxRow+1}; 
    x = new HypreParVector(comm, globalNumRows, x_data, rows);
}
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/* Constructor */
SpatialDisretization::SpatialDisretization(MPI_Comm spatialComm, bool M_exists) 
    : m_spatialComm{spatialComm}, m_M_exists{M_exists}, 
    m_M(NULL), m_L(NULL), m_u0(NULL), m_g(NULL), m_z(NULL), 
    m_t_L{0.0}, m_t_g{0.0}, 
    m_useSpatialParallel(false)
{
    // Get number of processes
    MPI_Comm_rank(m_spatialComm, &m_spatialRank);
    MPI_Comm_size(m_spatialComm, &m_spatialCommSize);
    
    if (m_spatialCommSize > 1) m_useSpatialParallel = true;
};


/* Destructor: Clean up memory */
SpatialDisretization::~SpatialDisretization() {
    if (m_M) delete m_M;
    if (m_L) delete m_L;
    
    // Do I need to first destroy the data in these vectors?
    if (m_u0) delete m_u0;
    if (m_g) delete m_g;
}

/* Functions setting HYPRE matrix/vector member variables  */
void SpatialDisretization::SetM() {
    if (!m_M) GetSpatialDiscretizationM(m_M);
}

void SpatialDisretization::SetL(double t) {
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

/* Set mass matrix variable if a mass matrix exists and hasn't previously been set */
void SpatialDisretization::SetU0() {
    if (!m_u0) GetSpatialDiscretizationU0(m_u0);
}

void SpatialDisretization::SetG(double t) {
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
}

void SpatialDisretization::GetSpatialDiscretizationM(HypreParMatrix * &M) {
    if (m_M_exists) {
        std::cout << "WARNING: If a mass matrix exists (as indicated), the derived class must implement it!" << '\n';
        MPI_Finalize();
        exit(1);
    }
}

void SpatialDisretization::Test(double t) {
    SetU0();
    SetG(t);
    SetL(t);
    SetM();
    
    Vector z = Vector(m_u0->Size());
    
    m_M->Mult(*m_u0, z);
    
    z.Print(cout);
}

// 
// 
// void SetL();
// void SetG();
// void SetU0();
// 
// virtual void GetM() = 0;
// virtual void GetL() = 0;
// virtual void GetG() = 0;
// virtual void GetU0() = 0;