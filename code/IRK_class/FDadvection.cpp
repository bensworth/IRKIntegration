#include "FDadvection.hpp"


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
#endif


/* -------------------------------------------------------------------------- */
/* --------- Implementation of pure virtual functions in base class --------- */
/* -------------------------------------------------------------------------- */
/* These functions essentially just wrap the CSR spatial discretization into  */
/* MFEM-implemented HYPRE matrices and vectors                                */
/* -------------------------------------------------------------------------- */

void FDadvection::ExplicitMult(const Vector &x, Vector &y)
{
    m_L->Mult(x, y);
}


/* Get identity mass matrix; parallel distribution is based on what's saved in the
associated member variables of this class */
void FDadvection::GetSpatialDiscretizationM(HypreParMatrix * &M) 
{
    int    * M_rowptr  = new int[m_onProcSize+1];
    int    * M_colinds = new int[m_onProcSize];
    double * M_data    = new double[m_onProcSize];
    M_rowptr[0] = 0;
    for (int i = 0; i < m_onProcSize; i++) {
        M_colinds[i]   = i + m_localMinRow;
        M_data[i]      = 1.0;
        M_rowptr[i+1]  = i+1;
    }
    GetHypreParMatrixFromCSRData(m_globComm,  
                                    m_localMinRow, m_localMinRow + m_onProcSize-1, m_spatialDOFs, 
                                    M_rowptr, M_colinds, M_data,
                                    M); 
    
    /* These are copied into M, so can delete */
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
} 

void FDadvection::GetSpatialDiscretizationL(double t, HypreParMatrix * &L) {
    
    double * U = NULL;
    bool getU = false;
    int U_ID = -1;
    int m_size = -1;
    
    int      ilower;
    int      iupper;
    int      spatialDOFs;
    int      onProcSize;
    int    * L_rowptr;
    int    * L_colinds;
    double * L_data;
    
    // No parallelism: Spatial discretization on single processor
    if (!m_useSpatialParallel) {
        getSpatialDiscretizationL(L_rowptr, L_colinds, L_data, U,  getU, U_ID, spatialDOFs, t, m_size);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator    
    } else {
        getSpatialDiscretizationL(m_globComm, L_rowptr, L_colinds, L_data, 
                                    U,  getU, U_ID, ilower, iupper, spatialDOFs, 
                                    t, m_size);
    }

    // Flip the sign of L since the base class expects L from du/dt = L*u and not du/dt + L*u = 0
    NegateData(L_rowptr[0], L_rowptr[iupper-ilower+1], L_data);

    GetHypreParMatrixFromCSRData(m_globComm,  
                                    ilower, iupper, spatialDOFs, 
                                    L_rowptr, L_colinds, L_data,
                                    L); 
    
    /* These are copied into L, so can delete */
    delete L_rowptr;
    delete L_colinds;
    delete L_data;
} 


void FDadvection::NegateData(int start, int stop, double * &data) {
    for (int i = start; i < stop; i++) {
        data[i] *= -1.0;
    }
}

/* Get initial condition in an MFEM HypreParVector */
void FDadvection::GetSpatialDiscretizationU0(HypreParVector * &u0) {
    
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    double * U;

    // No parallelism: Spatial discretization on single processor
    if (!m_useSpatialParallel) {
        getInitialCondition(U, spatialDOFs);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator
    } else {
        getInitialCondition(m_globComm, U, ilower, iupper, spatialDOFs);    
    }    
    
    GetHypreParVectorFromData(m_globComm, 
                             ilower, iupper, spatialDOFs, 
                             U, u0);
}


/* Get solution-independent source term in an MFEM HypreParVector */
void FDadvection::GetSpatialDiscretizationG(double t, HypreParVector * &g) {
    
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    double * G;

    // No parallelism: Spatial discretization on single processor
    if (!m_useSpatialParallel) {
        // Call when NOT using spatial parallelism                                        
        getSpatialDiscretizationG(G, spatialDOFs, t); 
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator
    } else {
        // Call when using spatial parallelism                          
        getSpatialDiscretizationG(m_globComm, G, ilower, iupper, spatialDOFs, t); 
    }    
    
    GetHypreParVectorFromData(m_globComm, 
                             ilower, iupper, spatialDOFs, 
                             G, g);
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
        height = m_u->Size(); // TODO : Need a  better way  of setting this...
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

/* Functions setting HYPRE matrix/vector member variables  */
void SpatialDiscretization::SetM() {
    if (!m_M) GetSpatialDiscretizationM(m_M);
}


/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */




/* Copy external numerical dissipation parameters into member variable */
void FDadvection::SetNumDissipation(Num_dissipation dissipation_params) 
{
    m_dissipation = true;
    m_dissipation_params = dissipation_params;
    
    if (m_dim > 1) {
        if (m_spatialRank == 0) std::cout << "WARNING: Numerical dissipation only implemented for 1D problems..." << '\n';
        MPI_Finalize();
        exit(1);
    }
}

FDadvection::FDadvection(MPI_Comm globComm, bool M_exists) :
    IRKOperator(globComm, M_exists), m_M_exists{M_exists},
{
    // Get number of processes
    MPI_Comm_rank(m_globComm, &m_spatialRank);
    MPI_Comm_size(m_globComm, &m_globCommSize);
    
    if (m_globCommSize > 1) m_useSpatialParallel = true;
}


FDadvection::FDadvection(MPI_Comm globComm, bool M_exists, int dim, int refLevels, int order, 
                        int problemID, std::vector<int> px)
    : IRKOperator(globComm), m_M_exists{M_exists},
    m_dim{dim}, m_refLevels{refLevels}, m_problemID{problemID}, m_px{px},
    m_periodic(false), m_inflow(false), m_PDE_soln_implemented(false), m_dissipation(false),
    m_M(NULL), m_L(NULL), m_u(NULL), m_g(NULL),
    m_t_L{0.0}, m_t_g{0.0}, m_t_u{0.0}, 
    m_useSpatialParallel(false), height(-1)
{    
    // Seed random number generator so results are consistent!
    srand(0);

    // Get number of processes
    MPI_Comm_rank(m_globComm, &m_spatialRank);
    MPI_Comm_size(m_globComm, &m_globCommSize);
    
    if (m_globCommSize > 1) m_useSpatialParallel = true;
    
    /* ----------------------------------------------------------------------------------------------------- */
    /* --- Check specified proc distribution is consistent with the number of procs passed by base class --- */
    /* ----------------------------------------------------------------------------------------------------- */
    if (!m_px.empty()) {        
        // Doesn't make sense to prescribe proc grid if not using spatial parallelism
        if (!m_useSpatialParallel) {
            if (m_spatialRank == 0) std::cout << "WARNING: Trying to prescribe spatial processor grid layout while not using spatial parallelism!" << '\n';
        // Ensure the product of number of procs in each direction matches the number of procs assigned in space in the base class
        } else {
            int num_procs = 1;
            for (const int &element: m_px)
                num_procs *= element;
            if (num_procs != m_globCommSize) {
                std::string proc_grid_dims = "";
                for (const int &element: m_px)
                    proc_grid_dims += std::to_string(element) + "x";
                proc_grid_dims.pop_back();
                if (m_spatialRank == 0) std::cout << "WARNING: Prescribed spatial processor grid (P=" << proc_grid_dims << ") having " << num_procs << " processors does not contain same number of procs as specified in base class (" << m_globCommSize << ")! \n";
                MPI_Finalize();
                exit(1);
            }
        }
    }
    
    /* ------------------------------ */
    /* --- Setup grid information --- */
    /* ------------------------------ */
    // Can generalize this if you like to pass in distinct order and nDOFs in y-direction. This by default just makes them the same as in the x-direction    
    double nx = pow(2, refLevels);
    double ny = nx;
    //double nx = 10;
    double dx = 2.0 / nx; 
    double dy = 2.0 / ny; 
    double xboundary0 = -1.0; // Assume x \in [-1,1].
    
    if (dim >= 1) {
        m_nx.push_back(nx);
        m_dx.push_back(dx);
        m_boundary0.push_back(xboundary0);
        m_order.push_back(order);
        m_spatialDOFs = m_nx[0];
        
        // These will be updated below if using spatial parallelism.
        m_onProcSize  = m_spatialDOFs; 
        m_localMinRow = 0;
    }
    
    // Just make domain in y-direction the same as in x-direction
    if (dim == 2) {
        //m_nx.push_back(nx + 7);
        m_nx.push_back(ny);
        m_dx.push_back(dy);
        m_boundary0.push_back(xboundary0);
        m_order.push_back(order);
        m_spatialDOFs = m_nx[0] * m_nx[1];
        
        // These will be updated below if using spatial parallelism.
        m_onProcSize  = m_spatialDOFs; 
        m_localMinRow = 0;
    }
    
    /* Set variables based on form of PDE */
    /* Test problems with periodic boundaries */
    if (m_problemID == 1) { /* Constant-coefficient */
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_G_isTimedependent = false;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 2) { /* Variable-coefficient in convervative form */
        m_conservativeForm  = true; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 3) { /* Variable-coefficient in non-convervative form */
        m_conservativeForm  = false; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 4) { /* Spatially variable-coefficient in convervative form */
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
        
    /* Test problems with inflow/outflow boundaries */
    } else if (m_problemID == 101) { /* Constant-coefficient */
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_G_isTimedependent = true; 
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 102) { /* Variable-coefficient in convervative form */
        m_conservativeForm  = true; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 103) { /* Variable-coefficient in non-convervative form */
        m_conservativeForm  = false; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;    
        
    // Quit because wave speeds, sources, IC's etc are not implemented for a `general` problem
    } else { 
        if (m_spatialRank == 0) std::cout << "WARNING: FD problemID == " << m_problemID << " not recognised!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    
    // Set BC flag
    if (m_problemID < 100) {
        m_periodic = true;
    } else if (m_problemID >= 100) {
        m_inflow = true;
        if (m_dim > 1) {
            std::cout << "WARNING: FD with inflow BCs not implemented in 2D" << '\n';
            MPI_Finalize();
            exit(1);
        }
    }
    
    
    /* -------------------------------------------------------- */
    /* -------------- Set up spatial parallelism -------------- */
    /* -------------------------------------------------------- */
    /* Ensure spatial parallelism setup is permissible and 
    decide which variables current process and its neighbours own, etc */
    if (m_useSpatialParallel) {
        //  To be safe, just do a check to ensure there isn't more procs than DOFs
        if (m_globCommSize > m_spatialDOFs) {
            if (m_spatialRank == 0) std::cout << "WARNING: Number of processors must exceed number of spatial DOFs!" << '\n';
            MPI_Finalize();
            exit(1);
        }
        
        /* --- One spatial dimension --- */
        if (m_dim == 1) 
        {
            m_pGridInd.push_back(m_spatialRank);
            if (m_px.empty()) m_px.push_back(m_globCommSize);
            m_nxOnProcInt.push_back(m_nx[0]/m_px[0]);
            
            // Compute number of DOFs on proc
            if (m_spatialRank < m_px[0]-1)  {
                m_nxOnProc.push_back(m_nxOnProcInt[0]);  // All procs in interior have same number of DOFS
            } else {
                m_nxOnProc.push_back(m_nx[0] - (m_px[0]-1)*m_nxOnProcInt[0]); // Proc on EAST boundary take the remainder of DOFs
            }
            m_localMinRow = m_pGridInd[0] * m_nxOnProcInt[0]; // Index of first DOF on proc
            m_onProcSize  = m_nxOnProc[0];
            
        /* --- Two spatial dimensions --- */
        } 
        else if (m_dim == 2) 
        {
            /* If a square number of procs not set by base class, the user must 
            manually pass dimensions of proc grid */
            if (m_px.empty()) {
                int temp = sqrt(m_globCommSize);
                if (temp * temp != m_globCommSize) {
                    std::cout << "WARNING: Spatial processor grid dimensions must be specified if non-square grid is to be used (using P=" << m_globCommSize << " procs in space)" << '\n';
                    MPI_Finalize();
                    exit(1);
                /* Setup default square process grid */
                } else {
                    m_px.push_back(temp); // In x-direction have sqrt of number of total procs
                    m_px.push_back(temp); // In y-direction: ditto
                }
            }
            
            // Get indices on proc grid
            m_pGridInd.push_back(m_spatialRank % m_px[0]); // x proc grid index
            m_pGridInd.push_back(m_spatialRank / m_px[0]); // y proc grid index
            
            // Number of DOFs on procs in interior of domain
            m_nxOnProcInt.push_back(m_nx[0]/m_px[0]);
            m_nxOnProcInt.push_back(m_nx[1]/m_px[1]);
            
            // Number of DOFs on procs on boundary of proc domain 
            m_nxOnProcBnd.push_back(m_nx[0] - (m_px[0]-1)*m_nxOnProcInt[0]); // East boundary
            m_nxOnProcBnd.push_back(m_nx[1] - (m_px[1]-1)*m_nxOnProcInt[1]); // North boundary
            
            // Compute number of DOFs on proc
            if (m_pGridInd[0] < m_px[0] - 1) {
                m_nxOnProc.push_back( m_nxOnProcInt[0] ); // All procs in interior have same number of DOFS
            } else {
                m_nxOnProc.push_back( m_nxOnProcBnd[0] ); // Procs on EAST boundary take the remainder of DOFs
            }
            if (m_pGridInd[1] < m_px[1] - 1) {
                m_nxOnProc.push_back( m_nxOnProcInt[1] ); // All procs in interior have same number of DOFS
            } else {
                m_nxOnProc.push_back( m_nxOnProcBnd[1] ); // All procs in interior have same number of DOFS
            }
            m_onProcSize = m_nxOnProc[0] * m_nxOnProc[1]; 
            
            // Compute global index of first DOF on proc
            m_localMinRow = m_pGridInd[0]*m_nxOnProcInt[0]*m_nxOnProc[1] + m_pGridInd[1]*m_nx[0]*m_nxOnProcInt[1];
                        
            /* --- Communicate size information to my four nearest neighbours --- */
            // Assumes we do not have to communicate further than our nearest neighbouring procs...
            // Note: we could work this information out just using the grid setup but it's more fun to send/retrieve from other procs
            // Global proc indices of my nearest neighbours; the processor grid is assumed periodic here to enforce periodic BCs
            int pNInd = m_pGridInd[0]  + ((m_pGridInd[1]+1) % m_px[1]) * m_px[0];
            int pSInd = m_pGridInd[0]  + ((m_pGridInd[1]-1 + m_px[1]) % m_px[1]) * m_px[0];
            int pEInd = (m_pGridInd[0] + 1) % m_px[0] + m_pGridInd[1] * m_px[0];
            int pWInd = (m_pGridInd[0] - 1 + m_px[0]) % m_px[0] + m_pGridInd[1] * m_px[0];
            
            // Send out index of first DOF I own to my nearest neighbours
            MPI_Send(&m_localMinRow, 1, MPI_INT, pNInd, 0, m_globComm);
            MPI_Send(&m_localMinRow, 1, MPI_INT, pSInd, 0, m_globComm);
            MPI_Send(&m_localMinRow, 1, MPI_INT, pEInd, 0, m_globComm);
            MPI_Send(&m_localMinRow, 1, MPI_INT, pWInd, 0, m_globComm);
            
            // Recieve index of first DOF owned by my nearest neighbours
            m_neighboursLocalMinRow.reserve(4); // Neighbours are ordered as NORTH, SOUTH, EAST, WEST
            MPI_Recv(&m_neighboursLocalMinRow[0], 1, MPI_INT, pNInd, 0, m_globComm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_neighboursLocalMinRow[1], 1, MPI_INT, pSInd, 0, m_globComm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_neighboursLocalMinRow[2], 1, MPI_INT, pEInd, 0, m_globComm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_neighboursLocalMinRow[3], 1, MPI_INT, pWInd, 0, m_globComm, MPI_STATUS_IGNORE);
            
            // Send out dimensions of DOFs I own to nearest neighbours
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pNInd, 0, m_globComm);
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pSInd, 0, m_globComm);
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pEInd, 0, m_globComm);
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pWInd, 0, m_globComm);
            
            // Receive dimensions of DOFs that my nearest neighbours own
            m_neighboursNxOnProc.reserve(8); // Just stack the nx and ny on top of one another in pairs in same vector
            MPI_Recv(&m_neighboursNxOnProc[0], 2, MPI_INT, pNInd, 0, m_globComm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_neighboursNxOnProc[2], 2, MPI_INT, pSInd, 0, m_globComm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_neighboursNxOnProc[4], 2, MPI_INT, pEInd, 0, m_globComm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_neighboursNxOnProc[6], 2, MPI_INT, pWInd, 0, m_globComm, MPI_STATUS_IGNORE);
        }
    }
    //std::cout << "I made it through constructor..." << '\n';
}


FDadvection::~FDadvection()
{
    if (m_M) delete m_M;
    if (m_L) delete m_L;
    if (m_u) delete m_u;
    if (m_g) delete m_g;
    //if (m_z) delete m_z;
}


// Integer ceiling division
int FDadvection::div_ceil(int numerator, int denominator)
{
        std::div_t res = std::div(numerator, denominator);
        return res.rem ? (res.quot + 1) : res.quot;
}

/* Return data to be used as an initial iterate. Data depends on integer U0ID */
double FDadvection::GetInitialIterate(double x, int U0ID) {
    if (U0ID == -1) {  // PDE initial condition
        return InitCond(x);
    } else if (U0ID == 0) { // Zero  
        return 0.0; 
    } else { // Random number in [0,1]
        return (double)rand() / (double)RAND_MAX;
    }
}

/* Return data to be used as an initial iterate. Data depends on integer U0ID */
double FDadvection::GetInitialIterate(double x, double y, int U0ID) {
    if (U0ID == -1) {  // PDE initial condition
        return InitCond(x, y);
    } else if (U0ID == 0) { // Zero  
        return 0.0; 
    } else { // Random number in [0,1]
        return (double)rand() / (double)RAND_MAX;
    }
}


/* Exact solution for model problems. 

This depends on initial conditions, source terms, wave speeds, and  mesh
So if any of these are updated the solutions given here will be wrong...  */
double FDadvection::PDE_Solution(double x, double t) {
    if (m_problemID == 1 || m_problemID == 101) {
        return InitCond( std::fmod(x + 1 - t, 2)  - 1 );
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4 || m_problemID == 102 || m_problemID == 103) {
        return cos( PI*(x-t) ) * exp( cos( 2*PI*t ) - 1 );     
    } else {
        return 0.0; // Just so we're not given a compilation warning
    }
}


double FDadvection::PDE_Solution(double x, double y, double t) {
    if (m_problemID == 1) {
        return InitCond( std::fmod(x + 1 - t, 2) - 1, std::fmod(y + 1 - t, 2)  - 1 );
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4) {
        return cos( PI*(x-t) ) * cos( PI*(y-t) ) * exp( cos( 4*PI*t ) - 1 );     
    } else {
        return 0.0; // Just so we're not given a compilation warning
    }
}

 
double FDadvection::InitCond(double x) 
{        
    if (m_problemID == 1 || m_problemID == 101) {
        return pow(sin(PI * x), 4.0);
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4 || m_problemID == 102 || m_problemID == 103) {
        return cos(PI * x);
    } else {
        return 0.0;
    }
}

double FDadvection::InflowBoundary(double t) 
{
    if (m_problemID == 101) {
        return InitCond(1.0 - WaveSpeed(1.0, t)*t); // Set to be the analytical solution at the RHS boundary
    }  else if (m_problemID == 102 || m_problemID == 103) {
        return PDE_Solution(m_boundary0[0], t);     // Just evaluate the analytical PDE soln on the boundary
    } else {
        return 0.0;
    }
}

double FDadvection::InitCond(double x, double y) 
{        
    if (m_problemID == 1) {
        return pow(cos(PI * x), 4.0) * pow(sin(PI * y), 2.0);
        //return ;
        // if ((x >= 0) && (y >= 0)) return 1.0;
        // if ((x < 0) && (y >= 0)) return 2.0;
        // if ((x < 0) && (y < 0)) return 3.0;
        // if ((x >= 0) && (y < 0)) return 4.0;
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4) {
        return cos(PI * x) * cos(PI * y);
    } else {
        return 0.0;
    }
}


// Wave speed for 1D problem
// For inflow problems, this MUST be positive near and on the inflow and outflow boundaries!
double FDadvection::WaveSpeed(double x, double t) {
    if (m_problemID == 1 || m_problemID == 101) {
        return 1.0;
    } else if (m_problemID == 2 || m_problemID == 3) {
        return cos( PI*(x-t) ) * exp( -pow(sin(2*PI*t), 2.0) );
    } else if (m_problemID == 4) {
        return 0.5*(1.0 + pow(cos(PI*x), 2.0));
    } else if (m_problemID == 102 || m_problemID == 103) {
        return 0.5*(1.0 + pow(cos(PI*(x-t)), 2.0)) * exp( -pow(sin(2*PI*t), 2.0) ); 
    }  else  {
        return 0.0;
    }
}


// Wave speed for 2D problem; need to choose component as 1 or 2.
double FDadvection::WaveSpeed(double x, double y, double t, int component) {
    if (m_problemID == 1) {
        return 1.0;
    } else if (m_problemID == 2 || m_problemID == 3) {
        if (component == 0) {
            return cos( PI*(x-t) ) * cos(PI*y) * exp( -pow(sin(2*PI*t), 2.0) );
        } else {
            return sin(PI*x) * cos( PI*(y-t) ) * exp( -pow(sin(2*PI*t), 2.0) );
        }
    } else if (m_problemID == 4) {
        if (component == 0) {
            return 0.5*(1.0 + pow(cos(PI*x), 2.0)) * 0.5*(1.0 + pow(sin(PI*y), 2.0));
        } else {
            return 0.5*(1.0 + pow(sin(PI*x), 2.0)) * 0.5*(1.0 + pow(cos(PI*y), 2.0));
        }
    } else {
        return 0.0;
    }
}



// Map grid index to grid point in specified dimension
double FDadvection::MeshIndToPoint(int meshInd, int dim)
{
    return m_boundary0[dim] + meshInd * m_dx[dim];
}


// Mapping between global indexing of unknowns and true mesh indices
// TODO: Add in support here for 2D problem both with and without spatial parallel...
int FDadvection::GlobalIndToMeshInd(int globInd)
{
    if (m_periodic) {
        return globInd;
    } else {
        return globInd+1; // The solution at inflow boundary is eliminated since it's prescribed by the boundary condition
    }
}


// RHS of PDE 
double FDadvection::PDE_Source(double x, double t)
{
    if (m_problemID == 1 || m_problemID == 101) {
        return 0.0;
    } else if (m_problemID == 2) {
        return PI * exp( -2*pow(sin(PI*t), 2.0)*(cos(2*PI*t) + 2) ) * ( sin(2*PI*(t-x)) 
                    - exp( pow(sin(2*PI*t), 2.0) )*(  sin(PI*(t-x)) + 2*sin(2*PI*t)*cos(PI*(t-x)) ) );
    } else if (m_problemID == 3) {
        return 0.5 * PI * exp( -2*pow(sin(PI*t), 2.0)*(cos(2*PI*t) + 2) ) * ( sin(2*PI*(t-x)) 
                    - 2*exp( pow(sin(2*PI*t), 2.0) )*(  sin(PI*(t-x)) + 2*sin(2*PI*t)*cos(PI*(t-x)) ) );
    } else if (m_problemID == 4) {
        return 0.5 * exp( -1.0 + cos(2*PI*t) ) * (-PI*cos(PI*(x-t))*(4*sin(2*PI*t) + sin(2*PI*x)) + PI*pow(sin(PI*x), 2.0)*sin(PI*(x-t)) );
    } else if (m_problemID == 102) {
        return 0.5 * exp(-2.0*(2.0 + cos(2.0*PI*t))*pow(sin(PI*t), 2.0))
                    * ( PI*(1.0 + 3.0*pow(cos(PI*(t-x)), 2.0))*sin(PI*(t-x))
                    - 2.0*PI*exp(pow(sin(2*PI*t), 2.0))*(2.0*cos(PI*(t-x))*sin(2.0*PI*t) + sin(PI*(t-x))) );
    
    } else if (m_problemID == 103) {
        return 0.5 * exp(-2.0*(2.0 + cos(2.0*PI*t))*pow(sin(PI*t), 2.0))
                    * ( PI*(1.0 + pow(cos(PI*(t-x)), 2.0))*sin(PI*(t-x))
                    - 2.0*PI*exp(pow(sin(2*PI*t), 2.0))*(2.0*cos(PI*(t-x))*sin(2.0*PI*t) + sin(PI*(t-x))) );
                    
    } else {
        return 0.0;
    }
}

// RHS of PDE 
double FDadvection::PDE_Source(double x, double y, double t)
{
    if (m_problemID == 1) {
        return 0.0;
    } else if (m_problemID == 2) {
        return PI*exp(-3*pow(sin(2*PI*t), 2.0)) * 
            (
            cos(PI*(t-y))*( -exp(pow(sin(2*PI*t), 2.0)) * sin(PI*(t-x)) + cos(PI*y)*sin(2*PI*(t-x)) ) +
            cos(PI*(t-x))*( -exp(pow(sin(2*PI*t), 2.0)) * (4*cos(PI*(t-y))*sin(4*PI*t) + sin(PI*(t-y))) + sin(PI*x)*sin(2*PI*(t-y)) )
            );
    } else if (m_problemID == 3) {
        return 0.5*PI*exp(-3*pow(sin(2*PI*t), 2.0)) * 
            (
            cos(PI*(t-y))*( -2*exp(pow(sin(2*PI*t), 2.0)) * sin(PI*(t-x)) + cos(PI*y)*sin(2*PI*(t-x)) ) +
            cos(PI*(t-x))*( -2*exp(pow(sin(2*PI*t), 2.0)) * (4*cos(PI*(t-y))*sin(4*PI*t) + sin(PI*(t-y))) + sin(PI*x)*sin(2*PI*(t-y)) )
            );
    } else if (m_problemID == 4) {
        return -0.25*PI*exp(-1.0 + cos(4*PI*t)) * 
            (
            cos(PI*(y-t))*sin(PI*(x-t))*(-3.0 + pow(sin(PI*y),2.0) + pow(cos(PI*x),2.0)*(1.0+pow(sin(PI*y),2.0)) )
            +
            cos(PI*(x-t)) * 
            (
                (-3.0 + pow(sin(PI*x),2.0) + pow(cos(PI*y),2.0)*(1.0+pow(sin(PI*x),2.0)))*sin(PI*(y-t)) 
                +
                0.5*cos(PI*(y-t))*(32.0*sin(4*PI*t) + 3.0*(sin(2*PI*x) + sin(2*PI*y)) - sin(2*PI*(x+y)) )
            )
            ); 
            
    } else {
        return 0.0;
    }
}



// NO SPATIAL PARALLELISM: Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::getSpatialDiscretizationL(int * &L_rowptr, int * &L_colinds,
                                           double * &L_data, double * &U0, bool getU0, int U0ID,
                                           int &spatialDOFs, double t, int &bsize)
{
    if (m_dim == 1) {
        // Simply call the same routine as if using spatial parallelism
        int dummy1, dummy2;
        get1DSpatialDiscretizationL(NULL, L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID, dummy1, dummy2,
                                      spatialDOFs, t, bsize);
    } else if (m_dim == 2) {
        // Call a serial implementation of 2D code
        get2DSpatialDiscretizationL(L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID,
                                      spatialDOFs, t, bsize);
    }
}

// USING SPATIAL PARALLELISM: Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::getSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) 
{
    if (m_dim == 1) {
        get1DSpatialDiscretizationL(globComm, L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID, localMinRow, localMaxRow,
                                      spatialDOFs, t, bsize);
    } else if (m_dim == 2) {
        get2DSpatialDiscretizationL(globComm, L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID, localMinRow, localMaxRow,
                                      spatialDOFs, t, bsize);
    }
}




// USING SPATIAL PARALLELISM: Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::get2DSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize)
{
    // Unpack variables frequently used
    // x-related variables
    int nx          = m_nx[0];
    double dx       = m_dx[0];
    int xFD_Order   = m_order[0];
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    // y-related variables
    int ny          = m_nx[1];
    double dy       = m_dx[1];
    int yFD_Order   = m_order[1];
    int yStencilNnz = yFD_Order + 1; // Width of the FD stencil
    int yDim        = 1;
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    spatialDOFs   = m_spatialDOFs;                      
    localMinRow   = m_localMinRow;                   // First row on proc
    localMaxRow   = localMinRow + m_onProcSize - 1;  // Last row on proc 
    int L_nnz     = (xStencilNnz + yStencilNnz - 1) * m_onProcSize; // Nnz on proc. Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    L_rowptr      = new int[m_onProcSize + 1];
    L_colinds     = new int[L_nnz];
    L_data        = new double[L_nnz];
    L_rowptr[0]   = 0;
    if (getU0) U0 = new double[m_onProcSize]; // Initial guesss at solution
    int rowcount  = 0;
    int dataInd   = 0;
    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * xPlusInds;
    int * yPlusInds;
    double * xPlusWeights;
    double * yPlusWeights;
    get1DUpwindStencil(xPlusInds, xPlusWeights, xDim);
    get1DUpwindStencil(yPlusInds, yPlusWeights, yDim);
    
    // Generate stencils for wind blowing right to left by reversing stencils
    int * xMinusInds       = new int[xStencilNnz];
    int * yMinusInds       = new int[yStencilNnz];
    double * xMinusWeights = new double[xStencilNnz];
    double * yMinusWeights = new double[yStencilNnz];
    for (int i = 0; i < xStencilNnz; i++) {
        xMinusInds[i]    = -xPlusInds[xFD_Order-i];
        xMinusWeights[i] = -xPlusWeights[xFD_Order-i];
    } 
    for (int i = 0; i < yStencilNnz; i++) {
        yMinusInds[i]    = -yPlusInds[yFD_Order-i];
        yMinusWeights[i] = -yPlusWeights[yFD_Order-i];
    } 
    
    // Placeholder for weights to discretize derivatives at each point 
    double * xLocalWeights = new double[xStencilNnz];
    double * yLocalWeights = new double[yStencilNnz];
    int    * xLocalInds; // This will just point to an existing array, doesn't need memory allocated!
    int    * yLocalInds; // This will just point to an existing array, doesn't need memory allocated!
    int      xIndOnProc; 
    int      yIndOnProc; 
    int      xIndGlobal;
    int      yIndGlobal;
    double   x;
    double   y;
    
    // Compute x- and y-components of wavespeed given some dx or dy perturbation away from the current point
    std::function<double(int)> xLocalWaveSpeed;
    std::function<double(int)> yLocalWaveSpeed; 
    
    // Given local indices on current process return global index
    std::function<int(int, int, int)> MeshIndsOnProcToGlobalInd = [this, localMinRow](int row, int xIndOnProc, int yIndOnProc) { return localMinRow + xIndOnProc + yIndOnProc*m_nxOnProc[0]; };
    
    // Given connection that overflows in some direction onto a neighbouring process, return global index of that connection. OverFlow variables are positive integers.
    std::function<int(int, int)> MeshIndsOnNorthProcToGlobalInd = [this](int xIndOnProc, int yOverFlow)  { return m_neighboursLocalMinRow[0] + xIndOnProc + (yOverFlow-1)*m_neighboursNxOnProc[0]; };
    std::function<int(int, int)> MeshIndsOnSouthProcToGlobalInd = [this](int xIndOnProc, int yOverFlow)  { return m_neighboursLocalMinRow[1] + xIndOnProc + (m_neighboursNxOnProc[3]-yOverFlow)*m_neighboursNxOnProc[2]; };
    std::function<int(int, int)> MeshIndsOnEastProcToGlobalInd  = [this](int xOverFlow,  int yIndOnProc) { return m_neighboursLocalMinRow[2] + xOverFlow-1  + yIndOnProc*m_neighboursNxOnProc[4]; };
    std::function<int(int, int)> MeshIndsOnWestProcToGlobalInd  = [this](int xOverFlow,  int yIndOnProc) { return m_neighboursLocalMinRow[3] + m_neighboursNxOnProc[6]-1 - (xOverFlow-1) + yIndOnProc*m_neighboursNxOnProc[6]; };
    
    
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {                                          
        xIndOnProc = rowcount % m_nxOnProc[0];                      // x-index on proc
        yIndOnProc = rowcount / m_nxOnProc[0];                      // y-index on proc
        xIndGlobal = m_pGridInd[0] * m_nxOnProcInt[0] + xIndOnProc; // Global x-index
        yIndGlobal = m_pGridInd[1] * m_nxOnProcInt[1] + yIndOnProc; // Global y-index
        y          = MeshIndToPoint(yIndGlobal, yDim);              // y-value of current point
        x          = MeshIndToPoint(xIndGlobal, xDim);              // x-value of current point

        // Compute x- and y-components of wavespeed given some dx or dy perturbation away from the current point
        xLocalWaveSpeed = [this, x, dx, y, t, xDim](int xOffset) { return WaveSpeed(x + dx * xOffset, y, t, xDim); };
        yLocalWaveSpeed = [this, x, y, dy, t, yDim](int yOffset) { return WaveSpeed(x, y + dy * yOffset, t, yDim); };
    
        // Get stencil for discretizing x-derivative at current point 
        getLocalUpwindDiscretization(xLocalWeights, xLocalInds,
                                        xLocalWaveSpeed, 
                                        xPlusWeights, xPlusInds, 
                                        xMinusWeights, xMinusInds, 
                                        xStencilNnz);
        // Get stencil for discretizing y-derivative at current point 
        getLocalUpwindDiscretization(yLocalWeights, yLocalInds,
                                        yLocalWaveSpeed, 
                                        yPlusWeights, yPlusInds, 
                                        yMinusWeights, yMinusInds, 
                                        yStencilNnz);
    
        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, then continuing to loop over y
        // Actually, periodicity stuffs this up I think...
        for (int yNzInd = 0; yNzInd < yStencilNnz; yNzInd++) {

            // The two stencils will intersect somewhere at this y-point
            if (yLocalInds[yNzInd] == 0) {
                for (int xNzInd = 0; xNzInd < xStencilNnz; xNzInd++) {
                    int temp = xIndOnProc + xLocalInds[xNzInd]; // Local x-index of current connection
                    // Connection to process on WEST side
                    if (temp < 0) {
                        L_colinds[dataInd] = MeshIndsOnWestProcToGlobalInd(abs(temp), yIndOnProc);
                    // Connection to process on EAST side
                    } else if (temp > m_nxOnProc[0]-1) {
                        L_colinds[dataInd] = MeshIndsOnEastProcToGlobalInd(temp - (m_nxOnProc[0]-1), yIndOnProc);
                    // Connection is on processor
                    } else {
                        L_colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, temp, yIndOnProc);
                    }
                    L_data[dataInd]    = xLocalWeights[xNzInd];

                    // The two stencils intersect at this point x-y-point, i.e. they share a 
                    // column in L, so add y-derivative information to x-derivative information that exists there
                    if (xLocalInds[xNzInd] == 0) L_data[dataInd] += yLocalWeights[yNzInd]; 
                    dataInd += 1;
                }
    
            // There is no possible intersection between between x- and y-stencils
            } else {
                int temp = yIndOnProc + yLocalInds[yNzInd]; // Local y-index of current connection
                // Connection to process on SOUTH side
                if (temp < 0) {
                    L_colinds[dataInd] = MeshIndsOnSouthProcToGlobalInd(xIndOnProc, abs(temp));
                // Connection to process on NORTH side
                } else if (temp > m_nxOnProc[1]-1) {
                    L_colinds[dataInd] = MeshIndsOnNorthProcToGlobalInd(xIndOnProc, temp - (m_nxOnProc[1]-1));
                // Connection is on processor
                } else {
                    L_colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, xIndOnProc, temp);
                }
                
                L_data[dataInd]    = yLocalWeights[yNzInd];
                dataInd += 1;
            }
        }    
    
        // Set initial guess at the solution
        if (getU0) U0[rowcount] = GetInitialIterate(x, y, U0ID); 
    
        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    }    
    
    // Check that sufficient data was allocated
    if (dataInd > L_nnz) {
        std::cout << "WARNING: FD spatial discretization matrix has more nonzeros than allocated.\n";
    }
    
    // Clean up
    delete[] xPlusInds;
    delete[] xPlusWeights;
    delete[] yPlusInds;
    delete[] yPlusWeights;
    delete[] xMinusInds;
    delete[] xMinusWeights;
    delete[] xLocalWeights;
    delete[] yLocalWeights;
}
                             


/* Serial implementation of 2D spatial discretization. Is essentially the same as the
parallel version, but the indexing is made simpler */
void FDadvection::get2DSpatialDiscretizationL(int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID,
                                              int &spatialDOFs, double t, int &bsize)
{
    // Unpack variables frequently used
    // x-related variables
    int nx          = m_nx[0];
    double dx       = m_dx[0];
    int xFD_Order   = m_order[0];
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    // y-related variables
    int ny          = m_nx[1];
    double dy       = m_dx[1];
    int yFD_Order   = m_order[1];
    int yStencilNnz = yFD_Order + 1; // Width of the FD stencil
    int yDim        = 1;


    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    spatialDOFs     =  m_spatialDOFs;
    int localMinRow = 0;
    int localMaxRow = m_spatialDOFs - 1;
    int L_nnz       = (xStencilNnz + yStencilNnz - 1) * m_onProcSize; // Nnz on proc. Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    L_rowptr        = new int[m_onProcSize + 1];
    L_colinds       = new int[L_nnz];
    L_data          = new double[L_nnz];
    int rowcount    = 0;
    int dataInd     = 0;
    L_rowptr[0]     = 0;
    if (getU0) U0   = new double[m_onProcSize]; // Initial guesss at solution


    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * xPlusInds;
    int * yPlusInds;
    double * xPlusWeights;
    double * yPlusWeights;
    get1DUpwindStencil(xPlusInds, xPlusWeights, xDim);
    get1DUpwindStencil(yPlusInds, yPlusWeights, yDim);

    // Generate stencils for wind blowing right to left by reversing stencils
    int * xMinusInds       = new int[xStencilNnz];
    int * yMinusInds       = new int[yStencilNnz];
    double * xMinusWeights = new double[xStencilNnz];
    double * yMinusWeights = new double[yStencilNnz];
    for (int i = 0; i < xStencilNnz; i++) {
        xMinusInds[i]    = -xPlusInds[xFD_Order-i];
        xMinusWeights[i] = -xPlusWeights[xFD_Order-i];
    } 
    for (int i = 0; i < yStencilNnz; i++) {
        yMinusInds[i]    = -yPlusInds[yFD_Order-i];
        yMinusWeights[i] = -yPlusWeights[yFD_Order-i];
    } 

    // Placeholder for weights to discretize derivatives at point 
    double * xLocalWeights = new double[xStencilNnz];
    double * yLocalWeights = new double[yStencilNnz];
    int * xLocalInds; // This will just point to an existing array, doesn't need memory allocated!
    int * yLocalInds; // This will just point to an existing array, doesn't need memory allocated!
    int xInd;
    int yInd;
    double x;
    double y;

    // Get functions that compute x- and y-components of wavespeed given some dx or dy perturbation away from the current point
    std::function<double(int)> xLocalWaveSpeed;
    std::function<double(int)> yLocalWaveSpeed;    

    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {
        xInd = row % nx;                   // x-index of current point
        yInd = row / nx;                   // y-index of current point
        x    = MeshIndToPoint(xInd, xDim); // x-value of current point
        y    = MeshIndToPoint(yInd, yDim); // y-value of current point

        // Get functions that compute x- and y-components of wavespeed given some dx or dy perturbation away from the current point
        xLocalWaveSpeed = [this, x, dx, y, t, xDim](int xOffset) { return WaveSpeed(x + dx * xOffset, y, t, xDim); };
        yLocalWaveSpeed = [this, x, y, dy, t, yDim](int yOffset) { return WaveSpeed(x, y + dy * yOffset, t, yDim); };

        // Get stencil for discretizing x-derivative at current point 
        getLocalUpwindDiscretization(xLocalWeights, xLocalInds,
                                        xLocalWaveSpeed, 
                                        xPlusWeights, xPlusInds, 
                                        xMinusWeights, xMinusInds, 
                                        xStencilNnz);
        // Get stencil for discretizing y-derivative at current point 
        getLocalUpwindDiscretization(yLocalWeights, yLocalInds,
                                        yLocalWaveSpeed, 
                                        yPlusWeights, yPlusInds, 
                                        yMinusWeights, yMinusInds, 
                                        yStencilNnz);

        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, then continuing to loop over y
        // Actually, periodicity stuffs this up I think...
        for (int yNzInd = 0; yNzInd < yStencilNnz; yNzInd++) {
            
            // The two stencils will intersect somewhere at this y-point
            if (yLocalInds[yNzInd] == 0) {

                for (int xNzInd = 0; xNzInd < xStencilNnz; xNzInd++) {
                    // Account for periodicity here. This always puts resulting x-index in range 0,nx-1
                    L_colinds[dataInd] = ((xInd + xLocalInds[xNzInd] + nx) % nx) + yInd*nx; 
                    L_data[dataInd]    = xLocalWeights[xNzInd];

                    // The two stencils intersect at this point x-y-point, i.e. they share a 
                    // column in L, so add y-derivative information to x-derivative information that exists there
                    if (xLocalInds[xNzInd] == 0) {
                        L_data[dataInd] += yLocalWeights[yNzInd]; 
                    }
                    dataInd += 1;
                }

            // There is no possible intersection between between x- and y-stencils
            } else {
                // Account for periodicity here. This always puts resulting y-index in range 0,ny-1
                L_colinds[dataInd] = xInd + ((yInd + yLocalInds[yNzInd] + ny) % ny)*nx;
                L_data[dataInd]    = yLocalWeights[yNzInd];
                dataInd += 1;
            }
        }    
        
        // Set initial guess at the solution
        if (getU0) U0[rowcount] = GetInitialIterate(x, y, U0ID);

        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    } 
    
    // Check that sufficient data was allocated
    if (dataInd > L_nnz) {
        std::cout << "WARNING: FD spatial discretization matrix has more nonzeros than allocated.\n";
    }
    
    // Clean up
    delete[] xPlusInds;
    delete[] xPlusWeights;
    delete[] yPlusInds;
    delete[] yPlusWeights;
    delete[] xMinusInds;
    delete[] xMinusWeights;
    delete[] xLocalWeights;
    delete[] yLocalWeights;    
}




// // Get local CSR structure of FD spatial discretization matrix, L
// void FDadvection::get1DSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
//                                               int *&L_colinds, double *&L_data, double *&U0,
//                                               bool getU0, int &localMinRow, int &localMaxRow,
//                                               int &spatialDOFs, double t, int &bsize) 
// {
//     // Unpack variables frequently used
//     int nx          = m_nx[0];
//     double dx       = m_dx[0];
//     int xFD_Order   = m_order[0];
//     int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
//     int xDim        = 0;
// 
// 
//     /* ----------------------------------------------------------------------- */
//     /* ------ Initialize variables needed to compute CSR structure of L ------ */
//     /* ----------------------------------------------------------------------- */
//     localMinRow  = m_localMinRow;                    // First row on proc
//     localMaxRow  = m_localMinRow + m_onProcSize - 1; // Last row on proc
//     spatialDOFs  = m_spatialDOFs;
//     int L_nnz    = xStencilNnz * m_onProcSize;  // Nnz on proc
//     L_rowptr     = new int[m_onProcSize + 1];
//     L_colinds    = new int[L_nnz];
//     L_data       = new double[L_nnz];
//     int rowcount = 0;
//     int dataInd  = 0;
//     L_rowptr[0]  = 0;
//     if (getU0) U0 = new double[m_onProcSize]; // Initial guesss at solution    
// 
//     /* ---------------------------------------------------------------- */
//     /* ------ Get components required to approximate derivatives ------ */
//     /* ---------------------------------------------------------------- */
//     // Get stencils for upwind discretizations, wind blowing left to right
//     int * plusInds;
//     double * plusWeights;
//     get1DUpwindStencil(plusInds, plusWeights, xDim);
// 
// 
//     // Generate stencils for wind blowing right to left by reversing stencils
//     int * minusInds       = new int[xStencilNnz];
//     double * minusWeights = new double[xStencilNnz];
//     for (int i = 0; i < xStencilNnz; i++) {
//         minusInds[i]    = -plusInds[xFD_Order-i];
//         minusWeights[i] = -plusWeights[xFD_Order-i];
//     } 
// 
//     // Placeholder for weights and indices to discretize derivative at each point
//     double * localWeights = new double[xStencilNnz];
//     int * localInds; // This will just point to an existing array, doesn't need memory allocated!
//     double x;
// 
//     // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
//     std::function<double(int)> localWaveSpeed;    
// 
// 
//     /* ------------------------------------------------------------------- */
//     /* ------ Get CSR structure of L for all rows on this processor ------ */
//     /* ------------------------------------------------------------------- */
//     for (int row = localMinRow; row <= localMaxRow; row++) {
//         x = MeshIndToPoint(row, xDim); // Mesh point we're discretizing at 
// 
//         // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
//         localWaveSpeed = [this, x, dx, t](int offset) { return WaveSpeed(x + dx * offset, t); };
// 
//         // Get weights for discretizing spatial component at current point 
//         getLocalUpwindDiscretization(localWeights, localInds,
//                                         localWaveSpeed, 
//                                         plusWeights, plusInds, 
//                                         minusWeights, minusInds, 
//                                         xStencilNnz);
// 
// 
// 
// 
// 
//         for (int count = 0; count < xStencilNnz; count++) {
//             L_colinds[dataInd] = (localInds[count] + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
//             L_data[dataInd]    = localWeights[count];
//             dataInd += 1;
//         }        
// 
//         // Set initial guess at the solution
//         if (getU0) U0[rowcount] = 1.0; // TODO : Set this to a random value?    
// 
//         L_rowptr[rowcount+1] = dataInd;
//         rowcount += 1;
//     }    
// 
//     // Clean up
//     delete[] plusInds;
//     delete[] plusWeights;
//     delete[] minusInds;
//     delete[] minusWeights;
//     delete[] localWeights;
// }


// Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::get1DSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) 
{
    // Unpack variables frequently used
    int nx          = m_nx[0];
    double dx       = m_dx[0];
    int xFD_Order   = m_order[0];
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    
    
    int NnzPerRow = xStencilNnz; // Estimate of NNZ per row of L
    // Get stencil information for numerical dissipation term if there is one, so total NNZ estimate can be updated */
    int      dissNnz     = -1;
    int    * dissInds    = NULL;
    double * dissWeights = NULL;
    if (m_dissipation) {
        dissNnz = m_dissipation_params.degree + 1;
        Get1DDissipationStencil(dissInds, dissWeights, dissNnz);
        
        // kth-degree dissipation uses k/2 points in both directions
        // pth-order upwind uses floor[(p+2)/2] DOFs in upwind direction
        NnzPerRow = 2 * std::max( (xFD_Order + 2)/2, m_dissipation_params.degree/2 ) + 1; // So this is a bound on nnz of total stencil
    } 
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    localMinRow  = m_localMinRow;                    // First row on proc
    localMaxRow  = m_localMinRow + m_onProcSize - 1; // Last row on proc
    spatialDOFs  = m_spatialDOFs;
    int L_nnz    = NnzPerRow * m_onProcSize;  // Nnz on proc. This is a bound. Will always be slightly less than this for inflow/outflow boudaries
    L_rowptr     = new int[m_onProcSize + 1];
    L_colinds    = new int[L_nnz];
    L_data       = new double[L_nnz];
    int rowcount = 0;
    int dataInd  = 0;
    L_rowptr[0]  = 0;
    if (getU0) U0 = new double[m_onProcSize]; // Initial guesss at solution    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * plusInds;
    double * plusWeights;
    get1DUpwindStencil(plusInds, plusWeights, xDim);
    
    
    // Generate stencils for wind blowing right to left by reversing stencils
    int * minusInds       = new int[xStencilNnz];
    double * minusWeights = new double[xStencilNnz];
    for (int i = 0; i < xStencilNnz; i++) {
        minusInds[i]    = -plusInds[xFD_Order-i];
        minusWeights[i] = -plusWeights[xFD_Order-i];
    } 
    
    // Placeholder for weights and indices to discretize derivative at each point
    double * localWeights = new double[xStencilNnz];
    int * localInds; // This will just point to an existing array, doesn't need memory allocated!
    double x;
    int xInd;
    
    // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
    std::function<double(int)> localWaveSpeed;    
    
         
    // Different components of the domain for inflow/outflow boundaries
    double xIntLeftBndry  = MeshIndToPoint(m_order[0]/2 + 2, 0); // For x < this, stencil has some dependence on inflow
    double xIntRightBndry = MeshIndToPoint(m_nx[0] - div_ceil(m_order[0], 2) + 1, 0); // For x > this, stencil has some dependence on outflow ghost points
        
         
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {
        xInd = GlobalIndToMeshInd(row);    // Mesh index of point we're discretizing at
        x    = MeshIndToPoint(xInd, xDim);   // Value of point we're discretizing at

        // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
        localWaveSpeed = [this, x, dx, t](int offset) { return WaveSpeed(x + dx * offset, t); };
                
        // Get weights for discretizing spatial component at current point 
        getLocalUpwindDiscretization(localWeights, localInds,
                                        localWaveSpeed, 
                                        plusWeights, plusInds, 
                                        minusWeights, minusInds, 
                                        xStencilNnz);
                                        
        // Periodic BCs simply wrap stencil at both boundaries
        if (m_periodic) {
            if (!m_dissipation) {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = (localInds[count] + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }
            
            /* Add numerical dissipation term to advection term. 
                Note: It's easiest to just create new arrays and delete them everytime since the nnz 
                and structure of the summed stencil can change depending on the upwinding direction */
            } else {
                std::map<int, double> sum; // The summed/combined stencil
                Merge1DStencilsIntoMap(localInds, localWeights, xStencilNnz, dissInds, dissWeights, dissNnz, sum);
                std::map<int, double>::iterator it;
                for (it = sum.begin(); it != sum.end(); it++) {
                    L_colinds[dataInd] = (it->first + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
                    L_data[dataInd]    = it->second;
                    dataInd += 1;
                }
            }  
        
        // Inflow/outflow boundaries; need to adapt each boundary
        } else if (m_inflow) {
            // Ensure no numerical dissipation: Code set up to handle this... Inflow BCs hard enough on their own
            if (m_dissipation) {
                std::cout << "WARNING: Numerical dissipation not implemented for inflow/outlow BCs!" << '\n';
                MPI_Finalize();
                exit(1);
            }
            
            // DOFs stencil is influenced by inflow and potentially ghost points 
            if (x < xIntLeftBndry) {
                
                /* We simply use the stencil as normal but we truncating it to the interior points  
                with the remaining components are picked up in the solution-independent vector */
                for (int count = 0; count < xStencilNnz; count++) {
                    if (localInds[count] + row >= 0) { // Only allow dependencies on interior points
                        L_colinds[dataInd] = localInds[count] + row;
                        L_data[dataInd]    = localWeights[count];
                        dataInd += 1;
                    }
                } 
                
            // DOFs stencil is influenced by ghost points at outflow; need to modify stencil based on extrapolation
            } else if (x > xIntRightBndry) {

                // New stencil for discretization at outflow boundary
                int      xOutflowStencilNnz;
                int    * localOutflowInds;
                double * localOutflowWeights;
                GetOutflowDiscretization(xOutflowStencilNnz, localOutflowWeights, localOutflowInds, xStencilNnz, localWeights, localInds, 0, xInd); 
                
                // Add in stencil after potentially performing extrapolation
                for (int count = 0; count < xOutflowStencilNnz; count++) {
                    L_colinds[dataInd] = localOutflowInds[count] + row;
                    L_data[dataInd]    = localOutflowWeights[count];
                    dataInd += 1;
                }
                
                delete[] localOutflowInds;
                delete[] localOutflowWeights;
                
            // DOFs stencil only depends on interior DOFs (proceed as normal)
            } else {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = localInds[count] + row;
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }     
            }
        }     
    
        // Set initial guess at the solution
        if (getU0) U0[rowcount] = GetInitialIterate(x, U0ID);
        
        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
            
    }  
      
    
    // Clean up
    delete[] plusInds;
    delete[] plusWeights;
    delete[] minusInds;
    delete[] minusWeights;
    delete[] localWeights;
}


/* Merge two 1D stencils */
void FDadvection::Merge1DStencilsIntoMap(int * indsIn1, double * weightsIn1, int nnzIn1, 
                            int * indsIn2, double * weightsIn2, int nnzIn2,
                            std::map<int, double> &out) 
{
    // Insert first stencil into map
    for (int i = 0; i < nnzIn1; i++) {
        out[indsIn1[i]] = weightsIn1[i];
    }
    
    // Add second stencil into map
    for (int i = 0; i < nnzIn2; i++) {
        out[indsIn2[i]] += weightsIn2[i];
    }
}

// Update stencil at outflow boundary by performing extrapolation of the solution from the interior

// Hard-coded to assume that stencil of any point uses wind blowing from left to right.

// DOFInd is the index of x_{DOFInd}. This is the point we're discretizing at
void FDadvection::GetOutflowDiscretization(int &outflowStencilNnz, double * &localOutflowWeights, int * &localOutflowInds, 
                                    int stencilNnz, double * localWeights, int * localInds, int dim, int xInd) 
{
    
    int p = m_order[dim]; // Interpolation polynomial is of degree at most p-1 (interpolates p DOFs closest to boundary)
    outflowStencilNnz = p;
    
    localOutflowInds    = new int[outflowStencilNnz];
    localOutflowWeights = new double[outflowStencilNnz];
    
    std::map<int, double>::iterator it;
    std::map<int, double> entries; 
    
    // Populate dictionary with stencil information depending on interior DOFs
    int count = 0;
    for (int j = xInd - p/2 - 1; j <= m_nx[dim]; j++) {
        entries[localInds[count]] = localWeights[count];
        count += 1;
    }
    count -= 1; // Note that localInds[count] == connection to u_[nx]
    
    // Extrapolation leads to additional coupling to the p DOFs closest to the boundary
    for (int k = 0; k <= p-1; k++) {
        // Coefficient for u_{nx-p+1+k} connection from extrapolation
        double delta = 0.0;
        for (int j = 1; j <= xInd + div_ceil(p, 2) - 1 - m_nx[dim]; j++) {
            delta += localWeights[count + j] * LagrangeOutflowCoefficient(j, k, p);
        }
        // Add weighting for this DOF to interior stencil weights
        entries[ m_nx[dim] - xInd - p + 1 + k ] += delta;
    }
    
    // Copy data from dictionary into array to be returned.
    int dataInd = 0;
    for (it = entries.begin(); it != entries.end(); it++) {
        localOutflowInds[dataInd]    = it->first;
        localOutflowWeights[dataInd] = it->second;
        dataInd += 1;
    }
    
}


// Coefficients of DOFs arising from evaluating Lagrange polynomial at a ghost point
double FDadvection::LagrangeOutflowCoefficient(int i, int k, int p)
{
    double phi = 1.0;
    for (int ell = 0; ell <= p-1; ell++) {
        if (ell != k) {
            phi *= (i+p-1.0-ell)/(k-ell);
        }
    }
    return phi;
}





// Compute upwind weights to provide upwind discretization of linear flux function
// Note that localInds is just directed to point at the right set of indices
void FDadvection::getLocalUpwindDiscretization(double * &localWeights, int * &localInds,
                                    std::function<double(int)> localWaveSpeed,
                                    double * const &plusWeights, int * const &plusInds, 
                                    double * const &minusWeights, int * const &minusInds,
                                    int nWeights)
{    
    // Wave speed at point in question; the sign of this determines the upwind direction
    double waveSpeed0 = localWaveSpeed(0); 
    
    // Wind blows from minus to plus
    if (waveSpeed0 >= 0.0) {
        localInds = plusInds;
    
        // PDE is in conservation form: Need to discretize (wavespeed*u)_x
        if (m_conservativeForm) {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = localWaveSpeed(plusInds[ind]) * plusWeights[ind];
            }
    
        // PDE is in non-conservation form: Need to discretize wavespeed*u_x    
        } else {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = waveSpeed0 * plusWeights[ind];
            }
        }
    
    // Wind blows from plus to minus
    } else {        
        localInds = minusInds;
        
        // PDE is in conservation form: Need to discretize (wavespeed*u)_x
        if (m_conservativeForm) {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = localWaveSpeed(minusInds[ind]) * minusWeights[ind];
            }
    
        // PDE is in non-conservation form: Need to discretize wavespeed*u_x      
        } else {
            for (int ind = 0; ind < nWeights; ind++) {
                localWeights[ind] = waveSpeed0 * minusWeights[ind];
            }
        }
    }    
}


// Evaluate grid-function when grid is distributed on a single process
void FDadvection::GetGridFunction(void * GridFunction, 
                                    double * &B, 
                                    int &spatialDOFs)
{
    spatialDOFs = m_spatialDOFs;
    B = new double[m_spatialDOFs];
    
    // One spatial dimension
    if (m_dim == 1) {
        // Cast function to the correct format
        std::function<double(double)> GridFunction1D = *(std::function<double(double)> *) GridFunction;
        
        for (int xInd = 0; xInd < m_nx[0]; xInd++) {
            B[xInd] = GridFunction1D(MeshIndToPoint(GlobalIndToMeshInd(xInd), 0));
        }
        
    // Two spatial dimensions
    } else if (m_dim == 2) {
        // Cast function to the correct format
        std::function<double(double, double)> GridFunction2D = *(std::function<double(double, double)> *) GridFunction;
        
        int rowInd = 0;
        for (int yInd = 0; yInd < m_nx[1]; yInd++) {
            for (int xInd = 0; xInd < m_nx[0]; xInd++) {
                B[rowInd] = GridFunction2D(MeshIndToPoint(xInd, 0), MeshIndToPoint(yInd, 1));
                rowInd += 1;
            }
        }
    }
}


// Evaluate grid-function when grid is distributed across multiple processes
void FDadvection::GetGridFunction(void * GridFunction, 
                                    const MPI_Comm &globComm, 
                                    double * &B, 
                                    int &localMinRow, 
                                    int &localMaxRow, 
                                    int &spatialDOFs) 
{
    spatialDOFs  = m_spatialDOFs;
    localMinRow  = m_localMinRow;                    // First row on process
    localMaxRow  = m_localMinRow + m_onProcSize - 1; // Last row on process
    int rowcount = 0;
    B            = new double[m_onProcSize]; 

    // One spatial dimension
    if (m_dim == 1) {
        // Cast function to the correct format
        std::function<double(double)> GridFunction1D = *(std::function<double(double)> *) GridFunction;

        for (int row = localMinRow; row <= localMaxRow; row++) {
            B[rowcount] = GridFunction1D(MeshIndToPoint(GlobalIndToMeshInd(row), 0));
            rowcount += 1;
        }
        
    // Two spatial dimensions
    } else if  (m_dim == 2) {
        // Cast function to the correct format
        std::function<double(double, double)> GridFunction2D = *(std::function<double(double, double)> *) GridFunction;

        int xInd, yInd;      
        for (int row = localMinRow; row <= localMaxRow; row++) {
            xInd = m_pGridInd[0] * m_nxOnProcInt[0] + rowcount % m_nxOnProc[0]; // x-index of current point
            yInd = m_pGridInd[1] * m_nxOnProcInt[1] + rowcount / m_nxOnProc[0]; // y-index of current point
            B[rowcount] = GridFunction2D(MeshIndToPoint(xInd, 0), MeshIndToPoint(yInd, 1));
            rowcount += 1;
        }
    }
}


// Get PDE solution
bool FDadvection::GetExactPDESolution(const MPI_Comm &globComm, 
                                            double * &U, 
                                            int &localMinRow, 
                                            int &localMaxRow, 
                                            int &spatialDOFs, 
                                            double t)
{
    if (m_PDE_soln_implemented) {
        // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
        if (m_dim == 1) {
            std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Solution(x, t); };
            GetGridFunction((void *) &GridFunction, globComm, U, localMinRow, localMaxRow, spatialDOFs);
        } else if (m_dim == 2) {
            std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Solution(x, y, t); };
            GetGridFunction((void *) &GridFunction, globComm, U, localMinRow, localMaxRow, spatialDOFs);
        }  
        return true;
    } else {
        return false;
    }
}

bool FDadvection::GetExactPDESolution(double * &U, int &spatialDOFs, double t)
{
    if (m_PDE_soln_implemented) {
        // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
        if (m_dim == 1) {
            std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Solution(x, t); };
            GetGridFunction((void *) &GridFunction, U, spatialDOFs);
        } else if (m_dim == 2) {
            std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Solution(x, y, t); };
            GetGridFunction((void *) &GridFunction, U, spatialDOFs);
        }  
        return true;
    } else {
        return false;
    }
}

// Get solution-independent component of spatial discretization in vector  G
void FDadvection::getSpatialDiscretizationG(const MPI_Comm &globComm, 
                                            double * &G, 
                                            int &localMinRow, 
                                            int &localMaxRow, 
                                            int &spatialDOFs, 
                                            double t)
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Source(x, t); };
        GetGridFunction((void *) &GridFunction, globComm, G, localMinRow, localMaxRow, spatialDOFs);
        
        // Update G with inflow boundary information if necessary
        // All DOFs with coupling to inflow boundary are assumed to be on process 0 (there are very few of them)
        if (m_spatialRank == 0 && m_inflow) AppendInflowStencil1D(G, t);
        
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Source(x, y, t); };
        GetGridFunction((void *) &GridFunction, globComm, G, localMinRow, localMaxRow, spatialDOFs);
    }  
}


// Get solution-independent component of spatial discretization in vector  G
void FDadvection::getSpatialDiscretizationG(double * &G, int &spatialDOFs, double t)
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Source(x, t); };
        GetGridFunction((void *) &GridFunction, G, spatialDOFs);
        
        // Update G with inflow boundary information if necessary
        if (m_inflow) AppendInflowStencil1D(G, t);
        
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Source(x, y, t); };
        GetGridFunction((void *) &GridFunction, G, spatialDOFs);
    }  
}


/* Get the first p-1 derivatives of u at the inflow boundary at time t0

NOTES:
    -All derivatives are approximated via 2nd-order centred finite differences
    
    -Lots of terms here are explicitly written as functions of time so that higher-order
        derivatives that use them can perform numerical differentiation in time
*/
void FDadvection::GetInflowBoundaryDerivatives1D(double * &du, double t0)
{
    //int p = m_order[0]-1; // Hack for seeing if I really need all the derivatives...
    int p = m_order[0]; // Order of spatial discretization
    du    = new double[p]; 
    du[0] = InflowBoundary(t0); // The boundary itself. i.e, the 0th-derivative 
    
    /* ------------------- */
    /* --- Compute u_x --- */
    /* ------------------- */
    if (p >= 2) {
        double h = 1e-6;            // Spacing used in FD approximations of derivatives
        double x0 = m_boundary0[0]; // Coordinate of inflow boundary
        
        std::function<double(double)> s_x0 = [&, this](double t) { return PDE_Source(x0, t); };
        std::function<double(double)> a_x0 = [&, this](double t) { return WaveSpeed(x0, t); };
        
        std::function<double(double, double)> a = [this](double x, double t) { return WaveSpeed(x, t); };
        
        // x-derivative of wave speed evaluated at point x0 as a function of time
        std::function<double(double)> dadx_x0 = [&, this](double t) 
                { return GetCentralFDApprox([&, this](double x) { return a(x, t); }, x0, 1, h); };
        
        std::function<double(double)> z    = [this](double t) { return InflowBoundary(t); };
        
        // t-derivative of BC as a function of time
        std::function<double(double)> dzdt = [&, this](double t) { return GetCentralFDApprox(z, t, 1, h); };
        
        // x-derivative of u as a function of time
        std::function<double(double)> dudx;
        
        // Get u_x(x0,t) as a function of time depending on form of PDE
        if (m_conservativeForm) {
            dudx = [&, this](double t) { return (s_x0(t) - dadx_x0(t)*z(t) - dzdt(t))/a_x0(t); };
        } else {
            dudx = [&, this](double t) { return (s_x0(t) - dzdt(t))/a_x0(t); };
        }
        
        // Evaluate u_x(x0,t) at time t0
        du[1] = dudx(t0);
        
        /* -------------------- */
        /* --- Compute u_xx --- */
        /* -------------------- */
        if (p >= 3) {
            std::function<double(double, double)> s = [this](double x, double t) { return PDE_Source(x, t); };
            // x-derivative of source evaluated at point x0 as a function of time
            std::function<double(double)> dsdx_x0 = [&, this](double t) 
                    { return GetCentralFDApprox([&, this](double x) { return s(x, t); }, x0, 1, h); };
            
            // xx-derivative of wave speed evaluated at point x0 as a function of time
            std::function<double(double)> d2adx2_x0 = [&, this](double t) 
                    { return GetCentralFDApprox([&, this](double x) { return a(x, t); }, x0, 2, h); };
            
            // x-derivative of reciprocal of wave speed evaluated at point x0 as a function of time
            std::function<double(double)> dradx_x0 = [&, this](double t) 
                    { return GetCentralFDApprox([&, this](double x) { return 1.0/a(x, t); }, x0, 1, h); };
            
            // xt-derivative of u as a function of time
            std::function<double(double)> d2udxdt = [&, this](double t) { return GetCentralFDApprox(dudx, t, 1, h); };
            
            // xx-derivative of u as a function of time
            std::function<double(double)> d2udx2;
            
            // Get u_xx(x0,t) as a function of time depending on form of PDE
            if (m_conservativeForm) {
                d2udx2 = [&, this](double t) { return (dsdx_x0(t) - d2adx2_x0(t)*z(t) - dadx_x0(t)*dudx(t) - d2udxdt(t))/a_x0(t) 
                                                + dradx_x0(t)*(s_x0(t) - dadx_x0(t)*z(t) - dzdt(t)); };
            } else {
                d2udx2 = [&, this](double t) { return (dsdx_x0(t) - d2udxdt(t))/a_x0(t) 
                                                + dradx_x0(t)*(s_x0(t) - dzdt(t)); };
            }
            
            // Evaluate u_xx(x0,t) at time t0
            du[2] = d2udx2(t0);
            //du[0] = 0.0;
            
            if (p >= 4)  {
                std::cout << "WARNING: Inflow derivatives only implemented up to degree 2\n";
                MPI_Finalize();
                exit(1);
            }
        }
    }
}

// Return a central approximation of order-th derivative of f centred at x0
double FDadvection::GetCentralFDApprox(std::function<double(double)> f, double x0, int order, double h) {
    
    // Just use 2nd-order approximations
    if (order == 1) {
        return (- 0.5*f(x0-h) + 0.5*f(x0+h))/h;
    } else if (order == 2) {
        return (f(x0-h) -2*f(x0) + f(x0+h))/(h*h);
    } else if (order == 2) {
        return (-0.5*f(x0-2*h) + f(x0-h) - f(x0+h) + 0.5*f(x0+2*h))/(h*h*h);
    } else {
        std::cout << "WARNING: FD approximations for derivative of order " << order << " not implemented!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
}


// Get values at inflow boundary and ghost points associated with it using inverse Lax--Wendroff procedure
void FDadvection::GetInflowValues(std::map<int, double> &uGhost, double t, int dim) 
{
    
    double * du;
    GetInflowBoundaryDerivatives1D(du, t);
    uGhost[0] = du[0]; // The inflow boundary value itself
    
    // Approximate solution at p/2 ghost points using Taylor series based at inflow
    for (int i = -1; i >= -m_order[dim]/2; i--) {
        uGhost[i] = du[0]; // 0th-order derivative
        for (int k = 1; k <= m_order[dim]-1; k++) { // True value??
        //for (int k = 1; k <= m_order[dim]-2; k++) { // Works almost identically???
            uGhost[i] += pow(i*m_dx[dim], k) / factorial(k) * du[k];
        }
    }
}


// Update solution-independent term discretization information pertaining to inflow boundary
// Hard-coded to assume that wind blows left to right for these points near the boundary...
void FDadvection::AppendInflowStencil1D(double * &G, double t) {
    
    // Unpack variables frequently used
    int nx          = m_nx[0];
    double dx       = m_dx[0];
    int xFD_Order   = m_order[0];
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    
    
    /* --- Get solution at inflow and ghost points --- */
    std::map<int, double> uGhost; // Use dictionary so we can access data via its physical grid index    
    GetInflowValues(uGhost, t, xDim);
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // Get stencils for upwind discretizations, wind blowing left to right
    int * plusInds;
    double * plusWeights;
    get1DUpwindStencil(plusInds, plusWeights, xDim);

    // Generate stencils for wind blowing right to left by reversing stencils
    // NOTE: We shouldn't need these here since wind is assumed to blow left to right at all these points near
    // the boundary, but we need these arrays for the implementation that gets the stencil 
    int * minusInds       = NULL;
    double * minusWeights = NULL;
    
    // Placeholder for weights and indices to discretize derivative at each point
    double * localWeights = new double[xStencilNnz];
    int * localInds; // This will just point to an existing array, doesn't need memory allocated!
    int xInd;
    double x;
    
    
    // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
    std::function<double(int)> localWaveSpeed;  
    
    // There are p/2+1 DOFs whose stencil depends on inflow and potentially ghost points
    for (int row = 0; row <= m_order[0]/2; row++) {
        
        xInd = GlobalIndToMeshInd(row);
            
        // Value of grid point we're discretizing at
        x = MeshIndToPoint(xInd, xDim); 
        
        // Get function, which given an integer offset, computes wavespeed(x + dx * offset, t)
        localWaveSpeed = [this, x, dx, t](int offset) { return WaveSpeed(x + dx * offset, t); };
                
        // Get weights for discretizing spatial component at current point 
        getLocalUpwindDiscretization(localWeights, localInds,
                                        localWaveSpeed, plusWeights, plusInds, 
                                        minusWeights, minusInds, xStencilNnz);
            
        // Loop over entries in stencil, adding couplings to boundary point or ghost points                            
        // TODO: Why do I have to subtract and not add here??
        for (int count = 0; count < xStencilNnz; count++) {
            if (xInd + localInds[count] <= 0) G[row] -= localWeights[count] * uGhost[xInd + localInds[count]]; 
        }                                 
    }
}


// Allocate vector U0 memory and populate it with initial condition.
void FDadvection::getInitialCondition(const MPI_Comm &globComm, 
                                        double * &U0, 
                                        int &localMinRow, 
                                        int &localMaxRow, 
                                        int &spatialDOFs) 
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this](double x) { return InitCond(x); };
        GetGridFunction((void *) &GridFunction, globComm, U0, localMinRow, localMaxRow, spatialDOFs);
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this](double x, double y) { return InitCond(x, y); };
        GetGridFunction((void *) &GridFunction, globComm, U0, localMinRow, localMaxRow, spatialDOFs);
    }   
}


// Allocate vector U0 memory and populate it with initial condition.
void FDadvection::getInitialCondition(double * &U0, int &spatialDOFs)
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this](double x) { return InitCond(x); };
        GetGridFunction((void *) &GridFunction, U0, spatialDOFs);
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this](double x, double y) { return InitCond(x, y); };
        GetGridFunction((void *) &GridFunction, U0, spatialDOFs);
    }  
}


/* Stencils for centred discretizations of diffusion operator, d^degree/dx^degree. degree \in {2,4} 

NOTES:
    degree == 2 uses a 2nd-order FD stencil
    degree == 4 uses a 2nd-order FD stencil
    assumes mesh spacing is same in all grid dimensions
    Uses centred differences
*/
void FDadvection::Get1DDissipationStencil(int * &inds, double * &weights, int &nnz)
{
    // Using a 2nd-order difference means the stencil will use degree+1 nodes
    nnz     = m_dissipation_params.degree + 1;
    inds    = new int[nnz];
    weights = new double[nnz];
    
    if (m_dissipation_params.degree == 2) {
        inds[0] = -1;
        inds[1] = +0;
        inds[2] = +1;
        weights[0] = +1.0;
        weights[1] = -2.0;
        weights[2] = +1.0;
        
    }  else if (m_dissipation_params.degree == 4) {
        inds[0] = -2;
        inds[1] = -1;
        inds[2] = +0;
        inds[3] = +1;
        inds[4] = +2;
        weights[0] = +1.0;
        weights[1] = -4.0;
        weights[2] = +6.0;
        weights[3] = -4.0;
        weights[4] = +1.0;
        
    } else {
        std::cout << "WARNING: FD-advection numerical dissipation must be of degree 2 or 4" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    // Mesh-dependent weighting of dissipation (c0*dx^c1) \times FD coefficient (dx^{-degree})
    // Note the -1 here means that c0 > 0 corresponds to dissipation, while c0 < 0 means soln will blow up
    double c = -1.0 * m_dissipation_params.c0 * pow(m_dx[0], 1.0*m_dissipation_params.c1) * pow(m_dx[0], -1.0*m_dissipation_params.degree);
    
    /* Scale discretization weights */
    for (int i = 0; i < nnz; i++) {
        weights[i] *= c;
    }
}


// Stencils for upwind discretizations of d/dx. Wind is assumed to blow left to right. 
void FDadvection::get1DUpwindStencil(int * &inds, double * &weights, int dim)
{    
    // Just check that there are sufficiently many DOFs to discretize derivative
    if (m_nx[dim] < m_order[dim] + 1) {
        std::cout << "WARNING: FD stencil requires more grid points than are on grid! Increase nx!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    
    inds    = new int[m_order[dim]+1];
    weights = new double[m_order[dim]+1];
    
    if (m_order[dim] ==  1) 
    {
        inds[0] = -1;
        inds[1] =  0;
        weights[0] = -1.0;
        weights[1] =  1.0;
    }
    else if (m_order[dim] == 2) 
    {
        inds[0] = -2;
        inds[1] = -1;
        inds[2] =  0;
        weights[0] =  1.0/2.0;
        weights[1] = -4.0/2.0;
        weights[2] =  3.0/2.0;
    }
    else if (m_order[dim] == 3) 
    {
        inds[0] = -2;
        inds[1] = -1;
        inds[2] =  0;
        inds[3] =  1;
        weights[0] =  1.0/6.0;
        weights[1] = -6.0/6.0;
        weights[2] =  3.0/6.0;
        weights[3] =  2.0/6.0;
    }
    else if (m_order[dim] == 4) 
    {
        inds[0] = -3;
        inds[1] = -2;
        inds[2] = -1;
        inds[3] =  0;
        inds[4] =  1;
        weights[0] = -1.0/12.0;
        weights[1] =  6.0/12.0;
        weights[2] = -18.0/12.0;
        weights[3] =  10.0/12.0;
        weights[4] =  3.0/12.0;
    }
    else if (m_order[dim] == 5) 
    {    
        inds[0] = -3;
        inds[1] = -2;
        inds[2] = -1;
        inds[3] =  0;
        inds[4] =  1;
        inds[5] =  2;
        weights[0] = -2.0/60.0;
        weights[1] =  15.0/60.0;
        weights[2] = -60.0/60.0;
        weights[3] =  20.0/60.0;
        weights[4] =  30.0/60.0;
        weights[5] = -3.0/60.0;
    } 
    else 
    {
        std::cout << "WARNING: invalid choice of spatial discretization. Upwind discretizations of orders 1--5 only implemented.\n";
        MPI_Finalize();
        exit(1);
    }
    
    for (int i = 0; i < m_order[dim]+1; i++) {
        weights[i] /= m_dx[dim];
    }
}


/* -------------------------------------------------------------------------- */
/* ----- Some utility functions that may be helpful for derived classes ----- */
/* -------------------------------------------------------------------------- */

/** Get parallel (square) matrix A from its local CSR data
NOTES: HypreParMatrix makes copies of the data, so it can be deleted */
void FDadvection::GetHypreParMatrixFromCSRData(MPI_Comm comm,  
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
void FDadvection::GetHypreParVectorFromData(MPI_Comm comm, 
                                            int localMinRow, int localMaxRow, HYPRE_Int globalNumRows, 
                                            double * x_data, HypreParVector * &x)
{
    int rows[2] = {localMinRow, localMaxRow+1}; 
    x = new HypreParVector(comm, globalNumRows, x_data, rows);
}

/* Get identity matrix that's compatible with A */
void FDadvection::GetHypreParIdentityMatrix(const HypreParMatrix &A, HypreParMatrix * &I) 
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

/* Get identity mass matrix operator */
void FDadvection::GetIRKOperatorM(HypreParMatrix * &M) {
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

// void FDadvection::Test(double t) {
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




