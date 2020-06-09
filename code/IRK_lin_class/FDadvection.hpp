#ifndef IRK_H
    #include "IRK.hpp"
#endif
// 
// #ifndef SPATIALDISCRETIZATION_H
// 
// #endif

//#include "IRK.hpp"

#include "mfem.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <map>

#define PI 3.14159265358979323846


enum FDBias {
    CENTRAL = 0, UPWIND = 1
};

// class FDStencil {
// public:
//     int size;
//     int * nodes;
//     double * weights;
//     FDStencil() : size(0), nodes(NULL), weights(NULL) {};    
//     ~FDStencil() {
//         if (nodes) delete nodes; nodes = NULL;
//         if (weights) delete weights; weights = NULL;
//     }
// };


// TODO: I want to have the stencil data be read only... But not really sure how to do it...
/* Abstract class defining FD approximations of constant and variable-coefficient differential operators */
class FDApprox
{
private:
    int m_derivative;   /* Degree of derivative to be approximated */
    int m_order;        /* Order of FD approximation */
    double m_dx;        /* Mesh width used in approximation */
    
    FDBias m_bias;      /* Bias of FD stencil */
    
    bool m_c_set;       /* Has coefficient of operator been set? */
    bool m_c_current;   /* Do member stencils use current coefficient? */
    bool m_variable;    /* Is operator constant- or variable-coefficient? */
    double m_constant_c;/* Coefficient of constant-coefficient operator */
    std::function<double(double)> m_variable_c; /* Variable coefficient of variable-coefficient operator */
    bool m_conservative;/* Is variable coefficient in conservative or non-conservative form? */
    
    double m_x;         /* Point FD approximation is being applied at */
    bool m_x_current;   /* Do member stencils use current x? */
    
    int m_size; /* Number of entries in stencils. NOTE: Cannot change over lifetime of object. */
    
    int * m_nodes, * m_plusNodes, * m_minusNodes;
    double * m_weights, * m_localWeights, * m_plusWeights, * m_minusWeights;
    
    bool m_delete_nodes;
    
    /* Stencils for upwind discretizations of D1 == d/dx for stencil biased to the left. */
    void GetD1UpwindPlus(int * &nodes, double * &weights) const;
    
    /* Stencils for upwind discretizations of D1 == d/dx for stencil biased to the right. */
    void GetD1UpwindMinus(int * &nodes, double * &weights) const;
    
    /* Stencils for upwind discretizations of variable-coefficient D1 == D1(x). */
    void GetVariableD1Upwind(int * &localNodes, double * &localWeights) const;
    
    /* Stencils for central discretizations of D1 == d/dx. */
    void GetD1Central(int * &inds, double * &weights) const;
    
    /* Stencils for central discretizations variable-coefficient of D1 == D1(x). */
    void GetVariableD1Central(double * &localWeights) const;
                                     
    /* Stencils for central discretizations of D2 == d^2/dx^2. */
    void GetD2Central(int * &nodes, double * &weights) const;
    
    /* Stencils for central discretizations variable-coefficient of D2 == D2(x). */
    void GetVariableD2Central(double * &localWeights) const;
    
public:    
    
    /// Operators
    FDApprox(int derivative, int order, double dx); // Bias is N/A
    FDApprox(int derivative, int order, double dx, FDBias bias); // Bias applies                                             
    
    ~FDApprox();
    
    // Set location of point to be discretized for variable-coefficient operator
    inline void SetX(double x) { m_x = x; m_x_current = false; };
    // Get location of point to be discretized for variable-coefficient operator
    inline double GetX(double x) const { return m_x; } ;
    
    // Set constant coefficient
    inline void SetCoefficient(double c) {
        m_constant_c = c;
        m_variable = false;
        m_c_set = true;
        m_c_current = false;
        m_x_current = true; // Discretization doesn't depend on x
    } 
    
    // Set variable coefficient and its type
    inline void SetCoefficient(std::function<double(double)> const &c, bool conservative) { 
        m_variable_c = c;
        m_conservative = conservative;
        m_variable = true;
        m_c_set = true;
        m_c_current = false;
     }
     
    // Set variable coefficient
    inline void SetCoefficient(std::function<double(double)> const &c) { 
        SetCoefficient(c, m_conservative);
    }

    // Set type of variable coefficient
    inline void SetVarCoefficientType(bool conservative) {
        if (m_conservative != conservative) {
            m_c_current = false;
            m_conservative = conservative;
        }
    }
    
    // Size of stencil (i.e., nodes/weights arrays)
    inline int GetSize() const { return m_size; };
    
    void GetApprox(int * &nodes, double * &weights);

};



/* Abstract class defining an equidistant Cartesian mesh */
class FDMesh
{
private:
    
    friend class FDadvection;
    
    /// --- Characteristic properties of mesh --- ///
    int m_dim;                /* Number of dimensions */
    std::vector<int>    m_nx; /* Number of points in each dimension */
    std::vector<double> m_dx; /* Mesh spacing in each dimension */
    int m_refLevels;          /* 2^refLevels points in each dimension */
    int m_nxTotal;            /* Total number of points within mesh */
    std::vector<double> m_boundary0; /* Lower boundary in each direction */
    
    /// --- MPI info --- ///
    MPI_Comm m_comm; /* communicator grid is distributed on */ 
    int m_rank;      /* Rank of each proc */
    int m_commSize;  /* Number of procs */
    bool m_parallel; /* Is grid distributed across multiple procs? */
    
    /// --- Parallel distribution of mesh --- ///
    int m_globOffset;                  /* Global index of first point on process */
    
    std::vector<int>    m_np;                   /* Number of procs in each dimension */
    std::vector<int>    m_pIdx;             /* Grid indices of proc in each dimension */
    std::vector<int>    m_nxOnProc;             /* Number of points on proc in each dimension */
    int                 m_nxOnProcTotal;        /* Total number of points on proc */
    
    std::vector<int>    m_nxOnProcInt;          /* Number of DOFs in each direction on procs in INTERIOR of proc domain */
    std::vector<int>    m_nxOnProcBnd;          /* Number of DOFs in each direction on procs on BOUNDARY of proc domain */
    

    std::vector<int>    m_nborGlobOffset; /* Global index of first DOF owned by neighbouring procs */
    std::vector<int>    m_nborNxOnProc;   /* Number of DOFs in each direction owned by neighbouring procs */


public:
    FDMesh(MPI_Comm comm, int dim = 1, int refLevels = 5, std::vector<int> np = {});
    //~FDMesh();

    double MeshIndToPoint(int meshInd, int dim) const;
    

    /* Get mesh info */
    double Get_dx(int dim = 0) const { return m_dx[dim]; };
    int    Get_nx(int dim = 0) const { return m_nx[dim]; };
    int    Get_dim() const { return m_dim; };
};


/* I provide finite-difference discretizations to advection PDEs. Two forms of advection PDE are supported:
      u_t + \grad_x (wavespeed(x,t)*u)   = source(x,t)      (CONSERVATIVE form)
      u_t + wavespeed(x,t) \cdot \grad u = source(x,t)      (NON-CONSERVATIVE form)

Several test problems implemented, as follows:
        u_t + u_x = 0,                             problemID == 1
        u_t + \nabla \cdot (a(x,t)u)_x = s2(x,t),  problemID == 2
        u_t + a(x,t) \cdot \nabla u    = s3(x,t),  problemID == 3
    where the wave speeds and sources can be found in the main file... Note both 1D and 2D in space versions are implememted.  
    
    
NOTES:
    -In higher dimensions, the finite-differences are done in a dimension by dimension fashion    
    
    -The wavespeed is an m-dimensional vector function whose m components are each functions 
        of the m spatial variables and time
    
    -Only periodic boundary conditions are implemented. This means the wavespeed has to be periodic in space!
        Non-periodic is too nasty.
    
    -In 2D: If a non-square number of processors is to be used, the user 
        must(!) specify the number of processors in each of the x- and y-directions
*/


/* Parameters for adding numerical dissipation into advection discretization */
struct Num_dissipation {
    int degree; /* Degree of dissipation/derivative. Can be 2 or 4 */
    
    /* Dissipation coefficient takes the form c0*dx^c1. c0 > 0 corresponds to 
        dissipation while c0 < 0 corresponds to anti-dissipation (growth!) */
    double c0;
    int    c1;
};


// {1.5, "", "FA", 0.1, 0.01, 0.0, 100, 10, 0.e-4, 6}; // w/ on-proc relaxation
// {1.5, "", "FFC", 0.1, 0.01, 0.0, 100, 0, 0.e-4, 6};    // w/ Jacobi relaxation
struct AMG_parameters {
   double distance = 1.5;
   std::string prerelax = "";
   std::string postrelax = "FFC";
   double strength_tolC = 0.1;
   double strength_tolR = 0.01;
   double filter_tolR = 0.0;
   int interp_type = 100;
   int relax_type = 0;
   double filterA_tol = 0.e-4;
   int coarsening = 6;
   int maxiter = 1;
};

/* information used to assemble a matrix A == gamma*I - dt*L */
struct A_parameters {
    double gamma;
    double dt;     
    int index; // There is potentially a list of different A
};

class FDadvection : public IRKOperator
{
private:
    // TODO: Do we want the ability to set AMG parameters for every linear system? 
    // E.g., maybe it's more important they be robust for type 2 with small eta/beta?
    
    // Preconditioners for matrices of the form A == gamma*I - dt*L
    mfem::Array<HypreBoomerAMG *> m_A_precs; // Preconditioner for each A
    mfem::Array<A_parameters>     m_A_info;  // Information about each A
    int m_A_idx;                             // Index of current A
    AMG_parameters m_AMG_params_type1;       // AMG parameters for type 1 preconditioners
    AMG_parameters m_AMG_params_type2;       // AMG parameters for type 2 preconditioners
    
    
    const FDMesh &m_mesh;
    
    
    bool m_conservativeForm;                    /* TRUE == PDE in conservative form; FALSE == PDE in non-conservative form */
    bool m_periodic;                            /* Periodic boundaries */
    bool m_inflow;                              /* Inflow/outflow boundaries */
    
    int m_dim;                                  /* Number of spatial dimensions */
    int m_problemID;                            /* ID for test problems */
    
    int m_refLevels;                            /* Have nx == 2^(refLevels + 2) spatial DOFs */
    int m_onProcSize;                           /* Number of DOFs on proc */
    int m_spatialDOFs;                          /* Total number of DOFs in spatial disc */
    int m_localMinRow;                          /* Global index of first DOF on proc */
    
    bool m_PDE_soln_implemented;                /* Exact solution of PDE is implemented */
    int m_order;                /* Order of discretization in each direction */
    
    
    
    
    bool                m_dissipation;          /* Numerical dissipation added to advection terms */
    Num_dissipation     m_dissipation_params;   /* Parameters describing numerical dissipation */

    /* Components of the spatial discretization */
    HypreParMatrix * m_I;  /* Compatible identity matrix */
    HypreParMatrix * m_L;  /* Linear solution-dependent operator */
    
    // double  m_t_L; /* Time the current (linear) solution-dependent operator is evaluated at */
    // double  m_t_g; /* Time the current solution-independent source term is evaluated at */
    // double  m_t_u; /* Time the current solution is evaluated at */
    // 
    /* Space discretization is assumed fully time-dependent; must be over ridden in derived class if this is not the case */
    bool    m_L_isTimedependent;    /* Is L time dependent? */
    bool    m_G_isTimedependent;    /* Is g time dependent? */
    
    bool     m_parallel;  /* Hmm...  DO we need this?? */
    MPI_Comm m_comm;         /* Spatial communicator; the spatial discretization code has access to this */
    int      m_commSize;     /* Num processes in spatial communicator */
    int      m_rank;         /* Process rank in spatial communicator */    

    


    // Call when using spatial parallelism                          
    void GetSpatialDiscretizationG(const MPI_Comm &globComm, double* &G, 
                                    int &localMinRow, int &localMaxRow, int &spatialDOFs, double t) const;                               
    void GetSpatialDiscretizationL(const MPI_Comm &globComm, int* &L_rowptr, 
                                    int* &L_colinds, double* &L_data,
                                    double* &U0, bool getU0, int U0ID,
                                    int &localMinRow, int &localMaxRow, int &spatialDOFs,
                                    double t, int &bsize) const;                                            
    
    // Call when NOT using spatial parallelism                                        
    void GetSpatialDiscretizationG(double* &G, int &spatialDOFs, double t) const; 
    void GetSpatialDiscretizationL(int* &L_rowptr, int* &L_colinds, double* &L_data,
                                    double* &U0, bool getU0, int U0ID, 
                                    int &spatialDOFs, double t, int &bsize) const;                                            
                                         
    /* Uses spatial parallelism */                                
    void Get2DSpatialDiscretizationL(const MPI_Comm &globComm, 
                                        int *&L_rowptr, int *&L_colinds, double *&L_data, 
                                        double * &U0, bool getU0, int U0ID,
                                        int &localMinRow, int &localMaxRow, int &spatialDOFs, 
                                        double t, int &bsize) const;
    void Get1DSpatialDiscretizationL(const MPI_Comm &globComm, 
                                        int *&L_rowptr, int *&L_colinds, double *&L_data, 
                                        double * &U0, bool getU0, int U0ID,
                                        int &localMinRow, int &localMaxRow, int &spatialDOFs, 
                                        double t, int &bsize) const;                                
                                    
    /* No spatial parallelism */
    void Get2DSpatialDiscretizationL(int *&L_rowptr, int *&L_colinds, double *&L_data, 
                                        double *&U0, bool getU0, int U0ID,
                                        int &spatialDOFs, double t, int &bsize) const;
                                
    /* Uses spatial parallelism */  
    void GetInitialCondition(const MPI_Comm &globComm, 
                                   double * &U0, 
                                   int      &localMinRow, 
                                   int      &localMaxRow, 
                                   int      &spatialDOFs) const;
                                
    void GetInitialCondition(double * &U0, 
                             int      &spatialDOFs) const;
    
    bool GetExactPDESolution(const MPI_Comm &globComm, 
                                                double * &U, 
                                                int &localMinRow, 
                                                int &localMaxRow, 
                                                int &spatialDOFs, 
                                                double t) const;
    
                             
    bool GetExactPDESolution(double * &U0, 
                            int      &spatialDOFs, double t) const;

    void GetGridFunction(void   *  GridFunction, 
                         double * &B, 
                         int      &spatialDOFs) const;

    void GetGridFunction(      void     *  GridFunction, 
                         const MPI_Comm   &globComm, 
                               double   * &B, 
                               int        &localMinRow, 
                               int        &localMaxRow, 
                               int        &spatialDOFs) const;


    
    
    
    
    void Get1DDissipationStencil(int * &inds, double *&weights, int &nnz) const; 
    
    /* Initial iterate for iterative solver */
    double GetInitialIterate(double x, int U0ID) const;       
    double GetInitialIterate(double x, double y, int U0ID) const;
     
    double InitCond(double x) const;                          /* 1D initial condition */
    double InitCond(double x, double y) const;                /* 2D initial condition */
    double WaveSpeed(double x, double t) const;               /* 1D wave speed */
    double WaveSpeed(double x, double y, double t,      /* 2D wave speed */
                        int component) const;  
    double PDE_Source(double x, double t) const;              /* 1D source */
    double PDE_Source(double x, double y, double t) const;    /* 2D source */

    double PDE_Solution(double x, double t) const;
    double PDE_Solution(double x, double y, double t) const;

    double LagrangeOutflowCoefficient(int i, int k, int p) const;
    double InflowBoundary(double t) const;
    void AppendInflowStencil1D(double * &G, double t) const;
    void GetOutflowDiscretization(int &outflowStencilNnz, double * &localOutflowWeights, 
                                    int * &localOutflowInds, int stencilNnz, double * localWeights, 
                                    int * localInds, int dim, int DOFInd) const; 

    void GetInflowBoundaryDerivatives1D(double * &du, double t) const;
    void GetInflowValues(std::map<int, double> &uGhost, double t, int dim) const;
    double GetCentralFDApprox(std::function<double(double)> f, double x0, int order, double h) const;

    int GlobalIndToMeshInd(int globInd) const;
    
    /* Utility-type functions */
    void Merge1DStencilsIntoMap(int * indsIn1, double * weightsIn1, int nnzIn1, 
                                int * indsIn2, double * weightsIn2, int nnzIn2,
                                std::map<int, double> &out) const;
                                   
    void NegateData(int start, int stop, double * &data) const;
    
    
    
    /* -------------------------------------------------------------------------- */
    /* ----- Some utility functions that may be helpful for derived classes ----- */
    /* -------------------------------------------------------------------------- */
    void GetHypreParMatrixFromCSRData(MPI_Comm comm,  
                                        int localMinRow, int localMaxRow, int globalNumRows, 
                                        int * A_rowptr, int * A_colinds, double * A_data,
                                        HypreParMatrix * &A) const; 

    void GetHypreParVectorFromData(MPI_Comm comm, 
                                    int localMinRow, int localMaxRow, int globalNumRows, 
                                    double * x_data, HypreParVector * &x) const;

    void GetHypreParIdentityMatrix(const HypreParMatrix &A, HypreParMatrix * &I) const;


    void GetG(double t, HypreParVector * &g) const;

    void GetHypreParMatrixL(double t, HypreParMatrix * &L) const;
    void GetHypreParMatrixI(HypreParMatrix * &I) const;
    void SetI(); // Set identity, m_I 
    /* -------------------------------------------------------------------------- */

                    
public:
    
    /* Constructors */
    FDadvection(MPI_Comm globComm, const FDMesh &mesh, int order, int problemID);
    ~FDadvection();
    
    /* Add numerical dissipation into pure advection discretization  */
    void SetNumDissipation(Num_dissipation dissipation_params);
    
    // Set member variables
    void SetL(double t); // Set spatial disc. matrix, m_L 
    
    /* Get initial conditon */
    void GetU0(HypreParVector * &u0) const;
    
    /* Get exact solution (if available) */
    bool GetUExact(double t, HypreParVector * &u) const;
    
    /* --- Virtual functions from base class requiring implementation --- */
    /* Compute y <- L*x + g(t) */
    void Mult(const Vector &x, Vector &y) const;
    
    /* Compute y <- L*x */
    void ApplyL(const Vector &x, Vector &y) const;
    
    /* Precondition A == (\gamma*I - dt*L) */
    void ImplicitPrec(const Vector &x, Vector &y) const;

    // Function to ensure that ImplicitPrec preconditions A == (\gamma*I - dt*L)
    // with gamma and dt as passed to this function.
    //      + index -> index of eigenvalue (pair) in IRK storage
    //      + type -> eigenvalue type, 1 = real, 2 = complex pair
    //      + t -> time.
    // These additional parameters are to provide ways to track when
    // A == (\gamma*I - dt*L) must be reconstructed or not to minimize setup.
    void SetSystem(int index, double t, double dt, double gamma, int type);
    
    /* Set member variables holding parameters for AMG solve. 
    Pass type == 0 to set both type 1 and 2 with same parameters */
    void SetAMG_parameters(AMG_parameters parameters, int type = 0);
};


