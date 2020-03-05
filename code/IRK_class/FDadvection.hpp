// #ifndef IRK_H
//     #include "IRK.hpp"
// #endif
// 
// #ifndef SPATIALDISCRETIZATION_H
// 
// #endif

#include "SpatialDiscretization.hpp"

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


class FDadvection : public SpatialDiscretization
{
private:
    
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
    std::vector<int>    m_order;                /* Order of discretization in each direction */
    std::vector<int>    m_nx;                   /* Number of DOFs in each direction */
    std::vector<double> m_dx;                   /* Mesh spacing in each direction */
    std::vector<double> m_boundary0;            /* Lower boundary of domain in each direction. I.e., WEST (1D, 2D, 3D), SOUTH (2D, 3D), DOWN (3D) */
    std::vector<int>    m_px;                   /* Number of procs in each grid direction */
    std::vector<int>    m_pGridInd;             /* Grid indices of proc in each direction */
    std::vector<int>    m_nxOnProc;             /* Number of DOFs in each direction on proc */
    std::vector<int>    m_nxOnProcInt;          /* Number of DOFs in each direction on procs in INTERIOR of proc domain */
    std::vector<int>    m_nxOnProcBnd;          /* Number of DOFs in each direction on procs on BOUNDARY of proc domain */
    std::vector<int>    m_neighboursLocalMinRow;/* Global index of first DOF owned by neighbouring procs */
    std::vector<int>    m_neighboursNxOnProc;   /* Number of DOFs in each direction owned by neighbouring procs */
    
    bool                m_dissipation;          /* Numerical dissipation added to advection terms */
    Num_dissipation     m_dissipation_params;   /* Parameters describing numerical dissipation */


    // /* Some hacks so I can explicitly assemble identity mass matrix for testing... */
    // int m_Mass_localMinRow;
    // int m_Mass_localMaxRow;

    // Call when using spatial parallelism                          
    void getSpatialDiscretizationG(const MPI_Comm &spatialComm, double* &G, 
                                    int &localMinRow, int &localMaxRow, int &spatialDOFs, double t);                               
    void getSpatialDiscretizationL(const MPI_Comm &spatialComm, int* &L_rowptr, 
                                    int* &L_colinds, double* &L_data,
                                    double* &U0, bool getU0, int U0ID,
                                    int &localMinRow, int &localMaxRow, int &spatialDOFs,
                                    double t, int &bsize);                                            
    
    // Call when NOT using spatial parallelism                                        
    void getSpatialDiscretizationG(double* &G, int &spatialDOFs, double t); 
    void getSpatialDiscretizationL(int* &L_rowptr, int* &L_colinds, double* &L_data,
                                    double* &U0, bool getU0, int U0ID, 
                                    int &spatialDOFs, double t, int &bsize);                                            
                                         
    /* Uses spatial parallelism */                                
    void get2DSpatialDiscretizationL(const MPI_Comm &spatialComm, 
                                        int *&L_rowptr, int *&L_colinds, double *&L_data, 
                                        double * &U0, bool getU0, int U0ID,
                                        int &localMinRow, int &localMaxRow, int &spatialDOFs, 
                                        double t, int &bsize);
    void get1DSpatialDiscretizationL(const MPI_Comm &spatialComm, 
                                        int *&L_rowptr, int *&L_colinds, double *&L_data, 
                                        double * &U0, bool getU0, int U0ID,
                                        int &localMinRow, int &localMaxRow, int &spatialDOFs, 
                                        double t, int &bsize);                                
                                    
    /* No spatial parallelism */
    void get2DSpatialDiscretizationL(int *&L_rowptr, int *&L_colinds, double *&L_data, 
                                        double *&U0, bool getU0, int U0ID,
                                        int &spatialDOFs, double t, int &bsize);
                                
    /* Uses spatial parallelism */  
    void getInitialCondition(const MPI_Comm &spatialComm, 
                                   double * &U0, 
                                   int      &localMinRow, 
                                   int      &localMaxRow, 
                                   int      &spatialDOFs);
                                
    void getInitialCondition(double * &U0, 
                             int      &spatialDOFs);
    
    bool GetExactPDESolution(const MPI_Comm &spatialComm, 
                                                double * &U, 
                                                int &localMinRow, 
                                                int &localMaxRow, 
                                                int &spatialDOFs, 
                                                double t);
    
                             
    bool GetExactPDESolution(double * &U0, 
                            int      &spatialDOFs, double t);
                            
    

    void GetGridFunction(void   *  GridFunction, 
                         double * &B, 
                         int      &spatialDOFs);

    void GetGridFunction(      void     *  GridFunction, 
                         const MPI_Comm   &spatialComm, 
                               double   * &B, 
                               int        &localMinRow, 
                               int        &localMaxRow, 
                               int        &spatialDOFs);


    /* Helper functions; shouldn't really be called outside of this class */
    void getLocalUpwindDiscretization(double * &localWeights, int * &localInds,
                                        std::function<double(int)> localWaveSpeed,
                                        double * const &plusWeights, int * const &plusInds, 
                                        double * const &minusWeights, int * const &minusInds,
                                        int nWeights);
    double MeshIndToPoint(int meshInd, int dim);
    void get1DUpwindStencil(int * &inds, double * &weight, int dim);
    void Get1DDissipationStencil(int * &inds, double *&weights, int &nnz); 
    
    double GetInitialIterate(double x, int U0ID);       /* 1D initial iterate for iterative solver */
    double GetInitialIterate(double x, double y,        /* 2D initial iterate for iterative solver */
                        int U0ID); 
    double InitCond(double x);                          /* 1D initial condition */
    double InitCond(double x, double y);                /* 2D initial condition */
    double WaveSpeed(double x, double t);               /* 1D wave speed */
    double WaveSpeed(double x, double y, double t,      /* 2D wave speed */
                        int component);  
    double PDE_Source(double x, double t);              /* 1D source */
    double PDE_Source(double x, double y, double t);    /* 2D source */

    double PDE_Solution(double x, double t);
    double PDE_Solution(double x, double y, double t);

    double LagrangeOutflowCoefficient(int i, int k, int p);
    double InflowBoundary(double t);
    
    void AppendInflowStencil1D(double * &G, double t);
    
    void GetOutflowDiscretization(int &outflowStencilNnz, double * &localOutflowWeights, int * &localOutflowInds, int stencilNnz, double * localWeights, int * localInds, int dim, int DOFInd); 

    int factorial(int n) { return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n; };
    
    int GlobalIndToMeshInd(int globInd);
    void GetInflowBoundaryDerivatives1D(double * &du, double t);
    void GetInflowValues(std::map<int, double> &uGhost, double t, int dim);
    double GetCentralFDApprox(std::function<double(double)> f, double x0, int order, double h);

    
    /* Utility-type functions */
    void Merge1DStencilsIntoMap(int * indsIn1, double * weightsIn1, int nnzIn1, 
                                int * indsIn2, double * weightsIn2, int nnzIn2,
                                std::map<int, double> &out);
                                
    int div_ceil(int numerator, int denominator);    
    
    
    /* Pure virtual functions requiring implementation from the base class */
    void GetSpatialDiscretizationU0(HypreParVector * &u0); 
    void GetSpatialDiscretizationG(double t, HypreParVector * &g); 
    void GetSpatialDiscretizationL(double t, HypreParMatrix * &L); 
    void GetSpatialDiscretizationM(HypreParMatrix * &M); 
                           
public:
    /* Constructors */
	FDadvection(MPI_Comm globComm, bool M_exists);
	FDadvection(MPI_Comm globComm, bool M_exists, int dim, int refLevels, int order, int problemID, std::vector<int> px = {});
    ~FDadvection();
    
    /* Add numerical dissipation into pure advection discretization  */
    void SetNumDissipation(Num_dissipation dissipation_params);
};