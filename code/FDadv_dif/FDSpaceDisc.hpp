#include "mfem.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>
#include <vector>
#include <cmath>
#include <map>

using namespace std;
using namespace mfem;


enum Direction {
    EAST = 0, WEST = 1, NORTH = 2, SOUTH = 3
};

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
public:
    
    friend class FDSpaceDisc;
    
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
    std::vector<int>    m_pIdx;                 /* Grid indices of proc in each dimension */
    std::vector<int>    m_nxLocal;              /* Number of points on proc in each dimension */
    int                 m_nxLocalTotal;         /* Total number of points on proc */
    
    std::vector<int>    m_nborRanks;            /* Rank of neighbouring processes. W,E,S,N */
    std::vector<int>    m_nborpIdxWest;         /* Indices of west neighbour */
    std::vector<int>    m_nborpIdxEast;         /* Indices of east neighbour */
    std::vector<int>    m_nborpIdxSouth;        /* Indices of south neighbour */
    std::vector<int>    m_nborpIdxNorth;        /* Indices of north neighbour */
    
    std::vector<int>    m_nxLocalInt;          /* Number of DOFs in each direction on procs in INTERIOR of proc domain */
    std::vector<int>    m_nxLocalBnd;          /* Number of DOFs in each direction on procs on BOUNDARY of proc domain */

    std::vector<int>    m_nborGlobOffset; /* Global index of first DOF owned by neighbouring procs */
    std::vector<int>    m_nborNxLocal;   /* Number of DOFs in each direction owned by neighbouring procs */

    
    FDMesh(MPI_Comm comm, int dim = 1, int refLevels = 5, std::vector<int> np = {});
    //~FDMesh();

    double MeshIndToPoint(int meshInd, int dim) const;
    
    void LocalDOFIndToMeshPoints(int i, Vector &x) const;
    int LocalOverflowToGlobalDOFInd(int locInd, int overflow, Direction nbor) const;
    void LocalDOFIndToLocalMeshInds(int i, Array<int> &meshInds) const;
    int LocalMeshIndsToGlobalDOFInd(const Array<int> &meshInds) const;

    void EvalFunction(double (*Function)(const Vector &x), 
                                      Vector * &u) const;

    void EvalFunction(double (*TDFunction)(const Vector &x, double t),
                                      double t, Vector * &u) const;
                                      
    void EvalFunction(double (*Function)(const Vector &x), 
                                      Vector &u) const;

    void EvalFunction(double (*TDFunction)(const Vector &x, double t),
                                      double t, Vector &u) const;                                  

    /* Get mesh info */
    inline double Get_dx(int dim = 0) const { return m_dx[dim]; };
    inline int    Get_nx(int dim = 0) const { return m_nx[dim]; };
    inline int    Get_dim() const { return m_dim; };
    inline MPI_Comm GetComm() const { return m_comm; };
    inline int GetCommSize() const { return m_commSize; };
    
    inline int GetGlobalSize() const { return m_nxTotal; };
    inline int GetLocalSize() const { return m_nxLocalTotal; };
};


class FDSpaceDisc : public Operator
{
protected:
    const FDMesh &m_mesh;
    
    int m_derivative;
    int m_dim;                                  /* Number of spatial dimensions */
    int m_order;                                /* Order of discretization*/
    FDBias m_bias;
    
    int m_localSize;                           /* Number of DOFs on proc */
    int m_globSize;
    int m_spatialDOFs;                          /* Total number of DOFs in spatial disc */
    int m_globMinRow;                          /* Global index of first DOF on proc */
    int m_globMaxRow;                          /* Global index of last DOF on proc */
    
    bool     m_parallel;      
    MPI_Comm m_comm;         
    int      m_commSize;     
    int      m_rank;       

    void Get1DConstantOperatorCSR(int derivative, Vector c,
                                  int order, FDBias bias,
                                  int * &rowptr, int * &colinds, double * &data) const;
    
    void Get2DConstantOperatorCSR(int derivative, Vector c,
                                  int order, FDBias bias,
                                  int * &rowptr, int * &colinds, double * &data) const;
    
    
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
    
    /* -------------------------------------------------------------------------- */
                        
public:
    
    /* Constructors */
    FDSpaceDisc(const FDMesh &mesh, int derivative, int order, FDBias bias);
    ~FDSpaceDisc();
    
    /* --- Functions for derived classes to implement --- */
    virtual void Mult(const Vector &x, Vector &y) const = 0;
    
    virtual void AssembleGradient(const Vector &u) 
        { mfem_error("GetGradient:: Not overridden in derived class"); };
    
    // virtual Operator &GetGradient(const Vector &u) const
    // {
    //    mfem_error("Operator::GetGradient() is not overloaded!");
    //    return const_cast<Operator &>(*this);
    // }
    
    HypreParMatrix * GetHypreParIdentityMatrix() const;
};


// c.grad(u)
class FDLinearOp : public FDSpaceDisc 
{
    
private:
    Vector m_c;
    mutable HypreParMatrix * m_Op;    
    
public:
    /* Constructors */
    FDLinearOp(const FDMesh &mesh, int derivative, Vector c, int order, FDBias bias);
    ~FDLinearOp();
    
    void Assemble() const;
    
    void Mult(const Vector &x, Vector &y) const { m_Op->Mult(x, y); };
    
    // HypreParMatrix &GetOp() const 
    // { 
    //     if (m_Op) return *m_Op; 
    //     else mfem_error("GetOp:: Operator not assembled"); 
    //     return NULL;
    // };
    
    HypreParMatrix & Get() const { 
        if (!m_Op) Assemble(); 
        return *m_Op; 
    };
    
    // Gradient of linear operator is just the operator...
    HypreParMatrix & GetGradient() const 
    {   
        if (!m_Op) { Assemble(); }
        return *m_Op; 
    };
    HypreParMatrix & GetGradient(const Vector &u) const { return GetGradient(); };
    
    
};


// c.grad(f(u)) for potentially nonlinear f
class FDNonlinearOp : public FDSpaceDisc 
{
    private:
        double (*m_f)(double u);
        double (*m_df)(double u);
        Vector m_c;
        mutable HypreParMatrix * m_Gradient;  

        
        void Mult1DSerial(const Vector &x, Vector &y) const;
        void Mult1DParallel(const Vector &x, Vector &y) const;
        void Mult2DSerial(const Vector &x, Vector &y) const;
        void Mult2DParallel(const Vector &x, Vector &y) const;
        HypreParMatrix &GetGradient1DSerial(const Vector &a) const;
        HypreParMatrix &GetGradient1DParallel(const Vector &a) const;
        HypreParMatrix &GetGradient2D(const Vector &u) const;

    public:
        /* Constructors */
        FDNonlinearOp(const FDMesh &mesh, int derivative, Vector c, 
                        double (*f)(double), 
                        int order, FDBias bias);
        FDNonlinearOp(const FDMesh &mesh, int derivative, Vector c, 
                        double (*f)(double), double (*df)(double), 
                        int order, FDBias bias);                
        
        void Mult(const Vector &x, Vector &y) const;
        
        void SetGradientFunction(double (*df)(double u)) { m_df = df; };
        
        HypreParMatrix & GetGradient(const Vector &u) const;
};

