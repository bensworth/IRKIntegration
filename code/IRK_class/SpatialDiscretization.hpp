#ifndef SPATIALDISCRETIZATION_H
#define SPATIALDISCRETIZATION_H

#include <mpi.h>
#include "HYPRE.h"
#include "mfem.hpp"
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;



// /* Compute actions of operators involving the spatial discretization */
// class SpatialDisretizationOp : public TimeDependentOperator
// {
// private:
// 
// 
// protected:
//     HypreParMatrix &m_M, &m_L;
//     Vector &m_g;
//     CGSolver M_solver;   
//     mutable Vector z; // A placeholder vector
// 
// public:
//     double m_dt;
//     SpatialDisretizationOp();
// 
//     // Get y <- M^{-1}*L*x
//     virtual void Mult(const Vector &x, Vector &y) const;
// 
//     //// Get y <- M^-1*g
//     //virtual void GetSolIndepComponent(Vector y, double t)  const;
// 
//     // Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients (in ascending order of dregree)
//     virtual void PolyMult(const Vector coefficients, const Vector &x, Vector &y) const;
//     virtual ~SpatialDisretizationOp() { }
// };


/* 

Class for arbitrary spatial discretization resulting in the time-dependent ODE M*du/dt = L(t)*u + g(t), u(0) = u0. 
    
That is, spatial discretizations should be a sub class of this class
    
The main thing is basically having to     
*/
class SpatialDisretization : public TimeDependentOperator
{
    
private:    
    
protected:
    
    double  m_t_L;
    double  m_t_g;
    
    bool    m_M_exists;
    int     m_bsize; /* Do we need this??? Maybe not...?  Does it get stored in the hypre matrix??  I.e.,  if M is a block matrix, it should be stored as a block matrix */
    
    bool    m_L_isTimedependent;    /* Is L time dependent? */
    bool    m_G_isTimedependent;    /* Is g time dependent? */
    
    bool     m_useSpatialParallel;  /*  */
    MPI_Comm m_spatialComm;         /* Spatial communicator; the spatial discretization code has access to this */
    int      m_spatialCommSize;     /* Num processes in spatial communicator */
    int      m_spatialRank;         /* Process rank in spatial communicator */    
    
    
    HypreParMatrix * m_M;  /* Mass matrix */
    HypreParMatrix * m_L;  /* Linear spatial discretization matrix */
    HypreParVector * m_g;  /* Solution-independent source term */
    HypreParVector * m_u0; /* Solution-independent source term  */    

    mutable HypreParVector * m_z;

    /* Set member variables  */
    void SetM();
    void SetL(double t);
    void SetG(double t);
    void SetU0();

    

    /* -------------------------------------------------------------------------- */
    /* ----- Some utility functions that may be helpful for derived classes ----- */
    /* -------------------------------------------------------------------------- */
    void GetHypreParMatrixFromCSRData(MPI_Comm comm,  
                                        int localMinRow, int localMaxRow, int globalNumRows, 
                                        int * A_rowptr, int * A_colinds, double * A_data,
                                        HypreParMatrix * &A); 

    void GetHypreParVectorFromData(MPI_Comm comm, 
                                    int localMinRow, int localMaxRow, int globalNumRows, 
                                    double * x_data, HypreParVector * &x);
    /* -------------------------------------------------------------------------- */
    /* -------------------------------------------------------------------------- */
    /* -------------------------------------------------------------------------- */


    /* Functions requiring implementation in derived class, with the possible exception of getting the mass matrix */
    virtual void GetSpatialDiscretizationM(HypreParMatrix * &M);
    virtual void GetSpatialDiscretizationL(double t, HypreParMatrix * &L) = 0;
    virtual void GetSpatialDiscretizationG(double t, HypreParVector * &g) = 0;
    virtual void GetSpatialDiscretizationU0(HypreParVector * &u0) = 0;
    
public:
    
    void PrintL(){  if (m_L) m_L->Print("L.txt"); else std::cout << "WARNING: m_L == NULL, cannot be printed!\n"; };
    void PrintM(){  if (m_M) m_M->Print("M.txt"); else std::cout << "WARNING: m_M == NULL, cannot be printed!\n"; };
    void PrintG(){  if (m_g) m_g->Print("g.txt"); else std::cout << "WARNING: m_g == NULL, cannot be printed!\n"; };
    void PrintU0(){ if (m_u0) m_u0->Print("u0.txt"); else std::cout << "WARNING: m_u0 == NULL, cannot be printed!\n"; };
    
    SpatialDisretization(MPI_Comm spatialComm, bool M_exists);
    ~SpatialDisretization();
    
    void Test(double t);
};


                            
#endif                            