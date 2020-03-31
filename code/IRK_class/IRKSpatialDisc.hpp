#ifndef IRKSpatialDisc_H
#define IRKSpatialDisc_H

//#include "IRK.hpp"

#include <mpi.h>
#include "HYPRE.h"
#include "mfem.hpp"
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;


/* 
Abstract base class for linear spatial discretizations of a PDE resulting in the 
time-dependent ODE M*du/dt = L(t)*u + g(t), u(0) = u0. 
*/
class IRKSpatialDisc : public TimeDependentOperator
// Must implement:
//  void TimeDependentOperator::ExplicitMult(const Vector &x, Vector &y) const
//  void TimeDependentOperator::ImplicitMult(const Vector &x, Vector &y) const

{
    
private:    
    
    // Hmm, IRK is a friend class which means it can access the private and protected members of an instance of this class!
    friend class IRK;
    
    void GetHypreParIdentityMatrix(const HypreParMatrix &A, HypreParMatrix * &I);
    
protected:
    
    /* Components of the spatial discretization */
    //HypreParMatrix * m_M;  /* Mass matrix */
    //HypreParMatrix * m_L;  /* Linear solution-dependent operator */
    HypreParVector * m_g;  /* M^{-1} * solution-independent source term  */
    HypreParVector * m_u;  /* Solution */
    
    
    /* TODO : Need a mass matrix solver... Not sure of the most effective way for doing  this..
    Its setup  will likely require some knowledge from  the derived class. E.g., there may be no 
    mass matrix... Or if it's a DG  disc, maybe we actually store the inverse and just do multiplication
    with it everytime, or if  it's Cont. Galerkin, maybe we use Conj. Gradient  to invert...
    */
    //Solver M_solver;       /* Required for computing action of M^-1 */
    
    double  m_t_L; /* Time the current (linear) solution-dependent operator is evaluated at */
    double  m_t_g; /* Time the current solution-independent source term is evaluated at */
    double  m_t_u; /* Time the current solution is evaluated at */
    
    bool    m_M_exists;
    int     m_bsize; /* Do we need this??? Maybe not...?  Does it get stored in the hypre matrix??  I.e.,  if M is a block matrix, it should be stored as a block matrix */
    
    
    /* Space discretization is assumed fully time-dependent; must be over ridden in derived class if this is not the case */
    bool    m_L_isTimedependent;    /* Is L time dependent? */
    bool    m_G_isTimedependent;    /* Is g time dependent? */
    
    bool     m_useSpatialParallel;  /* Hmm...  DO we need this?? */
    MPI_Comm m_spatialComm;         /* Spatial communicator; the spatial discretization code has access to this */
    int      m_spatialCommSize;     /* Num processes in spatial communicator */
    int      m_spatialRank;         /* Process rank in spatial communicator */    
    

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
    virtual void GetIRKSpatialDiscM(HypreParMatrix * &M);
    virtual void GetIRKSpatialDiscL(double t, HypreParMatrix * &L) = 0;
    virtual void GetIRKSpatialDiscG(double t, HypreParVector * &g) = 0;
    virtual void GetIRKSpatialDiscU0(HypreParVector * &u0) = 0;
    
public:
    
    HypreParMatrix * m_M;  /* Mass matrix */
    HypreParMatrix * m_L;  /* Linear solution-dependent operator */
    
    
    
    /* Set member variables  */
    void SetM();
    void SetL(double t);
    void SetG(double t);
    void SetU0();
    
    
    int m_nDOFs; /* Size of system */
    
    void SaveL(){ if (m_L) m_L->Print("L.txt"); else std::cout << "WARNING: m_L == NULL, cannot be printed!\n"; };
    void SaveM(){ if (m_M) m_M->Print("M.txt"); else std::cout << "WARNING: m_M == NULL, cannot be printed!\n"; };
    void SaveG(){ if (m_g) m_g->Print("g.txt"); else std::cout << "WARNING: m_g == NULL, cannot be printed!\n"; };
    void SaveU(const char * fname){ if (m_u) m_u->Print(fname); else std::cout << "WARNING: m_u == NULL, cannot be printed!\n"; };
    
    IRKSpatialDisc(MPI_Comm spatialComm, bool M_exists);
    ~IRKSpatialDisc();
    
    /* Get y <- M^{-1}*L(t)*x */
    void SolDepMult(const Vector &x, Vector &y);
    
    /* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients.
    Coefficients must be provided for all monomial terms (even if they're 0) and 
    in increasing order (from 0th to nth) */
    void SolDepPolyMult(Vector coefficients, double alpha, const Vector &x, Vector &y);
    
    /* Get y <- M^{-1}*g(t) ... Do  i need this... This  is like just setting G and then accessing... */
    void SolIndepComp(Vector y, double t);

    void Test(double t);
};
                            
#endif                            