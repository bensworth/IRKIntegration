#ifndef IRK_H
#define IRK_H

//#ifndef SPATIALDISCRETIZATION_H

//#endif

#include "SpatialDiscretization.hpp"

#include "HYPRE.h"
#include "mfem.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;



// /* AMG preconditioner for real linear systems [alpha*I - beta*L] */
// class EigPreconditioner : public Solver
// {
// public:
//     EigPreconditioner(double alpha, double beta, HypreParMatrix &L, int numiters);
// };


/* The real operators, 
    (zeta*I - dt*L), 
    or [(eta+i*beta)*I - dt*L]*[(eta-i*beta)*I - dt*L], which is equivalent to 
        [(eta^2+beta^2)*I - 2*eta*dt*L + (dt*L)^2] 
*/
class CharPolyFactorOperator : public Operator
{
private:
           
    
public:
    MPI_Comm m_comm;
    bool m_conjPair;
    double m_zeta;
    double m_eta;
    double m_beta;
    double m_dt;
    Vector m_c;     /* Coefficients describing operator as polynomial in L*/
    SpatialDiscretization &m_S;
    
    Solver * m_solver; /* Solver */
    Solver * m_precon; /* Preconditioner */ 
    
    CharPolyFactorOperator(MPI_Comm comm, double dt, double zeta, double eta, double beta, SpatialDiscretization &S);
    
    /* y <- char. poly factor(dt*L) * x */
    inline virtual void Mult(const Vector &x, Vector &y) const { m_S.SolDepPolyMult(m_c, m_dt, x, y); }

    virtual ~CharPolyFactorOperator();
};


/* Class implementing conjugate pair preconditioned solution of fully implicit 
RK schemes for the linear ODE system M*du/dt = L*u + g(t), as implemented in 
SpatialDiscretization */
class IRK
{
    
private:    
    
    SpatialDiscretization * m_S;    /* Pointer to an instance of a SpatialDiscretization */
    Vector * m_z;                   /* RHS of linear system */
    Vector * m_y;                   /* Solution of linear system */
    Vector * m_w;                   /* Auxillary vector */
    
    /* Factor in char poly that we need to invert */
    Array<CharPolyFactorOperator *> m_CharPolyFactorOperators;
    
    double m_dt; /* Time step size */
    int    m_nt; /* Number of time steps */
    
    /* Runge-Kutta Butcher tableaux variables */
    int m_RK_ID;
    
    int m_s;            /* Number of RK stages */
    int m_zetaSize;     /* Number of real eigenvalues of inv(A0) */
    int m_etaSize;      /* Number of complex conjugate pairs of eigenvalues of inv(A0) */
    
    DenseMatrix m_A0;   /* Butcher tableaux matrix A0 */
    DenseMatrix m_invA0;/* Inverse of Butcher tableaux matrix A0 */
    Vector m_b0;        /* Butcher tableaux weights */
    Vector m_c0;        /* Butcher tableaux nodes */
    Vector m_d0;        /* The vector b0^\top * inv(A0) */
    Vector m_zeta;      /* REAL eigenvalues of inv(A0) */
    Vector m_beta;      /* IMAGINARY parts of complex conjugate pairs of eigenvalues of inv(A0) */
    Vector m_eta;       /* REAL parts of complex conjugate pairs of eigenvalues of inv(A0) */
    
    Vector * m_XCoeffs; /* Coefficients of polynomials {X_j}_{j=1}^s */
    
    /* --- Relating to HYPRE solution of linear systems --- */
    int m_numProcess;
    int m_rank;
    MPI_Comm m_comm;            /* Global communicator */
    
    
    void SetButcherCoeffs();    /* Set Butcher tableaux coefficients */
    void SetXCoeffs();          /* Set coefficients of polynomials X_j */
    void PolyAction();          /* Compute action of a polynomial on a vector */
    
    /* Setting elements in arrays */
    inline void Set(double * A, int i, int j, double aij) { A[i + j*m_s] = aij; }; // 2D array embedded in 1D array of size s, using rowmjr ordering (columns ordered contiguously) 
    inline void Set(double * A, int i, double ai) { A[i] = ai; }; // 1D array
    
    /* Initialize and set Butcher arrays to  correct dimensions */
    void SizeButcherArrays(double * &A, double * &invA, double * &b, double * &c, double * &d, 
                            double * &zeta, double * &eta, double * &beta);
    
    
    /* Form and set RHS of linear system, m_z */
    void SetRHSLinearSystem(double t);
    
protected:    
    
    
public:
    IRK(MPI_Comm globComm, int RK_ID, SpatialDiscretization * S, double dt, int nt);
    ~IRK();
    
    void TimeStep();
    
    void SaveSolInfo(string filename, map<string, string> additionalInfo);
    
    //inline void PrintU(string filename) {if (m_u)}
    
    void Test();
};

#endif