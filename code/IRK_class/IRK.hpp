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


/* Describes the conjugate pair operator ([eta+i*beta]*I - M^{-1}*L)*([eta-i*beta]*I - M^{-1}*L) */
class ConjEigOp : public TimeDependentOperator
{
    
private:
    //double m_dt;  /* This class needs access to the time step */
      
    
public:
    ConjEigOp(double dt, double eta, double beta);
    
    double m_eta0;
    double m_beta0;  
    double m_dt; // Why can't I  access this from IRK???
    
    void Mult(const Vector &x, Vector &y);
    
    GMRESSolver MySolver;
};

/* Describes the real operator (zeta*I - M^{-1}*L) */
class RealEigOp : public TimeDependentOperator
{
    
private:
    //double m_dt;  /* This class needs access to the time step */
     
    
public:
    
    double m_zeta0;   
    double m_dt; // Why can't I  access this from IRK???
    
    RealEigOp(double dt, double zeta);
    
    void Mult(const Vector &x, Vector &y);
    
    GMRESSolver MySolver;
};


/* Class implementing conjugate pair preconditioning of fully implicit RK schemes
for the linear ODE system du/dt = L*u + g(t) */
class IRK
{
    
    friend class ConjEigOp;
    friend class RealEigOp;
    
private:    
    
    SpatialDiscretization * m_S;    /* Pointer to an instance of a SpatialDiscretization */
    Vector * m_z;                   /* RHS of linear system */
    Vector * m_y;                   /* Solution of linear system */
    Vector * m_w;                   /* Auxillary vector */
    
    /* Operator classes; need one for every operator */
    vector<RealEigOp> RealEigOps;
    vector<ConjEigOp> ConjEigOps;
    
    double m_dt; /* Time step size */
    int    m_nt; /* Number of time steps */
    
    /* Runge-Kutta Butcher tableaux variables */
    int m_RK_ID;
    
    int m_s;            /* Number of RK stages */
    int m_zetaSize;     /* Number of real eigenvalues of inv(A0) */
    int m_etaSize;      /* Number of complex conjusgate pairs of eigenvalues of inv(A0) */
    
    DenseMatrix m_A0;   /* Butcher tableaux matrix A0 */
    DenseMatrix m_invA0;/* Inverse of Butcher tableaux matrix A0 */
    Vector m_b0;        /* Butcher tableaux weights */
    Vector m_c0;        /* Butcher tableaux nodes */
    Vector m_d0;        /* The vector b0^\top * inv(A0) */
    Vector m_zeta;      /* REAL eigenvalues of A^-1 */
    Vector m_beta;      /* IMAGINARY parts of complex conjugate pairs of eigenvalues of A0^-1 */
    Vector m_eta;       /* REAL parts of complex conjugate pairs of eigenvalues of A0^-1 */
    
    Vector * m_XCoeffs;/* Coefficients of polynomials {X_j}_{j=1}^s */
    
    
    /* --- Relating to HYPRE solution of linear systems --- */
    MPI_Comm            m_globComm;            /* Global communicator */
    
    
    void SetButcherCoeffs();    /* Set Butcher tableaux coefficients */
    void SetXCoeffs();          /* Set coefficients of polynomials X_j */
    void PolyAction();          /* Compute action of a polynomial on a vector */
    
    /* Setting elements arrays */
    void Set(double * A, int i, int j, double aij) { A[i*m_s + j] = aij; }; // 2D array embedded in 1D array of size s, using rowmjr ordering (rows ordered contiguously) 
    void Set(double * A, int i, double ai) { A[i] = ai; }; // 1D array
    
    /* Initialize and set Butcher arrays to  correct dimensions */
    void SizeButcherArrays(double * &A, double * &invA, double * &b, double * &c, double * &d, 
                            double * &zeta, double * &eta, double * &beta);
    
    
    /* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients.
    Coefficients must be provided for all monomial terms (even if they're 0) and 
    in increasing order (from 0th to nth) */
    void PolyMult(Vector coefficients, double alpha, const Vector &x, Vector &y);
    
    /* Form RHS of linear system, m_z */
    void SetRHSLinearSystem(double t);
    
protected:    
    
    
public:
    IRK(MPI_Comm globComm, int RK_ID, SpatialDiscretization * S, double dt, int nt);
    ~IRK();
    
    void TimeStep();
    
    void Test();
};

#endif