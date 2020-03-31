#ifndef IRK_H
#define IRK_H

//#ifndef IRKSpatialDisc_H

//#endif

#include "IRKSpatialDisc.hpp"

#include "HYPRE.h"
#include "mfem.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;


struct AMG_parameters {
   // Options: LAIR: {1, 1.5, 2}, NAIR: {3, 4, 5} - default 1.5
   //    LAIR is more robust on non-triangular matrices, NAIR should be
   //    faster to setup at the cost of slightly degraded convergence. 
   //    E.g., NAIR 4 corresponds to distance 4-2=2 restriction.
   double distance;
   // Options string consisting of F, C, or A; e.g., "FFC" or "FA"
   //    Default: prerelax = "", postrelax = "FA" (FA specifically
   //    for type 10 relax)
   std::string prerelax;
   std::string postrelax;
   // Strength tolerance for coarsening. Default 0.1, can be problem
   // dependent. Try between 0.01 and 0.5, typically in the middle.
   double strength_tolC;
   // Tolerance for building restriction. Smaller is better, typically 
   // 0.01 (default) or so. 
   double strength_tolR;
   // Remove small entries from R. Default 0, strategy we are testing to
   // reduce complexity. 
   double filter_tolR;
   // Default 100 (simple interpolation). Can also use 6 (classical interp).
   int interp_type;
   // Default 10 (processor block Jacobi, i.e., ordered on processor Gauss-
   // Seidel). Can also use 0 (Jacobi) or 3 (on processor, unordered Gauss-
   // Seidel)
   int relax_type;
   // Eliminate small entries from the matrix to reduce memory/complexity.
   // Default 1e-4. Can degrade convergence, if so make smaller.
   double filterA_tol;
   // Default 6, coarsens not aggressively. Can also use 3 (slower in 
   // parallel, also not aggressive coarsening) or 10 (more aggressive
   // coarsening).
   int coarsening;
};


/* Preconditioner for Krylov solution of char. poly factors, which are:
    TYPE 1. [zeta*I - dt*M^{-1}*L]
    TYPE 2. [(eta^2+beta^2)*I - 2*eta*dt*M^{-1}*L + (dt*M^{-1}*L)^2]
    
This classe's MULT function should be called once per Krylov iteration of the above system

Define operator M^{-1}*J(gamma) == M^{-1}*[gamma*M - dt*L], then:
    If TYPE 1:
        Approximately solve M^{-1}*J(zeta), where J(zeta) is approximately inverted with AMG.
        
    If TYPE 2:
        Approximately solve [M^{-1}*J(eta)]^2 by approximately inverting
        M^{-1}*J(eta) TWICE, where J(eta) is approximately inverted with either AMG or Krylov preconditioned AMG.
*/
class CharPolyPrecon : public Solver
{
    
private:
    int m_type; /* 1 or 2; type of preconditioner to provide */
    
    Operator * m_op;
    
    IRKSpatialDisc &m_S; /* Holds all information about spatial discretization */
    HypreParMatrix  * m_J;      /* Matrix to be approximately inverted: J == gamma*M - dt*L */ 
    
    Solver  * m_solver; /* Solver for J */
    Solver  * m_precon; /* Preconditioner for J */
    
    //AIROptions m_amg_options;
public:
    // Make virtual functions to set and setup solver and preconditioner. E.g., implement GMRES
    // and amg preconditioner by default, but user can override w/ more appropriate solver  should they  wish
    
    CharPolyPrecon(MPI_Comm comm, double gamma, double dt, int type, IRKSpatialDisc &S);
    ~CharPolyPrecon();

    virtual void Mult(const Vector &x, Vector &y) const;
    
    // This is a pure virtual function, not sure why. Don't think we really need it...
    virtual void SetOperator(const Operator &op) {  };
};

/* Char. poly factors, F:
    TYPE 1. F == [zeta*I - dt*L]
    TYPE 2. F == [(eta^2+beta^2)*I - 2*eta*dt*L + (dt*L)^2] 
*/
class CharPolyOp : public Operator
{
private:
    
public:
    
    int m_type; /* 1 or 2; type of factor */
    double m_zeta;
    double m_eta;
    double m_beta;
    double m_dt;
    Vector m_c;     /* Coefficients describing operator as polynomial in L */
    IRKSpatialDisc &m_S;
    
    MPI_Comm m_comm;
    
    Solver * m_solver; /* Solver for factor */
    Solver * m_precon; /* Preconditioner for factor */ 
    
    /* Type 1 operator */
    CharPolyOp(MPI_Comm comm, double dt, double zeta, IRKSpatialDisc &S);
    
    /* Type 2 operator */
    CharPolyOp(MPI_Comm comm, double dt, double eta, double beta, IRKSpatialDisc &S);
    
    /* y <- char. poly factor(dt*M^{-1}*L) * x */
    inline virtual void Mult(const Vector &x, Vector &y) const { m_S.SolDepPolyMult(m_c, m_dt, x, y); }

    ~CharPolyOp();
};


/* Class implementing conjugate pair preconditioned solution of fully implicit 
RK schemes for the linear ODE system M*du/dt = L*u + g(t), as implemented in 
IRKSpatialDisc */
class IRK : public ODESolver
{
// Must implement:
//  void Step(Vector &x, double &t, double &dt)
//  void Run(Vector &x, double &t, double &dt, double tf) 
//

private:    
    
    IRKSpatialDisc * m_S;    /* Holds all information about THE spatial discretization */
    Vector * m_z;                   /* RHS of linear system */
    Vector * m_y;                   /* Solution of linear system */
    Vector * m_w;                   /* Auxillary vector */
    
    /* Char. poly factors needed to be inverted */
    Array<CharPolyOp *> m_CharPolyOps;
    
    /* Runge-Kutta Butcher tableaux variables */
    int m_RK_ID;
    
    int m_s;            /* Number of RK stages */
    int m_zetaSize;     /* Number of real eigenvalues of inv(A0) */
    int m_etaSize;      /* Number of complex conjugate pairs of eigenvalues of inv(A0) */
    
    DenseMatrix m_A0;   /* Butcher tableaux matrix A0 */
    DenseMatrix m_invA0;/* Inverse of A0 */
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

    /* Construct right-hand side, m_z, for IRK integration, including applying
    the block Adjugate and Butcher inverse */
    void ConstructRHS(double t, double dt);

protected:    
    
    
public:

    // Implicit Runge Kutta type. Enumeration:
    //  First digit: group of schemes
    //  + 0 = L-stable SDIRK
    //  + 1 = Gauss-Legendre
    //  + 2 = RadauIIA
    //  + 3 = Lobatto IIIC
    //  Second digit: order of scheme
    enum Type { 
      SDIRK2 = 02, SDIRK3 = 03, SDIRK4 = 04,
      Gauss4 = 14, Gauss6 = 16, Gauss8 = 18,
      RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27,
      LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36
    };

    IRK(IRKSpatialDisc *S, IRK::Type RK_ID, MPI_Comm globComm);
    ~IRK();
    
    void IRK::Step(Vector &x, double &t, double &dt);
    void IRK::Run(Vector &x, double &t, double &dt, double tf);
    
    void SaveSolInfo(string filename, map<string, string> additionalInfo);
    
    //inline void PrintU(string filename) {if (m_u)}
};

#endif