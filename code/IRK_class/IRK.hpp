#ifndef IRK_H
#define IRK_H

//#ifndef IRKOperator_H

//#endif

//#include "IRKOperator.hpp"

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
   double distance;
   std::string prerelax;
   std::string postrelax;
   double strength_tolC;
   double strength_tolR;
   double filter_tolR;
   int interp_type;
   int relax_type;
   double filterA_tol;
   int coarsening;
};

/* 
Abstract base class for linear spatial discretizations of a PDE resulting in the 
time-dependent ODE M*du/dt = L*u + g(t), u(0) = u0. 

If no mass matrix exists, M_exists=false must be passed to the constructor. 
If a mass matrix exists (default), the virtual function ApplyM() must be implemented.
*/
class IRKOperator : public TimeDependentOperator
{    
protected:
    MPI_Comm m_globComm;
    
public:
    // NOTE: By default, assume there's a mass matrix
    IRKOperator(MPI_Comm comm, bool M_exists = true) : TimeDependentOperator(), m_globComm{comm}, m_M_exists{M_exists} {};
    ~IRKOperator() { };

    /* Get y <- M^{-1}*L*x OR y <- L*x */
    virtual void ExplicitMult(const Vector &x, Vector &y) const = 0;
    
    /* Apply mass matrix, y = M*x. If not re-implemented, this method simply generates an error. */
    virtual void ApplyM(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ApplyM() is not overridden!");
    }
    
    /* Precondition (\gamma*M - dt*L) OR (\gamma*I - dt*L) */
    virtual void ImplicitPrec(const Vector &x, Vector &y) const = 0;
    
    
    // // BEN: I'm confused here. You say in the notes that M^{-1} is the same as the Mult() function the user has to provide?
    // /* Apply mass matrix inverse, y = M^{-1}*x. If not re-implemented, this method simply generates an error. */
    // virtual void ApplyMinv(const Vector &x, Vector &y) const
    // {
    //     mfem_error("IRKOperator::ApplyMinv() is not overridden!");
    // }
    
    // Function to ensure that ImplicitPrec preconditions (\gamma*M - dt*L) OR (\gamma*I - dt*L)
    // with gamma and dt as passed to this function.
    //      + index -> index of eigenvalue (pair) in IRK storage
    //      + type -> eigenvalue type, 1 = real, 2 = complex pair
    //      + t -> time.
    // These additional parameters are to provide ways to track when
    // (\gamma*M - dt*L) or (\gamma*I - dt*L) must be reconstructed or not to minimize setup.
    virtual void SetSystem(int index, double t, double dt,
                           double gamma, int type) = 0;
    
    /* Get RHS vector scaled by inverse of mass matrix, M^{-1}*g(t) OR I*g(t).
        Default behaviour assumes g doesn't exist by setting it to zero */
    virtual void GetG(double t, Vector &g) { g = 0.0; };
    
    /* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients:
            P(x) := c(0)*x^0 + c(1)*x^1 + ... + c(n)*x^n, where c == coefficients
        Coefficients must be provided for all monomial terms (even if they're 0) */
    inline void PolynomialMult(Vector coefficients, double alpha, const Vector &x, Vector &y) const
    {
        int n = coefficients.Size() - 1;
        y.Set(coefficients[n], x); // y <- coefficients[n]*x
        Vector z(y.Size()); // An auxillary vector
        for (int ell = n-1; ell >= 0; ell--) {
            this->ExplicitMult(y, z); // z <- M^{-1}*L*y OR z <- L*y       
            add(coefficients[ell], x, alpha, z, y); // y <- coefficients[ell]*x + alpha*z
        } 
    };
    
    // Does a mass matrix exist for this discretization; this needs to be public so IRK can access it
    bool m_M_exists; 
};


/* Wrapper to preconditioner factors in a polynomial by preconditioning either
    TYPE 1. [zeta*M - dt*L]
    TYPE 2. [zeta*M - dt*L]M^{-1}[zeta*M - dt*L]
For Type 1, there should be an M^{-1}, but M is applied to both sides of
the equation before solving. [zeta*M - dt*L] is then preconditioned using
IRKOperator.ImplicitPrec(). Type 2 involves two IRKOperator.ImplicitPrec()
applications, with an application of M in between, IRKOperator.ApplyM().
*/
class CharPolyPrecon : public Solver
{

private:
    int m_type; /* 1 or 2; type of preconditioner to provide */
    IRKOperator &m_S; /* Holds all information about spatial discretization */

public:

    CharPolyPrecon(IRKOperator &S)
        : Solver(S.Height(), false), m_S(S), m_type(-1) { };

    ~CharPolyPrecon() { };

    void SetType(int type) { m_type = type; };

    inline void Mult(const Vector &x, Vector &y) const
    {
        if (m_type == 1) {
            // With mass matrix
            if (m_S.m_M_exists) {
                m_S.ImplicitPrec(x, y);   // Precondition [gamma*M - dt*L]
            // Without mass matrix
            }
            else {
                m_S.ImplicitPrec(x, y);   // Precondition [gamma*I - dt*L]
            }
        }
        else if (m_type == 2) {
            Vector z(x);
            // With mass matrix 
            if (m_S.m_M_exists) {
                m_S.ImplicitPrec(x, y);     // Precondition [gamma*M - dt*L]
                m_S.ApplyM(y, z);           // Apply M
                m_S.ImplicitPrec(z, y);     // Precondition [gamma*M - dt*L]
            // Without mass matrix
            }
            else {
                m_S.ImplicitPrec(x, z);     // Precondition [gamma*I - dt*L]
                m_S.ImplicitPrec(z, y);     // Precondition [gamma*I - dt*L]
            }
        }
        else {
            mfem_error("Must set polynomial type 1 or 2!\n");
        }
    };

    // Purely virtual function we must implement but do not use
    virtual void SetOperator(const Operator &op) {  };
};

/* Char. poly factors, F:
    TYPE 1. F == [zeta*I - dt*L]
    TYPE 2. F == [(eta^2+beta^2)*I - 2*eta*dt*L + (dt*L)^2] 
*/
class CharPolyOp : public Operator
{
private:

    int m_type; // 1 or 2; type of factor
    double m_gamma; // Constant in preconditioner
    double m_dt;
    Vector m_c;     // Coefficients describing operator as polynomial in L
    IRKOperator &m_S;

public:

    /* Constructor for TYPE 1 char. polynomial factor */
    CharPolyOp(double dt, double zeta, IRKOperator &S) 
        : Operator(S.Height()), m_c(2), m_dt{dt}, m_S{S},
            m_gamma(zeta), m_type(1)
    {
        // Coefficients of operator as a polynomial in L
        m_c(0) = zeta;
        m_c(1) = -1.0;
    };

    /* Constructor for TYPE 2 char. polynomial factor */
    CharPolyOp(double dt, double eta, double beta, IRKOperator &S) 
        : Operator(S.Height()), m_dt{dt}, m_S{S},
        m_c(3), m_gamma(eta), m_type(2)
    {
        // Coefficients of operator as a polynomial in L
        m_c(0) = eta*eta + beta*beta;
        m_c(1) = -2.0*eta;
        m_c(2) = 1.0;
    };

    inline int Type() {return m_type; };
    inline double Gamma() {return m_gamma; };
    inline double dt() {return m_dt; };
    inline void Setdt(double dt) { m_dt = dt; };

    /* y <- char. poly factor(dt*M^{-1}*L) * x */
    inline void Mult(const Vector &x, Vector &y) const { m_S.PolynomialMult(m_c, m_dt, x, y); }

    ~CharPolyOp() { };
};


/* Class implementing conjugate pair preconditioned solution of fully implicit 
RK schemes for the linear ODE system M*du/dt = L*u + g(t), as implemented in 
IRKOperator */
class IRK : public ODESolver
{
private:    

    IRKOperator * m_S;          // Holds all information about THE spatial discretization
    Vector * m_z;               // RHS of linear system
    Vector * m_y;               // Solution of linear system
    Vector * m_w;               // Auxillary vector

    // Char. poly factors and preconditioner wrapper
    Array<CharPolyOp  *> m_CharPolyOps;
    CharPolyPrecon  m_CharPolyPrec;
    IterativeSolver * m_krylov;

    // Runge-Kutta variables
    int m_RK_ID;        // Type of IRK scheme
    int m_solveID;      // Type of Krylov acceleration
    int m_s;            // Number of RK stages
    int m_zetaSize;     // Number of real eigenvalues of inv(A0)
    int m_etaSize;      // Number of complex conjugate pairs of eigenvalues of inv(A0)    
    DenseMatrix m_A0;   // Butcher tableaux matrix A0
    DenseMatrix m_invA0;// Inverse of A0
    Vector m_b0;        // Butcher tableaux weights
    Vector m_c0;        // Butcher tableaux nodes
    Vector m_d0;        // The vector b0^\top * inv(A0)
    Vector m_zeta;      // REAL eigenvalues of inv(A0)
    Vector m_beta;      // IMAGINARY parts of complex pairs of eigenvalues of inv(A0)
    Vector m_eta;       // REAL parts of complex pairs of eigenvalues of inv(A0)
    Vector *m_XCoeffs;  // Coefficients of polynomials {X_j}_{j=1}^s

    // --- Relating to HYPRE solution of linear systems ---
    int m_numProcess;
    int m_rank;
    MPI_Comm m_comm;            // Global communicator

    void SetButcherCoeffs();    // Set Butcher tableaux coefficients
    void SetXCoeffs();          // Set coefficients of polynomials X_j
    void PolyAction();          // Compute action of a polynomial on a vector

    // Setting elements in arrays
    inline void Set(double * A, int i, int j, double aij) { A[i + j*m_s] = aij; }; // 2D array embedded in 1D array of size s, using rowmjr ordering (columns ordered contiguously) 
    inline void Set(double * A, int i, double ai) { A[i] = ai; }; // 1D array

    // Initialize and set Butcher arrays to  correct dimensions
    void SizeButcherArrays(double * &A, double * &invA, double * &b, double * &c, double * &d, 
                            double * &zeta, double * &eta, double * &beta);

    // Construct right-hand side, m_z, for IRK integration, including applying
    // the block Adjugate and Butcher inverse 
    void ConstructRHS(const Vector &x, double t, double dt);

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

    // Krylov solve type for IRK system
    enum Solve {
        CG = 0, MINRES = 1, GMRES = 2, BICGSTAB = 3, FGMRES = 4
    };

    IRK(IRKOperator *S, IRK::Type RK_ID, MPI_Comm comm);
    ~IRK();

    void Run(Vector &x, double &t, double &dt, double tf);
    
    void Init(double dt0) {};
    
    void Step(Vector &x, double &t, double &dt);

    void SetSolve(IRK::Solve solveID=IRK::GMRES, double reltol=1e-6,
                  int maxiter=250, double abstol=1, int kdim=15,
                  int printlevel=-1);
    
    



    void SaveSolInfo(string filename, map<string, string> additionalInfo);

    //inline void PrintU(string filename) {if (m_u)}
};

#endif