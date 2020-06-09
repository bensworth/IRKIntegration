#ifndef IRK_H
#define IRK_H

#include "HYPRE.h"
#include "mfem.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;

/** Abstract base class for spatial discretizations of a PDE resulting in the 
quasi-time-dependent, linear ODEs
    M*du/dt = L*u + g(t)    [_OR_ du/dt = L*u + g(t) if no mass matrix exists]

This class uses TimeDependentOperator::EvalMode w.r.t. the ODE system:
    -EvalMode::NORMAL corresponds to the ODEs above,
    -EvalMode::ADDITIVE_TERM_1 corresponds to the ODEs
        M*du/dt = L*u   [_OR_ du/dt = L*u if no mass matrix exists]

If a mass matrix M exists, the following virtual functions must be implemented:
    ImplicitMult(x,y): y <- M*x
    ApplyMInv(x,y): y <- M^{-1}*x */
class IRKOperator : public TimeDependentOperator
{    
protected:
    MPI_Comm m_globComm;
    
    // Auxillary vectors
    mutable Vector temp1, temp2; 
    
public:
    IRKOperator(MPI_Comm comm, int n=0, double t=0.0, Type type=EXPLICIT) 
        : TimeDependentOperator(n, t, type), 
            m_globComm{comm}, temp1(n), temp2(n) {};
    
    ~IRKOperator() { };

    MPI_Comm GetComm() { return m_globComm; };

    /** Apply action of du/dt, y <- M^{-1}*[L*x + g(t)] */
    virtual void Mult(const Vector &x, Vector &y) const = 0;
    
    /** Apply action of M*du/dt, y <- [L*x + g(t)] */
    //virtual void ExplicitMult(const Vector &x, Vector &y) const = 0;
    
    
    /** Apply action mass matrix, y = M*x. 
    If not re-implemented, this method simply generates an error. */
    virtual void ImplicitMult(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ImplicitMult() is not overridden!");
    }
    
    /* TODO: Ben, not quite sure what to do here... Given that M defines the 
    implicit component of the system, and, so, "ImplicitMult()" computes its action, it
    it seems natural to me that "ImplicitSolve()" should compute the action of its 
    inverse... But we cannot use this name as it already means something different in 
    TimeDependentOperator... Should we just call it "ApplyInvM()"? It's just a little 
    awkard having such different names for two functions that do the inverse of each other...
    
    Maybe we should just call this ImplicitInvMult(), then it's consistsent with ImplicitMult()
    */
    /** Apply action of inverse of mass matrix, y = M^{-1}*x. 
    If not re-implemented, this method simply generates an error. */
    virtual void ApplyMInv(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ApplyMInv() is not overridden!");
    }
    
    // TODO: Remove in favour of EvalMode
    /** Apply action of L, y = L*x. */
    virtual void ApplyL(const Vector &x, Vector &y) const = 0;
    
    /** Precondition (\gamma*M - dt*L) */
    virtual void ImplicitPrec(const Vector &x, Vector &y) const = 0;
    
    
    // Function to ensure that ImplicitPrec preconditions (\gamma*M - dt*L) OR (\gamma*I - dt*L)
    // with gamma and dt as passed to this function.
    //      + index -> index of real char poly factor, [0,#number of real factors)
    //      + type -> eigenvalue type, 1 = real, 2 = complex pair
    //      + t -> time.
    // These additional parameters are to provide ways to track when
    // (\gamma*M - dt*L) or (\gamma*I - dt*L) must be reconstructed or not to minimize setup.
    virtual void SetSystem(int index, double t, double dt,
                           double gamma, int type) = 0;
    
    
    /* Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients, c:
            P(x) := c(0)*x^0 + c(1)*x^1 + ... + c(q)*x^q.
        Coefficients must be provided for all monomial terms, even if they're 0. */
    inline void PolynomialMult(Vector c, double alpha, const Vector &x, Vector &y) const
    {
        int q = c.Size() - 1;
        y.Set(c[q], x);
        
        for (int i = q-1; i >= 0; i--) {
            /* temp1 <- M^{-1}*L*y */
            if (this->isImplicit()) {
                this->ApplyL(y, temp2); 
                this->ApplyMInv(temp2, temp1);
            } else {
                this->ApplyL(y, temp1); 
            }
            // TODO:
            // this->SetEvalMode(EvalMode::ADDITIVE_TERM_1);
            // this->Mult(y, temp1);
            
            if (c[i] != 0.0) {
                add(c[i], x, alpha, temp1, y); // y <- c[i]*x + alpha*temp1
            } else {
                y.Set(alpha, temp1); // y <- alpha*temp1
            }                       
        } 
    }
};


/** Class holding RK Butcher tableau, and associated data, required by both 
LINEAR and NONLINEAR IRK solvers */
class RKData 
{
public:
    // Implicit Runge Kutta type. Enumeration:
    //  First digit: group of schemes
    //  - 1 = A-stable (but NOT L-stable) SDIRK schemes
    //  + 0 = L-stable SDIRK
    //  + 1 = Gauss-Legendre
    //  + 2 = RadauIIA
    //  + 3 = Lobatto IIIC
    //  Second digit: order of scheme
    enum Type { 
        ASDIRK4 = -14,
        LSDIRK1 = 01, LSDIRK2 = 02, LSDIRK3 = 03, LSDIRK4 = 04,
        Gauss2 = 12, Gauss4 = 14, Gauss6 = 16, Gauss8 = 18, Gauss10 = 110,
        RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27, RadauIIA9 = 29,
        LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36, LobIIIC8 = 38
    };  
    
private:    
    Type ID;
    
    /** Set data required by solvers */
    void SetData();     
    /** Set dimensions of data structures */
    void SizeData();    
    
public:
    RKData(Type ID_) : ID{ID_} { SetData(); };
    
    ~RKData() {};
    
    // Standard data required by all solvers
    int s;              // Number of stages
    int s_eff;          // Number of eigenvalues of A0 once complex conjugates have been combined
    DenseMatrix A0;     // The Butcher matrix
    DenseMatrix invA0;  // Inverse of Buthcher matrix
    Vector b0;          // Butcher tableau weights
    Vector c0;          // Butcher tableau nodes
    Vector d0;          // inv(A0^\top)*b0
    
    // Associated data required by LINEAR solver
    Vector zeta;        // REAL eigenvalues of inv(A0)
    Vector beta;        // IMAGINARY parts of complex pairs of eigenvalues of inv(A0)
    Vector eta;         // REAL parts of complex pairs of eigenvalues of inv(A0)
    
    // Associated data required by NONLINEAR solver
    DenseMatrix Q0;     // Orthogonal matrix in Schur decomposition of A0^-1
    DenseMatrix R0;     // Quasi-upper triangular matrix in Schur decomposition of A0^-1
    Array<int> R0_block_sizes; // From top of block diagonal, sizes of blocks
};


/* Wrapper to preconditioner factors in a polynomial by preconditioning either
    TYPE 1. [gamma*M - dt*L] _OR_ [gamma*I - dt*L]
    TYPE 2. [gamma*M - dt*L]M^{-1}[gamma*M - dt*L]  _OR_ [gamma*I - dt*L]^2

[gamma*M - dt*L] _OR_ [gamma*I - dt*L] is preconditioned using
IRKOperator.ImplicitPrec(). Type 2 involves two IRKOperator.ImplicitPrec()
applications, with an application of M in between, IRKOperator.ApplyM().
*/
class CharPolyPrecon : public Solver
{

private:
    
    const IRKOperator &m_IRKOper;   // Spatial discretization
    int m_degree;                   // Degree of preconditioner as polynomial in L
    mutable Vector m_temp;          // Auxillary vector
    
public:

    CharPolyPrecon(const IRKOperator &IRKOper_)
        : Solver(IRKOper_.Height(), false), 
        m_IRKOper(IRKOper_), m_temp(IRKOper_.Height()), m_degree(-1) { };

    ~CharPolyPrecon() { };

    void SetDegree(int degree_) { m_degree = degree_; };

    /** Apply action of solver */
    inline void Mult(const Vector &x, Vector &y) const
    {
        if (m_degree == 1) {
            m_IRKOper.ImplicitPrec(x, y);         // Precondition [gamma*M - dt*L] 
        
        } else if (m_degree == 2) {
            // MASS MATRIX 
            if (m_IRKOper.isImplicit()) {
                m_IRKOper.ImplicitPrec(x, y);     // Precondition [gamma*M - dt*L]
                m_IRKOper.ImplicitMult(y, m_temp);// Apply M
                m_IRKOper.ImplicitPrec(m_temp, y);// Precondition [gamma*M - dt*L]
            // NO MASS MATRIX
            } else {
                m_IRKOper.ImplicitPrec(x, m_temp);// Precondition [gamma*I - dt*L]
                m_IRKOper.ImplicitPrec(m_temp, y);// Precondition [gamma*I - dt*L]
            }
        } else {
            mfem_error("CharPolyPrecon::Must set polynomial degree to 1 or 2!\n");
        }
    };

    // Purely virtual function we must implement but do not use
    virtual void SetOperator(const Operator &op) {  };
};

/* Char. poly factors, F:
    Degree 1. F == [zeta*M - dt*L]
    Degree 2. F == [(eta^2+beta^2)*M - 2*eta*dt*L + dt^2*L*M^{-1}*L] 
*/
class CharPolyOp : public Operator
{
private:

    const IRKOperator &m_IRKOper;   // Spatial discretization
    int m_degree;                   // Degree of operator as polynomial in L
    double m_gamma;                 // Constant in preconditioner
    double m_dt;                    // Time step
    Vector m_c;                     // Coefficients describing operator as polynomial in L
    mutable Vector m_temp;          // Auxillary vector
    
public:

    /** Constructor for TYPE 1 char. polynomial factor */
    CharPolyOp(double dt_, double zeta_, IRKOperator &IRKOper_) 
        : Operator(IRKOper_.Height()), 
            m_degree(1), m_c(2), m_gamma(zeta_),
             m_dt{dt_}, m_IRKOper{IRKOper_}, m_temp(IRKOper_.Height())
    {
        m_c(0) = zeta_;
        m_c(1) = -1.0;
    };

    /** Constructor for TYPE 2 char. polynomial factor */
    CharPolyOp(double dt_, double eta_, double beta_, IRKOperator &IRKOper_) 
        : Operator(IRKOper_.Height()), 
            m_degree(2), m_c(3), m_gamma(eta_),
            m_dt{dt_}, m_IRKOper{IRKOper_}, m_temp(IRKOper_.Height())
    {
        m_c(0) = eta_*eta_ + beta_*beta_;
        m_c(1) = -2.0*eta_;
        m_c(2) = 1.0;
    };

    inline int Degree() {return m_degree; };
    inline double Gamma() {return m_gamma; };
    inline double dt() {return m_dt; };
    inline void Setdt(double dt_) { m_dt = dt_; };

    /** y <- char. poly factor(dt*M^{-1}*L)*x */
    inline void Mult(const Vector &x, Vector &y) const 
    {
        // MASS MATRIX: Factor is not quite a polynomial in dt*L
        if (m_IRKOper.isImplicit()) {
            // F == [zeta*M - dt*L]
            if (m_degree == 1) {
                m_IRKOper.ApplyL(x, y);
                y *= -m_dt;
                m_IRKOper.ImplicitMult(x, m_temp);
                y.Add(m_c(0), m_temp);
                
            // F == [(eta^2 + beta^2)*M - 2*dt*L + dt^2*L*M^{-1}*L]
            } else if (m_degree == 2) {
                m_IRKOper.ApplyL(x, y);
                m_IRKOper.ApplyMInv(y, m_temp);
                m_temp *= m_c(2)*m_dt;
                m_temp.Add(m_c(1), x);  // temp = [c(1)*I + c(2)*dt*M^{-1}*L]*x
                m_IRKOper.ApplyL(m_temp, y);
                y *= m_dt;              // y = dt*L*[c(1)*I + c(2)*dt*M^{-1}*L]*x
                m_IRKOper.ImplicitMult(x, m_temp);
                //m_temp *= m_c(0); 
                //y += m_temp;
                y.Add(m_c(0), m_temp);
            }
            
        // NO MASS MATRIX: Factor is simply a polynomial in dt*L    
        } else {
            m_IRKOper.PolynomialMult(m_c, m_dt, x, y); 
        }
    }
    ~CharPolyOp() { };
};


/** Class implementing conjugate-pair preconditioned solution of fully implicit 
RK schemes for the linear ODE system M*du/dt = L*u + g(t), as implemented in 
IRKOperator */
class IRK : public ODESolver
{
public:
    // Krylov solve type for IRK system
    enum Solve {
        CG = 0, MINRES = 1, GMRES = 2, BICGSTAB = 3, FGMRES = 4
    };    
    
private:    
    MPI_Comm m_comm;          
    int m_numProcess;
    int m_rank;
    
    RKData m_Butcher;       // Runge-Kutta Butcher tableau and associated data
    
    IRKOperator * m_IRKOper;// Spatial discretization
    
    Vector m_sol, m_rhs;    // Solution and RHS of linear systems
    Vector m_temp1, m_temp2;// Auxillary vectors

    // Char. poly factors and preconditioner wrapper
    Array<CharPolyOp  *> m_CharPolyOps;
    CharPolyPrecon  m_CharPolyPrec;
    IterativeSolver * m_krylov;

    // Runge-Kutta variables    
    int m_solveID;              // Type of Krylov acceleration
    vector<Vector> m_XCoeffs;  // Vectors for the coefficients of polynomials {X_j}_{j=1}^s
    // TODO: if I use MFEM::Array<Vector> rather than std::vector<Vector> I get compiler warnings whenever I size the MFEM::Array...
    
    /* --- Relating to solution of linear systems --- */
    int m_krylov_print;
    vector<int> m_avg_iter;  // Across whole integration, avg number of Krylov iters for each system
    vector<int> m_type;      // Type 1 or 2?
    vector<double> m_eig_ratio; // The ratio beta/eta
    
    void SetXCoeffs();        // Set coefficients of polynomials X_j
    void StiffAccSimplify();  // Modify XCoeffs in instance of stiffly accurate IRK scheme
    void PolyAction();        // Compute action of a polynomial on a vector

    /** Construct right-hand side vector z for IRK integration, including applying
     the block Adjugate and Butcher inverse */
    void ConstructRHS(const Vector &x, double t, double dt, Vector &rhs);

public:

    IRK(IRKOperator *S, RKData::Type RK_ID_);
    ~IRK();
 
    void Init(TimeDependentOperator &F);

    void Run(Vector &x, double &t, double &dt, double tf);
    
    void Step(Vector &x, double &t, double &dt);

    void SetSolve(IRK::Solve solveID=IRK::GMRES, double reltol=1e-6,
                  int maxiter=250, double abstol=1e-6, int kdim=15,
                  int printlevel=2);

    // Get statistics about solution of linear systems
    inline void GetSolveStats(vector<int> &avg_iter, vector<int> &type, 
                                vector<double> &eig_ratio) const {
                                    avg_iter  =  m_avg_iter;
                                    type      =  m_type;
                                    eig_ratio =  m_eig_ratio;
                                }
};

#endif