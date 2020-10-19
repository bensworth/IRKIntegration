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

/** Class for spatial discretizations of a PDE resulting in the 
    quasi-time-dependent, linear ODEs
        M*du/dt = L*u + g(t)    [_OR_ du/dt = L*u + g(t) if no mass matrix exists]

    If a mass matrix M exists, the following virtual functions must be implemented:
        ImplicitMult(x,y): y <- M*x
        ApplyMInv(x,y): y <- M^{-1}*x */

// Hmm????        
// This class uses TimeDependentOperator::EvalMode w.r.t. the ODE system:
//     -EvalMode::NORMAL corresponds to the ODEs above,
//     -EvalMode::ADDITIVE_TERM_1 corresponds to the ODEs
//         M*du/dt = L*u   [_OR_ du/dt = L*u if no mass matrix exists]        
//
//
// Maybe  to be consistent with  the NONLINEAR case, should call this ExplicitGradientMult and GradientMult
// since L is just the gradient of L*u w.r.t. u.
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

    /// Apply action of du/dt, y <- M^{-1}*[L*x + g(t)]
    virtual void Mult(const Vector &x, Vector &y) const = 0;
    
    /// Apply action of M*du/dt, y <- [L*x + g(t)] 
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
    /// Apply action of L, y = L*x. 
    virtual void ApplyL(const Vector &x, Vector &y) const = 0;
    
    /// Precondition (\gamma*M - dt*L) 
    virtual void ImplicitPrec(const Vector &x, Vector &y) const = 0;
    
    /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L) 
            + index: index of system to solve, [0,s_eff)
            + dt:    time step size
            + type:  eigenvalue type, 1 = real, 2 = complex pair
        These additional parameters are to provide ways to track when
        (\gamma*M - dt*L) must be reconstructed or not to minimize setup. */
    virtual void SetSystem(int index, double dt, double gamma, int type) = 0;
    
    
    /** Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients, c:
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
        ASDIRK3 = -13, ASDIRK4 = -14,
        LSDIRK1 = 01, LSDIRK2 = 02, LSDIRK3 = 03, LSDIRK4 = 04,
        Gauss2 = 12, Gauss4 = 14, Gauss6 = 16, Gauss8 = 18, Gauss10 = 110,
        RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27, RadauIIA9 = 29,
        LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36, LobIIIC8 = 38
    };  
    
private:    
    Type ID;
    
    /// Set data required by solvers 
    void SetData();     
    /// Set dimensions of data structures 
    void SizeData();    
    
public:
    RKData(Type ID_) : ID{ID_} { SetData(); };
    
    ~RKData() {};
    
    /// Is the scheme SDIRK? Assumes SDIRK schemes have Type's < 10...
    inline bool isSDIRK() const { return (static_cast<int>(ID) < 10); };
    
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


/** Preconditioner for characteristic polynomial factors
        Degree 1: [gamma*M - dt*L], 
        Degree 2: [gamma*M - dt*L]M^{-1}[gamma*M - dt*L],

    where [gamma*M - dt*L] is preconditioned using IRKOperator.ImplicitPrec(). */
class CharPolyFactorPrec : public Solver
{

private:
    
    const IRKOperator &m_IRKOper;   // Spatial discretization
    int m_degree;                   // Degree of preconditioner as polynomial in L
    mutable Vector m_temp;          // Auxillary vector
    
public:

    CharPolyFactorPrec(const IRKOperator &IRKOper_)
        : Solver(IRKOper_.Height(), false), 
        m_IRKOper(IRKOper_), m_temp(IRKOper_.Height()), m_degree(-1) { };

    ~CharPolyFactorPrec() { };

    void SetDegree(int degree_) { m_degree = degree_; };

    /// Apply action of preconditioner
    inline void Mult(const Vector &x, Vector &y) const
    {
        if (m_degree == 1) {
            m_IRKOper.ImplicitPrec(x, y);         // Precondition [gamma*M - dt*L] 
        
        } else if (m_degree == 2) {
            // MASS MATRIX: [gamma*M - dt*L]M^{-1}[gamma*M - dt*L]  
            if (m_IRKOper.isImplicit()) {
                m_IRKOper.ImplicitPrec(x, y);     // Precondition [gamma*M - dt*L]
                m_IRKOper.ImplicitMult(y, m_temp);// Apply M
                m_IRKOper.ImplicitPrec(m_temp, y);// Precondition [gamma*M - dt*L]
            // NO MASS MATRIX: [gamma*I - dt*L]^2
            } else {
                m_IRKOper.ImplicitPrec(x, m_temp);// Precondition [gamma*I - dt*L]
                m_IRKOper.ImplicitPrec(m_temp, y);// Precondition [gamma*I - dt*L]
            }
        } else {
            mfem_error("CharPolyFactorPrec::Must set polynomial degree to 1 or 2!\n");
        }
    };

    /// Purely virtual function we must implement but do not use
    virtual void SetOperator(const Operator &op) {  };
};


/** Characteristic polynomial factor after being scaled by mass matrix M:
        Degree 1: [zeta*M - dt*L],
        Degree 2: [(eta^2+beta^2)*M - 2*eta*dt*L + dt^2*L*M^{-1}*L].
    
    Or, more generally,
        [c(0)*M + c(1)*dt*L + c(2)*dt^2*L*M^{-1}*L] */
class CharPolyFactor : public Operator
{
private:

    const IRKOperator &m_IRKOper;   // Spatial discretization
    int m_degree;                   // Degree of operator as polynomial in L
    double m_gamma;                 // Constant in preconditioner
    double m_dt;                    // Time step
    Vector m_c;                     // Coefficients describing operator as polynomial in L
    mutable Vector m_temp;          // Auxillary vector
    
public:

    /// Constructor for degree 1 factor 
    CharPolyFactor(double dt_, double zeta_, IRKOperator &IRKOper_) 
        : Operator(IRKOper_.Height()), 
            m_degree(1), m_c(2), m_gamma(zeta_),
             m_dt{dt_}, m_IRKOper{IRKOper_}, m_temp(IRKOper_.Height())
    {
        m_c(0) = zeta_;
        m_c(1) = -1.0;
    };

    /// Constructor for degree 2 factor
    CharPolyFactor(double dt_, double eta_, double beta_, IRKOperator &IRKOper_, int mag_prec=0) 
        : Operator(IRKOper_.Height()), 
            m_degree(2), m_c(3), m_gamma(eta_),
            m_dt{dt_}, m_IRKOper{IRKOper_}, m_temp(IRKOper_.Height())
    {
        m_c(0) = eta_*eta_ + beta_*beta_;
        m_c(1) = -2.0*eta_;
        m_c(2) = 1.0;
        // Boolean to use constant \gamma = \eta or \gamma = \sqrt(eta^2+\beta^2)
        if (mag_prec==2) m_gamma = m_c(0);
        if (mag_prec==1) m_gamma = std::sqrt(m_c(0));
        else m_gamma = eta_;
    };

    /// Getters
    inline int Degree() {return m_degree; };
    inline double Gamma() {return m_gamma; };
    inline double dt() {return m_dt; };
    /// Set time step
    inline void SetTimeStep(double dt_) { m_dt = dt_; };

    /// Apply action of operator, y <- (char. poly factor)*x 
    inline void Mult(const Vector &x, Vector &y) const 
    {
        // MASS MATRIX: Factor is not quite a polynomial in dt*L
        if (m_IRKOper.isImplicit()) {
            // [c(0)*M - dt*L]
            if (m_degree == 1) {
                m_IRKOper.ApplyL(x, y);
                y *= -m_dt;
                m_IRKOper.ImplicitMult(x, m_temp);
                y.Add(m_c(0), m_temp);
                
            // [c(0)*M + c(1)*dt*L + c(2)*dt^2*L*M^{-1}*L]
            } else if (m_degree == 2) {
                m_IRKOper.ApplyL(x, y);
                m_IRKOper.ApplyMInv(y, m_temp);
                m_temp *= m_c(2)*m_dt;
                m_temp.Add(m_c(1), x);      // temp = [c(1)*I + c(2)*dt*M^{-1}*L]*x
                m_IRKOper.ApplyL(m_temp, y);
                y *= m_dt;                  // y = dt*L*[c(1)*I + c(2)*dt*M^{-1}*L]*x
                m_IRKOper.ImplicitMult(x, m_temp);
                y.Add(m_c(0), m_temp);      // y = [c(0)*M + dt*L*[c(1)*I + c(2)*dt*M^{-1}*L]*x
            }
            
        // NO MASS MATRIX: Factor is simply a polynomial in dt*L    
        } else {
            m_IRKOper.PolynomialMult(m_c, m_dt, x, y); 
        }
    }
    ~CharPolyFactor() { };
};



/** Class implementing conjugate-pair preconditioned solution of fully implicit 
    RK schemes for the quasi-time-dependent, linear ODE system 
        M*du/dt = L*u + g(t), 
    as implemented as an IRKOperator */
class IRK : public ODESolver
{
public:
    // Krylov solve type for IRK systems
    enum KrylovMethod {
        CG = 0, MINRES = 1, GMRES = 2, BICGSTAB = 3, FGMRES = 4
    };   
    
    // Parameters for Krylov solver
    struct KrylovParams {
        double abstol = 1e-10;
        double reltol = 1e-10;
        int maxiter = 100;
        int printlevel = 0;
        int kdim = 30;
        KrylovMethod solver = KrylovMethod::GMRES;
    }; 

private:    
    MPI_Comm m_comm;          
    int m_numProcess;
    int m_rank;
    
    RKData m_Butcher;           // Runge-Kutta Butcher tableau and associated data
    
    IRKOperator * m_IRKOper;    // Spatial discretization
    Vector m_sol, m_rhs;        // Solution and RHS of linear systems
    Vector m_temp1, m_temp2;    // Auxillary vectors

    /* Characteristic polynomial factors and preconditioner */
    Array<CharPolyFactor *> m_CharPolyOper; // Factored characteristic polynomial
    CharPolyFactorPrec  m_CharPolyPrec;     // Preconditioner for above factors
    IterativeSolver * m_krylov;             // Solver for inverting above factors
    KrylovParams m_krylov_params;           // Parameters for above solver
  
    vector<Vector> m_weightedAdjCoeffs; // Vectors for the coefficients of polynomials {X_j}_{j=1}^s
    // TODO: if I use MFEM::Array<Vector> rather than std::vector<Vector> I get compiler warnings whenever I size the MFEM::Array...
    
    /* Statistics on solution of linear systems */
    vector<int> m_avg_iter;         // Across whole integration, avg number of Krylov iters for each system
    vector<int> m_degree;           // Degree of system as polynomial in L
    vector<double> m_eig_ratio;     // The ratio beta/eta

    
    /// Build linear solver
    void SetSolver();
    
    /// Set coefficients of polynomials X_j         
    void SetWeightedAdjCoeffs();
    
    /// Modify XCoeffs in instance of stiffly accurate IRK scheme    
    void StiffAccSimplify();        
    
    /// Compute action of matrix polynomial on a vector
    void PolyAction();              
    
    /** Construct right-hand side vector for IRK integration, including applying
     the block Adjugate and Butcher inverse */
    void ConstructRHS(const Vector &x, double t, double dt, Vector &rhs);

public:
    IRK(IRKOperator *IRKOper_, RKData::Type RK_ID_, int mag_prec=0);
    ~IRK();
 
    void Init(TimeDependentOperator &F);

    void Run(Vector &x, double &t, double &dt, double tf);
    
    void Step(Vector &x, double &t, double &dt);

    /// Set parameters for Krylov solver
    inline void SetKrylovParams(KrylovParams params) { 
        MFEM_ASSERT(!m_krylov, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
        m_krylov_params = params;
    }

    /// Get statistics about solution of linear systems
    inline void GetSolveStats(vector<int> &avg_iter, vector<int> &type, 
                                vector<double> &eig_ratio) const {
                                    avg_iter  =  m_avg_iter;
                                    type      =  m_degree;
                                    eig_ratio =  m_eig_ratio;
                                }
};

#endif