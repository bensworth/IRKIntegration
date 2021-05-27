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

/// Kronecker transform between two block vectors: y <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, const BlockVector &x, BlockVector &y);

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

    /** Apply action of M*du/dt, y <- [L*x + g(t)]
        If not re-implemented, this method simply generates an error. 
        
        Note that this method is only required for the Staff algorithm */
    virtual void ExplicitMult(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ImplicitMult() is not overridden!");
    }


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
        (\gamma*M - dt*L) must be reconstructed or not to minimize setup. 
        This is required by conjugate-pair IRK */
    virtual void SetSystem(int index, double dt, double gamma, int type) = 0;
    
    /** Ensures that this->ImplicitPrec() preconditions (M - dt*L)
            + index: index of system to solve, [0,s-1)
            + dt:    time step size 
        This is required by StaffIRK */
    virtual void SetSystem(int index, double dt) = 0;


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
        if (mag_prec==3) m_gamma = m_c(0);
        else if (mag_prec==2) m_gamma = m_c(0)/(eta_);
        else if (mag_prec==1) m_gamma = std::sqrt(m_c(0));
        else m_gamma = eta_;

        int myid;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        if (myid ==0) std::cout << "eta = " << eta_ << ", beta = " << beta_ << ", gamma = " << m_gamma << "\n";
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
        CG = 0, MINRES = 1, GMRES = 2, BICGSTAB = 3, FGMRES = 4, FP = 5
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
        MFEM_ASSERT(!m_krylov, "IRK::SetKrylovParams:: Can only be called before IRK::Init()");
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








////////////////////////////////////////////////////////////////////////////////
/** --- The below algorithms pertain only to the implementation of the IRK 
    algorithms from Staff et al. (2006) --- */
////////////////////////////////////////////////////////////////////////////////

class StaffStagePreconditioner; // The class responsible for preconditioning this operator
/** Operator F defining the block-coupled s stage equations. They satisfy F*k = 0, 
    where 
                       [ M - a_11*dt*L ... -a_1s*dt*L ]
                F*k =  [ ...           ...        ... ]
                       [ M - a_s1*dt*L ... -a_ss*dt*L ]*/
class StaffStageOper : public BlockOperator
{
        
private:
    // Preconditioner needs access to this class
    friend class StaffStagePreconditioner; 
    
    Array<int> &offsets;            // Offsets of operator     
    
    mutable IRKOperator * IRKOper;  // Spatial discretization
    const RKData &Butcher;          // All RK information
    
    // Parameters that operator depends on
    double dt;                      // Current time step
    
    // Wrappers for scalar vectors
    mutable BlockVector k_block, y_block;
    
    // Auxillary vectors
    mutable BlockVector temp_block;
    
public:
    
    StaffStageOper(IRKOperator * S_, Array<int> &offsets_, const RKData &RK_) 
        : BlockOperator(offsets_), 
            IRKOper(S_), Butcher(RK_), 
            dt(0.0), 
            offsets(offsets_),
            k_block(offsets_), y_block(offsets_),
            temp_block(offsets_)
            { };
    
    
    // Set time step
    inline void SetTimeStep(double dt_) { dt = dt_; };
    // Get time step
    inline double GetTimeStep() { return dt; };
    
    
    /// Compute action of operator, y <- F*k
    inline void Mult(const Vector &k_scalar, Vector &y_scalar) const
    {
        // Wrap scalar Vectors with BlockVectors
        k_block.Update(k_scalar.GetData(), offsets);
        y_block.Update(y_scalar.GetData(), offsets);
        
        
        /* temp <- (I \otimes L)*k */
        for (int i = 0; i < Butcher.s; i++) {        
            IRKOper->ApplyL(k_block.GetBlock(i), temp_block.GetBlock(i)); // temp <- L*k
        }
        
        // y <- (A0 \otimes I)(I \otimes L)*k ==  (A0 \otimes I)*temp
        KronTransform(Butcher.A0, temp_block, y_block); 
        
        
        /* y <- -dt*y + (I \otimes M)k */
        for (int i = 0; i < Butcher.s; i++) {        
            y_block.GetBlock(i) *= -dt;
            // MASS MATRIX
            if (IRKOper->isImplicit()) {
                IRKOper->ImplicitMult(k_block.GetBlock(i), temp_block.GetBlock(0)); //  temp <- M*k  
                y_block.GetBlock(i) += temp_block.GetBlock(0);
            // NO MASS MATRIX    
            } else {
                y_block.GetBlock(i) += k_block.GetBlock(i);
            }
        }
    }
};


/** Class implementing the Staff et al. (2006) algorithm for fully implicit
    RK schemes for the quasi-time-dependent, linear ODE system
        M*du/dt = L*u + g(t),
    as implemented as an IRKOperator */
class StaffIRK : public ODESolver
{
public:
    // // Krylov solve type for IRK systems
    // enum KrylovMethod {
    //     GMRES = 2, FGMRES = 4
    // };

    // // Parameters for Krylov solver
    // struct KrylovParams {
    //     double abstol = 1e-10;
    //     double reltol = 1e-10;
    //     int maxiter = 100;
    //     int printlevel = 0;
    //     int kdim = 30;
    //     KrylovMethod solver = KrylovMethod::GMRES;
    // };

    // Block preconditioner
    enum PreconditionerID {
        JACOBI = 0, // Jacobi
        GSL = 1,    // Gauss--Seidel LOWER
        GSU = 2     // Gauss--Seidel UPPER
    };
    
    // Block sparsity pattern of preconditioner
    enum PreconditionerSparsity {
        DIAGONAL = 0, 
        LOWERTRIANGULAR = 1, 
        UPPERTRIANGULAR = 2
    };

private:
    MPI_Comm m_comm;
    int m_numProcess;
    int m_rank;

    RKData m_Butcher;           // Runge-Kutta Butcher tableau and associated data

    IRKOperator * m_IRKOper;    // Spatial discretization. 
    BlockVector m_k;            // The stage vectors
    BlockVector m_rhs;          // RHS of stage system
    Array<int> m_stageOffsets;  // Offsets 
    StaffStageOper * m_stageOper; // Operator F encoding stages, F*w = rhs

    StaffStagePreconditioner * m_stagePreconditioner;       // Block preconditioner for block stage equations
    PreconditionerID m_preconditionerID;    // Sparsity pattern of block preconditioner
    PreconditionerSparsity m_preconditionerSparsity;    // Sparsity pattern of block preconditioner
    DenseMatrix m_preconditionerCoefficients; // Coefficients of block preconditioner
    IterativeSolver * m_krylov;         // Solver for inverting block stage equations
    IRK::KrylovParams m_krylov_params;  // Parameters for above solver

    /* Statistics on solution of linear systems */
    int m_avg_iter;         // Across whole integration, avg number of Krylov iters per time step


    /// Build linear solver
    void SetSolver();

    /** Construct right-hand side vector for IRK integration, including applying
     the block Adjugate and Butcher inverse */
    void ConstructRHS(const Vector &x, double t, double dt, BlockVector &rhs);

    /* Set the preconditioner sparsity variable based on the the preconditioner ID */
    void SetPreconditionerSparsity() {
        if (m_preconditionerID == 0) {
            m_preconditionerSparsity = PreconditionerSparsity::DIAGONAL;
        } 
        else if (m_preconditionerID == 1) 
        {
            m_preconditionerSparsity = PreconditionerSparsity::LOWERTRIANGULAR;
        } 
        else if (m_preconditionerID == 2) 
        {
            m_preconditionerSparsity = PreconditionerSparsity::UPPERTRIANGULAR;    
        }
    } 

public:
    StaffIRK(IRKOperator *IRKOper_, RKData::Type RK_ID_, 
            StaffIRK::PreconditionerID preconditionerID_);
    ~StaffIRK();

    void Init(TimeDependentOperator &F);

    void Run(Vector &x, double &t, double &dt, double tf);

    void Step(Vector &x, double &t, double &dt);

    /// Set parameters for Krylov solver
    inline void SetKrylovParams(IRK::KrylovParams params) {
        MFEM_ASSERT(!m_krylov, "IRK::SetKrylovParams:: Can only be called before IRK::Init()");
        m_krylov_params = params;
    }

    /// Get statistics about solution of linear systems
    inline void GetSolveStats(int &avg_iter) const {
                                    avg_iter  =  m_avg_iter;
                                }
};



/** System matrix of block stage equations is approximated to be block triangular 
    (e.g., block Gauss--Seidel) or block diagonal (e.g., block Jacobi) 
    
    
    NOTE: The matrix of preconditioner coefficients used here does not necessarily 
    share the same sparsity pattern as the true preconditioner coefficients. An 
    additional PreconditionerSparsity variable is passed to this method indicating 
    whether the (true) preconditioner coefficients are diagonal, upper triangular, 
    or lower triangular, and according to this, only the necessary parts of the 
    passed preconditioner coefficients are accessed. 
    For example, for a block Jacobi method, the preconditioner coefficients might 
    actually contain all of A0, but because the PreconditionerSparsity::DIAGONAL
    is also passed, only the diagonal of the preconditioner coefficients array is 
    accessed.
    */
class StaffStagePreconditioner : public Solver
{
private:
    StaffStageOper &stageOper;
    
    Array<int> &offsets;    // Offsets for vectors with s blocks
    
    DenseMatrix &preconditionerCoefficients;
    
    StaffIRK::PreconditionerSparsity &preconditionerSparsity;
    
    // Auxillary vectors  
    mutable BlockVector x_block, b_block;   // s blocks
    mutable Vector temp; // 1 block/scalar vector

    
public:
    StaffStagePreconditioner(StaffStageOper &IRKStageOper_, 
        StaffIRK::PreconditionerSparsity &preconditionerSparsity_, 
        DenseMatrix &preconditionerCoefficients_) : 
        Solver(IRKStageOper_.Height()),
        stageOper(IRKStageOper_), 
        preconditionerSparsity(preconditionerSparsity_),
        preconditionerCoefficients(preconditionerCoefficients_),
        offsets(IRKStageOper_.offsets),
        x_block(IRKStageOper_.offsets), b_block(IRKStageOper_.offsets),
        temp(IRKStageOper_.Height())
    { };
    
    ~StaffStagePreconditioner() {};
    
    
    /** x = inv(A)*b, where A is the inexact, block preconditioner. */
    inline void Mult(const Vector &b_scalar, Vector &x_scalar) const
    {
        // Wrap scalar Vectors into BlockVectors
        b_block.Update(b_scalar.GetData(), offsets);
        x_block.Update(x_scalar.GetData(), offsets);
        
        switch (preconditionerSparsity) {
            case StaffIRK::PreconditionerSparsity::DIAGONAL:
                BlockDiagonalSubstitution(b_block, x_block);
                break;
            
            case StaffIRK::PreconditionerSparsity::LOWERTRIANGULAR:
                BlockForwardSubstiution(b_block, x_block);
                break;
                
            case StaffIRK::PreconditionerSparsity::UPPERTRIANGULAR:
                BlockBackwardSubstitution(b_block, x_block);
                break;
                
            default:
                mfem_error("StaffIRK:Preconditoner Sparsity must be diagonal, or triangular");
        }
    }
    
    /// Purely virtual function we must implement but do not use.
    virtual void SetOperator(const Operator &op) {  }
    
private:    
    
    // Solve block diagonal system
    inline void BlockDiagonalSubstitution(const BlockVector &b_block, BlockVector &x_block) const
    {
        // Note: These systems are decoupled; i.e. this loop could be parallelized over i
        for (int i = 0; i < stageOper.Butcher.s; i++) {
            // Set the scaled time-step for the diagonal block M - scaled_dt*L
            double scaled_dt = stageOper.GetTimeStep() * preconditionerCoefficients(i,i);
            stageOper.IRKOper->SetSystem(i, scaled_dt); // Ensure we precondition the correct system
            // Apply preconditioner to approximately invert diagonal block
            stageOper.IRKOper->ImplicitPrec(b_block.GetBlock(i), x_block.GetBlock(i));
            // x_block.GetBlock(i) -= b_block.GetBlock(i);
            // x_block.GetBlock(i).Print();
        }
    }
    
    // Solve block lower triangular system
    inline void BlockForwardSubstiution(const BlockVector &b_block, BlockVector &x_block) const
    {

        // Solve for the ith stage vector in sequence
        for (int i = 0; i < stageOper.Butcher.s; i++) {
            
            // Subtract previously computed blocks of the solution in the ith 
            // equation to the RHS. The new RHS vector is "temp"
            if (i > 0) {
                temp = b_block.GetBlock(i); // temp <- b(i)
                
                // Subtract out jth solution component
                for (int j = 0; j < i; j++) {
                    stageOper.IRKOper->ApplyL(x_block.GetBlock(j), x_block.GetBlock(i)); // Use x(i) as a temp variable here.
                    double scaled_dt = stageOper.GetTimeStep() * preconditionerCoefficients(i,j);
                    temp.Add(scaled_dt, x_block.GetBlock(i));
                }
            }
            
            // Set the scaled time-step for the diagonal block M - scaled_dt*L
            double scaled_dt = stageOper.GetTimeStep() * preconditionerCoefficients(i,i);
            stageOper.IRKOper->SetSystem(i, scaled_dt); // Ensure we precondition the correct system
            // Apply preconditioner to approximately invert diagonal block
            if (i > 0) {
                stageOper.IRKOper->ImplicitPrec(temp, x_block.GetBlock(i));
            } else {
                stageOper.IRKOper->ImplicitPrec(b_block.GetBlock(i), x_block.GetBlock(i));
            }
        }
    }
    
    // Solve block upper triangular system
    inline void BlockBackwardSubstitution(const BlockVector &b_block, BlockVector &x_block) const 
    {
        // Solve for the ith stage vector in sequence
        for (int i = stageOper.Butcher.s-1; i >= 0; i--) {
            
            // Subtract previously computed blocks of the solution in the ith 
            // equation to the RHS. The new RHS vector is "temp"
            if (i < stageOper.Butcher.s-1) {
                temp = b_block.GetBlock(i); // temp <- b(i)
                
                // Subtract out jth solution component
                for (int j = stageOper.Butcher.s-1; j > i; j--) {
                    stageOper.IRKOper->ApplyL(x_block.GetBlock(j), x_block.GetBlock(i)); // Use x(i) as a temp variable here.
                    double scaled_dt = stageOper.GetTimeStep() * preconditionerCoefficients(i,j);
                    temp.Add(scaled_dt, x_block.GetBlock(i));
                }
            }
            
            // Set the scaled time-step for the diagonal block M - scaled_dt*L
            double scaled_dt = stageOper.GetTimeStep() * preconditionerCoefficients(i,i);
            stageOper.IRKOper->SetSystem(stageOper.Butcher.s-1-i, scaled_dt); // Ensure we precondition the correct system
            // Apply preconditioner to approximately invert diagonal block
            if (i < stageOper.Butcher.s-1) {
                stageOper.IRKOper->ImplicitPrec(temp, x_block.GetBlock(i));
            } else {
                stageOper.IRKOper->ImplicitPrec(b_block.GetBlock(i), x_block.GetBlock(i));
            }
        }
    }
    
};




#endif
