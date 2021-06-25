#ifndef IRK_UTILS_H
#define IRK_UTILS_H

#include "HYPRE.h"
#include "mfem.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;


// TODO : Check function definitions. The obvious definition to me is
// having ExplicitMult() apply the explicit splitting, and  likewise for
// ImplicitMult() and the implicit splitting. Right now in the code
// though, we have ImplicitMult() applies a mass matrix and
// ExplicitMult() applies the operator that we are solving implicitly.


/// Kronecker transform between two block vectors: y <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, const BlockVector &x, BlockVector &y);

/// Kronecker transform in place: x <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, BlockVector &x);

/// Kronecker transform between two block vectors using A transpose: y <- (A^\top \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, const BlockVector &x, BlockVector &y);

/// Kronecker transform using A transpose in place: x <- (A^T \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, BlockVector &x);

/// Krylov solve type for IRK systems
enum KrylovMethod {
    FP = -1, CG = 0, MINRES = 1, GMRES = 2, BICGSTAB = 3, FGMRES = 4
};

/// Parameters for Krylov solver(s)
struct KrylovParams {
    double abstol = 1e-10;
    double reltol = 1e-10;
    int maxiter = 100;
    int printlevel = 0;
    int kdim = 30;
    bool iterative_mode = false;
    KrylovMethod solver = KrylovMethod::GMRES;
}; 

/** Class for spatial discretizations of a PDE resulting in the time-dependent, 
    nonlinear ODEs
        M*du/dt = N(u,t)    [OR du/dt = N(u,t) if the mass is the identity]
    MFEM typically treats time integration as 
        du/dt = F^{-1} G(u),
    Here F represents what MFEM calls the “implicit” part, and G represents the
    “explicit” part. I think these are largely used for PETSc integration.
    Usually the implicit part F is just a mass matrix. 

    Here, we have redefined the Mult functions to make more sense and work
    for IMEX schemes as well. For non-identity mass matrices, the following
    virtual function must be implemented:
        MasstMult(x,y): y <- M*x
    */
class IRKOperator : public TimeDependentOperator
{   
public:
    /** Describes how the s explicit gradient operators 
            {N'} == (N'(u+dt*x1),...,N'(u+dt*xs))
        are provided by the user. This determine's the solver used by IRK, and 
        which set of virtual functions from this class must be implemented.
    
        APPROXIMATE - The s operators are approximated with the operator Na' 
        that is (in some sense) representative of all s of them, e.g., 
            Na' = N'(u+dt*avg(w1,...,ws)).
        
        EXACT - The s different operators  {N'} are provided */ 
    enum ExplicitGradients {
        APPROXIMATE = 0,// Na'
        EXACT = 1       // {N'}
    };     
     
protected:
    MPI_Comm m_comm;
    ExplicitGradients m_gradients; 
    mutable Vector temp;    // Auxillary vector
    bool m_linearly_imp;      // Linearly implicit (or fully linear)
    
public:
    IRKOperator(MPI_Comm comm, bool linearly_imp, int n=0, double t=0.0,
        Type type=EXPLICIT, ExplicitGradients ex_gradients=EXACT);

    // Sets linearly implicit to false by default
    IRKOperator(MPI_Comm comm, int n=0, double t=0.0,
        Type type=EXPLICIT, ExplicitGradients ex_gradients=EXACT)
        : IRKOperator(comm, false, n, t, type, ex_gradients) { };

    ~IRKOperator() { };

    /// Get MPI communicator
    inline MPI_Comm GetComm() { return m_comm; }

    inline bool IsLinearlyImplicit() { return m_linearly_imp; }

    /// Get type of explicit gradients 
    ExplicitGradients GetExplicitGradientsType() const { return m_gradients; }

    /* ---------------------------------------------------------------------- */
    /* ----------------------- Pure virtual functions ----------------------- */
    /* ---------------------------------------------------------------------- */
    /** Apply action of implicit part of operator y <- N_I(x,y). For fully
        implicit schemes, this just corresponds to applying the time-dependent
        (nonlinear) operator.
        PREVIOUSLY CALLED ExplicitMult */
    virtual void ImplicitMult(const Vector &x, Vector &y) const = 0;

    /** Apply action of explicit part of operator y <- N_E(x,y) */
    virtual void ExplicitMult(const Vector &x, Vector &y) const;

    /** Apply preconditioner set with previous call to SetPreconditioner()
        If not re-implemented, this method simply generates an error. */    
    virtual void ImplicitPrec(const Vector &x, Vector &y) const;

    /** Apply preconditioner set with call to SetPreconditioner() using `index.`
        If not re-implemented, this method simply generates an error. */  
    virtual void ImplicitPrec(int index, const Vector &x, Vector &y) const;

    /** Add implicit forcing function to rhs,
            rhs += r*f(t + r*z),
        for time-dependent forcing function f.
        If not reimplemented, rhs is not modified.
        - Important for linearly implicit methods, where forcing function
        cannot be included in ImplicitMult function. */
    virtual void AddImplicitForcing(Vector &rhs, double t, double r, double z) { };

    
    /* Only used for standard RK IMEX schemes. */
    virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k) { };

    /** Solve M*x - dtf(x, t) = b, where f() is the implicit part of the operator. */
    virtual void ImplicitSolve2(const double dt, const Vector &b, Vector &x)
    { mfem_error("IMEXTimeDependentOperator::ImplicitSolve2() is not overridden!"); };

    /* ---------------------------------------------------------------------- */
    /* ---------------- Virtual functions for Type::IMPLICIT ---------------- */
    /* -- Note, this weird MFEM notation IMPLICIT means has a mass matrix. -- */
    /* ---------------------------------------------------------------------- */

    /** Apply action mass matrix, y = M*x. 
        If not re-implemented, this method simply generates an error.
        PREVIOUSLY CALLED ImplictMult */
    virtual void MassMult(const Vector &x, Vector &y) const;

    /** Apply action of inverse of mass matrix, y = M^{-1}*x. 
        If not re-implemented, this method simply generates an error.
        NOTE : only necessary for PolyIMEX methods. */
    virtual void MassInv(const Vector &x, Vector &y) const;

    /* ---------------------------------------------------------------------- */
    /* ------ Virtual functions for ExplicitGradients::APPROXIMATE ------ */
    /* ---------------------------------------------------------------------- */
    
    /** -- IRK -- Set approximate gradient Na' which is an approximation
        to the s explicit gradients 
            {N'} == {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
        Such that it is referenceable with ExplicitGradientMult() and 
        SetPreconditioner() 
        If not re-implemented, this method simply generates an error
        for problems with nonlinear implicit component.

        This function is called from TriJacSolver.SetOperator, which is
        called within each Newton iteration, with the updated nonlinear
        iterate. */
    virtual void SetExplicitGradient(const Vector &u, double dt, 
                                     const BlockVector &x, const Vector &c);

    /** -- PolyIMEX --  Set approximate gradient Na' which is an 
        approximation to the s explicit gradients 
            {N'} == {N'(x[i], this->GetTime() + r*z[i])}, i=0,...,s-1
        Such that it is referenceable with ExplicitGradientMult() and 
        SetPreconditioner() 
        If not re-implemented, this method simply generates an error
        for problems with nonlinear implicit component.
        
        This function is called from TriJacSolver.SetOperator, which is
        called within each Newton iteration, with the updated nonlinear
        iterate. */
    virtual void SetExplicitGradient(double r, const BlockVector &x,
                                     const Vector &z);

    /** Compute action of Na' explicit gradient operator.
        For problems with
            m_linearly_imp = true,
        this function should be equivalent to self->ImplicitMult()
        *without forcing functions.*
        If not re-implemented, this method simply generates an error. */    
    virtual void ExplicitGradientMult(const Vector &x, Vector &y) const;

    /** Assemble preconditioner for gamma*M - dt*Na' that's applied by
        by calling: 
            1. ImplicitPrec(.,.) if no further calls to SetPreconditioner() are made
            2. ImplicitPrec(index,.,.) */
    virtual void SetPreconditioner(int index, double dt, double gamma, int type);
    
    /* ---------------------------------------------------------------------- */
    /* --------- Virtual functions for ExplicitGradients::EXACT --------- */
    /* ---------------------------------------------------------------------- */

    /** -- IRK -- Set the explicit gradients 
            {N'} == {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
        Such that they are referenceable with ExplicitGradientMult() and 
        SetPreconditioner().
        If not re-implemented, this method simply generates an error
        for problems with nonlinear implicit component.

        This function is called from TriJacSolver.SetOperator, which is
        called within each Newton iteration, with the updated nonlinear
        iterate. */
    virtual void SetExplicitGradients(const Vector &u, double dt, 
                                      const BlockVector &x, const Vector &c);
    
    /** -- PolyIMEX -- Set the explicit gradients 
            {N'} == {N'(x[i], this->GetTime() + r*z[i])}, i=0,...,s-1
        Such that they are referenceable with ExplicitGradientMult() and 
        SetPreconditioner().
        If not re-implemented, this method simply generates an error
        for problems with nonlinear implicit component.

        This function is called from TriJacSolver.SetOperator, which is
        called within each Newton iteration, with the updated nonlinear
        iterate. */
    virtual void SetExplicitGradients(double r, const BlockVector &x,
                                      const Vector &z);

    /** Compute action of `index`-th explicit gradient operator.
        If not re-implemented, this method simply generates an error
        for problems with nonlinear implicit component. */    
    virtual void ExplicitGradientMult(int index, const Vector &x, Vector &y) const;

    /** Assemble preconditioner for matrix 
            gamma*M - dt*<weights,{N'}> 
        that's applied by by calling: 
            1. ImplicitPrec(.,.) if no further calls to SetPreconditioner() are made
            2. ImplicitPrec(index,.,.)  
        If not re-implemented, this method simply generates an error. */    
    virtual void SetPreconditioner(int index, double dt, double gamma, Vector weights);
    
    /* ---------------------------------------------------------------------- */
    /* ---------- Helper functions for ExplicitGradients::EXACT --------- */
    /* ---------------------------------------------------------------------- */
    
    /// Compute y <- y + c*<weights,{N'}>x
    void AddExplicitGradientsMult(double c, const Vector &weights, 
                                         const Vector &x, Vector &y) const;
    
    /** Compute y1 <- y1 + c1*<weights1,{N'}>x, 
                y2 <- y2 + c2*<weights2,{N'}>x */
    void AddExplicitGradientsMult(double c1, const Vector &weights1, 
                                         double c2, const Vector &weights2, 
                                         const Vector &x, Vector &y1, Vector &y2) const;
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
    //  IMEX schemes have same enum structure, with additional
    //  leading 1, e.g., 27 -> 127.
    enum Type { 
        DUMMY = -1000000000, 
        ASDIRK3 = -13, ASDIRK4 = -14,
        LSDIRK1 = 01, LSDIRK2 = 02, LSDIRK3 = 03, LSDIRK4 = 04,
        Gauss2 = 12, Gauss4 = 14, Gauss6 = 16, Gauss8 = 18, Gauss10 = 110,
        RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27, RadauIIA9 = 29,
        LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36, LobIIIC8 = 38,
        IMEXGauss4 = 114,
        IMEXRadauIIA3 = 123, 
        IMEXLobIIIC2 = 132
    };  
    
private:    
    Type ID;
    
    /// Set data required by solvers 
    void SetData();     
    /// Set dimensions of data structures 
    void SizeData();    
    
    /** Set dummy 2x2 RK data that has the specified values of 
        beta_on_eta == beta/eta and eta */
    void SetDummyData(double beta_on_eta, double eta);
    
public:
    bool IsIMEX() const {
        if (static_cast<int>(this->ID) > 100) return true;
        else return false;
    };

    /// Constructor for real RK schemes
    RKData(Type ID_) : ID(ID_) { SetData(); };
    
    /** Constructor for setting dummy RK data with 2x2 matrix having complex 
        conjugate eigenvalues with ratio beta_on_eta and real-component of eta */
    RKData(double beta_on_eta, double eta = 1.0) : ID(DUMMY) { 
        mfem_warning("This is not a valid RK scheme. Setting beta/eta is to be used for testing purposes only!\n"); 
        SetDummyData(beta_on_eta, eta); 
    };
    
    ~RKData() {};
    
    // Standard data required by all solvers
    int s;              // Number of stages
    int s_eff;          // Number of eigenvalues of A0 once complex conjugates have been combined
    DenseMatrix A0;     // The Butcher matrix (includes explicit stage for PolyIMEX)
    DenseMatrix invA0;  // Inverse of Buthcher matrix
    Vector b0;          // Butcher tableau weights
    Vector c0;          // Butcher tableau nodes
    Vector d0;          // inv(A0^\top)*b0
    
    // Associated data required by NONLINEAR solver
    DenseMatrix Q0;     // Orthogonal matrix in Schur decomposition of A0^-1
    DenseMatrix R0;     // Quasi-upper triangular matrix in Schur decomposition of A0^-1
    Array<int> R0_block_sizes; // From top of block diagonal, sizes of blocks

    // PolyIMEX data
    Vector z0;          // Quadrature nodes for PolyIMEX
    // DenseMatrix A0_it;           // *--- IF A0_it != A0 ---*
    DenseMatrix expA0;
    DenseMatrix expA0_it;
    double alpha;
    int order;          // Order of integration
    bool is_imex;

};    


class TriJacSolver;
/** Operator F defining the s stage equations for IRK or Polynomial
    IMEX methods.

    IRK: nonlinear equations satisfy F(w) = 0, where w = (A0 \otimes I)*k, and

                       [ F1 ]                         [N(u+dt*w1,t+c1*dt)]
        F(w;u,t,dt) =  [ .. ] = (inv(A0) \otimes M) - [        ....      ]
                       [ Fs ]                         [N(u+dt*ws,t+cs*dt)]

    Note, this operator exactly represents the original nonlinear stage
    equations with a change of variable; however, there is no other 
    scaling/modification. 

    PolyIMEX: nonlinear equations satisfy F(u) = 0, where

                    [ F1 ]                         [N(u1,t+c1*dt)]   [g1]
        F(u,t,dt) = [ .. ] = (inv(A0) \otimes M) - [    ....     ] - [..]
                    [ Fs ]                         [N(us,t+cs*dt)]   [gs]
    where {g1,...,gs} is a supplied right-hand side including forcing
    functions and explicit solution at previous time steps.
    */
class IRKStageOper : public BlockOperator
{
        
private:
    // Jacobian solver need access
    friend class TriJacSolver;     
    
    Array<int>& offsets;            // Offsets of operator     
    
    mutable IRKOperator* IRKOper;  // Spatial discretization
    const RKData& Butcher;          // All RK information
    
    // Parameters that operator depends on
    const Vector* u;    // Current state; NULL for PolyIMEX methods
    double t;           // Current time of state.
    double dt;          // Current time step
    
    // Wrappers for vectors
    mutable BlockVector w_block, y_block;
    
    // Auxillary vectors
    mutable BlockVector temp_block;
    mutable Vector temp_vector1, temp_vector2;    
    
    Operator* dummy_gradient; 
    
    // Current iterate that true Jacobian would linearize with (this is passed 
    // into GetGradient())
    mutable BlockVector current_iterate;
    
    // Number of times GetGradient() been called with current states (u, t, dt).
    mutable int getGradientCalls;
    
public:
    
    IRKStageOper(IRKOperator * S_, Array<int> &offsets_, const RKData &RK_) 
        : BlockOperator(offsets_), 
        IRKOper(S_), Butcher(RK_), 
        u(NULL), t(0.0), dt(0.0), 
        offsets(offsets_),
        w_block(offsets_), y_block(offsets_),
        temp_vector1(S_->Height()),
        temp_vector2(S_->Height()), 
        temp_block(offsets_),
        current_iterate(),
        dummy_gradient(NULL),
        getGradientCalls(0)
    { };
    
    /// Set current time, time step, and solution for IRK methods
    void SetParameters(const Vector* u_, double t_, double dt_);

    /// Set current time and time stepfor PolyIMEX methods
    void SetParameters(double t_, double dt_);

    inline double GetTimeStep() { return dt; };
    inline double GetTime() { return t; };
    
    /// Return reference to current iterate
    inline const BlockVector &GetCurrentIterate() { return current_iterate; };

    /// Return number of GetGradientCalls since setting parameters
    inline int GetGradientCalls() { return getGradientCalls; };

    /** Meant to return Jacobian of operator. This is called by Newton during 
        every iteration, and the result will be passed in to its linear solver 
        via its SetOperator(). We don't need this function in its current form,
        however. */
    virtual Operator &GetGradient(const Vector &w) const;
    
    /// Compute action of operator, y <- F(w)
    void Mult(const Vector &w_vector, Vector &y_vector) const;
};


/** Class describing the operator that's formed by taking the "quasi" product of
    an orthogonal matrix Q with itself */
class QuasiMatrixProduct : public Array2D<Vector> {
    
private:
    int height;

public: 
    QuasiMatrixProduct(DenseMatrix Q);
    
    void Sparsify(int sparsity);
    
    void Print() const;
    
    /** Lump elements in Vectors to the |largest| entry. Note by orthogonality 
        of Q, diagonal entries lump to 1, and off diagonal entries lump to 0 */
    void Lump();
    
    /// Set all Vectors not on diagonal equal to zero
    void TruncateOffDiags();
};


/** Defines diagonal blocks appearing in Jacobian. These take the form 1x1 or 
    2x2 blocks. The form of these operators depends on 
    IRKOper::GetExplicitGradientsType()

    If ExplicitGradients==APPROXIMATE:
        1x1: 
             [R(0,0)*M - dt*Na']

        2x2:
             [R(0,0)*M-dt*Na'     R(0,1)*M   ]
             [R(1,0)*M        R(1,1)*M-dt*Na']
    
    If ExplicitGradients==EXACT:
        1x1: 
             [R(0,0)*M-dt*<Z(0,0),{N'}>]

        2x2:
             [R(0,0)*M-dt*<Z(0,0),{N'}>   R(0,1)*M-dt*<Z(0,1),{N'}>]
             [R(1,0)*M-dt*<Z(1,0),{N'}>   R(1,1)*M-dt*<Z(1,1),{N'}>]

    TODO:
        Somehow check that z's are all of length s... Implementation assumes they are.
*/
class JacDiagBlock : public BlockOperator
{
private:
    // Allow preconditioner access so that it can use IRKOper
    friend class JacDiagBlockPrec;
    
    int size;                   // Block size
    const Array<int> &offsets;  // Block offsets for operator
    const IRKOperator &IRKOper; // Class defining M, and explicit gradients
    mutable double dt;          // Current time step 
    mutable Vector temp_vector; // Auxillary vector
    
    // Data defining 1x1 operator
    double R00;
    Vector Z00;
    
    // Additional data required to define 2x2 operator
    double R01, R10, R11;
    Vector Z01, Z10, Z11; 
    mutable BlockVector x_block, y_block;
    
public:

    /// ExplicitGradients==APPROXIMATE, 1x1 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_, 
        double R00_);

    /// ExplicitGradients==EXACT, 1x1 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_, 
        double R00_, Vector Z00_);

    /// ExplicitGradients==APPROXIMATE, 2x2 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_, 
        double R00_, double R01_, double R10_, double R11_);

    /// ExplicitGradients==EXACT, 2x2 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_, 
        double R00_, double R01_, double R10_, double R11_, 
        Vector Z00_, Vector Z01_, Vector Z10_, Vector Z11_);

    inline void SetTimeStep(double dt_) const { dt = dt_; };
    inline double GetTimeStep() const { return dt; };
    inline int Size() const { return size; };
    inline const Array<int> &Offsets() const { return offsets; };

    /// Compute action of diagonal block    
    void Mult(const Vector &x, Vector &y) const;
};


/** Preconditioner for assisting in the inversion of diagonal blocks of the 
    Jacobian matrix, i.e. JacDiagBlock's. As for the diagonal blocks themselves,
    the form of the preconditioner depends on IRKOper::GetExplicitGradientsType()
    
    If ExplicitGradients==APPROXIMATE:
        1x1: 
            [R(0,0)*M - dt*Na'] 
        Is preconditioned by the user's preconditioner applied to itself.    

        2x2:
            [R(0,0)*M-dt*Na'     R(0,1)*M   ]
            [R(1,0)*M        R(1,1)*M-dt*Na']
        Is preconditioned by  
            [R(0,0)*M-dt*Na'          0     ]
            [R(1,0)*M         gamma*M-dt*Na']    
                
    If ExplicitGradients==EXACT:
        1x1: 
            [R(0,0)*M-dt*<Z(0,0),{N'}>]
        Is preconditioned by the user's preconditioner applied to  
            [R(0,0)*M-dt*<Y(0,0),{N'}>]      

        2x2:
            [R(0,0)*M-dt*<Z(0,0),{N'}>  R(1,0)*M-dt*<Z(1,0),{N'}>]
            [R(1,0)*M-dt*<Z(1,0),{N'}>  R(0,0)*M-dt*<Z(1,1),{N'}>]
        Which is preconditioned by 
             [R(0,0)*M-dt*<Y(0,0),{N'}>                0          ]
             [R(1,0)*M-dt*<Y(1,0),{N'}>   gamma*M-dt*<Y(1,1),{N'}>]
        
        NOTE: 
            -In general, one can choose Y == Z, but there is the option to use a 
            different vector (this is why this class doesn't just take this data 
            straight from JacDiagBlock). 
    
    Where for all 2x2 systems, IRKOper.ImplicitPrec(i,x,y) is used to precondition 
    the ith diagonal block */
class JacDiagBlockPrec : public Solver
{
private:
    const JacDiagBlock &BlockOper;
    mutable Vector temp_vector; // Auxillary vector
    bool identity;              // Use identity preconditioner. Useful as a comparison.
    
    // Extra data required for 2x2 blocks
    double R10;                 
    Vector Y10;
    int prec00_idx;             // BlockOper.IRKOper.ImplicitPrec(prec00_idx,.,.) preconditions (0,0) block
    int prec11_idx;             // BlockOper.IRKOper.ImplicitPrec(prec11_idx,.,.) preconditions (1,1) block                 
    mutable BlockVector x_block, y_block;
        
public:
    /// 1x1 block
    JacDiagBlockPrec(const JacDiagBlock &BlockOper_, bool identity_=false);
        
    /// ExplicitGradients==APPROXIMATE, 2x2 block
    JacDiagBlockPrec(const JacDiagBlock &BlockOper_, double R10_, 
        int prec00_idx_, int prec11_idx_, bool identity_=false);

    /// ExplicitGradients==EXACT, 2x2 block
    JacDiagBlockPrec(const JacDiagBlock &BlockOper_, double R10_, Vector Y10_, 
        int prec00_idx_, int prec11_idx_, bool identity_=false);

    ~JacDiagBlockPrec() {}

    /// Apply action of preconditioner
    void Mult(const Vector &x_vector, Vector &y_vector) const;
    
    /// Purely virtual function we must implement but do not use.
    virtual void SetOperator(const Operator &op) {  }
};



/** Approximate Jacobian of (transformed) nonlinear stage equations in
    IRKStageOper via real Schur decomposition and block triangular
    approximation. The operator
        P = (Q0^\top \otimes I) * diag[N'(u+dt*w1),...,N'(u+dt*ws)] * (Q0 \otimes I)
    is approximated by the block upper triangular matrix \tilde{P}, and the 
    corresponding approximate Jacobian
        R0 \otimes M  - \tilde{P}
    is inverted "exactly" via backward substitution. The basic form of \tilde{P} is 
    determined via the IRKOper's type of ExplicitGradients.
    
    If ExplicitGradients==APPROXIMATE
        The s gradients {N'} == (N'(u+dt*w1),...,N'(u+dt*ws)) are each approximated
        by Na', such that 
            \tilde{P} = diag(Na',...,Na'),
        and the Jacobian is written as a difference of Kronecker products    
    
    If ExplicitGradients==EXACT
        The sparsity pattern of \tilde{P} is set by `Z_solver`, and the sparsity
        pattern of \tilde{P} that the preconditioners used to invert the diagonal 
        blocks are assembled on is set by `Z_prec`. 
        For example,
            -if Z_solver.Sparsity == LUMPED, then \tilde{P} is a block diagonal matrix,
                with each block being one of {N'(u+dt*w1),...,N'(u+dt*ws)} (the exact 
                one is the index of the |largest| weight in Z_solver before being 
                sparsified)
            
            -if Z_solver.Sparsity == DIAGONAL, then \tilde{P} is a block diagonal matrix,
                with each block a linear combination of {N'(u+dt*w1),...,N'(u+dt*ws)}

            -if Z_solver.Sparsity == DENSE, then \tillde{P} is the block upper triangular
                matrix formed by truncating P into the block sparsity pattern of R0 \otimes M

        For example, if the ith diagonal block is 1x1, then it is preconditioned by
        applying IRKOper.SetSystem applied to
            R0(i,i)*M - dt*<Z_prec(i,i),{N'}>. 
            
    NOTE:  
        If ExplicitGradients==APPROXIMATE, Z_solver==Z_prec==NULL is permissible
        since these variables are ignored regardless of their values. */
class TriJacSolver : public Solver
{

private:
    
    IRKStageOper &StageOper;
    
    int printlevel;
    int jac_update_rate;    // How frequently is Jacobian updated?
    int gamma_idx;          // Constant used to precondition Schur complement
    
    Array<int> &offsets;    // Offsets for vectors with s blocks
    Array<int> offsets_1;   // Offsets for vectors with 1 block
    Array<int> offsets_2;   // Offsets for vectors with 2 blocks
    
    // Auxillary vectors  
    mutable BlockVector x_block, b_block, b_block_temp, x_block_temp;   // s blocks
    mutable BlockVector y_2block, z_2block;                             //  2 blocks 
    mutable Vector temp_vector1, temp_vector2;  
     
    // Diagonal blocks inverted during backward substitution
    Array<JacDiagBlock *> DiagBlock;
    
    // Preconditioners to assist with inversion of diagonal blocks
    Array<JacDiagBlockPrec *> DiagBlockPrec;
    
    // Solvers for inverting diagonal blocks
    IterativeSolver * krylov_solver1; // 1x1 solver
    IterativeSolver * krylov_solver2; // 2x2 solver
    // NOTE: krylov_solver2 is just a pointer to krylov_solver1 so long as there aren't 
    // both 1x1 and 2x2 systems to solve AND different solver parameters weren't passed 
    bool multiple_krylov; // Do we really use two different solvers?
    
    // Number of Krylov iterations for each diagonal block
    mutable vector<int> krylov_iters;
    
    // Sparsity pattern of \tilde{P}
    bool kronecker_form;    // Structure of \tilde{P}: Just a short hand for StageOper.IRKOper->GetExplicitGradient
    const QuasiMatrixProduct * Z_solver;    // For solver
    const QuasiMatrixProduct * Z_prec;      // For diagonal preconditioners

    /// Set up Krylov solver for inverting diagonal blocks
    void GetKrylovSolver(IterativeSolver * &solver, const KrylovParams &params) const;
    
    /** Solve \tilde{J}*y = z via block backward substitution, where 
            \tilde{J} = R \otimes M - dt * \tilde{P}
        NOTE: RHS vector z is not const, since its data is overridden during the 
        solve */
    void BlockBackwardSubstitution(BlockVector &z_block, BlockVector &y_block) const;

public:

    /** General constructor, where 1x1 and 2x2 systems can use different Krylov 
        solvers.
        NOTE: To use only a single solver, requires &solver_params1==&solver_params2 */
    TriJacSolver(IRKStageOper &StageOper_, int jac_update_rate_, int gamma_idx_,
                    const KrylovParams &solver_params1, const KrylovParams &solver_params2,
                    const QuasiMatrixProduct * Z_solver_, const QuasiMatrixProduct * Z_prec_);
    
    /// Constructor for when 1x1 and 2x2 systems use same solver
    TriJacSolver(IRKStageOper &StageOper_, int jac_update_rate_, int gamma_idx_,
                    const KrylovParams &solver_params,
                    const QuasiMatrixProduct * Z_solver, const QuasiMatrixProduct * Z_prec)   
        : TriJacSolver(StageOper_, jac_update_rate_, gamma_idx_, solver_params, solver_params, 
        Z_solver, Z_prec) {};

    ~TriJacSolver();
    
    /// Functions to track solver progress
    inline vector<int> GetNumIterations() { return krylov_iters; };
    void ResetNumIterations();

    /** Newton method will pass the operator returned from its GetGradient() to 
        this, but we don't actually use it. Instead, we update the approximate 
        gradient Na' or the exact gradients {N'} if requested */
    void SetOperator (const Operator &op);

    /** Solve J*x = b for x, J=A^-1 \otimes M - dt * (Q \otimes I) * \tilde{P} * (Q^\top \otimes I)
        We first transform J*x=b into 
            [Q^\top J Q][Q^\top * x]=[Q^\top * b] 
                        <==> 
            \tilde{J} * x_temp = b_temp,
        i.e., \tilde{J} = R \otimes M - dt * \tilde{P}, 
        x_temp = Q^\top * x_block, b_temp = Q^\top * b_block */
    void Mult(const Vector &b_vector, Vector &x_vector) const;
};


#endif
