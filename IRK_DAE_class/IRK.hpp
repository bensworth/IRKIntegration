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

/// Kronecker transform between two block vectors using A transpose: y <- (A^\top \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, const BlockVector &x, BlockVector &y);


/** Class for spatial discretizations of a PDE resulting in the time-dependent,
    nonlinear ODEs
        M*du/dt = N(u,t)    [OR du/dt = N(u,t) if the mass is the identity]

    For non-identity mass matrices, the following virtual function must be implemented:
        ImplicitMult(x,y): y <- M*x */
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
    mutable Vector temp; // Auxillary vector

public:
    IRKOperator(MPI_Comm comm, int n=0, double t=0.0, Type type=EXPLICIT,
                ExplicitGradients ex_gradients=EXACT)
        : TimeDependentOperator(n, t, type),
            m_comm(comm), m_gradients(ex_gradients), temp(n) {};

    ~IRKOperator() { };

    /// Get MPI communicator
    inline MPI_Comm GetComm() { return m_comm; };

    /// Get type of explicit gradients
    inline ExplicitGradients GetExplicitGradientsType() const { return m_gradients; }


    /* ---------------------------------------------------------------------- */
    /* ----------------------- Pure virtual functions ----------------------- */
    /* ---------------------------------------------------------------------- */
    /// Apply action of M*du/dt, y <- L(x,y)
    virtual void ExplicitMult(const Vector &x, Vector &y) const = 0;


    /** Apply preconditioner set with previous call to BuildPreconditioner()
        If not re-implemented, this method simply generates an error. */
    virtual void ImplicitPrec(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ImplicitPrec() is not overridden!");
    }

    /** Apply preconditioner set with call to BuildPreconditioner() using `index.`
        If not re-implemented, this method simply generates an error. */
    virtual void ImplicitPrec(int index, const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ImplicitPrec() is not overridden!");
    }

    /* ---------------------------------------------------------------------- */
    /* ---------------- Virtual functions for Type::IMPLICIT ---------------- */
    /* ---------------------------------------------------------------------- */
    /** Apply action mass matrix, y = M*x.
        If not re-implemented, this method simply generates an error. */
    virtual void ImplicitMult(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ImplicitMult() is not overridden!");
    }


    /* ---------------------------------------------------------------------- */
    /* ------ Virtual functions for ExplicitGradients::APPROXIMATE ------ */
    /* ---------------------------------------------------------------------- */

    /** Set approximate gradient Na' which is an approximation to the s explicit
        gradients
            {N'} == {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
        Such that it is referenceable with ExplicitGradientMult() and
        BuildPreconditioner()
        If not re-implemented, this method simply generates an error. */
    virtual void SetExplicitGradient(const Vector &u, double dt,
                                     const BlockVector &x, const Vector &c)
    {
        MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE,
                    "IRKOperator::SetExplicitGradient() applies only for \
                    ExplicitGradients::APPROXIMATE");
        mfem_error("IRKOperator::SetExplicitGradient() is not overridden!");
    }

    /** Compute action of Na' explicit gradient operator.
        If not re-implemented, this method simply generates an error. */
    virtual void ExplicitGradientMult(const Vector &x, Vector &y) const
    {
        MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE,
                    "IRKOperator::ExplicitGradientMult() applies only for \
                    ExplicitGradients::APPROXIMATE");

        mfem_error("IRKOperator::ExplicitGradientMult() is not overridden!");
    }

    /** Assemble preconditioner for gamma*M - dt*Na' that's applied by
        by calling:
            1. ImplicitPrec(.,.) if no further calls to BuildPreconditioner() are made
            2. ImplicitPrec(index,.,.) */
    virtual void BuildPreconditioner(int index, double dt, double gamma, int type)
    {
        MFEM_ASSERT(m_gradients == ExplicitGradients::APPROXIMATE,
                    "IRKOperator::SetExplicitGradient() applies only for \
                    ExplicitGradients::APPROXIMATE");
        mfem_error("IRKOperator::BuildPreconditioner() is not overridden!");
    }

    /** Assemble preconditioner for Schur Complement that's applied by calling:
            ImplicitPrec(index,.,.)
        If not reimplemented, this function calls BuildPreconditioner(), which
        assembles a precondtioner for gamma*M - dt*Na'. */
    virtual void BuildPreconditionerSchur(int index, double dt, double gamma, int type)
    {
        BuildPreconditioner(index, dt, gamma, type);
    }

    /** Assemble preconditioner for full 1x1 or 2x2 block system in Kronecker form.
        Applied by calling
            ImplicitPrec(index,.,.)
        Only needed if schur_precondition=false. */
    virtual void BuildGenPreconditioner(int index, double dt, double eta, double phi, double beta, int type)
    {
        mfem_error("IRKOperator::BuildGenPreconditioner() is not overridden!");
    }

    /* ---------------------------------------------------------------------- */
    /* --------- Virtual functions for ExplicitGradients::EXACT --------- */
    /* ---------------------------------------------------------------------- */

    /** Set the explicit gradients
            {N'} == {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
        Such that they are referenceable with ExplicitGradientMult() and
        BuildPreconditioner()

        If not re-implemented, this method simply generates an error. */
    virtual void SetExplicitGradients(const Vector &u, double dt,
                                      const BlockVector &x, const Vector &c)
    {
        MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT,
                    "IRKOperator::SetExplicitGradients() applies only for \
                    ExplicitGradients::EXACT");
        mfem_error("IRKOperator::SetExplicitGradients() is not overridden!");
    }

    /** Compute action of `index`-th explicit gradient operator.
        If not re-implemented, this method simply generates an error. */
    virtual void ExplicitGradientMult(int index, const Vector &x, Vector &y) const
    {
        MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT,
                    "IRKOperator::ExplicitGradientMult() applies only for \
                    ExplicitGradients::EXACT");
        mfem_error("IRKOperator::ExplicitGradientMult() is not overridden!");
    }

    /** Assemble preconditioner for matrix
            gamma*M - dt*<weights,{N'}>
        that's applied by by calling:
            1. ImplicitPrec(.,.) if no further calls to BuildPreconditioner() are made
            2. ImplicitPrec(index,.,.)

        If not re-implemented, this method simply generates an error. */
    virtual void BuildPreconditioner(int index, double dt, double gamma, Vector weights)
    {
        mfem_error("IRKOperator::BuildPreconditioner() is not overridden!");
    }

    /** Assemble preconditioner for Schur Complement that's applied by calling:
            ImplicitPrec(index,.,.)
        If not reimplemented, this function calls BuildPreconditioner(), which
        assembles a precondtioner for gamma*M - dt*<weights,{N'}>. */
    virtual void BuildPreconditionerSchur(int index, double dt, double gamma, Vector weights)
    {
        BuildPreconditioner(index, dt, gamma, weights);
    }

    /** Assemble preconditioner for full 2x2 block system, in non-Kronecker form,
        with weight vectors weights1 and weights2. Applied by calling
            ImplicitPrec(index,.,.)
        Only needed if schur_precondition=false. */
    virtual void BuildGenPreconditioner(int index, double dt, double eta, double phi, double beta,
        Vector weights1, Vector weights2)
    {
        mfem_error("IRKOperator::BuildGenPreconditioner() is not overridden!");
    }

    /* ---------------------------------------------------------------------- */
    /* ---------- Helper functions for ExplicitGradients::EXACT --------- */
    /* ---------------------------------------------------------------------- */

    /// Compute y <- y + c*<weights,{N'}>x
    inline void AddExplicitGradientsMult(double c, const Vector &weights,
                                         const Vector &x, Vector &y) const
    {

        MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT,
                    "IRKOperator::AddExplicitGradientsMult() applies only for \
                    ExplicitGradients::EXACT");

        MFEM_ASSERT(weights.Size() > 0,
            "IRKOperator::AddExplicitGradientsMult() not defined for empty weights");

        for (int i = 0; i < weights.Size(); i++) {
            if (fabs(c*weights(i)) > 0.) {
                ExplicitGradientMult(i, x, temp);
                y.Add(c*weights(i), temp);
            }
        }
    }

    /** Compute y1 <- y1 + c1*<weights1,{N'}>x,
                y2 <- y2 + c2*<weights2,{N'}>x */
    inline void AddExplicitGradientsMult(double c1, const Vector &weights1,
                                         double c2, const Vector &weights2,
                                         const Vector &x,
                                         Vector &y1, Vector &y2) const
    {

        MFEM_ASSERT(m_gradients == ExplicitGradients::EXACT,
                    "IRKOperator::AddExplicitGradientsMult() applies only for \
                    ExplicitGradients::EXACT");
        MFEM_ASSERT(weights1.Size() > 0 && weights2.Size() > 0,
            "IRKOperator::AddExplicitGradientsMult() not defined for empty weights");
        MFEM_ASSERT(weights1.Size() == weights2.Size(),
            "IRKOperator::AddExplicitGradientsMult() weight vectors need to be of equal length");

        for (int i = 0; i < weights1.Size(); i++) {
            if (fabs(c1*weights1(i)) > 0. || fabs(c2*weights2(i)) > 0.) {
                ExplicitGradientMult(i, x, temp);
                if (fabs(c1*weights1(i)) > 0.) y1.Add(c1*weights1(i), temp);
                if (fabs(c2*weights2(i)) > 0.) y2.Add(c2*weights2(i), temp);
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
        DUMMY = -1000000000,
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

    /** Set dummy 2x2 RK data that has the specified values of
        beta_on_eta == beta/eta and eta */
    void SetDummyData(double beta_on_eta, double eta);

public:
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


class TriJacSolver;
/** Operator F defining the s stage equations. They satisfy F(w) = 0,
    where w = (A0 \otimes I)*k, and

                       [ F1 ]                         [N(u+dt*w1,t+c1*dt)]
                       [ .. ]                         [        ....      ]
        F(w;u,t,dt) =  [ .. ] = (inv(A0) \otimes M) - [        ....      ]
                       [ .. ]                         [        ....      ]
                       [ Fs ]                         [N(u+dt*ws,t+cs*dt)] */
class IRKStageOper : public BlockOperator
{

private:
    // Jacobian solver need access
    friend class TriJacSolver;

    mutable IRKOperator * IRKOper;  // Spatial discretization
    const RKData &Butcher;          // All RK information

    Array<int> &offsets;            // Offsets of operator

    // Parameters that operator depends on
    const Vector * u;             // Current state.
    double t;                     // Current time of state.
    double dt;                    // Current time step

    // Wrappers for scalar vectors
    mutable BlockVector w_block, y_block;

    // Auxillary vectors
    mutable Vector temp_vector1, temp_vector2;
    mutable BlockVector temp_block;


    // Current iterate that true Jacobian would linearize with (this is passed
    // into GetGradient())
    mutable BlockVector current_iterate;

    Operator * dummy_gradient;
    // Number of times GetGradient() been called with current states (u, t, dt).
    mutable int getGradientCalls;

public:

    IRKStageOper(IRKOperator * S_, Array<int> &offsets_, const RKData &RK_)
        : BlockOperator(offsets_),
            IRKOper(S_), Butcher(RK_),
            offsets(offsets_),
            u(NULL),
            t(0.0),
            dt(0.0),
            w_block(offsets_), y_block(offsets_),
            temp_vector1(S_->Height()), temp_vector2(S_->Height()),
            temp_block(offsets_),
            current_iterate(),
            dummy_gradient(NULL),
            getGradientCalls(0)
            { };

    inline void SetParameters(const Vector * u_, double t_, double dt_) {
        t = t_;
        dt = dt_;
        u = u_;
        getGradientCalls = 0; // Reset counter
    };

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
    inline virtual Operator &GetGradient(const Vector &w) const
    {
        // Update `current_iterate` so that its data points to the current iterate's
        current_iterate.Update(w.GetData(), offsets);

        // Increment counter
        getGradientCalls++;

        // To stop compiler complaining of no return value
        return *dummy_gradient;
    }

    /// Compute action of operator, y <- F(w)
    inline void Mult(const Vector &w_scalar, Vector &y_scalar) const
    {
        MFEM_ASSERT(u, "IRKStageOper::Mult() Requires states to be set, see SetParameters()");

        // Wrap scalar Vectors with BlockVectors
        w_block.Update(w_scalar.GetData(), offsets);
        y_block.Update(y_scalar.GetData(), offsets);

        /* y <- inv(A0)*M*w */
        for (int i = 0; i < Butcher.s; i++) {
            IRKOper->ImplicitMult(w_block.GetBlock(i), temp_block.GetBlock(i));
        }
        KronTransform(Butcher.invA0, temp_block, y_block); // y <- inv(A0)*temp

        /* y <- y - N(u + dt*w) */
        for (int i = 0; i < Butcher.s; i++) {
            add(*u, dt, w_block.GetBlock(i), temp_vector1); // temp1 <- u+dt*w(i)
            IRKOper->SetTime(t + Butcher.c0[i]*dt);
            IRKOper->ExplicitMult(temp_vector1, temp_vector2); // temp2 <- N(temp1, t)
            y_block.GetBlock(i).Add(-1., temp_vector2);
        }
    }
};


/** Class describing the operator that's formed by taking the "quasi" product of
    an orthogonal matrix Q with itself */
class QuasiMatrixProduct : public Array2D<Vector> {

private:
    int height;

public:
    QuasiMatrixProduct(DenseMatrix Q)
        : Array2D<Vector>(Q.Height(), Q.Height()), height(Q.Height())
    {
        MFEM_ASSERT(Q.Height() == Q.Width(), "QuasiMatrixProduct:: Matrix must be square");

        // Create Vectors of coefficients
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < height; col++) {
                (*this)(row, col) = Vector(height);
                for (int i = 0; i < height; i++) {
                    (*this)(row, col)[i] = Q(i,row)*Q(i,col);
                }
            }
        }
    };

    inline void Sparsify(int sparsity) {
        // Sparsify if need be
        switch (sparsity) {
            case 0:
                this->Lump();
                break;
            case 1:
                this->TruncateOffDiags();
                break;
            default:
                break;
        }
    }

    inline void Print() const {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < height; col++) {
                mfem::out << "(" << row << "," << col << "): ";
                (*this)(row, col).Print();
            }
        }
    }

    /** Lump elements in Vectors to the |largest| entry. Note by orthogonality
        of Q, diagonal entries lump to 1, and off diagonal entries lump to 0 */
    inline void Lump() {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < height; col++) {
                // Set all off diagonal entries to zero
                if (row != col) {
                    for (int i = 0; i < height; i++) {
                        (*this)(row, col)[i] = 0.;
                    }
                // Handle diagonal entries
                } else {
                    for (int i = 0; i < height; i++) {
                        (*this)(row, col)[i] = fabs((*this)(row, col)[i]);
                    }
                    double current = (*this)(row, col)[0], max = current;
                    int maxidx = 0;
                    for (int i = 1; i < height; i++) {
                        current = (*this)(row, col)[i];
                        if (current > max) {
                            (*this)(row, col)[maxidx] = 0.;
                            max = current;
                            maxidx = i;
                        } else {
                            (*this)(row, col)[i] = 0.;
                        }
                    }
                    (*this)(row,col)[maxidx] = 1.;
                }
            }
        }
    }

    /// Set all Vectors not on diagonal equal to zero
    inline void TruncateOffDiags() {
        for (int row = 0; row < height; row++) {
            for (int col = 0; col < height; col++) {
                if (row != col) {
                    for (int i = 0; i < height; i++) {
                        (*this)(row, col)[i] = 0.;
                    }
                }
            }
        }
    }
};


/** Class implementing conjugate-pair preconditioned solution of fully implicit
    RK schemes for the nonlinear ODE system
        M*du/dt = N(u,t),
    as implemented as an IRKOperator */
class IRK : public ODESolver
{
public:
    /// Krylov solve type for IRK systems
    enum AccelMethod {
        FP = -1, CG = 0, MINRES = 1, GMRES = 2, BICGSTAB = 3, FGMRES = 4
    };

    /// Parameters for Krylov solver(s)
    struct KrylovParams {
        double abstol = 1e-10;
        double reltol = 1e-10;
        int maxiter = 100;
        int printlevel = 0;
        int kdim = 30;
        AccelMethod solver = AccelMethod::GMRES;
    };

    /** Sparsity pattern of (Q^\top \otimes I) diag(N1',...,Ns') (Q \otimes I)
        to be used in block-upper-triangular Quasi-Newton method */
    enum JacSparsity {
        LUMPED = 0, DIAGONAL = 1, DENSE = 2
    };

    /// Parameters for Newton solver
    struct NewtonParams {
        double reltol = 1e-6;
        double abstol = 1e-6;
        int maxiter = 10;
        int printlevel = 2;

        // How frequently to we update the Jacobian matrix?
        // <=0 - First iteration only
        // 1 - Every iteration
        // x > 1 - Every x iterations
        int jac_update_rate = 0;

        JacSparsity jac_solver_sparsity = JacSparsity::DENSE;
        JacSparsity jac_prec_sparsity  = JacSparsity::DIAGONAL;

        int gamma_idx = 1; // Constant used when preconditioning (2,2) block [0==eta, 1==eta+beta^2/eta].
    };

private:
    MPI_Comm m_comm;
    int m_numProcess;
    int m_rank;

    const RKData &m_Butcher;    // Runge-Kutta Butcher tableau and associated data

    IRKOperator * m_IRKOper;    // Spatial discretization.
    BlockVector m_w;            // The stage vectors
    Array<int> m_stageOffsets;  // Offsets
    IRKStageOper * m_stageOper; // Operator encoding stages, F(w) = 0

    // Nonlinear and linear solvers for computing stage vectors
    bool m_solversInit;             // Have the solvers been initialized?
    NewtonSolver * m_newton_solver; // Nonlinear solver for inverting F
    TriJacSolver * m_tri_jac_solver;// Linear solver for inverting (approximate) Jacobian
    QuasiMatrixProduct * m_jac_solverSparsity;
    QuasiMatrixProduct * m_jac_precSparsity;

    NewtonParams m_newton_params;   // Parameters for Newton solver
    KrylovParams m_krylov_params;   // Parameters for Krylov solver.
    KrylovParams m_krylov_params2;  // Parameters for Krylov solver for 2x2 systems, if different than that for 1x1 systems
    bool m_krylov2;                 // Do we use a second solver?
    bool m_schur_precondition;

    /* Statistics on solution of nonlinear and linear systems across whole integration */
    int m_avg_newton_iter;          // Avg number of Newton iterations per time step
    vector<int> m_avg_krylov_iter;  // Avg number of Krylov iterations per time step (for every 1x1 and 2x2 system)
    vector<int> m_system_size;      // Associated linear system sizes: 1x1 or 2x2?
    vector<double> m_eig_ratio;     // The ratio beta/eta

    /// Build nonlinear and linear solvers
    void SetSolvers();

public:
    IRK(IRKOperator *IRKOper_, const RKData &ButcherTableau);
    ~IRK();

    void Init(TimeDependentOperator &F);

    void Run(Vector &x, double &t, double &dt, double tf);

    void Step(Vector &x, double &t, double &dt);

    /// Set parameters for Newton solver
    inline void SetNewtonParams(NewtonParams params) {
        MFEM_ASSERT(!m_solversInit, "IRK::SetNewtonParams:: Can only be called before IRK::Run()");
        m_newton_params = params;
    }
    /// Set parameters for Krylov solver
    inline void SetKrylovParams(KrylovParams params) {
        MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
        m_krylov_params = params;
        m_krylov2 = false; // Using single Krylov solver
    }
    /** Set parameters for Krylov solver, if different solver to be used to 1x1
        and 2x2 systems */
    // TODO: Really should check these structs are not equal by value here...
    inline void SetKrylovParams(KrylovParams params1, KrylovParams params2) {
        MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
        m_krylov_params  = params1;
        m_krylov_params2 = params2;
        m_krylov2 = true; // Using two Krylov solvers
    }

    inline void UseSchurPreconditioner(bool schur_precond) {
        m_schur_precondition = schur_precond;
    }

    /// Get statistics about solution of nonlinear and linear systems
    inline void GetSolveStats(int &avg_newton_iter,
                                vector<int> &avg_krylov_iter,
                                vector<int> &system_size,
                                vector<double> &eig_ratio) const {
                                    avg_newton_iter = m_avg_newton_iter;
                                    avg_krylov_iter = m_avg_krylov_iter;
                                    system_size     = m_system_size;
                                    eig_ratio       = m_eig_ratio;
                                }
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
    friend class JacDiagBlockTriPrec;
    friend class JacDiagGenPrec;

    int size;                   // Block size
    const Array<int> &offsets;  // Block offsets for operator
    const IRKOperator &IRKOper; // Class defining M, and explicit gradients
    mutable double dt;          // Current time step
    mutable Vector temp_vector; // Auxillary vector

    // Data defining 1x1 operator
    double R00;
    // Additional data required to define 2x2 operator
    double R01, R10, R11;

    Vector Z00;
    Vector Z01, Z10, Z11;
    mutable BlockVector x_block, y_block;

public:

    /// ExplicitGradients==APPROXIMATE, 1x1 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_,
        double R00_)
        : BlockOperator(offsets_),
        size(1), offsets(offsets_), IRKOper(IRKOper_), dt(0.0), temp_vector(IRKOper_.Height()),
        R00(R00_)
        {
            MFEM_ASSERT(IRKOper.GetExplicitGradientsType() == IRKOperator::ExplicitGradients::APPROXIMATE,
                        "JacDiagBlock:: This constructor is for IRKOperator's \
                        with ExplicitGradients::APPROXIMATE");
        }

    /// ExplicitGradients==EXACT, 1x1 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_,
        double R00_, Vector Z00_)
        : BlockOperator(offsets_),
        size(1), offsets(offsets_), IRKOper(IRKOper_), dt(0.0), temp_vector(IRKOper_.Height()),
        R00(R00_), Z00(Z00_)
        {
            MFEM_ASSERT(IRKOper.GetExplicitGradientsType() == IRKOperator::ExplicitGradients::EXACT,
                        "JacDiagBlock:: This constructor is for IRKOperator's \
                        with ExplicitGradients::EXACT");
        }

    /// ExplicitGradients==APPROXIMATE, 2x2 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_,
        double R00_, double R01_, double R10_, double R11_)
        : BlockOperator(offsets_),
        size(2), offsets(offsets_), IRKOper(IRKOper_), dt(0.0), temp_vector(IRKOper_.Height()),
        R00(R00_), R01(R01_), R10(R10_), R11(R11_)
        {
            MFEM_ASSERT(IRKOper.GetExplicitGradientsType() == IRKOperator::ExplicitGradients::APPROXIMATE,
                        "JacDiagBlock:: This constructor is for IRKOperator's \
                        with ExplicitGradients::APPROXIMATE");
        }

    /// ExplicitGradients==EXACT, 2x2 block
    JacDiagBlock(const Array<int> &offsets_, const IRKOperator &IRKOper_,
        double R00_, double R01_, double R10_, double R11_,
        Vector Z00_, Vector Z01_, Vector Z10_, Vector Z11_)
        : BlockOperator(offsets_),
        size(2), offsets(offsets_), IRKOper(IRKOper_), dt(0.0), temp_vector(IRKOper_.Height()),
        R00(R00_), R01(R01_), R10(R10_), R11(R11_),
        Z00(Z00_), Z01(Z01_), Z10(Z10_), Z11(Z11_)
        {
            MFEM_ASSERT(IRKOper.GetExplicitGradientsType() == IRKOperator::ExplicitGradients::EXACT,
                        "JacDiagBlock:: This constructor is for IRKOperator's \
                        with ExplicitGradients::EXACT");
        }

    inline void SetTimeStep(double dt_) const { dt = dt_; };
    inline double GetTimeStep() const { return dt; };
    inline int Size() const { return size; };
    inline const Array<int> &Offsets() const { return offsets; };


    /// Compute action of diagonal block
    inline void Mult(const Vector &x, Vector &y) const
    {
        MFEM_ASSERT(x.Size() == this->Height(), "JacDiagBlock::Mult() incorrect input Vector size");
        MFEM_ASSERT(y.Size() == this->Height(), "JacDiagBlock::Mult() incorrect output Vector size");


        switch (IRKOper.GetExplicitGradientsType()) {
            case IRKOperator::ExplicitGradients::APPROXIMATE:
                // 1x1 operator, y = [R(0)(0)*M - dt*Na']*x
                if (size == 1) {
                    IRKOper.ExplicitGradientMult(x, y);
                    y *= -dt;

                    // Apply mass matrix to x
                    IRKOper.ImplicitMult(x, temp_vector);
                    y.Add(R00, temp_vector);
                }
                // 2x2 operator,
                //  y(0) = [R(0,0)*M-dt*Na']*x(0) + [R(0,1)*M]*x(1)
                //  y(1) = [R(1,0)*M]*x(0)        + [R(1,1)*M-dt*Na']*x(1)
                else if (size == 2) {
                    // Wrap scalar Vectors with BlockVectors
                    x_block.Update(x.GetData(), offsets);
                    y_block.Update(y.GetData(), offsets);

                    // y(0)
                    IRKOper.ExplicitGradientMult(x_block.GetBlock(0), y_block.GetBlock(0));
                    y_block.GetBlock(0) *= -dt;

                    // y(1)
                    IRKOper.ExplicitGradientMult(x_block.GetBlock(1), y_block.GetBlock(1));
                    y_block.GetBlock(1) *= -dt;

                    // M*x(0) dependence
                    IRKOper.ImplicitMult(x_block.GetBlock(0), temp_vector);
                    y_block.GetBlock(0).Add(R00, temp_vector);
                    y_block.GetBlock(1).Add(R10, temp_vector);
                    // M*x(1) dependence
                    IRKOper.ImplicitMult(x_block.GetBlock(1), temp_vector);
                    y_block.GetBlock(0).Add(R01, temp_vector);
                    y_block.GetBlock(1).Add(R11, temp_vector);
                }
                break;


            case IRKOperator::ExplicitGradients::EXACT:
                /** 1x1 operator,
                        y = [R(0,0)*M-dt*<z(0,0),{N'}>*x */
                if (size == 1) {
                    y = 0.;
                    IRKOper.AddExplicitGradientsMult(-dt, Z00, x, y);

                    // Apply mass matrix to x
                    IRKOper.ImplicitMult(x, temp_vector);
                    y.Add(R00, temp_vector);
                }
                /** 2x2 operator,
                        y(0) = [R(0,0)*M-dt*<Z(0,0),{N'>]*x(0)  + [R(0,1)*M-dt*<Z(0,1),{N'}>]*x(1)
                        y(1) = [R(1,0)*M-dt*<Z(1,0),{N'}>]*x(0) + [R(1,1)*M-dt*<Z(1,1),{N'}>]*x(1) */
                else if (size == 2) {
                    // Wrap scalar Vectors with BlockVectors
                    x_block.Update(x.GetData(), offsets);
                    y_block.Update(y.GetData(), offsets);

                    // Initialize y to zero
                    y_block.GetBlock(0) = 0.;
                    y_block.GetBlock(1) = 0.;

                    // --- Dependence on x(0)
                    IRKOper.AddExplicitGradientsMult(-dt, Z00, -dt, Z10,
                                                    x_block.GetBlock(0),
                                                    y_block.GetBlock(0),
                                                    y_block.GetBlock(1));
                    // Apply mass matrix
                    IRKOper.ImplicitMult(x_block.GetBlock(0), temp_vector);
                    y_block.GetBlock(0).Add(R00, temp_vector);
                    y_block.GetBlock(1).Add(R10, temp_vector);

                    // --- Dependence on x(1)
                    IRKOper.AddExplicitGradientsMult(-dt, Z01, -dt, Z11,
                                                    x_block.GetBlock(1),
                                                    y_block.GetBlock(0),
                                                    y_block.GetBlock(1));
                    // Apply mass matrix
                    IRKOper.ImplicitMult(x_block.GetBlock(1), temp_vector);
                    y_block.GetBlock(0).Add(R01, temp_vector);
                    y_block.GetBlock(1).Add(R11, temp_vector);
                }
                break;
        }
    }
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
class JacDiagBlockTriPrec : public Solver
{
private:
    const JacDiagBlock &BlockOper;
    bool identity;              // Use identity preconditioner. Useful as a comparison.
    mutable Vector temp_vector; // Auxillary vector

    // Extra data required for 2x2 blocks
    double R10;
    Vector Y10;
    int prec00_idx;             // BlockOper.IRKOper.ImplicitPrec(prec00_idx,.,.) preconditions (0,0) block
    int prec11_idx;             // BlockOper.IRKOper.ImplicitPrec(prec11_idx,.,.) preconditions (1,1) block
    mutable BlockVector x_block, y_block;

public:
    /// 1x1 block
    JacDiagBlockTriPrec(const JacDiagBlock &BlockOper_,
        bool identity_=false)
        : Solver(BlockOper_.Height()), BlockOper(BlockOper_),
        identity(identity_)
        {}

    /// ExplicitGradients==APPROXIMATE, 2x2 block
    JacDiagBlockTriPrec(const JacDiagBlock &BlockOper_, double R10_,
        int prec00_idx_, int prec11_idx_,
        bool identity_=false)
        : Solver(BlockOper_.Height()), BlockOper(BlockOper_),
            identity(identity_), temp_vector(BlockOper_.Offsets()[1]),
            R10(R10_),
            prec00_idx(prec00_idx_), prec11_idx(prec11_idx_)
        {
            MFEM_ASSERT(BlockOper.IRKOper.GetExplicitGradientsType() == IRKOperator::ExplicitGradients::APPROXIMATE,
                        "JacDiagBlockTriPrec:: This constructor is for IRKOperator's \
                        with ExplicitGradients::APPROXIMATE");
        }

    /// ExplicitGradients==EXACT, 2x2 block
    JacDiagBlockTriPrec(const JacDiagBlock &BlockOper_, double R10_, Vector Y10_,
        int prec00_idx_, int prec11_idx_,
        bool identity_=false)
        : Solver(BlockOper_.Height()), BlockOper(BlockOper_),
            identity(identity_), temp_vector(BlockOper_.Offsets()[1]),
            R10(R10_), Y10(Y10_),
            prec00_idx(prec00_idx_), prec11_idx(prec11_idx_)
        {
            MFEM_ASSERT(BlockOper.IRKOper.GetExplicitGradientsType() == IRKOperator::ExplicitGradients::EXACT,
                        "JacDiagBlockTriPrec:: This constructor is for IRKOperator's \
                        with ExplicitGradients::EXACT");
        }

    ~JacDiagBlockTriPrec() {}

    /// Apply action of preconditioner
    inline void Mult(const Vector &x_scalar, Vector &y_scalar) const {
        // Use an identity preconditioner
        if (identity) {
            y_scalar = x_scalar;

        // Use a proper preconditioner
        }
        else {

            // 1x1 system
            if (BlockOper.Size() == 1) {
                BlockOper.IRKOper.ImplicitPrec(x_scalar, y_scalar);
            }
            /* 2x2 system uses 2x2 block lower triangular preconditioner,
                [A 0][y0] = x0  =>  y0 = A^{-1}*x0
                [C D][y1] = x1  =>  y1 = D^{-1}*(x1 - C*y0) */
            else if (BlockOper.Size() == 2) {
                // Wrap scalar Vectors with BlockVectors
                x_block.Update(x_scalar.GetData(), BlockOper.Offsets());
                y_block.Update(y_scalar.GetData(), BlockOper.Offsets());

                // Which system is solved depends on IRKOper::ExplicitGradients
                switch (BlockOper.IRKOper.GetExplicitGradientsType()) {

                    // C == R(1,0)*M
                    case IRKOperator::ExplicitGradients::APPROXIMATE:
                        // Approximately invert (0,0) block
                        BlockOper.IRKOper.ImplicitPrec(prec00_idx, x_block.GetBlock(0), y_block.GetBlock(0));

                        // Form RHS of next system, temp <- x(1) - C*y(0)
                        // Apply mass matrix
                        BlockOper.IRKOper.ImplicitMult(y_block.GetBlock(0), temp_vector);
                        temp_vector *= -R10;
                        temp_vector += x_block.GetBlock(1);

                        // Approximately invert (1,1) block
                        BlockOper.IRKOper.ImplicitPrec(prec11_idx, temp_vector, y_block.GetBlock(1));
                        break;

                    // C == R(1,0)*M-dt*<Y(1,0),{N'}>
                    case IRKOperator::ExplicitGradients::EXACT:
                        // Approximately invert (0,0) block
                        BlockOper.IRKOper.ImplicitPrec(prec00_idx, x_block.GetBlock(0), y_block.GetBlock(0));

                        // Form RHS of next system, temp <- x(1) - C*y(0)
                        temp_vector = x_block.GetBlock(1);
                        // Apply mass matrix
                        BlockOper.IRKOper.ImplicitMult(y_block.GetBlock(0), y_block.GetBlock(1));
                        temp_vector.Add(-R10, y_block.GetBlock(1));

                        BlockOper.IRKOper.AddExplicitGradientsMult(-BlockOper.GetTimeStep(), Y10,
                                                                    y_block.GetBlock(0), temp_vector);

                        // Approximately invert (1,1) block
                        BlockOper.IRKOper.ImplicitPrec(prec11_idx, temp_vector, y_block.GetBlock(1));
                        break;
                }
            }
        }
    }

    /// Purely virtual function we must implement but do not use.
    virtual void SetOperator(const Operator &op) {  }
};


class JacDiagGenPrec : public Solver
{
private:
    const JacDiagBlock &BlockOper;
    int Schur_block_id;
    int num_blocks;

public:
    /// 1x1 block
    JacDiagGenPrec(const JacDiagBlock &BlockOper_, int Schur_block_id_, int num_blocks_)
        : Solver(BlockOper_.Height()), BlockOper(BlockOper_),
        Schur_block_id(Schur_block_id_), num_blocks(num_blocks_)
    {

    }

    ~JacDiagGenPrec() {}

    /// Apply action of preconditioner
    void Mult(const Vector &x_scalar, Vector &y_scalar) const {
        BlockOper.IRKOper.ImplicitPrec(Schur_block_id, x_scalar, y_scalar);
    }

    /// Purely virtual function we must implement but do not use.
    virtual void SetOperator(const Operator &op) {  }
};



/** Jacobian is approximated to be block upper triangular. The operator
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
    Array<Solver *> DiagBlockPrec;

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

    bool schur_precondition;

public:

    /** General constructor, where 1x1 and 2x2 systems can use different Krylov
        solvers.
        NOTE: To use only a single solver, requires &solver_params1==&solver_params2 */
    TriJacSolver(IRKStageOper &StageOper_, int jac_update_rate_, int gamma_idx_,
                const IRK::KrylovParams &solver_params1, const IRK::KrylovParams &solver_params2,
                const QuasiMatrixProduct * Z_solver_, const QuasiMatrixProduct * Z_prec_,
                bool schur_precondition_=true)
        : Solver(StageOper_.Height()),
        StageOper(StageOper_),
        printlevel(solver_params1.printlevel),
        jac_update_rate(jac_update_rate_),
        gamma_idx(gamma_idx_),
        offsets(StageOper_.RowOffsets()),
        x_block(StageOper_.RowOffsets()), b_block(StageOper_.RowOffsets()),
        b_block_temp(StageOper_.RowOffsets()), x_block_temp(StageOper_.RowOffsets()),
        temp_vector1(StageOper_.RowOffsets()[1]), temp_vector2(StageOper_.RowOffsets()[1]),
        krylov_solver1(NULL), krylov_solver2(NULL), multiple_krylov(false),
        Z_solver(Z_solver_), Z_prec(Z_prec_), schur_precondition(schur_precondition_)
    {

        kronecker_form = (StageOper.IRKOper->GetExplicitGradientsType() == IRKOperator::ExplicitGradients::APPROXIMATE);

        // Ensure that valid Z_solver and Z_prec provided for non Kronecker Jacobian
        if (!kronecker_form) {
            MFEM_ASSERT((Z_solver) && (Z_prec), "TriJacSolver:: IRKOperator using \
            exact gradients requires non NULL sparsity patterns Z_solver and Z_prec");
        }

        // Create offset arrays for 1x1 and 2x2 operators
        offsets.GetSubArray(0, 2, offsets_1);
        if (offsets.Size() > 2) offsets.GetSubArray(0, 3, offsets_2);

        // Create operators describing diagonal blocks
        bool size1_solves = false; // Do we solve any 1x1 systems?
        bool size2_solves = false; // Do we solve any 2x2 systems?
        double R00, R01, R10, R11; // Elements from diagonal block of R0
        bool identity = false;     // Use identity preconditioners as test that preconditioners are doing something

        int s_eff = StageOper.Butcher.s_eff;
        Array<int> size = StageOper.Butcher.R0_block_sizes;
        /*  Initialize operators describing diagonal blocks and their
            preconditioners, going from top left to bottom right. */
        DiagBlock.SetSize(s_eff);
        DiagBlockPrec.SetSize(s_eff);
        int row = 0;
        for (int block = 0; block < s_eff; block++) {

            // 1x1 diagonal block spanning row=row,col=row
            if (size[block] == 1) {
                size1_solves = true;
                R00 = StageOper.Butcher.R0(row,row);

                // Form of operator depends on IRKOperator::ExplicitGradients
                if (kronecker_form) {
                    DiagBlock[block] = new JacDiagBlock(offsets_1, *(StageOper.IRKOper), R00);
                } else {
                    DiagBlock[block] = new JacDiagBlock(offsets_1, *(StageOper.IRKOper), R00, (*Z_solver)(row,row));
                }

                // Precondition using predefined block triangular preconditioning
                // (schur) or using custom approach for full system.
                if (schur_precondition) {
                    DiagBlockPrec[block] = new JacDiagBlockTriPrec(*(DiagBlock[block]), identity);
                }
                else {
                    DiagBlockPrec[block] = new JacDiagGenPrec(*(DiagBlock[block]), row, 1);
                }
            }
            // 2x2 diagonal block spanning rows=(row,row+1),cols=(row,row+1)
            else if (size[block] == 2) {
                size2_solves = true;
                R00 = StageOper.Butcher.R0(row,row);
                R01 = StageOper.Butcher.R0(row,row+1);
                R10 = StageOper.Butcher.R0(row+1,row);
                R11 = StageOper.Butcher.R0(row+1,row+1);

                // Form of operator and preconditioner depends on IRKOperator::ExplicitGradients
                if (kronecker_form) {
                    DiagBlock[block] = new JacDiagBlock(offsets_2, *(StageOper.IRKOper),
                                                    R00, R01, R10, R11);

                    // Precondition using predefined block triangular preconditioning
                    // (schur) or using custom approach for full system.
                    if (schur_precondition) {
                        // Diagonal blocks in preconditioner are the same
                        if (gamma_idx == 0) {
                            DiagBlockPrec[block] = new JacDiagBlockTriPrec(*(DiagBlock[block]),
                                                            R10,
                                                            row,  // Precondition (row,row) block with IRKOper.ImplicitPrec(row,.,.)
                                                            row,  // Precondition (row+1,row+1) block with IRKOper.ImplicitPrec(row,.,.)
                                                            identity);
                        }
                        // Diagonal blocks in preconditioner are different
                        else {
                            DiagBlockPrec[block] = new JacDiagBlockTriPrec(*(DiagBlock[block]),
                                                            R10,
                                                            row,  // Precondition (row,row) block with IRKOper.ImplicitPrec(row,.,.)
                                                            row+1,// Precondition (row+1,row+1) block with IRKOper.ImplicitPrec(row+1,.,.)
                                                            identity);
                        }
                    }
                    else {
                        DiagBlockPrec[block] = new JacDiagGenPrec(*(DiagBlock[block]), row, 2);
                    }
                }
                else {
                    DiagBlock[block] = new JacDiagBlock(offsets_2, *(StageOper.IRKOper),
                                                    R00, R01, R10, R11,
                                                    (*Z_solver)(row,row),
                                                    (*Z_solver)(row,row+1),
                                                    (*Z_solver)(row+1,row),
                                                    (*Z_solver)(row+1,row+1));

                    // Precondition using predefined block triangular preconditioning
                    // (schur) or using custom approach for full system.
                    if (schur_precondition) {
                        DiagBlockPrec[block] = new JacDiagBlockTriPrec(*(DiagBlock[block]),
                                                        R10,
                                                        (*Z_prec)(row+1,row),
                                                        row,  // Precondition (row,row) block with IRKOper.ImplicitPrec(row,.,.)
                                                        row+1,// Precondition (row+1,row+1) block with IRKOper.ImplicitPrec(row+1,.,.)
                                                        identity);
                    }
                    else {
                        DiagBlockPrec[block] = new JacDiagGenPrec(*(DiagBlock[block]), row, 2);
                    }
                }
            }
            else {
                mfem_error("TriJacSolver:: R0 block sizes must be 1 or 2");
            }

            // Increment row counter by current block size
            row += size[block];
        }

        // Set up Krylov solver
        GetKrylovSolver(krylov_solver1, solver_params1);
        krylov_solver2 = krylov_solver1; // By default, 2x2 systems solved with krylov_solver1.

        /*  Setup different solver for 2x2 blocks if needed (solving both 1x1 and
            2x2 systems AND references to solver parameters are not identical) */
        if ((size1_solves && size2_solves) && (&solver_params1 != &solver_params2)) {
            MFEM_ASSERT(solver_params2.solver == IRK::AccelMethod::GMRES,
                            "TriJacSolver:: 2x2 systems must use GMRES.\n");
            GetKrylovSolver(krylov_solver2, solver_params2);
            multiple_krylov = true;
        }

        krylov_iters.resize(s_eff, 0);
    };

    /// Constructor for when 1x1 and 2x2 systems use same solver
    TriJacSolver(IRKStageOper &StageOper_, int jac_update_rate_, int gamma_idx_,
                const IRK::KrylovParams &solver_params,
                const QuasiMatrixProduct * Z_solver, const QuasiMatrixProduct * Z_prec,
                bool schur_precondition_=true)
        : TriJacSolver(StageOper_, jac_update_rate_, gamma_idx_, solver_params, solver_params,
                        Z_solver, Z_prec, schur_precondition_) {};

    ~TriJacSolver()
    {
        for (int i = 0; i < DiagBlock.Size(); i++) {
            delete DiagBlockPrec[i];
            delete DiagBlock[i];
        }
        delete krylov_solver1;
        if (multiple_krylov) delete krylov_solver2;
    };

    /// Functions to track solver progress
    inline vector<int> GetNumIterations() { return krylov_iters; };
    inline void ResetNumIterations() {
        for (int i = 0; i < krylov_iters.size(); i++) krylov_iters[i] = 0;
    };

    /** Newton method will pass the operator returned from its GetGradient() to
        this, but we don't actually use it. Instead, we update the approximate
        gradient Na' or the exact gradients {N'} if requested */
    inline void SetOperator (const Operator &op) {

        // Update gradient(s) if: First Newton iteration, OR current iteration
        // is a multiple of update rate
        if (StageOper.GetGradientCalls() == 1 ||
            (jac_update_rate > 0 && (StageOper.GetGradientCalls()+1) % jac_update_rate == 0))
        {
            if (kronecker_form) {
                // Set approximate gradient Na'
                StageOper.IRKOper->SetExplicitGradient(*(StageOper.u), StageOper.GetTimeStep(),
                                        StageOper.GetCurrentIterate(), StageOper.Butcher.c0);
            } else {
                // Set exact gradients {N'}
                StageOper.IRKOper->SetExplicitGradients(*(StageOper.u), StageOper.GetTimeStep(),
                                        StageOper.GetCurrentIterate(), StageOper.Butcher.c0);
            }
        }
    }

    /** Solve J*x = b for x, J=A^-1 \otimes M - dt * (Q \otimes I) * \tilde{P} * (Q^\top \otimes I)
        We first transform J*x=b into
            [Q^\top J Q][Q^\top * x]=[Q^\top * b]
                        <==>
            \tilde{J} * x_temp = b_temp,
        i.e., \tilde{J} = R \otimes M - dt * \tilde{P},
        x_temp = Q^\top * x_block, b_temp = Q^\top * b_block */
    inline void Mult(const Vector &b_scalar, Vector &x_scalar) const
    {
        // Wrap scalar Vectors into BlockVectors
        b_block.Update(b_scalar.GetData(), offsets);
        x_block.Update(x_scalar.GetData(), offsets);

        // Transform initial guess and RHS
        KronTransformTranspose(StageOper.Butcher.Q0, x_block, x_block_temp);
        KronTransformTranspose(StageOper.Butcher.Q0, b_block, b_block_temp);

        // Solve \tilde{J}*x_block_temp=b_block_temp,
        BlockBackwardSubstitution(b_block_temp, x_block_temp);

        // Transform to get original x
        KronTransform(StageOper.Butcher.Q0, x_block_temp, x_block);
    }



private:

    /// Set up Krylov solver for inverting diagonal blocks
    inline void GetKrylovSolver(IterativeSolver * &solver, const IRK::KrylovParams &params) const
    {
        switch (params.solver) {
            case IRK::AccelMethod::FP:
                solver = new SLISolver(StageOper.IRKOper->GetComm());
                break;
            case IRK::AccelMethod::CG:
                solver = new CGSolver(StageOper.IRKOper->GetComm());
                break;
            case IRK::AccelMethod::MINRES:
                solver = new MINRESSolver(StageOper.IRKOper->GetComm());
                break;
            case IRK::AccelMethod::GMRES:
                solver = new GMRESSolver(StageOper.IRKOper->GetComm());
                static_cast<GMRESSolver*>(solver)->SetKDim(params.kdim);
                break;
            case IRK::AccelMethod::BICGSTAB:
                solver = new MINRESSolver(StageOper.IRKOper->GetComm());
                break;
            case IRK::AccelMethod::FGMRES:
                solver = new FGMRESSolver(StageOper.IRKOper->GetComm());
                static_cast<FGMRESSolver*>(solver)->SetKDim(params.kdim);
                break;
            default:
                mfem_error("IRK::Invalid Krylov solve type.\n");
        }

        solver->iterative_mode = false;
        solver->SetAbsTol(params.abstol);
        solver->SetRelTol(params.reltol);
        solver->SetMaxIter(params.maxiter);
        solver->SetPrintLevel(params.printlevel);
    }


    /** Solve \tilde{J}*y = z via block backward substitution, where
            \tilde{J} = R \otimes M - dt * \tilde{P}
        NOTE: RHS vector z is not const, since its data is overridden during the
        solve */
    inline void BlockBackwardSubstitution(BlockVector &z_block, BlockVector &y_block) const
    {
        if (printlevel > 0) mfem::out << "  ---Backward solve---" << '\n';

        // Short hands
        int s = StageOper.Butcher.s;
        int s_eff = StageOper.Butcher.s_eff;
        DenseMatrix R = StageOper.Butcher.R0;
        Array<int> size = StageOper.Butcher.R0_block_sizes;
        bool krylov_converged;

        double dt = StageOper.GetTimeStep();

        /** Backward substitution: Invert diagonal blocks, which are:
            -1x1 systems for y(row), or
            -2x2 systems for (y(row),y(row+1)) */
        int row = s;
        for (int diagBlock = s_eff-1; diagBlock >= 0; diagBlock--)
        {
            int solve = s_eff - diagBlock;
            if (printlevel > 0) mfem::out << "    Block solve " << solve << " of " << s_eff;

            // Decrement row counter by current block size.
            row -= size[diagBlock];

            // Update parameters for diagonal blocks
            DiagBlock[diagBlock]->SetTimeStep(StageOper.GetTimeStep());

            // Compute constant gamma used to precondition Schur complement of 2x2 block
            double gamma = 0.;
            if (size[diagBlock] == 2) {
                double eta = R(row,row), beta = std::sqrt(-R(row,row+1)*R(row+1,row));
                if (gamma_idx == 0) {
                    gamma = eta;
                }
                else if (gamma_idx == 1) {
                    gamma = eta + beta*beta/eta;
                }
                else {
                    mfem_error("gamma must be 0, 1");
                }
            }

            // Assemble preconditioner(s) for diag block
            if (kronecker_form) {
                if (schur_precondition) {
                    // Preconditioner for R(row,row)*M-dt*Na'
                    StageOper.IRKOper->BuildPreconditioner(row, dt, R(row,row), size[diagBlock]);

                    /* Inverting 2x2 block: Assemble a 2nd preconditioner for
                        gamma*M-dt*Na' if gamma is not eta */
                    if (size[diagBlock] == 2 && gamma_idx != 0) {
                        StageOper.IRKOper->BuildPreconditionerSchur(row+1, dt, gamma, size[diagBlock]);
                    }
                }
                else {
                    double beta = StageOper.Butcher.beta[row];
                    StageOper.IRKOper->BuildGenPreconditioner(row, dt, R(row,row), R(row,row+1), beta, size[diagBlock]);
                }
            }
            else {
                // Preconditioner for R(row,row)*M-dt*<Z_prec(row,row),{N'}>
                if (size[diagBlock] == 2) {
                    /* Inverting 2x2 block: Assemble a 2nd preconditioner for
                        gamma*M-dt*<Z_prec(row+1,row+1),{N'}> */
                    if (schur_precondition) {
                        StageOper.IRKOper->BuildPreconditioner(row, dt, R(row,row), (*Z_prec)(row,row));
                        StageOper.IRKOper->BuildPreconditionerSchur(row+1, dt, gamma, (*Z_prec)(row+1,row+1));
                    } else {
                        double beta = StageOper.Butcher.beta[row];
                        StageOper.IRKOper->BuildGenPreconditioner(row, dt, R(row,row), R(row,row+1), beta,
                                                                (*Z_prec)(row,row),
                                                                (*Z_prec)(row+1,row+1));
                    }
                } else {
                    StageOper.IRKOper->BuildPreconditioner(row, dt, R(row,row), (*Z_prec)(row,row));
                }
            }

            // Invert 1x1 diagonal block
            if (size[diagBlock] == 1)
            {
                if (printlevel > 0) {
                    mfem::out << ": 1x1 block  -->  ";
                    if (printlevel != 2) mfem::out << '\n';
                }
                // --- Form RHS vector (this overrides z_block(row)) --- //
                // Subtract out known information from LHS of equations
                if (row+1 < s) {
                    /// R0 component
                    // Apply mass matrix
                    temp_vector1.Set(-R(row,row+1), y_block.GetBlock(row+1));
                    for (int j = row+2; j < s; j++) {
                        temp_vector1.Add(-R(row,j), y_block.GetBlock(j));
                    }
                    StageOper.IRKOper->ImplicitMult(temp_vector1, temp_vector2);
                    z_block.GetBlock(row) += temp_vector2; // Add to existing RHS

                    /// {N'} components (only appear for non-Kronecker product form)
                    if (!kronecker_form) {
                        for (int j = row+1; j < s; j++) {
                            StageOper.IRKOper->AddExplicitGradientsMult(
                                                    dt, (*Z_solver)(row,j),
                                                    y_block.GetBlock(j),
                                                    z_block.GetBlock(row));
                        }
                    }
                }

                // --- Solve 1x1 system ---
                // Pass preconditioner for diagonal block to Krylov solver
                krylov_solver1->SetPreconditioner(*DiagBlockPrec[diagBlock]);
                // Pass diagonal block to Krylov solver
                krylov_solver1->SetOperator(*DiagBlock[diagBlock]);
                // Solve
                krylov_solver1->Mult(z_block.GetBlock(row), y_block.GetBlock(row));
                krylov_converged = krylov_solver1->GetConverged();
                krylov_iters[diagBlock] += krylov_solver1->GetNumIterations();
            }
            // Invert 2x2 diagonal block
            else if (size[diagBlock] == 2)
            {
                if (printlevel > 0) {
                    mfem::out << ": 2x2 block  -->  ";
                    if (printlevel != 2) mfem::out << '\n';
                }

                // --- Form RHS vector (this overrides z_block(row),z_block(row+1)) --- //
                // Point z_2block to the appropriate data from z_block
                // (note data arrays for blocks are stored contiguously)
                z_2block.Update(z_block.GetBlock(row).GetData(), offsets_2);

                // Subtract out known information from LHS of equations
                if (row+2 < s) {
                    /// R0 component
                    // Apply mass matrix
                    temp_vector1.Set(-R(row,row+2), y_block.GetBlock(row+2));
                    for (int j = row+3; j < s; j++) {
                        temp_vector1.Add(-R(row,j), y_block.GetBlock(j));
                    }
                    StageOper.IRKOper->ImplicitMult(temp_vector1, temp_vector2);
                    z_2block.GetBlock(0) += temp_vector2; // Add to existing RHS
                    // Second component
                    temp_vector1.Set(-R(row+1,row+2), y_block.GetBlock(row+2));
                    for (int j = row+3; j < s; j++) {
                        temp_vector1.Add(-R(row+1,j), y_block.GetBlock(j));
                    }
                    StageOper.IRKOper->ImplicitMult(temp_vector1, temp_vector2);
                    z_2block.GetBlock(1) += temp_vector2; // Add to existing RHS

                    /// {N'} components (only appear for non-Kronecker product form)
                    if (!kronecker_form) {
                        for (int j = row+2; j < s; j++) {
                            StageOper.IRKOper->AddExplicitGradientsMult(
                                                    dt, (*Z_solver)(row,j),
                                                    dt, (*Z_solver)(row+1,j),
                                                    y_block.GetBlock(j),
                                                    z_2block.GetBlock(0), z_2block.GetBlock(1));
                        }
                    }
                }

                // Point y_2block to data array of solution vector
                y_2block.Update(y_block.GetBlock(row).GetData(), offsets_2);

                // --- Solve 2x2 system ---
                // Pass preconditioner for diagonal block to Krylov solver
                krylov_solver2->SetPreconditioner(*DiagBlockPrec[diagBlock]);
                // Pass diagonal block to Krylov solver
                krylov_solver2->SetOperator(*DiagBlock[diagBlock]);
                // Solve
                krylov_solver2->Mult(z_2block, y_2block);
                krylov_converged = krylov_solver2->GetConverged();
                krylov_iters[diagBlock] += krylov_solver1->GetNumIterations();
            }

            // Check convergence
            if (!krylov_converged) {
                string msg = "KronJacSolver::BlockBackwardSubstitution() Krylov solver at t="
                                + to_string(StageOper.IRKOper->GetTime())
                                + " not converged [system " + to_string(solve)
                                + "/" + to_string(s_eff)
                                + ", size=" + to_string(size[diagBlock]) + ")]\n";
                mfem_error(msg.c_str());
            }
        }
    }
};


#endif
