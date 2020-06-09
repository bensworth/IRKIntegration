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

// Implicit Runge Kutta type. Enumeration:
//  First digit: group of schemes
//  - 1 = A-stable (but NOT L-stable) SDIRK schemes
//  + 0 = L-stable SDIRK
//  + 1 = Gauss-Legendre
//  + 2 = RadauIIA
//  + 3 = Lobatto IIIC
//  Second digit: order of scheme
enum IRKType { 
    ASDIRK4 = -14,
    LSDIRK1 = 01, LSDIRK2 = 02, LSDIRK3 = 03, LSDIRK4 = 04,
    Gauss2 = 12, Gauss4 = 14, Gauss6 = 16, Gauss8 = 18, Gauss10 = 110,
    RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27, RadauIIA9 = 29,
    LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36, LobIIIC8 = 38
};

// Parameters for Newton solver
struct NEWTON_params {
    double reltol = 1e-6;
    double abstol = 1e-6;
    int maxiter = 10;
    int printlevel = 2; 
};

// Different types of Krylov solvers
enum class KrylovMethod {CG, MINRES, GMRES, BICGSTAB, FGMRES};

// Parameters for Krylov solver
struct Krylov_params {
    double abstol = 1e-10;
    double reltol = 1e-10;
    int maxiter = 100;
    int printlevel = 0;
    int kdim = 30;
    KrylovMethod solver = KrylovMethod::GMRES;
};



/// Kronecker transform between two block vectors: y <- (A \otimes I)*x
void KronTransform(const DenseMatrix &A, const BlockVector &x, BlockVector &y);

/// Kronecker transform between two block vectors using A transpose: y <- (A^\top \otimes I)*x
void KronTransformTranspose(const DenseMatrix &A, const BlockVector &x, BlockVector &y);


/** Abstract base class for spatial discretizations of a PDE resulting in the 
time-dependent, nonlinear ODEs
    M*du/dt = L(u,t)    _OR_    du/dt = L(u,t) [if no mass matrix exists]

If a mass matrix M exists, the following virtual function must be implemented:
    ImplicitMult(x,y): y <- M*x */
class IRKOperator : public TimeDependentOperator
{    
protected:
    MPI_Comm m_globComm;
    
public:
    IRKOperator(MPI_Comm comm, int n=0, double t=0.0, Type type=EXPLICIT) 
        : TimeDependentOperator(n, t, type), 
            m_globComm{comm} {};
    
    ~IRKOperator() { };

    MPI_Comm GetComm() { return m_globComm; };

    /** Apply action of M*du/dt, y <- L(x,y) */
    virtual void ExplicitMult(const Vector &x, Vector &y) const = 0;
    
    /** Gradient of L(u, t) w.r.t u evaluated at x */
    virtual Operator &GetExplicitGradient(const Vector &x) const = 0;
    
    /** Apply action mass matrix, y = M*x. 
    If not re-implemented, this method simply generates an error. */
    virtual void ImplicitMult(const Vector &x, Vector &y) const
    {
        mfem_error("IRKOperator::ImplicitMult() is not overridden!");
    }
    
    /** Precondition (\gamma*M - dt*L') */
    virtual void ImplicitPrec(const Vector &x, Vector &y) const = 0;
    
    
    // Function to ensure that ImplicitPrec preconditions (\gamma*M - dt*L')
    // with gamma and dt as passed to this function.
    //      + index -> index of system, [0,s_eff)
    //      + type -> eigenvalue type, 1 = real, 2 = complex pair
    //      + t -> time.
    // These additional parameters are to provide ways to track when
    // (\gamma*M - dt*L') _OR_ (\gamma*I - dt*L') must be reconstructed or not to minimize setup.
    // NOTE: Must use the Operator that a reference is returned to via calling
    // GetExplicitGradient().
    virtual void SetSystem(int index, double dt, double gamma, int type) = 0;
};

    
class RKButcherData 
{
private:    
    void SetData(IRKType RK_ID);// Set Butcher tableau coefficients
    void SizeData();            // Set dimensions of Butcher arrays
    
public:
    RKButcherData(IRKType RK_ID) { 
        SetData(RK_ID);
        invA0 = A0;
        invA0.Invert();
     };
    
    ~RKButcherData() {};
    
    int s;            // Number of stages
    int s_eff;        // Number of eigenvalues of A0 once complex conjugates have been combined
    DenseMatrix A0;   // The Butcher matrix
    DenseMatrix invA0;// Inverse of Buthcer matrix
    DenseMatrix Q0;   // Orthogonal matrix in Schur decomposition of A0^-1
    DenseMatrix R0;   // Quasi-upper triangular matrix in Schur decomposition of A0^-1
    Array<int> R0_block_sizes; // From top of block diagonal, sizes of blocks
    Vector b0;        // Butcher tableau weights
    Vector c0;        // Butcher tableau nodes
    Vector d0;        // inv(A0^\top)*b0
};    

// Type of Jacobian of to use when inverting an IRKStageOper
enum JacobianType {
    FULL = 0,       // The true Jacobian, uses stage-dependent N'.
    KRONECKER = 1   // Kronecker-product Jacobian. Ignores stage dependence of N' and assumes its constant.
};

class KronJacSolver;

/* Operator F defining the s stage equations. They satisfy F(w) = 0, 
    where w = (A0 \otimes I)*k, and
    
            [F1(w)]                         [N(u+dt*w1)]
            [ ... ]                         [   ....   ]
    F(w) =  [ ... ] = (inv(A0) \otimes M) - [   ....   ]
            [ ... ]                         [   ....   ]
            [Fs(w)]                         [N(u+dt*ws)] */
class IRKStageOper : public BlockOperator
{
private:
    friend class KronJacSolver;   // Approximate Jacobian solver needs access
    
    Array<int> &offsets;          // Offsets of operator     
    
    mutable IRKOperator * IRKOper;// Spatial discretization
    const RKButcherData &Butcher; // All RK information
    
    // Parameters that operator depends on
    const Vector * u;             // Current state.
    double t;                     // Current time of state.
    double dt;                    // Current time step
    
    // Wrappers for scalar vectors
    mutable BlockVector w_block, y_block;
    
    // Auxillary vectors
    mutable BlockVector temp_block;
    mutable Vector temp_scalar1, temp_scalar2;    
    
    Operator * dummy_gradient; 
    
    // Current iterate that true Jacobian would linearize with (this is passed 
    // into GetGradient())
    mutable BlockVector current_iterate;
    
    // Number of times GetGradient() been called with current states (u, t, dt).
    mutable int getGradientCalls;
public:
    
    IRKStageOper(IRKOperator * S_, Array<int> &offsets_, const RKButcherData &RK_) 
        : BlockOperator(offsets_), 
            IRKOper{S_}, Butcher{RK_}, 
            u(NULL), t{0.0}, dt{0.0}, 
            offsets(offsets_),
            w_block(offsets_), y_block(offsets_),
            temp_scalar1(S_->Height()), temp_scalar2(S_->Height()), 
            temp_block(offsets_),
            current_iterate(),
            dummy_gradient(NULL),
            getGradientCalls{0}
            { 
            
            };
    
    inline void SetParameters(const Vector * u_, double t_, double dt_) { 
        t = t_;
        dt = dt_;
        u = u_;
        getGradientCalls = 0; // Reset counter
    };

    inline double GetTimeStep() {return dt;};
    inline double GetTime() {return t;};
    
    // Return reference to current iterate
    inline const BlockVector &GetCurrentIterate() { return current_iterate; };

    /// Meant to return Jacobian of operator. This is called by Newton during 
    /// every iteration, and the result will be passed in to its linear solver 
    /// via its SetOperator().
    /// dummy function at the moment... Is required tho.
    inline virtual Operator &GetGradient(const Vector &w) const
    {
        // Update `current_iterate` so that its data points to the current iterate's
        current_iterate.Update(w.GetData(), offsets);
        
        // Increment counter
        getGradientCalls++;
            
        return *dummy_gradient; // To stop compiler complaining of no return type..
    }
    
    /// Compute action of operator, y <- F(w)
    inline void Mult(const Vector &w_scalar, Vector &y_scalar) const
    {
        MFEM_ASSERT(u, "IRKStageOper::Mult() Requires states to be set, see SetParameters()");
        
        // Wrap scalar Vectors with BlockVectors
        w_block.Update(w_scalar.GetData(), offsets);
        y_block.Update(y_scalar.GetData(), offsets);
        
        /* y <- inv(A0)*M*w */
        // MASS MATRIX
        if (IRKOper->isImplicit()) {
            // temp <- M*w
            for (int i = 0; i < Butcher.s; i++) {
                IRKOper->ImplicitMult(w_block.GetBlock(i), temp_block.GetBlock(i));
            }
            KronTransform(Butcher.invA0, temp_block, y_block); // y <- inv(A0)*temp
        // NO MASS MATRIX
        } else {
            KronTransform(Butcher.invA0, w_block, y_block); // y <- inv(A0)*w
        }
        
        /* y <- y - N(u + dt*w) */
        for (int i = 0; i < Butcher.s; i++) {        
            add(*u, dt, w_block.GetBlock(i), temp_scalar1); // temp1 <- u+dt*w(i)
            IRKOper->SetTime(t + Butcher.c0[i]*dt);
            IRKOper->ExplicitMult(temp_scalar1, temp_scalar2); // temp2 <- N(temp1, t)
            y_block.GetBlock(i).Add(-1., temp_scalar2);
        } 
    }
};


// Defines diagonal blocks appearing in the Kronecker product Jacobian. 
// These take the form 1x1 or 2x2 blocks.
//
// 1x1: 
//      [R(0,0)*M - dt*N']
// 
// 2x2:
//      [R(0,0)*M-dt*N'    R(0,1)*M   ]
//      [R(1,0)*M       R(1,1)*M-dt*N']
//
// NOTE: 
//      -Jacobian block N' is assumed to be the same on both diagonals
//      -R(0,0)==R(1,1) (up to machine precision anyways)
//
// TODO:
//  -make BlockOperator rather than Operator??
//  -Probably make into JacDiagBlock without the Kron part. Can just pass two separate blocks, which are potentially the same.
class KronJacDiagBlock : public Operator
{
private:
    
    int size;                   // Block size, 1x1 or 2x2
    const IRKOperator &IRKOper; // Class defining mass matrix M
    mutable Vector temp_scalar; // Auxillary vector
    
    // Data required for 1x1 operators
    double R00;
    
    // Data required for 2x2 operators
    Array<int> offsets; 
    DenseMatrix R;
    mutable BlockVector x_block, y_block;
    
    // Current time step and Jacobian block 
    mutable double dt;
    mutable const Operator * N_jac;
    
public:

    /// 1x1 block
    KronJacDiagBlock(int height, const IRKOperator &IRKOper_, double R00_) 
        : Operator(height), 
        size{1}, IRKOper{IRKOper_}, temp_scalar(IRKOper_.Height()),
        R00{R00_},
        dt{0.0}, N_jac(NULL)
        {};

    /// 2x2 block
    KronJacDiagBlock(int height, const IRKOperator &IRKOper_, DenseMatrix R_, 
        Array<int> offsets_) : Operator(height), 
        size{2}, IRKOper{IRKOper_}, temp_scalar(IRKOper_.Height()),
        R(R_), offsets(offsets_),
        dt{0.0}, N_jac(NULL)
        {};

    /// Update parameters required to compute action
    inline void SetParameters(double dt_, const Operator * N_jac_) const
    {
        dt = dt_;
        N_jac = N_jac_;
    };
        
    /// Compute action of diagonal block    
    inline void Mult(const Vector &x, Vector &y) const 
    {    
        MFEM_ASSERT(N_jac, "KronJacDiagBlock::Mult() must set Jacobian block with SetParameters()");
        MFEM_ASSERT(x.Size() == this->Height(), "KronJacDiagBlock::Mult() incorrect input Vector size");
        MFEM_ASSERT(y.Size() == this->Height(), "KronJacDiagBlock::Mult() incorrect output Vector size");
        
        // 1x1 operator, y = [R(0)(0)*M - dt*N']*x
        if (size == 1) {
            N_jac->Mult(x, y);
            y *= -dt;
        
        // MASS MATRIX    
        if (IRKOper.isImplicit()) {
            IRKOper.ImplicitMult(x, temp_scalar);
            y.Add(R00, temp_scalar);
        // NO MASS MATRIX    
        } else {
            y.Add(R00, x);
        }
        
            
        // 2x2 operator,
        //  y(0) = [R(0,0)*M-dt*N']*x(0) + [R(0,1)*M]*x(1)
        //  y(1) = [R(1,0)*M]*x(0)       + [R(1,1)*M-dt*N']*x(1)
        } else if (size == 2) {
            // Wrap scalar Vectors with BlockVectors
            x_block.Update(x.GetData(), offsets);
            y_block.Update(y.GetData(), offsets);
            
            // y(0)
            N_jac->Mult(x_block.GetBlock(0), y_block.GetBlock(0));
            y_block.GetBlock(0) *= -dt;
            
            // y(1)
            N_jac->Mult(x_block.GetBlock(1), y_block.GetBlock(1));
            y_block.GetBlock(1) *= -dt;
            
            // MASS MATRIX
            if (IRKOper.isImplicit()) {
                // M*x(0) dependence
                IRKOper.ImplicitMult(x_block.GetBlock(0), temp_scalar);
                y_block.GetBlock(0).Add(R(0,0), temp_scalar);
                y_block.GetBlock(1).Add(R(1,0), temp_scalar);
                // M*x(1) dependence
                IRKOper.ImplicitMult(x_block.GetBlock(1), temp_scalar);
                y_block.GetBlock(0).Add(R(0,1), temp_scalar);
                y_block.GetBlock(1).Add(R(1,1), temp_scalar);
            
            // NO MASS MATRIX    
            } else {
                // x(0) dependence
                y_block.GetBlock(0).Add(R(0,0), x_block.GetBlock(0));
                y_block.GetBlock(1).Add(R(1,0), x_block.GetBlock(0));
                // x(1) dependence
                y_block.GetBlock(0).Add(R(0,1), x_block.GetBlock(1));
                y_block.GetBlock(1).Add(R(1,1), x_block.GetBlock(1));
            }
        }
    };     
};


// Preconditioner for Kronecker-product diagonal blocks taking the form
// 1x1: 
//      [R(0,0)*M - dt*N']
// 
// 2x2:
//      [R(0,0)*M-dt*N'    R(0,1)*M   ]
//      [R(1,0)*M       R(0,0)*M-dt*N']
//
// The 2x2 operator is preconditioned by the INVERSE of
//      [R(0,0)*M-dt*N'      0        ]
//      [R(1,0)*M       R(0,0)*M-dt*N']
//
// Where, in all cases, IRKOper.ImplicitPrec(x,y) is used to approximately 
//      solve [R(0,0)*M-dt*N']*y=x
class KronJacDiagBlockPrec : public Solver
{
private:
    const IRKOperator &IRKOper;
    int size;
    bool identity; // Use identity preconditioner. Useful as a comparison.
    
    // Data required for 2x2 operators
    Array<int> offsets; 
    double R10;
    mutable BlockVector x_block, y_block;
    mutable Vector temp_scalar; // Auxillary vector
    
public:
    
    /// 1x1 block
    KronJacDiagBlockPrec(int height, const IRKOperator &IRKOper_, 
        bool identity_=false) 
        : Solver(height), IRKOper(IRKOper_), size{1}, identity(identity_) {};

    /// 2x2 block
    KronJacDiagBlockPrec(int height, const IRKOperator &IRKOper_, 
        double R10_, Array<int> offsets_, bool identity_=false) 
        : Solver(height), IRKOper(IRKOper_), size{2}, identity(identity_),
            R10{R10_}, offsets(offsets_), temp_scalar(offsets_[1]) {};
    
    ~KronJacDiagBlockPrec() {};
    
    /// Apply action of preconditioner
    inline void Mult(const Vector &x_scalar, Vector &y_scalar) const {
        // Use an identity preconditioner
        if (identity) {
            y_scalar = x_scalar;
            
        // Use a proper preconditioner    
        } else {
            // 1x1 system
            if (size == 1) {
                IRKOper.ImplicitPrec(x_scalar, y_scalar);
            }
            // 2x2 system uses 2x2 block lower triangular preconditioner 
            else if (size == 2) {
                // Wrap scalar Vectors with BlockVectors
                x_block.Update(x_scalar.GetData(), offsets);
                y_block.Update(y_scalar.GetData(), offsets);
                
                // Approximately invert (0,0) block 
                IRKOper.ImplicitPrec(x_block.GetBlock(0), y_block.GetBlock(0));
                
                /* Form RHS of next system, temp <- x(1) - R10*M*y(0) */
                // MASS MATRIX
                if (IRKOper.isImplicit()) {
                    IRKOper.ImplicitMult(y_block.GetBlock(0), temp_scalar);
                    temp_scalar *= -R10;
                    temp_scalar += x_block.GetBlock(1);
                // NO MASS MATRIX    
                } else {
                    add(x_block.GetBlock(1), -R10, y_block.GetBlock(0), temp_scalar); 
                }
                
                // Approximately invert (1,1) block
                IRKOper.ImplicitPrec(temp_scalar, y_block.GetBlock(1));
            }
        }
    };
    
    /// Purely virtual function we must implement but do not use.
    virtual void SetOperator(const Operator &op) {  };
};


// N' (the Jacobian of N) is assumed constant w.r.t w, so that
// the diagonal block of Jacobians can be written as a Kronecker product,
//      I \otimes N' ~ diag(N'(u+dt*w1), ..., N'(u+dt*ws))
// Letting J = A^-1 \otimes I - dt * I \otimes N', here we solve J*x=b via 
// block backward substitution, making use of the Schur decomposition of A^-1.
class KronJacSolver : public Solver
{

private:
    int printlevel;
    
    IRKStageOper &StageOper;
    
    Array<int> &offsets;    // Offsets for vectors with s blocks
    Array<int> offsets_2;   // Offsets for vectors with 2 blocks
    
    // Auxillary vectors  
    mutable BlockVector x_block, b_block, b_block_temp, x_block_temp;   // s blocks
    mutable BlockVector y_2block, z_2block;                             //  2 blocks 
    mutable Vector temp_scalar1, temp_scalar2;  
     
    // Diagonal blocks inverted during backward substitution
    Array<KronJacDiagBlock *> JacDiagBlock;
    
    // Solvers for inverting diagonal blocks
    IterativeSolver * krylov_solver1; // 1x1 solver
    IterativeSolver * krylov_solver2; // 2x2 solver
    // NOTE: krylov_solver2 is just a pointer to krylov_solver1 so long as there aren't 
    // both 1x1 and 2x2 systems to solve AND different solver parameters weren't passed 
    bool multiple_krylov; // Do we really use two different solvers?
    
    // Number of Krylov iterations for each diagonal block
    mutable vector<int> krylov_iters;
    
    // Preconditioners to assist with inversion of diagonal blocks
    Array<KronJacDiagBlockPrec *> JacDiagBlockPrec;
    
    // Jacobian of N.
    Operator * N_jac;
    
    // At what point do we linearize the Jacobian of N:
    //  0: u, t
    //  0<j<=s: u+dt*w(j),t+dt*c0(j)
    //      Best option(?): Use last stage value (j=s). This Get's correct 
    //          for single stage schemes, and get's Jacobian right for last DOF 
    //          which is important since we're doing back substitution.
    int N_jac_lin;
    
    // How often do we update the Jacobian:
    //  0:  At start of each new time step
    //  1:  Every Newton iteration
    int N_jac_update_rate;
    
public:

    // General constructor, where 1x1 and 2x2 systems can use different Krylov solvers
    // NOTE: To use only a single solver, requires &solver_params1==&solver_params2
    KronJacSolver(IRKStageOper &StageOper_, 
                    const Krylov_params &solver_params1, const Krylov_params &solver_params2, 
                    int N_jac_lin_, int N_jac_update_rate_) 
        : Solver(StageOper_.Height()),
        printlevel(solver_params1.printlevel),
        StageOper(StageOper_),
        offsets(StageOper_.RowOffsets()), offsets_2(0),
        x_block(StageOper_.RowOffsets()), b_block(StageOper_.RowOffsets()), 
        b_block_temp(StageOper_.RowOffsets()), x_block_temp(StageOper_.RowOffsets()), 
        y_2block(), z_2block(),
        temp_scalar1(StageOper_.RowOffsets()[1]), temp_scalar2(StageOper_.RowOffsets()[1]),
        krylov_solver1(NULL), krylov_solver2(NULL), multiple_krylov(false),
        N_jac(NULL), 
        N_jac_lin{N_jac_lin_},
        N_jac_update_rate{N_jac_update_rate_}
    {    
        MFEM_ASSERT(N_jac_lin > 0 && N_jac_lin <= StageOper.Butcher.s, 
                "KronJacSolver: Require 0 < N_jac_linearization <= s");
        MFEM_ASSERT(N_jac_update_rate == 0 || N_jac_update_rate == 1, 
                "KronJacSolver: Require N_jac_update_rate==0 or 1");
        
        // Create operators describing diagonal blocks
        bool type1_solves = false; // Do we solve any 1x1 systems?
        bool type2_solves = false; // Do we solve any 2x2 systems?
        double R00;         // 1x1 diagonal block of R0
        DenseMatrix R(2);   // 2x2 diagonal block of R0
        int row = 0;        // Row of R0 we're accessing
        JacDiagBlock.SetSize(StageOper.Butcher.s_eff);
        JacDiagBlockPrec.SetSize(StageOper.Butcher.s_eff);
        bool identity = !true; // Use identity preconditioners...
        for (int i = 0; i < StageOper.Butcher.s_eff; i++) {
            
            // 1x1 diagonal block
            if (StageOper.Butcher.R0_block_sizes[i] == 1)
            {
                type1_solves = true;
                R00 = StageOper.Butcher.R0(row,row);
                JacDiagBlock[i] = new KronJacDiagBlock(offsets[1], *(StageOper.IRKOper), R00);    
                JacDiagBlockPrec[i] = new KronJacDiagBlockPrec(offsets[1], *(StageOper.IRKOper), identity);
            } 
            // 2x2 diagonal block
            else 
            {
                type2_solves = true;
                if (offsets_2.Size() == 0) {
                    offsets_2.SetSize(3);
                    offsets_2[0] = offsets[0];
                    offsets_2[1] = offsets[1];
                    offsets_2[2] = offsets[2];
                }
                
                R(0,0) = StageOper.Butcher.R0(row,row);
                R(0,1) = StageOper.Butcher.R0(row,row+1);
                R(1,0) = StageOper.Butcher.R0(row+1,row);
                R(1,1) = StageOper.Butcher.R0(row+1,row+1);
                JacDiagBlock[i] = new KronJacDiagBlock(2*offsets[1], *(StageOper.IRKOper), R, offsets_2);
                JacDiagBlockPrec[i] = new KronJacDiagBlockPrec(2*offsets[1], *(StageOper.IRKOper),
                                            R(1,0), offsets_2, identity);
                row++; // We've processed 2 rows of R0 here.
            }
            
            row++; // Increment to next row of R0
        }
        
        // Set up Krylov solver 
        GetKrylovSolver(krylov_solver1, solver_params1);
        krylov_solver2 = krylov_solver1; // By default, 2x2 systems solved with krylov_solver1.
        
        // Setup different solver for 2x2 blocks if needed (solving both 1x1 and 
        // 2x2 systems AND references to solver parameters are not identical)
        if ((type1_solves && type2_solves) && (&solver_params1 != &solver_params2)) {
            MFEM_ASSERT(solver_params2.solver == KrylovMethod::GMRES, "IRK:: 2x2 systems must use GMRES.\n");
            GetKrylovSolver(krylov_solver2, solver_params2);
            multiple_krylov = true;
        }
        
        krylov_iters.resize(StageOper.Butcher.s_eff, 0);
    };
    
    /// Functions to track solver progress
    inline vector<int> GetNumIterations() { return krylov_iters; };
    inline void ResetNumIterations() { 
        for (int i = 0; i < krylov_iters.size(); i++) krylov_iters[i] = 0; 
    };
    
    // Constructor for when 1x1 and 2x2 systems use same solver
    KronJacSolver(IRKStageOper &StageOper_, const Krylov_params &solver_params,
                  int N_jac_lin_, int N_jac_update_rate_)   
                  : KronJacSolver(StageOper_, solver_params, solver_params, 
                                    N_jac_lin_, N_jac_update_rate_) {};
    
    ~KronJacSolver()
    {
        for (int i = 0; i < JacDiagBlock.Size(); i++) {
            delete JacDiagBlockPrec[i];
            delete JacDiagBlock[i];
        }
        delete krylov_solver1;
        if (multiple_krylov) delete krylov_solver2;
    };
    
    /// Set up Krylov solver for inverting diagonal blocks
    inline void GetKrylovSolver(IterativeSolver * &solver, const Krylov_params &params) {
        switch (params.solver) {
            case KrylovMethod::CG:
                solver = new CGSolver(StageOper.IRKOper->GetComm());
                break;
            case KrylovMethod::MINRES:
                solver = new MINRESSolver(StageOper.IRKOper->GetComm());
                break;
            case KrylovMethod::GMRES:
                solver = new GMRESSolver(StageOper.IRKOper->GetComm());
                static_cast<GMRESSolver*>(solver)->SetKDim(params.kdim);
                break;
            case KrylovMethod::BICGSTAB:
                solver = new MINRESSolver(StageOper.IRKOper->GetComm());
                break;    
            case KrylovMethod::FGMRES:
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
    
    
    /// Newton method will pass the operator returned from its GetGradient() to 
    /// this, but we don't actually require it. However, we do access whether
    /// we need to update our Jacobian, N_jac.
    void SetOperator (const Operator &op) { 
    
        double dt = StageOper.dt;
        double t = StageOper.t;
    
        // Update N_jac if necessary (1st Newton iteration, or all Newton iterations)
        if (StageOper.getGradientCalls == 1 || N_jac_update_rate == 1) {
    
            // Basic option: Eval jacobian of N at u
            if (N_jac_lin == 0) {
                temp_scalar1 = *(StageOper.u);
                StageOper.IRKOper->SetTime(t);
            // Eval jacobian of N at u + dt*w(N_jac_lin)
            } else {
                int idx = N_jac_lin-1; 
                add(*(StageOper.u), dt, StageOper.GetCurrentIterate().GetBlock(idx), temp_scalar1);
                StageOper.IRKOper->SetTime(t + dt*StageOper.Butcher.c0[idx]);
            } 
            
            N_jac = &(StageOper.IRKOper->GetExplicitGradient(temp_scalar1));
        }
    };
    
    /* Solve J*x = b for x, J=A^-1 \otimes I - dt * I \otimes L' 
    We first transform J*x=b into 
        [Q^\top J Q][Q^\top * x]=[Q^\top * b] 
                    <==> 
        \tilde{J} * x_temp = b_temp,
    i.e., x_temp = Q^\top * x_block, b_temp = Q^\top * b_block */
    void Mult(const Vector &b_scalar, Vector &x_scalar) const
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
    
    /* Solve \tilde{J}*y = z via block backward substitution, where 
        \tilde{J} = R \otimes M - I \otimes dt*N' 
    NOTE: RHS vector z is not const, since its data is overridden during the solve */
    void BlockBackwardSubstitution(BlockVector &z_block, BlockVector &y_block) const
    {
        if (printlevel > 0) mfem::out << "  ---Backward solve---" << '\n';
        
        // Short hands
        int s = StageOper.Butcher.s; 
        int s_eff = StageOper.Butcher.s_eff; 
        DenseMatrix R = StageOper.Butcher.R0; 
        bool krylov_converged;
        
        // Block index of current unknown (index w.r.t y/z block sizes, not Rs)
        int idx = s-1;
        
        /* Backward substitution: Invert diagonal blocks, which are 1x1 systems 
        for y[idx], or 2x2 systems for (y[idx],y[idx+1]) */
        for (int diagBlock = s_eff-1; diagBlock >= 0; diagBlock--)
        {
            if (printlevel > 0) mfem::out << "    Block solve " << s_eff-diagBlock << " of " << s_eff;
            
            // Update parameters for the diag block to be inverted
            JacDiagBlock[diagBlock]->SetParameters(StageOper.dt, N_jac);
            
            // Ensure correct preconditioner is set up for R(idx, idx)*M - dt*N'.
            int system_idx = s_eff-diagBlock-1; // Note the reverse ordering...
            StageOper.IRKOper->SetSystem(system_idx, StageOper.dt, R(idx, idx), 
                                        StageOper.Butcher.R0_block_sizes[diagBlock]);
            
            // Invert 1x1 diagonal block
            if (StageOper.Butcher.R0_block_sizes[diagBlock] == 1) 
            {
                if (printlevel > 0) {
                    mfem::out << ": 1x1 block  -->  ";
                    if (printlevel != 2) mfem::out << '\n';
                }
                // --- Form RHS vector (this overrides z_block(idx)) --- //
                // Subtract out known information from LHS of equations
                if (idx+1 < s) {
                    // MASS MATRIX
                    if (StageOper.IRKOper->isImplicit()) {
                        temp_scalar1.Set(-R(idx, idx+1), y_block.GetBlock(idx+1));
                        for (int j = idx+2; j < s; j++) {
                            temp_scalar1.Add(-R(idx, j), y_block.GetBlock(j));
                        }
                        StageOper.IRKOper->ImplicitMult(temp_scalar1, temp_scalar2);
                        z_block.GetBlock(idx) += temp_scalar2; // Add to existing RHS
                        
                    // NO MASS MATRIX    
                    } else {
                        for (int j = idx+1; j < s; j++) {
                            z_block.GetBlock(idx).Add(-R(idx, j), y_block.GetBlock(j));
                        }
                    }
                }
                
                // --- Solve 1x1 system --- //
                // [R(idx, idx)*M - dt*L]*y(idx) = augmented(z(idx))
                // Pass preconditioner for diagonal block to Krylov solver
                krylov_solver1->SetPreconditioner(*JacDiagBlockPrec[diagBlock]);
                // Pass diagonal block to Krylov solver
                krylov_solver1->SetOperator(*JacDiagBlock[diagBlock]);
                // Solve
                krylov_solver1->Mult(z_block.GetBlock(idx), y_block.GetBlock(idx));
                krylov_converged = krylov_solver1->GetConverged();
                krylov_iters[diagBlock] += krylov_solver1->GetNumIterations();
            } 
            // Invert 2x2 diagonal block
            else if (StageOper.Butcher.R0_block_sizes[diagBlock] == 2) 
            {
                if (printlevel > 0) {
                    mfem::out << ": 2x2 block  -->  ";
                    if (printlevel != 2) mfem::out << '\n';
                }
                idx--; // Index first unknown in pair rather than second
        
                // --- Form RHS vector (this overrides z_block[idx], z_block[idx+1]) --- //
                // Point z_2block to the appropriate data from z_block 
                // (note data arrays for blocks are stored contiguously)
                z_2block.Update(z_block.GetBlock(idx).GetData(), offsets_2);
                
                // Subtract out known information from LHS of equations
                if (idx+2 < s) {
                    // MASS MATRIX
                    if (StageOper.IRKOper->isImplicit()) {
                        // First component
                        temp_scalar1.Set(-R(idx, idx+2), y_block.GetBlock(idx+2));
                        for (int j = idx+3; j < s; j++) {
                            temp_scalar1.Add(-R(idx, j), y_block.GetBlock(j));
                        }
                        StageOper.IRKOper->ImplicitMult(temp_scalar1, temp_scalar2);
                        z_2block.GetBlock(0) += temp_scalar2; // Add to existing RHS
                        // Second component
                        temp_scalar1.Set(-R(idx+1, idx+2), y_block.GetBlock(idx+2)); 
                        for (int j = idx+3; j < s; j++) {
                            temp_scalar1.Add(-R(idx+1, j), y_block.GetBlock(j)); 
                        }
                        StageOper.IRKOper->ImplicitMult(temp_scalar1, temp_scalar2);
                        z_2block.GetBlock(1) += temp_scalar2; // Add to existing RHS
                    
                    // NO MASS MATRIX    
                    } else {
                        for (int j = idx+2; j < s; j++) {
                            z_2block.GetBlock(0).Add(-R(idx, j), y_block.GetBlock(j)); // First component
                            z_2block.GetBlock(1).Add(-R(idx+1, j), y_block.GetBlock(j)); // Second component
                        }
                    }
                }
                
                // Point y_2block to data array of solution vector
                y_2block.Update(y_block.GetBlock(idx).GetData(), offsets_2);
                
                // --- Solve 2x2 system --- //
                // [R(idx,idx)*M-dt*N'     R(idx,idx+1)*M    ][y(idx)  ] = augmented(z(idx))
                // [R(idx+1,idx)*M     R(idx+1,idx+1)*M-dt*N'][y(idx+1)] = augmented(z(idx+1))
                // Pass preconditioner for diagonal block to Krylov solver
                krylov_solver2->SetPreconditioner(*JacDiagBlockPrec[diagBlock]);
                // Pass diagonal block to Krylov solver
                krylov_solver2->SetOperator(*JacDiagBlock[diagBlock]);
                // Solve
                krylov_solver2->Mult(z_2block, y_2block);
                krylov_converged = krylov_solver2->GetConverged();    
                krylov_iters[diagBlock] += krylov_solver1->GetNumIterations();
            }
            
            // Check convergence 
            if (!krylov_converged) {
                string msg = "KronJacSolver::BlockBackwardSubstitution() Krylov solver at t=" 
                                + to_string(StageOper.IRKOper->GetTime()) 
                                + " not converged [system " + to_string(s_eff-diagBlock) 
                                + "/" + to_string(s_eff) 
                                + ", size=" + to_string(StageOper.Butcher.R0_block_sizes[diagBlock]) + ")]\n";
                mfem_error(msg.c_str());
            }
            
            idx--; // Decrement to refer to next unknown
        }
    }
};



/* Class implementing preconditioned solution of fully implicit RK schemes for 
the nonlinear ODE system M*du/dt = N(u,t), where M and N are implemented as an 
IRKOperator */
class IRK : public ODESolver
{
private:    
    MPI_Comm m_comm;          
    int m_numProcess;
    int m_rank;
    
    RKButcherData m_Butcher;    // Runge-Kutta Butcher tableau and associated data

    IRKOperator * m_IRKOper;    // All information about spatial discretization. 
    BlockVector m_w;            // The stage vectors
    Array<int> m_stageOffsets;  // Offsets 
    IRKStageOper * m_stageOper; // Operator encoding stages, F(w) = 0
    
    // Nonlinear and linear solvers for computing stage vectors
    bool m_solversInit;                 // Have the solvers been initialized?
    NewtonSolver * m_nonlinear_solver;  // Nonlinear solver for inverting F
    KronJacSolver * m_jacobian_solver;  // Linear solver for inverting approximate Jacobian
    NEWTON_params m_newton_params;      // Parameters for Newton solver
    Krylov_params m_krylov_params;      // Parameters for Krylov solver.
    Krylov_params m_krylov_params2;     // Parameters for Krylov solver for 2x2 systems, if different than that for 1x1 systems
    bool m_krylov2;                     // Do we use a second solver?

    
    /* --- Statistics on solution of nonlinear and linear systems --- */
    int m_avg_newton_iter;          // Across whole integration, avg number of Newton iterations per time step
    vector<int> m_avg_krylov_iter;  // Across whole integration, avg number of Krylov iterations per time step (for every 1x1 and 2x2 system)
    vector<int> m_system_size;      // Associated linear system sizes: 1x1 or 2x2?
    vector<double> m_eig_ratio;     // The ratio beta/eta
public:


    IRK(IRKOperator *IRKOper_, IRKType RK_ID_);
    ~IRK();
 
    void Init(TimeDependentOperator &F);

    void Run(Vector &x, double &t, double &dt, double tf);
    
    void Step(Vector &x, double &t, double &dt);

    void SimplifiedJacobianSolve(BlockVector &w);

    void InitSolvers();

    /* --- Set solver parameters fot implicit time-stepping --- */
    /// Newton solver
    inline void SetNewtonParams(NEWTON_params params) { 
        MFEM_ASSERT(!m_solversInit, "IRK::SetNewtonParams:: Can only be called before IRK::Run()");
        m_newton_params = params;
    }
    /// Krylov solver
    inline void SetKrylovParams(Krylov_params params) { 
        MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
        m_krylov_params = params;
        m_krylov2 = false; // Using single Krylov solver
    }
    /// Krylov solvers, if different solver to be used to 1x1 and 2x2 systems
    // TODO: Really should check these structs are not equal by value here...
    inline void SetKrylovParams(Krylov_params params1, Krylov_params params2) { 
        MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Run()");
        m_krylov_params  = params1;
        m_krylov_params2 = params2;
        m_krylov2 = true; // Using two Krylov solvers
    }
    
    // Get statistics about solution of nonlinear and linear systems
    inline void GetSolveStats(int &avg_newton_iter, 
                                vector<int> &avg_krylov_iter, 
                                vector<int> &system_size, 
                                vector<double> &eig_ratio) const {
                                    avg_newton_iter = m_avg_newton_iter;
                                    avg_krylov_iter = m_avg_krylov_iter;
                                    system_size     = m_system_size;
                                    eig_ratio       =  m_eig_ratio;
                                }
};

#endif