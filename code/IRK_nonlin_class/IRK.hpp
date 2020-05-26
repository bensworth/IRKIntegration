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
//  + 0 = L-stable SDIRK
//  + 1 = Gauss-Legendre
//  + 2 = RadauIIA
//  + 3 = Lobatto IIIC
//  Second digit: order of scheme
enum IRKType { 
    SDIRK1 = 01, SDIRK2 = 02, SDIRK3 = 03, SDIRK4 = 04,
    Gauss2 = 12, Gauss4 = 14, Gauss6 = 16, Gauss8 = 18, Gauss10 = 110,
    RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27, RadauIIA9 = 29,
    LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36, LobIIIC8 = 38
};

// Parameters for Newton solver
struct NEWTON_params {
    double reltol = 1e-6;
    int maxiter = 10;
    int printlevel = 2; 
};

// Different type of Krylov solvers
enum class KrylovMethod {CG, MINRES, GMRES, BICGSTAB, FGMRES};

// Parameters for Krylov solver
struct Krylov_params {
    double abstol = 1e-13;
    double reltol = 1e-13;
    int maxiter = 100;
    int printlevel = 0;
    int kdim = 30;
    KrylovMethod solver = KrylovMethod::GMRES;
};


/// Kronecker transform between two block vectors
void KronTransform(const DenseMatrix &A, const BlockVector &x, BlockVector &y);

/// Kronecker transform between two block vectors using A transpose
void KronTransformTranspose(const DenseMatrix &A, const BlockVector &x, BlockVector &y);


/* 
Abstract base class for spatial discretizations of a PDE resulting in the 
time-dependent ODE 
    M*du/dt = L(u,t)    _OR_    du/dt = L(u,t)

TODO: rework comments below once we've finalized what we're doing

If no mass matrix exists, M_exists=false must be passed to the constructor. 
If a mass matrix exists (default), the virtual function ApplyM() must be implemented.
*/
class IRKOperator : public TimeDependentOperator
{    
protected:
    MPI_Comm m_globComm;
    
public:
    // NOTE: Knowledge of a mass matrix is decuded from the value of TimeDependentOperator::Type == IMPLICIT
    IRKOperator(MPI_Comm comm, int n=0, double t=0.0, Type type=EXPLICIT) 
        : TimeDependentOperator(n, t, type), 
            m_globComm{comm}, m_M_exists{type == IMPLICIT} {};
    
    ~IRKOperator() { };

    MPI_Comm GetComm() { return m_globComm; };

    /** Apply action of du/dt, y <- M^{-1}*L(x,t) _OR_ y <- L(x,y) */
    //virtual void Mult(const Vector &x, Vector &y) const = 0;
    
    /** Apply action of M*du/dt _OR_ du/dt, y <- L(x,y) */
    virtual void ExplicitMult(const Vector &x, Vector &y) const = 0;
    
    /** Gradient of L(u, t) w.r.t u evaluated at x */
    virtual Operator &GetExplicitGradient(const Vector &x) const = 0;
    
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
    
    // /** Precondition (\gamma*M - dt*L) OR (\gamma*I - dt*L) */
    // virtual void ImplicitPrec(const Vector &x, Vector &y) const = 0;
    
    
    // Function to ensure that ImplicitPrec preconditions (\gamma*M - dt*L) OR (\gamma*I - dt*L)
    // with gamma and dt as passed to this function.
    //      + index -> index of real char poly factor, [0,#number of real factors)
    //      + type -> eigenvalue type, 1 = real, 2 = complex pair
    //      + t -> time.
    // These additional parameters are to provide ways to track when
    // (\gamma*M - dt*L) or (\gamma*I - dt*L') must be reconstructed or not to minimize setup.
    // virtual void SetSystem(int index, double t, double dt,
    //                        double gamma, int type) = 0;
    
    // Does a mass matrix exist for this discretization; this needs to be public so IRK can access it
    bool m_M_exists; 
};

// 
// /* Wrapper to preconditioner factors in a polynomial by preconditioning either
//     TYPE 1. [gamma*M - dt*L] _OR_ [gamma*I - dt*L]
//     TYPE 2. [gamma*M - dt*L]M^{-1}[gamma*M - dt*L]  _OR_ [gamma*I - dt*L]^2
// 
// [gamma*M - dt*L] _OR_ [gamma*I - dt*L] is preconditioned using
// IRKOperator.ImplicitPrec(). Type 2 involves two IRKOperator.ImplicitPrec()
// applications, with an application of M in between, IRKOperator.ApplyM().
// */
// class CharPolyPrecon : public Solver
// {
// 
// private:
//     int m_type; /* 1 or 2; type of preconditioner to provide */
//     IRKOperator &m_IRKOper; /* Holds all information about spatial discretization */
// 
// public:
// 
//     CharPolyPrecon(IRKOperator &S)
//         : Solver(S.Height(), false), m_IRKOper(S), m_type(-1) { };
// 
//     ~CharPolyPrecon() { };
// 
//     void SetType(int type) { m_type = type; };
// 
//     inline void Mult(const Vector &x, Vector &y) const
//     {
//         if (m_type == 1) {
//             m_IRKOper.ImplicitPrec(x, y); // Precondition [gamma*M - dt*L] _OR_ [gamma*I - dt*L]
// 
//         } else if (m_type == 2) {
//             Vector z(x);
//             // With mass matrix 
//             if (m_IRKOper.m_M_exists) {
//                 m_IRKOper.ImplicitPrec(z, y);     // Precondition [gamma*M - dt*L]
//                 m_IRKOper.ImplicitMult(y, z);     // Apply M
//                 m_IRKOper.ImplicitPrec(z, y);     // Precondition [gamma*M - dt*L]
//             // Without mass matrix
//             } else {
//                 m_IRKOper.ImplicitPrec(x, z);     // Precondition [gamma*I - dt*L]
//                 m_IRKOper.ImplicitPrec(z, y);     // Precondition [gamma*I - dt*L]
//             }
//         }
//         else {
//             mfem_error("CharPolyPrecon::Must set polynomial type 1 or 2!\n");
//         }
//     };
// 
//     // Purely virtual function we must implement but do not use
//     virtual void SetOperator(const Operator &op) {  };
// };

/* Char. poly factors, F:
    TYPE 1. F == [zeta*M - dt*L], _OR_ F == [zeta*I - dt*L]
    TYPE 2. F == [(eta^2+beta^2)*M - 2*eta*dt*L + dt^2*L*M^{-1}*L] _OR_ [(eta^2+beta^2)*I - 2*eta*dt*L + (dt*L)^2]
*/
// class CharPolyOp : public Operator
// {
// private:
// 
//     int m_type; // 1 or 2; type of factor
//     double m_gamma; // Constant in preconditioner
//     double m_dt;
//     Vector m_c;     // Coefficients describing operator as polynomial in L
//     IRKOperator &m_IRKOper;
// 
// public:
// 
//     /* Constructor for TYPE 1 char. polynomial factor */
//     CharPolyOp(double dt, double zeta, IRKOperator &S) 
//         : Operator(S.Height()), m_c(2), m_dt{dt}, m_IRKOper{S},
//             m_gamma(zeta), m_type(1)
//     {
//         // Coefficients of operator as a polynomial in L
//         m_c(0) = zeta;
//         m_c(1) = -1.0;
//     };
// 
//     /* Constructor for TYPE 2 char. polynomial factor */
//     CharPolyOp(double dt, double eta, double beta, IRKOperator &S) 
//         : Operator(S.Height()), m_dt{dt}, m_IRKOper{S},
//         m_c(3), m_gamma(eta), m_type(2)
//     {
//         // Coefficients of operator as a polynomial in L
//         m_c(0) = eta*eta + beta*beta;
//         m_c(1) = -2.0*eta;
//         m_c(2) = 1.0;
//     };
// 
//     inline int Type() {return m_type; };
//     inline double Gamma() {return m_gamma; };
//     inline double dt() {return m_dt; };
//     inline void Setdt(double dt) { m_dt = dt; };
// 
//     /* y <- char. poly factor(dt*M^{-1}*L)*x _OR_ y <- char. poly factor(dt*L)*x */
//     inline void Mult(const Vector &x, Vector &y) const 
//     {
//         // If no mass matrix, factor is simply a polynomial in dt*L
//         if (!m_IRKOper.m_M_exists) {
//             m_IRKOper.PolynomialMult(m_c, m_dt, x, y); 
// 
//         // If mass matrix exists, factor is not quite a polynomial in dt*L
//         } else {
//             Vector z(x); // Auxillary vector
// 
//             // F == [zeta*M - dt*L]
//             if (m_type == 1) {
//                 m_IRKOper.ImplicitMult(x, z);
//                 z *= m_c(0);
//                 m_IRKOper.ApplyL(x, y);
//                 y *= -m_dt;
//                 y += z;
// 
//             // F == [(eta^2 + beta^2)*M - 2*dt*L + dt^2*L*M^{-1}*L]
//             } else {
//                 m_IRKOper.ApplyL(x, y);
//                 m_IRKOper.ApplyMInv(y, z);
//                 z *= m_c(2)*m_dt;
//                 z.Add(m_c(1), x); // z = [c(1)*I + c(2)*dt*M^{-1}*L]*x
//                 m_IRKOper.ApplyL(z, y);
//                 y *= m_dt;   // y = dt*L*[c(1)*I + c(2)*dt*M^{-1}*L]*x
//                 m_IRKOper.ImplicitMult(x, z);
//                 z *= m_c(0); 
//                 y += z;
//             }
//         }
//     }
//     ~CharPolyOp() { };
// };


    
class RKButcherData 
{
private:    
    void SetData();     // Set Butcher tableau coefficients
    void SizeData();    // Set dimensions of Butcher arrays
    
    IRKType RK_ID;
    
public:
    
    RKButcherData(IRKType RK_ID_) : RK_ID{RK_ID_} { 
        SetData();
        invA0 = A0;
        invA0.Invert();
     };
    
    ~RKButcherData() {};
    
    int s;            // Number of stages
    int s_eff;        // Number of eigenvalues of A once complex conjugates have been combined
    DenseMatrix A0;   // The Butcher matrix
    DenseMatrix invA0;// Inverse of Buthcer matrix
    DenseMatrix Q0;   // Orthogonal matrix in Schur decomposition of A0^-1
    DenseMatrix R0;   // Quasi-upper triangular matrix in Schur decomposition of A0^-1
    Array<int> R0_block_sizes; // From top of block diagonal, sizes of blocks
    Vector b0;        // Butcher tableau weights
    Vector c0;        // Butcher tableau nodes
    Vector d0;        // b*inv(A)
};    

// Type of Jacobian of to use when inverting an IRKStageOper
enum JacobianType {
    FULL = 0,       // The true Jacobian, uses stage-dependent N'.
    KRONECKER = 1   // Kronecker-product Jacobian. Ignores stage dependence of N' and assumes its constant.
};

class KronJacSolver;
/* Operator F defining the s stage equations. They satisfy F(w) = 0, 
    where w = (A \otimes I)*k */
class IRKStageOper : public BlockOperator
{
private:
    friend class KronJacSolver;  // Approximate Jacobian solver needs access
    
    Array<int> &offsets;          // Offsets of operator     
    
    mutable IRKOperator * IRKOper;// Spatial discretization
    const RKButcherData &Butcher; // All RK information
    
    // Parameters that system depends on
    const Vector * u;             // Current state.
    double t;                     // Current time of state.
    double dt;                    // Current time step
    
    // Wrappers for scalar vectors
    mutable BlockVector w_block, y_block;
    
    // Auxillary vectors
    mutable Vector temp1, temp2;    
    
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
            temp1(S_->Height()), temp2(S_->Height()), 
            w_block(offsets_), y_block(offsets_),
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
    
    /// y <- F(w)
    inline void Mult(const Vector &w_scalar, Vector &y_scalar) const
    {
        // Wrap scalar Vectors with BlockVectors
        w_block.Update(w_scalar.GetData(), offsets);
        y_block.Update(y_scalar.GetData(), offsets);
        
        KronTransform(Butcher.invA0, w_block, y_block); // y <- inv(A)*w
        
        for (int i = 0; i < Butcher.s; i++) {        
            add(*u, dt, w_block.GetBlock(i), temp1); // temp1 <- u+dt*w(i)
            IRKOper->SetTime(t + Butcher.c0[i]*dt);
            IRKOper->ExplicitMult(temp1, temp2); // temp2 <- N(temp1, t)
            y_block.GetBlock(i).Add(-1., temp2);
        } 
    }
};


// Defines diagonal blocks appearing in the Kronecker product Jacobian. 
// These take the form 1x1 or 2x2 blocks.
//
// 1x1: 
//      [R(0,0)*I - dt*N']
// 
// 2x2:
//      [R(0,0)*I-dt*N'  R(0,1)*I  ]
//      [R(1,0)*I      R(1,1)-dt*N']
//
// NOTE: The Jacobian block N' is assumed to be the same on both diagonals
//
// TODO
//  -make BlockOperator rather than Operator??
//  -Probably make into JacDiagBlock without the Kron part. Can just pass two separate blocks, which are potentially the same.
class KronJacDiagBlock : public Operator
{
private:
    
    int size;
    mutable double dt;
    mutable const Operator * N_jac; 
    
    // Data required for 1x1 operators
    double R00;
    
    // Data required for 2x2 operators
    Array<int> offsets; 
    DenseMatrix R;
    mutable BlockVector x_block, y_block;
    
public:

    // 1x1 block
    KronJacDiagBlock(int height, double R00_) : Operator(height), 
        size{1}, dt{0.0}, N_jac(NULL),
        R00{R00_}
        {};

    // 2x2 block
    KronJacDiagBlock(int height, DenseMatrix R_, Array<int> offsets_) : Operator(height), 
        size{2}, dt{0.0}, N_jac(NULL),
        R(R_), offsets(offsets_)
        {};

    // Update parameters required to compute action
    inline void SetParameters(double dt_, const Operator * N_jac_) const
    {
        dt = dt_;
        N_jac = N_jac_;
    };
        
    // Compute action of diagonal block    
    inline void Mult(const Vector &x, Vector &y) const {
        
        MFEM_ASSERT(x.Size() == this->Height(), "KronJacDiagBlock::Mult() incorrect input Vector size");
        MFEM_ASSERT(y.Size() == this->Height(), "KronJacDiagBlock::Mult() incorrect output Vector size");
        
        if (!N_jac) mfem_error("KronJacDiagBlock::Mult() must set Jacobian block with SetParameters()");
        
        // 1x1 operator, y = [R(0)(0)*I - dt*N']*x
        if (size == 1) {
            N_jac->Mult(x, y);
            y *= -dt;
            y.Add(R00, x);
            
        // 2x2 operator,
        //  y(0) = [R(0,0)*I-dt*N']*x(0) + [R(0,1)*I]*x(1)
        //  y(1) = [R(1,0)*I]*x(0)       + [R(1,1)-dt*N']*x(1)
        } else if (size == 2) {
            // Wrap scalar Vectors with BlockVectors
            x_block.Update(x.GetData(), offsets);
            y_block.Update(y.GetData(), offsets);
            
            // 1st component of y
            N_jac->Mult(x_block.GetBlock(0), y_block.GetBlock(0));
            y_block.GetBlock(0) *= -dt;
            y_block.GetBlock(0).Add(R(0,0), x_block.GetBlock(0));
            y_block.GetBlock(0).Add(R(0,1), x_block.GetBlock(1));
            // 2nd component of y
            N_jac->Mult(x_block.GetBlock(1), y_block.GetBlock(1));
            y_block.GetBlock(1) *= -dt;
            y_block.GetBlock(1).Add(R(1,1), x_block.GetBlock(1));
            y_block.GetBlock(1).Add(R(1,0), x_block.GetBlock(0));
        }
    };     
};


// N' (the Jacobian of N) is assumed constant w.r.t w, so that
// the diagonal block of Jacobians can be written as a Kronecker product,
//      I \otimes N' ~ diag(N'(u+dt*w1), ..., N'(u+dt*ws))
// Letting J = A^-1 \otimes I - dt * I \otimes N', here we solve J*x=b via 
// block backward substitution, making use of the Schur decomposition of A^-1.
//
// TODO:
//  -Possibility, but maybe overkill. Allow two different Krylov solvers so that
//      a different solver can be used to invert 1x1 blocks (this could exploit 
//      SDP properties etc, which cannot be done for the 2x2 blocks)
class KronJacSolver : public Solver
{

private:
    IRKStageOper &StageOper;
    
    Array<int> &offsets;    // Offsets for vectors with s blocks
    Array<int> offsets_2;   // Offsets for vectors with 2 blocks
    
    // Auxillary vectors with s blocks
    mutable BlockVector x_block, b_block, b_block_temp, x_block_temp;   
    
    // Auxillary Vectors with 2 blocks 
    mutable BlockVector y_2block, z_2block;  
     
    // Auxillary vectors for evaluating N' 
    Vector temp_scalar; 
     
    // Diagonal blocks inverted during backward substitution
    Array<KronJacDiagBlock *> JacobianDiagBlock;
    
    // Solver for inverting diagonal blocks
    IterativeSolver * krylov_solver; 
    
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
    
    KronJacSolver(IRKStageOper &StageOper_, const Krylov_params &solver_params,
                  int N_jac_lin_, int N_jac_update_rate_) 
        : Solver(StageOper_.Height()),
        StageOper(StageOper_),
        offsets(StageOper_.RowOffsets()),
        x_block(StageOper_.RowOffsets()), b_block(StageOper_.RowOffsets()), 
        b_block_temp(StageOper_.RowOffsets()), x_block_temp(StageOper_.RowOffsets()), 
        y_2block(), z_2block(),
        temp_scalar(StageOper_.RowOffsets()[1]),
        offsets_2(0),
        N_jac(NULL), 
        N_jac_lin{N_jac_lin_},
        N_jac_update_rate{N_jac_update_rate_}
    {
        
        MFEM_ASSERT(N_jac_lin > 0 && N_jac_lin <= StageOper.Butcher.s, 
                "KronJacSolver: Require 0 < N_jac_linearization <= s");
        MFEM_ASSERT(N_jac_update_rate == 0 || N_jac_update_rate == 1, 
                "KronJacSolver: Require N_jac_update_rate==0 or 1");
        
        // Create operators describing diagonal blocks
        double R00;         // 1x1 diagonal block of R0
        DenseMatrix R(2);   // 2x2 diagonal block of R0
        int row = 0;        // Row of R0 we're accessing
        JacobianDiagBlock.SetSize(StageOper.Butcher.s_eff);
        for (int i = 0; i < StageOper.Butcher.s_eff; i++) {
            
            // 1x1 diagonal block
            if (StageOper.Butcher.R0_block_sizes[i] == 1)
            {
                R00 = StageOper.Butcher.R0(row,row);
                JacobianDiagBlock[i] = new KronJacDiagBlock(offsets[1], R00);    
            } 
            // 2x2 diagonal block
            else 
            {
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
                JacobianDiagBlock[i] = new KronJacDiagBlock(2*offsets[1], R, offsets_2);
                row++; // We've processed 2 rows of R0 here.
            }
            row++; // Increment to next row of R0
        }
        
        // Set up Krylov solver for inverting diagonal blocks
        GetKrylovSolver(krylov_solver, solver_params);
        
        // TODO Set the preconditioner... This will be done inside the backward 
        // substitution loop depending on the system being solved...
        //krylov_solver->SetPreconditioner(); // JacPrec is preconditioner for m_KRYLOV

        
    };
    
    
    ~KronJacSolver()
    {
        for (int i = 0; i < JacobianDiagBlock.Size(); i++) {
            delete JacobianDiagBlock[i];
        }
        delete krylov_solver;
    };
    
    /// Set up Krylov solver for inverting diagonal blocks
    //
    // NOTE: Rather than accessing member solver, leave open possibility of setting
    // up any Krylov solver for when I add another solver to handle 1x1/2x2 systems differently.
    inline void GetKrylovSolver(IterativeSolver * &solver, const Krylov_params &solver_params) {
        switch (solver_params.solver) {
            case KrylovMethod::CG:
                solver = new CGSolver(StageOper.IRKOper->GetComm());
                break;
            case KrylovMethod::MINRES:
                solver = new MINRESSolver(StageOper.IRKOper->GetComm());
                break;
            case KrylovMethod::GMRES:
                solver = new GMRESSolver(StageOper.IRKOper->GetComm());
                static_cast<GMRESSolver*>(solver)->SetKDim(solver_params.kdim);
                break;
            case KrylovMethod::BICGSTAB:
                krylov_solver = new MINRESSolver(StageOper.IRKOper->GetComm());
                break;    
            case KrylovMethod::FGMRES:
                solver = new FGMRESSolver(StageOper.IRKOper->GetComm());
                static_cast<FGMRESSolver*>(solver)->SetKDim(solver_params.kdim);
                break;
            default:
                mfem_error("IRK::Invalid Krylov solve type.\n");   
        }
        
        solver->iterative_mode = false;
        solver->SetAbsTol(solver_params.abstol);
        solver->SetRelTol(solver_params.reltol);
        solver->SetMaxIter(solver_params.maxiter);
        solver->SetPrintLevel(solver_params.printlevel);
    }
    
    
    // Newton method will pass the operator returned from its GetGradient() to this,
    // but we don't actually require this.
    void SetOperator (const Operator &op) { 
    
        double dt = StageOper.dt;
        double t = StageOper.t;
    
        // Update N_jac if necessary (1st Newton iteration, or all Newton iterations)
        if (StageOper.getGradientCalls == 1 || N_jac_update_rate == 1) {
    
            // Basic option: Eval jacobian of N at u
            if (N_jac_lin == 0) {
                temp_scalar = *(StageOper.u);
                StageOper.IRKOper->SetTime(t);
            // Eval jacobian of N at u + dt*w(N_jac_linearization)
            } else {
                int idx = N_jac_lin-1; 
                add(*(StageOper.u), dt, StageOper.GetCurrentIterate().GetBlock(idx), temp_scalar);
                StageOper.IRKOper->SetTime(t + dt*StageOper.Butcher.c0[idx]);
            } 
            
            N_jac = &(StageOper.IRKOper->GetExplicitGradient(temp_scalar));
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
        
        //mfem::out << "Passed KronJacSolver::Mult()" << '\n';
    }
    
    /* Solve \tilde{J}*y = z via block backward substitution, where 
        \tilde{J} = R \otimes I - dt*I \otimes N' 
    NOTE: RHS vector z is not const, since its data is overridden during the solve */
    void BlockBackwardSubstitution(BlockVector &z_block, BlockVector &y_block) const
    {
        mfem::out << "\t---Backward solve---" << '\n';
        
        // Short hands
        int s = StageOper.Butcher.s; 
        int s_eff = StageOper.Butcher.s_eff; 
        DenseMatrix R = StageOper.Butcher.R0; 
        
        // Block index of current unknown (index w.r.t y/z block sizes, not Rs)
        int idx = s-1;
        
        /* Backward substitution: Invert diagonal blocks, which are 1x1 systems 
        for y[idx], or 2x2 systems for (y[idx],y[idx+1]) */
        for (int diagBlock = s_eff-1; diagBlock >= 0; diagBlock--)
        {
            mfem::out << "\t  Block solve " << s_eff-diagBlock << " of " << s_eff;
            
            // Update parameters for the diag block to be inverted
            JacobianDiagBlock[diagBlock]->SetParameters(StageOper.dt, N_jac);
            
            // Pass diagonal block to GMRES 
            krylov_solver->SetOperator(*JacobianDiagBlock[diagBlock]);
            
            // Invert 1x1 diagonal block
            if (StageOper.Butcher.R0_block_sizes[diagBlock] == 1) 
            {
                mfem::out << ": 1x1 block" << '\n';
                // --- Form RHS vector (this overrides z_block[idx]) --- //
                // Substract out known information from LHS of equation
                for (int j = idx+1; j < s; j++) {
                    z_block.GetBlock(idx).Add(-R(idx, j), y_block.GetBlock(j));
                }
                
                // --- Solve 1x1 system --- //
                // [R(idx, idx)*I - dt*L]*y[idx] = augmented(z[idx])
                krylov_solver->Mult(z_block.GetBlock(idx), y_block.GetBlock(idx));
            
            } 
            // Invert 2 x 2 diagonal block
            else if (StageOper.Butcher.R0_block_sizes[diagBlock] == 2) 
            {
                mfem::out << ": 2x2 block" << '\n';
                idx--; // Index first unknown in pair rather than second
        
                // --- Form RHS vector (this overrides z_block[idx], z_block[idx+1]) --- //
                // Point z_2block to the appropriate data from z_block 
                // (note data arrays for blocks are stored contiguously)
                z_2block.Update(z_block.GetBlock(idx).GetData(), offsets_2);
                // Substract out known information from LHS of equations
                for (int j = idx+2; j < s; j++) {
                    z_2block.GetBlock(0).Add(-R(idx, j), y_block.GetBlock(j)); // First component
                    z_2block.GetBlock(1).Add(-R(idx+1, j), y_block.GetBlock(j)); // Second component
                }
                
                // Point y_2block to data array of solution vector
                y_2block.Update(y_block.GetBlock(idx).GetData(), offsets_2);
                
                // --- Solve 2x2 system --- //
                // [R(idx,idx)*I-dt*N'    R(idx,idx+1)    ][y[idx]  ] = augmented(z[idx])
                // [R(idx+1,idx)*I    R(idx+1,idx+1)-dt*N'][y[idx+1]] = augmented(z[idx+1])
                krylov_solver->Mult(z_2block, y_2block);
            }
            
            // Check for convergence of diagonal solver 
            if (!krylov_solver->GetConverged()) {
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
    
    IRKOperator * m_IRKOper; // Holds all information about THE spatial discretization. TODO: Maybe rename to avoid confusion with Butcher m_s...
    
    Vector m_y, m_z; // Solution and RHS of linear systems
    Vector m_f; // Auxillary vectors
    
    // Require linear/nonlinear solves for implicit time stepping
    IRKStageOper * m_stageOper; // Operator encoding the s stages, F(k) = 0
    bool m_solversInit;
    NewtonSolver * m_nonlinear_solver;
    KronJacSolver * m_jacobian_solver;
    NEWTON_params m_NEWTON; // Parameters for Newton solvers
    Krylov_params m_KRYLOV; // Parameters for Krylov solvers
    
    // // Char. poly factors and preconditioner wrapper
    // Array<CharPolyOp  *> m_CharPolyOps;
    // CharPolyPrecon  m_CharPolyPrec;
    // IterativeSolver * m_krylov;

    // Runge-Kutta variables
    RKButcherData m_RK;
    BlockVector m_k;    // Stage vectors
    BlockVector m_w;
    Array<int> m_stageOffsets;
    
    /* --- Relating to solution of linear systems --- */
    // vector<int> m_avg_iter;  // Across whole integration, avg number of Krylov iters for each system
    // vector<int> m_type;      // Type 1 or 2?
    // vector<double> m_eig_ratio; // The ratio beta/eta
    
    
    //void PolyAction();        // Compute action of a polynomial on a vector
    // 
    // // Construct right-hand side, z, for IRK integration, including applying
    // // the block Adjugate and Butcher inverse 
    // void ConstructRHS(const Vector &x, double t, double dt, Vector &z);

public:


    IRK(IRKOperator *S, IRKType RK_ID);
    ~IRK();
 
    void Init(TimeDependentOperator &F);

    void Run(Vector &x, double &t, double &dt, double tf);
    
    void Step(Vector &x, double &t, double &dt);

    void SimplifiedJacobianSolve(BlockVector &w);

    void InitSolvers();

    /// Set solver parameters fot implicit time-stepping; MUST be called before InitSolvers()
    inline void SetKrylovParams(Krylov_params params) { 
        if (!m_solversInit) m_KRYLOV = params; else mfem_error("IRK::SetKrylovParams:: Can only be called before IRK::Run()"); };
    inline void SetNewtonParams(NEWTON_params params) { 
        if (!m_solversInit) m_NEWTON = params; else mfem_error("IRK::SetNewtonParams:: Can only be called before IRK::Run()"); };

    // // Get statistics about solution of linear systems
    // inline void GetSolveStats(vector<int> &avg_iter, vector<int> &type, 
    //                             vector<double> &eig_ratio) const {
    //                                 avg_iter  =  m_avg_iter;
    //                                 type      =  m_type;
    //                                 eig_ratio =  m_eig_ratio;
    //                             }
};

#endif