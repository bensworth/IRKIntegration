#ifndef IRK_H
#define IRK_H

#include "HYPRE.h"
#include "mfem.hpp"
#include "IRK_utils.hpp"

#include <mpi.h>
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;

/** Sparsity pattern of (Q^T \otimes I) diag(N1',...,Ns') (Q \otimes I)
    to be used in block-upper-triangular Quasi-Newton method */
enum JacSparsity {
    LUMPED = 0, DIAGONAL = 1, DENSE = 2
};

/// Parameters for Newton solver
struct NewtonParams {
    double reltol = 1e-6;
    double abstol = 1e-6;
    int maxiter = 25;
    int printlevel = -1; 
    
    // How frequently to we update the Jacobian matrix?
    // <=0 - First iteration only
    // 1 - Every iteration
    // x > 1 - Every x iterations
    int jac_update_rate = 0;  
    JacSparsity jac_solver_sparsity = JacSparsity::DENSE;
    JacSparsity jac_prec_sparsity  = JacSparsity::DIAGONAL;
}; 

class IMEXEuler : public ODESolver
{
protected:
    Vector k;
    Vector z;
    IRKOperator *imex;    // Spatial discretization. 

public:
    void Init(IRKOperator &_imex);
    void Step(Vector &x, double &t, double &dt) override;
};


/** Class implementing conjugate-pair preconditioned solution of fully implicit 
    RK schemes for the nonlinear ODE system 
        M*du/dt = N(u,t), 
    as implemented as an IRKOperator */
class IRK : public ODESolver
{
protected:    
    MPI_Comm m_comm;          
    int m_numProcess;
    int m_rank;
    
    const RKData &m_Butcher;    // Runge-Kutta Butcher tableau and associated data

    IRKOperator * m_IRKOper;    // Spatial discretization. 
    BlockVector sol_imp;        // Block vector with s blocks for implicit solve of stages
    Array<int> m_stageOffsets;  // Offsets 
    IRKStageOper * m_stageOper; // Operator encoding stages, F(w) = 0
    
    // Nonlinear and linear solvers for computing stage vectors
    int gamma_idx;                  // (2,2) preconditioning constant, [0==eta, 1==eta+beta^2/eta].
    bool m_solversInit;             // Have the solvers been initialized?
    NewtonSolver * m_newton_solver; // Nonlinear solver for inverting F
    TriJacSolver * m_tri_jac_solver;// Linear solver for inverting (approximate) Jacobian
    QuasiMatrixProduct * m_jac_solverSparsity;
    QuasiMatrixProduct * m_jac_precSparsity;
    
    NewtonParams m_newton_params;   // Parameters for Newton solver
    KrylovParams m_krylov_params;   // Parameters for Krylov solver.
    KrylovParams m_krylov_params2;  // Parameters for Krylov solver for 2x2 systems, if different than that for 1x1 systems
    bool m_krylov2;                 // Do we use a second solver?

    /* Statistics on solution of nonlinear and linear systems across whole integration */
    int m_avg_newton_iter;          // Avg number of Newton iterations per time step
    vector<int> m_avg_krylov_iter;  // Avg number of Krylov iterations per time step (for every 1x1 and 2x2 system)
    vector<int> m_system_size;      // Associated linear system sizes: 1x1 or 2x2?
    vector<double> m_eig_ratio;     // The ratio beta/eta

    /// Build nonlinear and linear solvers
    void SetSolvers();

public:

    /// Constructor
    IRK(IRKOperator *IRKOper_, const RKData &ButcherTableau);
    
    /// Destructor
    ~IRK();
 
    /// Call base class' init and initialize remaing things here.
    void Init(TimeDependentOperator &F);

    /// Full time stepping
    void Run(Vector &x, double &t, double &dt, double tf);
    
    /** Apply RK update to take x from t to t+dt,
        x = x + (dt*b0^\top \otimes I)*k 
          = x + (dt*d0^\top \otimes I)*w,
    where w = (A0 x I)k. Note, w1 = a_11 k_1 + ... + a_1s k_s. */
    void Step(Vector &x, double &t, double &dt);

    /// Set parameters for Newton solver
    inline void SetNewtonParams(NewtonParams params) { 
        MFEM_ASSERT(!m_solversInit, "IRK::SetNewtonParams:: Can only be called before IRK::Init()");
        m_newton_params = params;
    }

    /// Set parameters for Krylov solver
    inline void SetKrylovParams(KrylovParams params) { 
        MFEM_ASSERT(!m_solversInit, "IRK::SetKrylovParams:: Can only be called before IRK::Init()");
        m_krylov_params = params;
        m_krylov2 = false; // Using single Krylov solver
    }

    /** Set parameters for Krylov solver, if different solver to be used to 1x1 
        and 2x2 systems */
    // TODO: Really should check these structs are not equal by value here...
    void SetKrylovParams(KrylovParams params1, KrylovParams params2);

    /// Get statistics about solution of nonlinear and linear systems
    void GetSolveStats(int &avg_newton_iter,
        vector<int> &avg_krylov_iter, vector<int> &system_size, 
        vector<double> &eig_ratio) const;

    // (2,2) preconditioning constant, [0==eta, 1==eta+beta^2/eta]
    void SetGammaID(int gamma_idx_) {gamma_idx = gamma_idx_; }
};

/** Class implementing conjugate-pair preconditioned solution of fully implicit 
    RK schemes for the nonlinear ODE system 
        M*du/dt = N(u,t), 
    as implemented as an IRKOperator */
class PolyIMEX : public IRK
{
private:
    Array<int> m_stageOffsets2; // Offsets for s+1 block vectors
    BlockVector exp_part;       // Block vector with s+1 blocks for explicit part of solution
    BlockVector rhs;            // Block vector with s blocks for implicit right-hand side
    Vector      sol_exp;        // Vector for explicit solution
    int exp_ind;
    int num_iters;

    bool initialized;
    bool linearly_imp;           // Linearly implicit

    /** Form rhs for polyomial IMEX; include scaling by (A0 x I)^{-1}, as
        applied to nonlinear system on right. */
    void FormImpRHS(Vector &x_prev, const double &t,
        const double &r, bool iterator);

    /// Update stored explicit components for future time steps/iterations. 
    void UpdateExplicitComponents(const double &t, const double &r);

    /** Update solution via one implicit-explicit pass; can be used as
        preconditioning for higher-order scheme or kernel of single time step. */
    void Iterate(Vector &x, const double &t, const double &r, bool iterator);

public:
    PolyIMEX(IRKOperator *IRKOper_, const RKData &ButcherTableau,
        bool linearly_imp_, int num_iters_=1);
    ~PolyIMEX() { };
 
    /// Call base class' init and initialize remaing things here.
    void Init(TimeDependentOperator &F);

    void SetSolvers();

    /// Full time stepping
    // void Run(Vector &x, double &t, double &dt, double tf);
    
    /** Apply RK update to take x from t to t+dt,
        x = x + (dt*b0^\top \otimes I)*k 
          = x + (dt*d0^\top \otimes I)*w,
    where w = (A0 x I)k. Note, w1 = a_11 k_1 + ... + a_1s k_s. */
    void Step(Vector &x, double &t, double &dt);
};

#endif
