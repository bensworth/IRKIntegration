#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "FDSpaceDisc.hpp"
#include "IRK.hpp"

using namespace mfem;
using namespace std;

#define PI 3.14159265358979323846

// TODO: Segfault associated w/ freeing HypreBoomerAMG... Currently not being
// deleted... See associated `TODO` comment...

// Sample runs:
//      *** Solve 2D in space problem w/ 4th-order discretizations in space & time ***
//      >> mpirun -np 4 ./driver_adv_dif_FD -f 1 -d 2 -ex 1 -ax 0.85 -ay 1 -mx 0.3 -my 0.25 -l 5 -o 4 -dt -2 -t 14 -save 2 -tf 2 -np 2 -nmaxit 20 -nrtol 1e-10 -natol 1e-10 -krtol 1e-13 -katol 1e-13 -kp 0
//
//      *** Solve 1D in space PDE with 10th-order discretizations in space & time with dt=10*dx ***
//      * Kronecker-product Jacobian, updated 1st Newton iteration only *
//      >> mpirun -np 4 ./driver_adv_dif_FD -jac 0 -jacu 0 -f 1 -d 1 -ex 1 -ax 0.85 -mx 0.3 -l 5 -o 10 -dt -10 -t 110 -save 2 -tf 4 -np 2 -nmaxit 20 -nrtol 1e-10 -natol 1e-10 -krtol 1e-13 -katol 1e-13 -kp 0
//
//      * Kronecker-product Jacobian, updated every Newton iteration *
//      >> mpirun -np 4 ./driver_adv_dif_FD -jac 0 -jacu 1 -f 1 -d 1 -ex 1 -ax 0.85 -mx 0.3 -l 5 -o 10 -dt -10 -t 110 -save 2 -tf 4 -np 2 -nmaxit 20 -nrtol 1e-10 -natol 1e-10 -krtol 1e-13 -katol 1e-13 -kp 0
//
//      * Jacobian truncated to be block upper triangular *
//      >> mpirun -np 4 ./driver_adv_dif_FD -jac 1 -jacu 1 -jacs 2 -jacp 2 -f 1 -d 1 -ex 1 -ax 0.85 -mx 0.3 -l 5 -o 10 -dt -10 -t 110 -save 2 -tf 4 -np 2 -nmaxit 20 -nrtol 1e-10 -natol 1e-10 -krtol 1e-13 -katol 1e-13 -kp 0
//
//      * From upper triangular Jacobian, truncate all off-diagonal N' components *
//      >> mpirun -np 4 ./driver_adv_dif_FD -jac 1 -jacu 1 -jacs 1 -jacp 1 -f 1 -d 1 -ex 1 -ax 0.85 -mx 0.3 -l 5 -o 10 -dt -10 -t 110 -save 2 -tf 4 -np 2 -nmaxit 20 -nrtol 1e-10 -natol 1e-10 -krtol 1e-13 -katol 1e-13 -kp 0
//
//      * From upper triangular Jacobian, truncate all N' components except the largest one from each diagonal *
//      >> mpirun -np 4 ./driver_adv_dif_FD -jac 1 -jacu 1 -jacs 0 -jacp 0 -f 1 -d 1 -ex 1 -ax 0.85 -mx 0.3 -l 5 -o 10 -dt -10 -t 110 -save 2 -tf 4 -np 2 -nmaxit 20 -nrtol 1e-10 -natol 1e-10 -krtol 1e-13 -katol 1e-13 -kp 0
//
//
// Solve scalar advection-diffusion equation,
//     u_t + alpha.grad(f(u)) = div( mu \odot grad(u) ) + s(x,t),
// for constant vectors alpha, and mu.
// 
// An initial condition is specified, and periodic spatial boundaries are used.
// 
// Options for advection flux f are,
//     f(u) = u   -- linear advection,                      [pass `-f 0`]
//     f(u) = u^2 -- nonlinear advection (Burgers' like).   [pass `-f 1`]
// 
// Implemented in 1 and 2 spatial dimensions.               
// 
// Test problems with manufactured solutions are implemented to facillitate 
// convergence studies.                                     [pass `-ex 1` (OR `-ex 2` for f(u)==u for a different example) to report error]
//
// TIME DISCRETIZATION: Fully implicit RK.
//
// SPATIAL DISCRETIZATION: Arbitrarily high-order FD is available. 
// NOTE: No sophisticated treatment is provided for the discretization of 
// f(u)=u^2/2, which means that for small diffusivities mu, the discrete solution
// will become unstable due to the development of sharp gradients (however, the
// development of such gradients is dependent on what s(x,t) is, of course).


/* Output dictionary entries via an ofstream */
class OutDict : public std::ofstream {
public:
    template <class T>
    inline void Print(string id, T entry) {*this << id << " " << entry << "\n";};
};

// Problem parameters global functions depend on
Vector alpha, mu;
int dim, example, fluxID;

/* Functions definining some parts of PDE */
double InitialCondition(const Vector &x);
double Source(const Vector &x, double t);
double PDESolution(const Vector &x, double t);


struct AMG_params {
    bool use_AIR = false; 
    double distance = 1.5;
    string prerelax = "";
    string postrelax = "FFC";
    double strength_tolC = 0.1;
    double strength_tolR = 0.01;
    double filter_tolR = 0.0;
    int interp_type = 100;
    int relax_type = 0;
    double filterA_tol = 0.e-4;
    int coarsening = 6;
    int agg_coarsening = 0; // Number of levels of aggressive coarsening
    int printlevel = 0;
};

/** Provides the time-dependent RHS of the ODEs after spatially discretizing the 
    PDE,
        du/dt = N(u,t) == - A(u) + D*u + s(t).
    where:
        A: The advection discretization,
        D: The diffusion discretization,
        s: The solution-independent source term discretization. */
class AdvDif : public IRKOperator
{
private:    

    int dim;                    // Spatial dimension
    FDMesh &Mesh;               // Mesh on which the PDE is discretized
    
    Vector alpha;               // Advection coefficients
    Vector mu;                  // Diffusion coefficients
    FDNonlinearOp * A;          // Nonlinear advection operator
    FDLinearOp * D;             // Diffusion operator
        
    mutable Vector source, temp;// Solution independent source term & auxillary vector

    // Preconditioners for systems of the form B = gamma*I-dt*Gradient
    // Parameters used to assemble B and its preconditioner
    struct Prec_params {
        double gamma = 0.;
        double dt = 0.;
        bool JacUpToDate = false;   // Have member Gradients been updated since preconditioner was built?
    };
    
    mutable Array<struct Prec_params> B_params;       
    Array<HypreBoomerAMG *> B_prec; // 
    Array<HypreParMatrix *> B_mats; // Seem's like it's only safe to free matrix once preconditioner is finished with it, so need to save these...
    map<int, int> B_hash;           // Hash table for mapping index passed with SetPreconditioner() or ImplicitPrec() to correct entries in array above
    int B_index;                    // Index that was set with previous call to SetPreconditioner()
    AMG_params AMG;                 // AMG solver params
    
    // The gradient operator(s), dN/du
    Array<HypreParMatrix *> Gradients; // Stage-dependent Jacobians
    HypreParMatrix * Gradient;         // Stage-independent Jacobian
    

    HypreParMatrix * identity;      // Compatible identity matrix
    
    /// Set solution independent source term at current time.
    inline void SetSource() const { Mesh.EvalFunction(&Source, GetTime(), source); };
    
    /// Compute J == N'(x)
    void GetGradient(const Vector &x, HypreParMatrix * &J) const;
    
public:
    
    AdvDif(IRKOperator::ExplicitGradients gradientsType, 
            FDMesh &Mesh_, int fluxID, Vector alpha_, Vector mu_, 
            int order, FDBias advection_bias);
    ~AdvDif();
    
    /// Allocate memory to and assign initial condition.
    inline void GetU0(Vector * &u) const { Mesh.EvalFunction(&InitialCondition, u); };
    
    /// Get error w.r.t. exact PDE solution (if available)
    bool GetError(int save, const char * out, double t, const Vector &u, 
                    double &eL1, double &eL2, double &eLinf);
    
    /** Set solver parameters for implicit time-stepping.
        MUST be called before InitSolvers() */
    inline void SetAMGParams(AMG_params params) { AMG = params; };
                    
    /* ---------------------------------------------------------------------- */
    /* -------------------------- Virtual functions ------------------------- */
    /* ---------------------------------------------------------------------- */                

    /// Compute the right-hand side of the ODE system.
    void Mult(const Vector &u, Vector &du_dt) const;
    
    /// Compute the right-hand side of the ODE system. (Some applications require this operator too)
    void ImplicitMult(const Vector &u, Vector &du_dt) const { Mult(u, du_dt); };

    /// Precondition B*x=y <==> (\gamma*I - dt*L')*x=y 
    void ImplicitPrec(const Vector &x, Vector &y) const {
        MFEM_ASSERT(B_prec[B_hash.at(B_index)], 
            "AdvDif::ImplicitPrec() Must first set system! See SetPreconditioner()");
        B_prec[B_hash.at(B_index)]->Mult(x, y);
    }  
    
    /// Precondition B*x=y with given index
    void ImplicitPrec(int index, const Vector &x, Vector &y) const {
        MFEM_ASSERT(B_prec[B_hash.at(index)], 
            "AdvDif::ImplicitPrec() Must first set system! See SetPreconditioner()");
        B_prec[B_hash.at(index)]->Mult(x, y);
    } 
    
    /// Apply action of identity mass matrix, y = M*x. 
    void MassMult(const Vector &x, Vector &y) const { y = x; };             

    /* ---------------------------------------------------------------------- */
    /* ---------- Virtual functions for approximate Jacobians of N  --------- */
    /* ---------------------------------------------------------------------- */
    
    /** Set approximate gradient Na' which is an approximation to the s explicit 
        gradients 
            {N'} = {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
        Such that it is referenceable with ExplicitGradientMult() and 
        SetPreconditioner() 
        
        If not re-implemented, this method simply generates an error. */
    void SetExplicitGradient(const Vector &u, double dt, 
                             const BlockVector &x, const Vector &c);


    /// Compute y <- Gradient*x                         
    void ExplicitGradientMult(const Vector &x, Vector &y) const 
    {  
        MFEM_ASSERT(Gradient, "AdvDif::ExplicitGradientMult() Gradient not set");
        Gradient->Mult(x, y);
    }                         
    
    /** Assemble preconditioner for gamma*M - dt*L' that's applied by
        by calling: 
            1. ImplicitPrec(.,.) if no further calls to SetPreconditioner() are made
            2. ImplicitPrec(index,.,.) */
    void SetPreconditioner(int index, double dt, double gamma, int type);    
    
    
    /* ---------------------------------------------------------------------- */
    /* ------------- Virtual functions for true Jacobians of N  ------------- */
    /* ---------------------------------------------------------------------- */
    
    /** Set the explicit gradients 
            {N'} = {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
        Or some approximation to them */
    void SetExplicitGradients(const Vector &u, double dt, 
                              const BlockVector &x, const Vector &c);

    /// Compute action of `index`-th gradient operator above
    void ExplicitGradientMult(int index, const Vector &x, Vector &y) const { 
        MFEM_ASSERT(Gradients.Size() > 0, 
            "AdvDif::ExplicitGradientMult() Gradients not yet set!");
        MFEM_ASSERT(index < Gradients.Size(), 
            "AdvDif::ExplicitGradientMult() index exceeds length of Gradients array!");    
        Gradients[index]->Mult(x, y);
    }
    
    /** Assemble preconditioner for gamma*M - dt*<weights,{N'}> that's applied by
        by calling: 
            1. ImplicitPrec(.,.) if no further calls to SetPreconditioner() are made
            2. ImplicitPrec(index,.,.) */
    void SetPreconditioner(int index, double dt, double gamma, Vector weights);

};



/* NOTES:
 - Cannot simultaneously specify dt, nt, and tf. Values of dt and tf take precidense 
 over nt. By default, however, nt and tf are specified and dt is determined from 
 them accordingly. */
int main(int argc, char *argv[])
{
    // Initialize parallel
    int myid, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);


    /* ------------------------------------------------------ */
    /* --- Set default values for command-line parameters --- */
    /* ------------------------------------------------------ */
    /* PDE */
    example = 1; // example ID: responsible for generating source and exact solution
    FDBias advection_bias = CENTRAL;
    int advection_bias_temp = static_cast<int>(advection_bias);
    double ax=1.0, ay=1.0; // Advection coefficients
    double mx=1.0, my=1.0; // Diffusion coefficients
    dim = 1; // Spatial dimension
    
    /* --- Time integration --- */
    int nt     = 10;  // Number of time steps
    double dt  = 0.0; // Time step size. dt < 0 means: dt == |dt|*dx
    double tf  = 2.0; // Final integration time
    int RK_ID  = 1;   // RK scheme (see below)

    /* --- Spatial discretization --- */
    int order     = 2; // Approximation order
    int refLevels = 4; // nx == 2^refLevels in every dimension
    int npx       = -1; // # procs in x-direction
    int npy       = -1; // # procs in y-direction
    
    // Solver parameters
    AMG_params AMG;
    int use_AIR_temp = (int) AMG.use_AIR;
    KrylovParams KRYLOV;
    int krylov_solver = static_cast<int>(KRYLOV.solver);
    NewtonParams NEWTON;
    int gradientsType = 1; // APPROXIMATE or EXACT Jacobians. 
    int newton_jacs = static_cast<int>(NEWTON.jac_solver_sparsity);
    int newton_jacp = static_cast<int>(NEWTON.jac_prec_sparsity);
    
    // Output paramaters
    int save         = 0;       // Save solution vector
    const char * out = "data/U"; // Filename of data to be saved...
    
    OptionsParser args(argc, argv);
    args.AddOption(&fluxID, "-f", "--flux-function",
                  "0==Linear==u, 1==Nonlinear==u^2.");
    args.AddOption(&example, "-ex", "--example-problem",
                  "1 (and 2 for linear problem)."); 
    args.AddOption(&ax, "-ax", "--alpha-x",
                  "Advection in x-direction."); 
    args.AddOption(&ay, "-ay", "--alpha-y",
                  "Advection in y-direction."); 
    args.AddOption(&mx, "-mx", "--mu-x",
                  "Diffusion in x-direction."); 
    args.AddOption(&my, "-my", "--mu-y",
                  "Diffusion in y-direction."); 
    args.AddOption(&dim, "-d", "--dim",
                  "Spatial dimension.");              

     /* Time integration */
    args.AddOption(&RK_ID, "-t", "--RK-method",
                  "Time discretization.");
    args.AddOption(&nt, "-nt", "--num-time-steps", "Number of time steps.");    
    args.AddOption(&tf, "-tf", "--tf", "0 <= t <= tf");
    args.AddOption(&dt, "-dt", "--time-step-size", "t_j = j*dt. NOTE: dt<0 means dt==|dt|*dx");

    /* Spatial discretization */
    args.AddOption(&order, "-o", "--order",
                  "FD: Approximation order."); 
    args.AddOption(&advection_bias_temp, "-b", "--bias",
                  "Advection bias: 0==CENTRAL, 1==UPWIND"); 
    args.AddOption(&refLevels, "-l", "--level",
                  "FD: Mesh refinement; 2^refLevels DOFs in each dimension.");
    args.AddOption(&npx, "-px", "--nprocs-in-x",
                  "FD: Number of procs in x-direction.");
    args.AddOption(&npy, "-py", "--nprocs-in-y",
                  "FD: Number of procs in y-direction.");                          
     
    /// --- Solver parameters --- ///   
    /* AMG parameters */
    args.AddOption(&use_AIR_temp, "-air", "--AMG-use-air", "AMG: 0=standard (default), 1=AIR");
    args.AddOption(&AMG.printlevel, "-ap", "--AMG-print", "AMG: Print level");
    args.AddOption(&AMG.agg_coarsening, "-agg", "--AMG-aggressive-coarseing", "AMG: Levels of aggressive coarsening");
    
    /* Krylov parameters */
    args.AddOption(&krylov_solver, "-ksol", "--krylov-method", "KRYLOV: Method (see KrylovMethod)");
    args.AddOption(&KRYLOV.reltol, "-krtol", "--krylov-rel-tol", "KRYLOV: Relative stopping tolerance");
    args.AddOption(&KRYLOV.abstol, "-katol", "--krylov-abs-tol", "KRYLOV: Absolute stopping tolerance");
    args.AddOption(&KRYLOV.maxiter, "-kmaxit", "--krylov-max-iterations", "KRYLOV: Maximum iterations");
    args.AddOption(&KRYLOV.kdim, "-kdim", "--krylov-dimension", "KRYLOV: Maximum subspace dimension");
    args.AddOption(&KRYLOV.printlevel, "-kp", "--krylov-print", "KRYLOV: Print level");
    
    /* Newton parameters */
    args.AddOption(&gradientsType, "-jac", "--ODEs-jacobian", "ODEs Jacobian: 0=Approximate/Kronecker form, 1=Exact form (see IRKOperator::ExplicitGradients)");
    args.AddOption(&NEWTON.reltol, "-nrtol", "--newton-rel-tol", "NEWTON: Relative stopping tolerance");
    args.AddOption(&NEWTON.abstol, "-natol", "--newton-abs-tol", "NEWTON: Absolute stopping tolerance");
    args.AddOption(&NEWTON.maxiter, "-nmaxit", "--newton-max-iterations", "NEWTON: Maximum iterations");
    args.AddOption(&NEWTON.printlevel, "-np", "--newton-print", "NEWTON: Print level");
    args.AddOption(&NEWTON.jac_update_rate, "-jacu", "--newton-jac-update-rate", "NEWTON: Rate that Jacobian is updated (see NewtonParams)");
    args.AddOption(&newton_jacs, "-jacs", "--newton-jac-solver-sparsity", "NEWTON: Jacobian solver sparsity (see NewtonParams)");
    args.AddOption(&newton_jacp, "-jacp", "--newton-jac-prec-sparsity", "NEWTON: Jacobian preconditioner sparsity (see NewtonParams)");
    args.AddOption(&NEWTON.gamma_idx, "-gamma", "--newton-prec-constant", "NEWTON: Constant used to precondition Schur complement");
    
    /* --- Text output of solution etc --- */              
    args.AddOption(&out, "-out", "--out-directory", "Name of output file."); 
    args.AddOption(&save, "-save", "--save-sol-data",
                  "save>0 will save solution info, save>1 also saves solution (and exact solution, if implemented).");              
    args.Parse();
    if (myid == 0) {
        args.PrintOptions(std::cout); 
    }
    // Set final forms of remaing params
    advection_bias = static_cast<FDBias>(advection_bias_temp);
    AMG.use_AIR = (bool) use_AIR_temp;
    KRYLOV.solver = static_cast<KrylovMethod>(krylov_solver);
    NEWTON.jac_solver_sparsity = static_cast<JacSparsity>(newton_jacs);
    NEWTON.jac_prec_sparsity = static_cast<JacSparsity>(newton_jacp);
    std::vector<int> np = {};
    if (npx != -1) {
        if (dim >= 1) np.push_back(npx);
        if (dim >= 2) np.push_back(npy);
    }
    alpha.SetSize(dim);
    mu.SetSize(dim);
    alpha(0) = ax;
    mu(0) = mx;
    if (dim > 1) {
        alpha(1) = ay;
        mu(1) = my;
    }
    
    
    ////////////////////////////////////////
    // Assemble mesh on which we discretize.
    ////////////////////////////////////////
    FDMesh Mesh(MPI_COMM_WORLD, dim, refLevels, np);
    double dx = Mesh.Get_dx();
    int nx = Mesh.Get_nx();
    
    /////////////////////////////////
    // Set up spatial discretization.
    /////////////////////////////////
    //for (int i = 0; i < mu.Size(); i++) mu(i) *= dx;
    AdvDif SpaceDisc(static_cast<IRKOperator::ExplicitGradients>(gradientsType), 
                     Mesh, fluxID, alpha, mu, order, advection_bias);
    SpaceDisc.SetAMGParams(AMG);
        
    // Get initial condition
    Vector * u = NULL;
    SpaceDisc.GetU0(u);
    
    
    //////////////
    // Time-march. 
    //////////////
    // dt < 0 means dt == |dt|*dx
    if (dt < 0.0) {
        dt *= -dx;
        nt = ceil(tf/dt);
    // dt not set, choose it according to nt and tf
    } else if (dt == 0.0) {
        dt = tf / (nt - 0);    
    // Adjust nt so that we (at least) step to tf    
    } else {
        nt = ceil(tf/dt);
    }
    
    
    // Build IRK object using spatial discretization
    RKData ButcherTableau(static_cast<RKData::Type>(RK_ID)); 
    IRK MyIRK(&SpaceDisc, ButcherTableau);        

    // Initialize IRK time-stepping solver
    MyIRK.Init(SpaceDisc);
    
    // Set solver settings 
    MyIRK.SetKrylovParams(KRYLOV);
    MyIRK.SetNewtonParams(NEWTON);
    
    
    // Time step 
    double t = 0.0;
    MyIRK.Run(*u, t, dt, tf);
    
    if (fabs(t-tf)>1e-14) {
        if (myid == 0) std::cout << "WARNING: Requested tf of " << tf << " adjusted to " << t << '\n';
    }
    tf = t; // Update final time


    /* ----------------------- */
    /* --- Save solve data --- */
    /* ----------------------- */
    
    // std::cout << "save = " << save << '\n';
    // std::cout << "out = " << out << '\n';
    // 
    // out = out + to_string(RK_ID);
    // 
    // std::cout << "out = " << out << '\n';
    
    if (save > 0) {
        if (save > 1) {
            ostringstream outt;
            outt << out << "." << myid;
            ofstream sol;
            sol.open(outt.str().c_str());
            u->Print_HYPRE(sol);
            sol.close();
        }
        
        /* Get error against exact PDE solution if available */
        double eL1, eL2, eLinf = 0.0; 
        bool got_error = SpaceDisc.GetError(save, out, tf, *u, eL1, eL2, eLinf);
        
        // Save data to file enabling easier inspection of solution            
        if (myid == 0) {
            OutDict solinfo;
            solinfo.open(out);
            solinfo << scientific;
            
            /* Temporal info */
            solinfo.Print("RK", RK_ID);
            solinfo.Print("dt", dt);
            solinfo.Print("nt", nt);
            solinfo.Print("tf", tf);
            
            /* Spatial info */
            solinfo.Print("dx", dx);
            solinfo.Print("nx", nx);
            solinfo.Print("space_dim", dim);
            solinfo.Print("space_refine", refLevels);
            solinfo.Print("exampleID", example); 
            
            /* Nonlinear and linear system/solve statistics */
            int avg_newton_iter;
            std::vector<int> avg_krylov_iter;
            std::vector<int> system_size;
            std::vector<double> eig_ratio;
            MyIRK.GetSolveStats(avg_newton_iter, avg_krylov_iter, system_size, eig_ratio);
            solinfo.Print("newton_gradients_type", gradientsType);
            solinfo.Print("newton_jac_update_rate", NEWTON.jac_update_rate);
            solinfo.Print("newton_jac_solver_sparsity", static_cast<int>(NEWTON.jac_solver_sparsity));
            solinfo.Print("newton_jac_prec_sparsity", static_cast<int>(NEWTON.jac_prec_sparsity));
            solinfo.Print("nrtol", NEWTON.reltol);
            solinfo.Print("natol", NEWTON.abstol);
            solinfo.Print("newton_iters", avg_newton_iter);
            solinfo.Print("gamma", NEWTON.gamma_idx);
            
            solinfo.Print("krtol", KRYLOV.reltol);
            solinfo.Print("katol", KRYLOV.abstol);
            solinfo.Print("kdim", KRYLOV.kdim);
            solinfo.Print("nsys", avg_krylov_iter.size());
            // Weighted total Krylov iterations: 1 for 1x1 systems, 2 for 2x2 systems
            int weighted_avg_krylov_iter_total = 0;
            for (int system = 0; system < avg_krylov_iter.size(); system++) {
                solinfo.Print("sys" + to_string(system+1) + "_iters", avg_krylov_iter[system]);
                solinfo.Print("sys" + to_string(system+1) + "_type", system_size[system]);
                solinfo.Print("sys" + to_string(system+1) + "_eig_ratio", eig_ratio[system]);
                if (system_size[system] == 1) {
                    weighted_avg_krylov_iter_total += avg_krylov_iter[system];
                } else if (system_size[system] == 2) {
                    weighted_avg_krylov_iter_total += 2*avg_krylov_iter[system];
                }    
            }
            solinfo.Print("total_weighted_iters", weighted_avg_krylov_iter_total);
            std::cout << "total_weighted_iters = " << weighted_avg_krylov_iter_total << '\n';
            
            /* Parallel info */
            solinfo.Print("P", numProcess);
            for (int d = 0; d < np.size(); d++) {
                solinfo.Print("np" + to_string(d), np[d]);
            }
            
            /* Error statistics */
            if (got_error) {
                solinfo.Print("eL1", eL1);
                solinfo.Print("eL2", eL2);
                solinfo.Print("eLinf", eLinf);
            }
            
            solinfo.close();
        }
    }
    delete u;
    MPI_Finalize();
    return 0;
}



/// Initial condition of PDE 
double InitialCondition(const Vector &x) {
    switch (x.Size()) {
        case 1:
            return pow(sin(PI/2.0*(x(0)-1.0)), 4.0);
        case 2:
            return pow(sin(PI/2.0*(x(0)-1.0)), 4.0) * pow(sin(PI/2.0*(x(1)-1.0)), 4.0);
        default:
            return 0.0;
    }
}


/// Solution-independent source term in PDE
double Source(const Vector &x, double t) {
    switch (example) {
        // Source is chosen for manufactured solution
        case 1:
            switch (fluxID) {
                // Linear advection flux
                case 0:
                    switch (x.Size()) {
                        case 1: 
                        {
                            double u = PI*(x(0)-alpha(0)*t);
                            return mu(0)/8.*exp(-mu(0)*t)*(
                                    -3.+4.*(PI*PI-1.)*cos(u)
                                    +(4.*PI*PI-1.)*cos(2*u));
                        }
                        case 2:
                        {
                            double u = 0.5*PI*(x(0)-alpha(0)*t-1.);
                            double v = 0.5*PI*(x(1)-alpha(1)*t-1.);
                            double z = (mu(0)+mu(1))*t;
                            return -0.5*exp(-z)*pow(sin(u)*sin(v), 2.)
                                    *(  mu(0)*(1.+2.*PI*PI+(4.*PI*PI-1.)*cos(2.*u))*pow(sin(v),2.)
                                      + mu(1)*(1.+2.*PI*PI+(4.*PI*PI-1.)*cos(2.*v))*pow(sin(u),2.) );
                        }
                        default:
                            return 0.0;
                    }
                // Nonlinear advection flux
                case 1:
                    switch (x.Size()) {
                        case 1: 
                        {
                            double u = 0.5*PI*(x(0)-alpha(0)*t);
                            double z = mu(0)*t;
                            return 0.5*exp(-2.*z)*(
                                    -4.*PI*alpha(0)*pow(cos(u),6.)*sin(2.*u)
                                    + exp(z)*pow(cos(u),2.)
                                    * (-mu(0)-2.*PI*PI*mu(0)+(4.*PI*PI-1.)*mu(0)*cos(2.*u) 
                                        + 2.*PI*alpha(0)*sin(2.*u)));
                        }
                        case 2:
                        {
                            double u = 0.5*PI*(x(0)-alpha(0)*t-1.);
                            double v = 0.5*PI*(x(1)-alpha(1)*t-1.);
                            double z = (mu(0)+mu(1))*t;
                            return -0.5*exp(-2.*z)*pow(sin(u)*sin(v), 2.)
                                    *(  2.*PI*alpha(0)*sin(2.*u)*pow(sin(v), 2.)*( exp(z)-2.*pow(sin(u)*sin(v), 4.) )
                                      + 2.*PI*alpha(1)*sin(2.*v)*pow(sin(u), 2.)*( exp(z)-2.*pow(sin(u)*sin(v), 4.) )
                                    + exp(z) 
                                    *(  mu(0)*(1.+2.*PI*PI+(4.*PI*PI-1.)*cos(2.*u))*pow(sin(v),2.)
                                      + mu(1)*(1.+2.*PI*PI+(4.*PI*PI-1.)*cos(2.*v))*pow(sin(u),2.) ));
                        }
                        default:
                            return 0.0;
                    }
            }
        // Default source is 0
        default:
            return 0.0;
    }
}


/// Manufactured PDE solution 
double PDESolution(const Vector &x, double t)
{    
    switch (example) {
        // Test problem where the initial condition is propagated with wave-speed alpha and dissipated with diffusivity mu
        case 1:
            switch (x.Size()) {
                case 1:
                    return pow(sin(PI/2.*(x(0)-1-alpha(0)*t)), 4.) * exp(-mu(0)*t);
                case 2:
                    return pow(sin(PI/2.*(x(0)-1-alpha(0)*t)) * sin(PI/2.*(x(1)-1-alpha(1)*t)), 4.) 
                            * exp(-(mu(0)+mu(1))*t);
                default:
                    return 0.0;
            }
            break;
        // 2nd test problem for linear-advection-diffusion that's more realistic (no forcing).
        case 2:
            if (fluxID == 1) cout <<  "PDESolution not implemented for NONLINEAR example " << example << "\n";
            switch (x.Size()) {
                case 1:
                {   
                    if (fabs(mu(0)) < 1e-15) mfem_error("PDESolution:: Solution only holds for non-zero diffusivity");
                    double u = PI*(x(0)-alpha(0)*t);
                    return 3./8. 
                            + 1./2.*cos(u)*exp(-mu(0)*PI*PI*t) 
                            + 1./8.*cos(2.*u)*exp(-mu(0)*4.*PI*PI*t);
                }
                case 2:
                {   
                    if (fabs(mu(0)) < 1e-15 || fabs(mu(1)) < 1e-15) mfem_error("PDESolution:: Solution only holds for non-zero diffusivity");
                    double u = PI*(x(0)-alpha(0)*t);
                    double v = PI*(x(1)-alpha(1)*t);
                    double p2t = PI*PI*t;
                    double m0 = mu(0);
                    double m1 = mu(1);
                    return 9./64. + 3./16.*cos(u)*exp(-m0*p2t) + 3./64.*cos(2*u)*exp(-4.*m0*p2t)
                            + cos(v)*(3./16.*exp(-m1*p2t) + 1./4.*cos(u)*exp(-(m0+m1)*p2t) + 1./16.*cos(2*u)*exp(-(4.*m0+m1)*p2t))
                            + cos(2.*v)*(3./64.*exp(-4.*m1*p2t) + 1./16.*cos(u)*exp(-(m0+4.*m1)*p2t) + 1./64.*cos(2*u)*exp(-4.*(m0+m1)*p2t));
                }
                default:
                    return 0.0;
            }
        default:
            cout <<  "PDESolution:: not implemented for example " << example << "\n";
            return 0.0;
    }
}


/// f(u) = u; fluxID == 0 
double LinearFlux(double u) { return u; }; 
double GradientLinearFlux(double u) { return 1.0; };
/// f(u) = u^2; fluxID == 1 
double NonlinearFlux(double u) { return u*u; }; 
double GradientNonlinearFlux(double u) { return 2.*u; };
/// Return pointer to flux function
typedef double (*ScalarFun)(double);
ScalarFun Flux(int fluxID) {
    switch (fluxID) {
        case 0:
            return LinearFlux;
        case 1:
            return NonlinearFlux;
        default:
            return NULL;
    }
}
/// Return pointer to gradient of flux function
ScalarFun GradientFlux(int fluxID) {
    switch (fluxID) {
        case 0:
            return GradientLinearFlux;
        case 1:
            return GradientNonlinearFlux;
        default:
            return NULL;
    }
}


AdvDif::AdvDif(IRKOperator::ExplicitGradients gradientsType, 
        FDMesh &Mesh_, int fluxID, Vector alpha_, Vector mu_, 
        int order, FDBias advection_bias) 
    : IRKOperator(Mesh_.GetComm(), Mesh_.GetLocalSize(), 0.0, 
        TimeDependentOperator::Type::IMPLICIT,
        //TimeDependentOperator::Type::EXPLICIT,
        gradientsType),
        Mesh{Mesh_},
        alpha(alpha_), mu(mu_),
        dim(Mesh_.m_dim),
        D(NULL), A(NULL),
        B_params(), B_prec(), B_mats(), B_index{0}, AMG(), 
        Gradients(), Gradient(NULL), 
        identity(NULL),
        source(Mesh_.GetLocalSize()), temp(Mesh_.GetLocalSize())
{
    // Assemble diffusion operator if non-zero
    if (mu.Normlp(infinity()) > 1e-15) {
        D = new FDLinearOp(Mesh, 2, mu, order, CENTRAL);
        D->Assemble();
    }
    
    // Assemble advection operator if non-zero
    if (alpha.Normlp(infinity()) > 1e-15) {
        A = new FDNonlinearOp(Mesh, 1, alpha, Flux(fluxID), GradientFlux(fluxID), 
                                order, advection_bias);
    }
    
    // std::cout << "alpha/dx = " << alpha(0)/Mesh.Get_dx() << '\n';
    // std::cout << "mu/dx^2 = " << mu(0)/(Mesh.Get_dx()*Mesh.Get_dx()) << '\n';
    
    
    if (!A && !D) mfem_error("AdvDif::AdvDif() Require at least one non-zero PDE coefficient.");
};


/// Evaluate RHS of ODEs: du_dt = -A(u) + D*u + s(t) 
void AdvDif::Mult(const Vector &u, Vector &du_dt) const
{
    if (D && A) {
        D->Mult(u, du_dt);
        A->Mult(u, source); // Use s as auxillary vector, is reset below anyways
        du_dt.Add(-1., source);
    } else if (D) {
        D->Mult(u, du_dt);
    } else {
        A->Mult(u, du_dt);
        du_dt.Neg();
    }
    // Solution-independent source term
    SetSource();
    du_dt += source;
}


/// Gradient of N(u, t) w.r.t u evaluated at x, dN/du(x) = -dA/du(x) + D 
void AdvDif::GetGradient(const Vector &x, HypreParMatrix * &J) const {
    if (J) delete J;
    
    if (A && D) {
        J = Add(-1., A->GetGradient(x), 1., D->Get()); 
    } else if (A) {
        J = &(A->GetGradient(x));
        *J *= -1.;
    } else if (D) {
        J = &(D->Get());
    }
}


// /** Gradient of L(u, t) w.r.t u evaluated at x, dL/du(x) = -dA/du(x) + D */
// // TODO:  deal with memory leaks here... who owns the jacobian, and who owns
// // the components that are added
// HypreParMatrix &AdvDif::GetExplicitGradient(const Vector &x) const {
//     if (Jacobian) delete Jacobian;
//     Jacobian = GetGradient2(x);
//     return *Jacobian;
// }

/** Set approximate gradient Na' which is an approximation to the s explicit 
    gradients 
        {N'} = {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
    
    Stored in the member variable "Gradient".
*/    
void AdvDif::SetExplicitGradient(const Vector &u, double dt, 
                                 const BlockVector &x, const Vector &c) 
{
    // Current member preconditioner no longer uses current Jacobians
    for (int i = 0; i < B_params.Size(); i++) B_params[i].JacUpToDate = false;
    
    // temp <- u + dt*x[s]
    add(u, dt, x.GetBlock(c.Size()-1), temp);
    
    if (Gradient) delete Gradient; Gradient = NULL;
                                    
    GetGradient(temp, Gradient);                
}
                                 
        
/** Set the explicit gradients 
        {N'} = {N'(u + dt*x[i], this->GetTime() + dt*c[i])}, i=0,...,s-1.
    Or some approximation to them.
    
    These are stored in the member variable "Gradients" */
void AdvDif::SetExplicitGradients(const Vector &u, double dt, 
                                  const BlockVector &x, const Vector &c) {
    
    int s = c.Size();
    
    // Free existing gradient operators
    if (Gradients.Size() > 0) {
        for (int i = 0; i < s; i++) {
            if (Gradients[i]) delete Gradients[i];
        }
    // Size array for the first time    
    } else {
        Gradients.SetSize(s);
    }
    
    // Set to NULL as required by some functions
    for (int i = 0; i < s; i++) {
        Gradients[i] = NULL;
    }
    
    // Current member preconditioner no longer uses current Jacobians
    for (int i = 0; i < B_params.Size(); i++) B_params[i].JacUpToDate = false;
    
    double t0 = this->GetTime();
    
    for (int i = 0; i < s; i++) {
        this->SetTime(t0 + dt*c(i));
        add(u, dt, x.GetBlock(i), temp); // temp <- u + dt*x(i)
        GetGradient(temp, Gradients[i]);
    }
    
    // Reset self to my original time
    this->SetTime(t0);
}                                 
                                 



// /// Gradient of N(u, t) w.r.t u evaluated at x, dL/du(x) = -dA/du(x) + D 
// HypreParMatrix * AdvDif::GetGradient(const Vector &x) const {
//     HypreParMatrix * J;
// 
//     if (A && D) {
//         J = Add(-1., A->GetGradient(x), 1., D->Get()); 
//     } else if (A) {
//         J = &(A->GetGradient(x));
//         *J *= -1.;
//     } else if (D) {
//         J = &(D->Get());
//     }
//     return J;
// }


/** Assemble preconditioner for gamma*M - dt*L' that's applied by
    by calling: 
        1. ImplicitPrec(.,.) if no further calls to SetPreconditioner() are made
        2. ImplicitPrec(index,.,.) */
void AdvDif::SetPreconditioner(int index, double dt, double gamma, int type) {
    MFEM_ASSERT(Gradient, "AdvDif::SetPreconditioner() Gradient not yet set!");

    // Update B_index
    B_index = index;

    // Preconditioner previously created with this index. Free it.
    if (B_hash.count(index) > 0) {
    
        // Only remove if: dt, gamma have changed, or if Gradient has been updated
        if (B_params[B_hash[index]].dt != dt || 
            B_params[B_hash[index]].gamma != gamma ||
            !B_params[B_hash[index]].JacUpToDate) { 
            delete B_prec[B_hash[index]];
            B_prec[B_hash[index]] = NULL;
            delete B_mats[B_hash[index]];
            B_mats[B_hash[index]] = NULL;
        }    
        
    /*  No preconditioner previously built with this index. Create space for a 
        new one. */
    } else {
        B_prec.Append(NULL);
        B_mats.Append(NULL);
        B_hash[index] = B_prec.Size()-1;
        B_params.Append(Prec_params()); 
    }    
    
    // Build a new preconditioner 
    if (!B_prec[B_hash[index]]) {
        
        // Update parameters struct
        B_params[B_hash[index]].dt = dt;
        B_params[B_hash[index]].gamma = gamma;
        B_params[B_hash[index]].JacUpToDate = true;
        
        // Assemble identity matrix 
        if (!identity) identity = (A) ? A->GetHypreParIdentityMatrix() : D->GetHypreParIdentityMatrix();
        
        // B = gamma*I - dt*Gradient
        HypreParMatrix * B = Add(-dt, *Gradient, gamma, *identity); 
        
        /* Build AMG preconditioner for B */
        HypreBoomerAMG * amg_solver = new HypreBoomerAMG(*B);
        
        amg_solver->SetMaxIter(1); 
        amg_solver->SetTol(0.0);
        amg_solver->SetMaxLevels(50); 
        amg_solver->SetPrintLevel(AMG.printlevel); 
        amg_solver->iterative_mode = false;
        if (AMG.use_AIR) {                        
            amg_solver->SetAdvectiveOptions(AMG.distance, AMG.prerelax, AMG.postrelax);
            amg_solver->SetStrongThresholdR(AMG.strength_tolR);
            amg_solver->SetFilterThresholdR(AMG.filter_tolR);
            amg_solver->SetStrengthThresh(AMG.strength_tolC);
            amg_solver->SetInterpolation(AMG.interp_type);
            amg_solver->SetRelaxType(AMG.relax_type);
            amg_solver->SetCoarsening(AMG.coarsening);
            // TODO - this option not there
            // amg_solver->            AMG.filterA_tol,                                      
        } else {
            amg_solver->SetInterpolation(0);
            amg_solver->SetCoarsening(AMG.coarsening);
            amg_solver->SetAggressiveCoarsening(AMG.agg_coarsening); 
        }  
        
        // Store pointers for AMG solver and matrix in Arrays
        B_prec[B_hash[index]] = amg_solver;
        B_mats[B_hash[index]] = B;
    }
}


/** Assemble preconditioner for (\gamma*I - dt*{weights}*{N'}) 
    and set B_index = index */
void AdvDif::SetPreconditioner(int index, double dt, double gamma, Vector weights) {
    MFEM_ASSERT(Gradients.Size() > 0, 
        "AdvDif::SetPreconditioner() Gradients not yet set!");

    // Update B_index
    B_index = index;

    // Preconditioner previously created with this index. Free it.
    if (B_hash.count(index) > 0) {
        
        // Only remove if: dt, gamma have changed, or if Gradients have been updated
        if (B_params[B_hash[index]].dt != dt || 
            B_params[B_hash[index]].gamma != gamma ||
            !B_params[B_hash[index]].JacUpToDate) { 
            delete B_prec[B_hash[index]];
            B_prec[B_hash[index]] = NULL;
            delete B_mats[B_hash[index]];
            B_mats[B_hash[index]] = NULL;
            //std::cout << "rebuilding..." << '\n';
        }
        
    /*  No preconditioner previously built with this index. Create space for a 
        new one. */
    } else {
        
        //std::cout << "first building..." << '\n';
        B_prec.Append(NULL);
        B_mats.Append(NULL);
        B_hash[index] = B_prec.Size()-1;
        B_params.Append(Prec_params()); 
    }   
    
    
    // Build a new preconditioner 
    if (!B_prec[B_hash[index]]) {
        // Update parameters struct
        B_params[B_hash[index]].dt = dt;
        B_params[B_hash[index]].gamma = gamma;
        B_params[B_hash[index]].JacUpToDate = true;
            
        // Assemble identity matrix 
        if (!identity) identity = (A) ? A->GetHypreParIdentityMatrix() : D->GetHypreParIdentityMatrix();

        // B = gamma*I - dt*{weights}*{Gradients}
        HypreParMatrix * B = NULL;
        for (int i = 0; i < weights.Size(); i++) {
            if (fabs(dt*weights(i)) > 0.) {
                if (B) {
                    B->Add(-dt*weights(i), *Gradients[i]);
                } else {
                    B = new HypreParMatrix(*Gradients[i]);
                    *B *= -dt*weights(i);
                }
            }
        }
        if (B) {
            B->Add(gamma, *identity);
        } else {
            B = new HypreParMatrix(*identity);
            *B *= gamma;
        }
        

        /* Build AMG preconditioner for B */
        HypreBoomerAMG * amg_solver = new HypreBoomerAMG(*B);

        amg_solver->SetMaxIter(1); 
        amg_solver->SetTol(0.0);
        amg_solver->SetMaxLevels(50); 
        amg_solver->SetPrintLevel(AMG.printlevel); 
        amg_solver->iterative_mode = false;
        if (AMG.use_AIR) {                        
            amg_solver->SetAdvectiveOptions(AMG.distance, AMG.prerelax, AMG.postrelax);
            amg_solver->SetStrongThresholdR(AMG.strength_tolR);
            amg_solver->SetFilterThresholdR(AMG.filter_tolR);
            amg_solver->SetStrengthThresh(AMG.strength_tolC);
            amg_solver->SetInterpolation(AMG.interp_type);
            amg_solver->SetRelaxType(AMG.relax_type);
            amg_solver->SetCoarsening(AMG.coarsening);
            // TODO - this option not there
            // amg_solver->            AMG.filterA_tol,                                    
        } else {
            amg_solver->SetInterpolation(0);
            amg_solver->SetCoarsening(AMG.coarsening);
            amg_solver->SetAggressiveCoarsening(AMG.agg_coarsening); 
        }  

        // Store pointers for AMG solver and matrix in Arrays
        B_prec[B_hash[index]] = amg_solver;
        B_mats[B_hash[index]] = B;
    }    
}



/* Get error against exact PDE solution if available. Also output if num solution is output */
bool AdvDif::GetError(int save, const char * out, double t, const Vector &u, double &eL1, double &eL2, double &eLinf) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    bool soln_implemented = false;
    Vector * u_exact = NULL;
    if (example == 1 || (example == 2 && fluxID == 0)) {
        if (myid == 0) std::cout << "PDE solution IS implemented for this example." << '\n';
        soln_implemented = true;
        Mesh.EvalFunction(&PDESolution, t, u_exact); 
    } else {
        if (myid == 0) std::cout << "PDE solution is NOT implemented for this example." << '\n';
    }
    
    if (soln_implemented) {
        if (save > 1) {
            ostringstream outt;
            outt << out << "_exact" << "." << myid;
            ofstream sol;
            sol.open(outt.str().c_str());
            u_exact->Print_HYPRE(sol);
            sol.close();
        }
        
        *u_exact -= u; // Error vector
        eL1   = u_exact->Normlp(1);
        eL2   = u_exact->Normlp(2);
        eLinf = u_exact->Normlp(infinity());
        delete u_exact;
        
        // Get global norms from on process norms computed above
        eL1   = GlobalLpNorm(1, eL1, MPI_COMM_WORLD);
        eL2   = GlobalLpNorm(2, eL2, MPI_COMM_WORLD);
        eLinf = GlobalLpNorm(infinity(), eLinf, MPI_COMM_WORLD);
        
        // Scale norms by mesh size
        double dx = Mesh.Get_dx(0);
        eL1 *= dx;
        eL2 *= sqrt(dx);
        int dim = Mesh.Get_dim();
        if (dim > 1) {
            double dy = Mesh.Get_dx(1);
            eL1 *= dy;
            eL2 *= sqrt(dy);
        }
        if (myid == 0) {
            std::cout << "Discrete error measured at final time:" << '\n';
            std::cout << "\teL1=" << eL1 << "\n\teL2=" << eL2 << "\n\teLinf=" << eLinf << '\n';    
        }
    }
    return soln_implemented;
}

AdvDif::~AdvDif()
{
    /* TODO: WHY can't I free amg_solver... What's odd is that I can delete these
    solvers elsewhere in the code...
        *** The MPI_Comm_free() function was called after MPI_FINALIZE was invoked.
        *** This is disallowed by the MPI standard.
    */
    
    
    //for (int i = 0; i < B_prec.Size(); i++) delete B_prec[i];
    
    // I can do this one, but not the one above... what gives?
    for (int i = 0; i < B_mats.Size(); i++) delete B_mats[i]; 
    
    if (identity) delete identity;
    if (D) delete D;
    if (A) delete A;
}