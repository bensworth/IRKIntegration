#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "FDSpaceDisc.hpp"
#include "IRK.hpp"

using namespace mfem;
using namespace std;

#define PI 3.14159265358979323846


/*  TODO: 
    -Segfault associated w/ freeing HypreBoomerAMG... Currently not being
        deleted... See associated `TODO` comment...
    -Scalability of IRK solver in terms of number of AMG iterations seems quite sensitive
        to the effecicency of the AMG preconditioner. E.g., if move from 0 to 1
        level of aggressive coarseing, I lose scalability  in terms of number of AMG iterations 
        as I refine the space-time mesh. Because HYPRE doesn't store statistics about the solver
        it's a bit hard to see what's happening, but going from 0 to 1 level of agressive coarsening
        decreases the grid complexity from ~1.7 to ~1.25 (and these numbers seem like they're roughly
        constant as the space-time mesh is refined). So I'm not sure if such a large grid complexity
        is acceptable. Probably should set up the AMG preconditioner to do a bit better job, but not 
        too sure what the right settings are.
        
        E.g., just doing 1 time step, 0 and 1 levels of aggressive coarsening 
        gives ~16 and ~40 GMRES iters and grid complexities of ~1.7 and ~1.2
            mpirun -np 4 ./driver_adv_dif_FD -d 2 -ex 1 -ax 0.85 -ay 1 -mx 0.3 -my 0.25 -o 4 -dt -2 -t 14 -save 2 -tf 0.00001 -krtol 1e-13 -katol 1e-13 -kp 2 -ap 2 -l 8 -ag 0
            mpirun -np 4 ./driver_adv_dif_FD -d 2 -ex 1 -ax 0.85 -ay 1 -mx 0.3 -my 0.25 -o 4 -dt -2 -t 14 -save 2 -tf 0.00001 -krtol 1e-13 -katol 1e-13 -kp 2 -ap 2 -l 8 -ag 1
            
        And, for example, taking a time step of dt=10*dx (-dt -10) rather than dt=2*dx (-dt -2) 
        as in the above example, the grid complexities of the two solvers remain the same as 
        above, yet 0 levels of aggressive coarsening only requires ~17 GMRES iters, 
        while 1 level of aggressive coarsening takes ~137 GMRES iters.
*/


// Sample runs:
//      *** Solve 2D in space problem w/ 4th-order discretizations in space & time ***
//      >> mpirun -np 4 ./driver_adv_dif_FD -d 2 -ex 1 -ax 0.85 -ay 1 -mx 0.3 -my 0.25 -l 5 -o 4 -dt -2 -t 14 -save 2 -tf 2 -krtol 1e-13 -katol 1e-13 -kp 2 -gamma 1
//      
//      ** Diffusion-only problem using GMRES solver
//      >> mpirun -np 4 ./driver_adv_dif_FD -d 2 -ex 1 -ax 0 -ay 0 -mx 0.3 -my 0.25 -l 5 -o 4 -dt -2 -t 14 -save 2 -tf 2 -krtol 1e-13 -katol 1e-13 -kp 2 -ksol 2 -gamma 1
//      ** Diffusion-only problem using CG solver
//      >> mpirun -np 4 ./driver_adv_dif_FD -d 2 -ex 1 -ax 0 -ay 0 -mx 0.3 -my 0.25 -l 5 -o 4 -dt -2 -t 14 -save 2 -tf 2 -krtol 1e-13 -katol 1e-13 -kp 2 -ksol 0 -gamma 1
// 
//
// Solve scalar linear advection-diffusion equation,
//     u_t + alpha.grad(u) = div( mu \odot grad(u) ) + s(x,t),
// for constant vectors alpha, and mu.
// 
// An initial condition is specified, and periodic spatial boundaries are used.
// 
// Implemented in 1 and 2 spatial dimensions.               
// 
// Test problems with manufactured solutions are implemented to facillitate 
// convergence studies.                                     [pass `-ex 1` OR `-ex 2` to report error]
//
// TIME DISCRETIZATION: Fully implicit RK.
//
// SPATIAL DISCRETIZATION: Arbitrarily high-order FD is available. 


/* Output dictionary entries via an ofstream */
class OutDict : public std::ofstream {
public:
    template <class T>
    inline void Print(string id, T entry) {*this << id << " " << entry << "\n";};
};

// Problem parameters global functions depend on
Vector alpha, mu;
int dim, example;

/* Functions definining some parts of PDE */
double InitialCondition(const Vector &x);
double Source(const Vector &x, double t);
double PDESolution(const Vector &x, double t);


struct AMGParams {    
    // AIR parameters
    bool use_AIR = false; 
    double distance = 1.5;
    string prerelax = "";
    string postrelax = "FFC";
    double strength_tolC = 0.1;
    double strength_tolR = 0.01;
    double filter_tolR = 0.0;
    double filterA_tol = 0.e-4;
    // Maybe these interp. and relax are better for AIR than those below for standard AMG?
    //int interp_type = 100; 
    //int relax_type = 0;
    
    // AMG coarsening options:
    int coarsen_type = 6;   // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
    int agg_levels   = 0;    // number of aggressive coarsening levels
    double theta     = 0.25; // strength threshold: 0.25, 0.5, 0.8

    // AMG interpolation options:
    int interp_type  = 6;    // 6 = extended+i, 0 = classical

    // AMG relaxation options:
    int relax_type   = 8;    // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi

    // Additional options:
    int printlevel  = 0;    
};


/** Provides the time-dependent RHS of the ODEs after spatially discretizing the 
    PDE,
        du/dt = L*u + s(t) == - A*u + D*u + s(t).
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
    FDLinearOp * A;             // Advection operator
    FDLinearOp * D;             // Diffusion operator
    
    mutable Vector source, temp;// Solution independent source term & auxillary vector

    // Preconditioners for systems of the form B = gamma*I-dt*L
    // Parameters used to assemble B and its preconditioner
    struct Prec_params {
        double gamma = 0.;
        double dt = 0.;
    };
    Array<struct Prec_params> B_params;
    Array<HypreBoomerAMG *> B_prec; // 
    Array<HypreParMatrix *> B_mats; // Seem's like it's only safe to free matrix once preconditioner is finished with it, so need to save these...
    AMGParams AMG;                  // AMG solver params
    int B_index;
    
    // Compatible identity matrix
    HypreParMatrix * identity;
    
    // Jacobian == L
    mutable HypreParMatrix * Jacobian;
    
    /// Set solution independent source term at current time.
    inline void SetSource() const { Mesh.EvalFunction(&Source, GetTime(), source); };
    
    /// Gradient of L(u, t) w.r.t u evaluated at x
    HypreParMatrix &SetJacobian() const;
    
public:
    
    AdvDif(FDMesh &Mesh_, Vector alpha_, Vector mu_, 
            int order, FDBias advection_bias);
    ~AdvDif();
    
    /// Allocate memory to and assign initial condition.
    inline void GetU0(Vector * &u) const { Mesh.EvalFunction(&InitialCondition, u); };
    
    /// Compute the right-hand side of the ODE system.
    void Mult(const Vector &u, Vector &du_dt) const;
    
    /// Compute the right-hand side of the ODE system.
    void ApplyL(const Vector &u, Vector &du_dt) const;
    
    
    /// Get error w.r.t. exact PDE solution (if available)
    bool GetError(int save, const char * out, double t, const Vector &u, 
                    double &eL1, double &eL2, double &eLinf);
    
    /// Precondition B*x=y <==> (\gamma*I - dt*L)*x=y
    inline void ImplicitPrec(const Vector &x, Vector &y) const {
        MFEM_ASSERT(B_prec[B_index], 
            "AdvDif::ImplicitPrec() Must first set system! See SetSystem()");
        B_prec[B_index]->Mult(x, y);
    }
    
    /** Ensures that this->ImplicitPrec() preconditions (\gamma*M - dt*L) 
            + index -> index of system to solve, [0,s_eff)
            + dt    -> time step size
            + type  -> eigenvalue type, 1 = real, 2 = complex pair
         These additional parameters are to provide ways to track when
         (\gamma*M - dt*L) must be reconstructed or not to minimize setup. */
    void SetSystem(int index, double dt, double gamma, int type);
    
    /** Set solver parameters fot implicit time-stepping.
        MUST be called before InitSolvers() */
    inline void SetAMGParams(AMGParams params) { AMG = params; };
    
    
    /// Apply action of identity mass matrix, y = M*x.
    inline void ImplicitMult(const Vector &x, Vector &y) const { y = x; };
    
    /// Mass matrix solve for identity mass matrix, y = M^{-1}*x.
    inline void ApplyMInv(const Vector &x, Vector &y) const { y = x; };
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
    int mag_prec = 0; // Index of constant used in preconditioner (see IRK.hpp)

    /* --- Spatial discretization --- */
    int order     = 2; // Approximation order
    int refLevels = 4; // nx == 2^refLevels in every dimension
    int npx       = -1; // # procs in x-direction
    int npy       = -1; // # procs in y-direction
    
    // Solver parameters
    AMGParams AMG;
    int use_AIR_temp = (int) AMG.use_AIR;
    IRK::KrylovParams KRYLOV;
    int krylov_solver = static_cast<int>(KRYLOV.solver);
    
    // Output paramaters
    int save         = 0;  // Save solution vector
    const char * out = "data/U"; // Filename of data to be saved...
    
    OptionsParser args(argc, argv);            
    args.AddOption(&example, "-ex", "--example-problem",
                  "1 or 2."); 
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
    args.AddOption(&mag_prec, "-gamma", "--gamma-value",
                  "Value of gamma in preconditioner: 0, 1, 2.");              
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
    args.AddOption(&AMG.coarsen_type, "-ac", "--AMG-print", "AMG: Coarsen type");
    args.AddOption(&AMG.agg_levels, "-ag", "--AMG-aggressive-coarseing", "AMG: Levels of aggressive coarsening");
    args.AddOption(&AMG.theta, "-at", "--AMG-theta", "AMG: Strength threshold");
    args.AddOption(&AMG.interp_type, "-ai", "--AMG-interp", "AMG: Interpolation");
    args.AddOption(&AMG.relax_type, "-ar", "--AMG-relax", "AMG: Relaxation");
    args.AddOption(&AMG.printlevel, "-ap", "--AMG-print", "AMG: Print level");

    /* Krylov parameters */
    args.AddOption(&krylov_solver, "-ksol", "--krylov-method", "KRYLOV: Method (see IRK::KrylovMethod)");
    args.AddOption(&KRYLOV.reltol, "-krtol", "--krylov-rel-tol", "KRYLOV: Relative stopping tolerance");
    args.AddOption(&KRYLOV.abstol, "-katol", "--krylov-abs-tol", "KRYLOV: Absolute stopping tolerance");
    args.AddOption(&KRYLOV.maxiter, "-kmaxit", "--krylov-max-iterations", "KRYLOV: Maximum iterations");
    args.AddOption(&KRYLOV.kdim, "-kdim", "--krylov-dimension", "KRYLOV: Maximum subspace dimension");
    args.AddOption(&KRYLOV.printlevel, "-kp", "--krylov-print", "KRYLOV: Print level");

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
    KRYLOV.solver = static_cast<IRK::KrylovMethod>(krylov_solver);
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
    
    /////////////////////////////////
    // Set up spatial discretization.
    /////////////////////////////////
    AdvDif SpaceDisc(Mesh, alpha, mu, order, advection_bias);
    SpaceDisc.SetAMGParams(AMG);
        
    // Get initial condition
    Vector * u = NULL;
    SpaceDisc.GetU0(u);
    
    // Get mesh info
    double dx = Mesh.Get_dx();
    int    nx = Mesh.Get_nx();
    
    
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
    IRK MyIRK(&SpaceDisc, static_cast<RKData::Type>(RK_ID), mag_prec);        

    // Initialize IRK time-stepping solver
    MyIRK.Init(SpaceDisc);
    
    // Set Krylov solver settings
    MyIRK.SetKrylovParams(KRYLOV);

    
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
            
            /* Linear system/solve statistics */
            std::vector<int> avg_krylov_iter;
            std::vector<int> system_size;
            std::vector<double> eig_ratio;
            MyIRK.GetSolveStats(avg_krylov_iter, system_size, eig_ratio);
            solinfo.Print("krtol", KRYLOV.reltol);
            solinfo.Print("katol", KRYLOV.abstol);
            solinfo.Print("kdim", KRYLOV.kdim);
            solinfo.Print("nsys", avg_krylov_iter.size());
            for (int system = 0; system < avg_krylov_iter.size(); system++) {
                solinfo.Print("sys" + to_string(system+1) + "_iters", avg_krylov_iter[system]);
                solinfo.Print("sys" + to_string(system+1) + "_type", system_size[system]);
                solinfo.Print("sys" + to_string(system+1) + "_eig_ratio", eig_ratio[system]);
            }
            
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
        case 2:
            return 0.0;
        // Default source is 0
        default:
            return 0.0;
    }
}


/// Manufactured PDE solution
double PDESolution(const Vector &x, double t)
{    
    switch (example) {
        // Test example where the initial condition is propagated with wave-speed alpha and dissipated with diffusivity mu
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
        // 2nd test example for linear-advection-diffusion that's more realistic (no forcing).
        case 2:
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


AdvDif::AdvDif(FDMesh &Mesh_, Vector alpha_, Vector mu_, 
        int order, FDBias advection_bias) 
    : IRKOperator(Mesh_.GetComm(), Mesh_.GetLocalSize(), 0.0, 
        //TimeDependentOperator::Type::IMPLICIT),
        TimeDependentOperator::Type::EXPLICIT),
        Mesh{Mesh_},
        alpha(alpha_), mu(mu_),
        dim(Mesh_.m_dim),
        D(NULL), A(NULL),
        B_prec(), B_mats(), B_index{0}, AMG(), 
        Jacobian(NULL), identity(NULL),
        source(Mesh_.GetLocalSize()), temp(Mesh_.GetLocalSize())
{
    // Assemble diffusion operator if non-zero
    if (mu.Normlp(infinity()) > 1e-15) {
        D = new FDLinearOp(Mesh, 2, mu, order, CENTRAL);
        D->Assemble();
    }
    
    // Assemble advection operator if non-zero
    if (alpha.Normlp(infinity()) > 1e-15) {
        A = new FDLinearOp(Mesh, 1, alpha, order, advection_bias);
        A->Assemble();
    }
    if (!A && !D) mfem_error("AdvDif::AdvDif() Require at least one non-zero PDE coefficient.");
};


/// Evaluate RHS of ODEs: du_dt = -A*u + D*u + s(t) 
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

/// Evaluate RHS of ODEs without source: du_dt = -A*u + D*u 
void AdvDif::ApplyL(const Vector &u, Vector &du_dt) const
{
    if (D && A) {
        D->Mult(u, du_dt);
        A->Mult(u, source); // Use source as auxillary vector
        du_dt.Add(-1., source);
    } else if (D) {
        D->Mult(u, du_dt);
    } else {
        A->Mult(u, du_dt);
        du_dt.Neg();
    }
}


/// Gradient of L*u w.r.t u evaluated (i.e., L == -A + D) 
HypreParMatrix &AdvDif::SetJacobian() const {
    if (Jacobian) delete Jacobian;

    if (A && D) {
        Jacobian = HypreParMatrixAdd(-1., A->Get(), 1., D->Get()); 
    } else if (A) {
        Jacobian = &(A->Get());
        *Jacobian *= -1.;
    } else if (D) {
        Jacobian = &(D->Get());
    }
    return *Jacobian;
}

void AdvDif::SetSystem(int index, double dt, double gamma, int type) {
    
    // Update B_index
    B_index = index;
    //std::cout << "index = " << index << '\n';

    // Preconditioner previously built with this index: Update it?
    if (index < B_params.Size()) {
        // if dt or gamma change, rebuild
        if (B_params[index].dt != dt ||
            B_params[index].gamma != gamma) {
            delete B_prec[index];
            B_prec[index] = NULL;
            delete B_mats[index];
            B_mats[index] = NULL;
            //std::cout << "Re-building index = " << index << '\n';    
        }
        
    // No preconditioner previously built with this index: Create space for new one. 
    } else {
        //std::cout << "Building index = " << index << '\n';
        B_prec.Append(NULL);
        B_mats.Append(NULL);
        B_params.Append(Prec_params()); 
    }  
    
    
    // Build a new preconditioner if needed
    if (!B_prec[index]) {
        // Assemble L/Jacobian (if not done previously)
        if (!Jacobian) SetJacobian();
        
        // Assemble identity matrix (if not done previously)
        if (!identity) identity = (A) ? A->GetHypreParIdentityMatrix() : D->GetHypreParIdentityMatrix();
        
        // Form matrix B = gamma*I - dt*Jacobian
        HypreParMatrix * B = HypreParMatrixAdd(-dt, *Jacobian, gamma, *identity); 
        
        // Build AMG preconditioner for B
        HypreBoomerAMG * amg_solver = new HypreBoomerAMG(*B);
        
        // Set options in AMG struct
        amg_solver->SetMaxIter(1); 
        amg_solver->SetTol(0.0);
        amg_solver->SetPrintLevel(AMG.printlevel); 
        amg_solver->iterative_mode = false;
        
        
        if (AMG.use_AIR) {                      
            amg_solver->SetLAIROptions(AMG.distance, 
                                        AMG.prerelax, AMG.postrelax,
                                        AMG.strength_tolC, AMG.strength_tolR, 
                                        AMG.filter_tolR, AMG.interp_type, 
                                        AMG.relax_type, AMG.filterA_tol,
                                        AMG.coarsen_type);                                       
        } else {
            amg_solver->SetCoarsening(AMG.coarsen_type);
            amg_solver->SetAggressiveCoarsening(AMG.agg_levels); 
            amg_solver->SetStrengthThresh(AMG.theta);
            amg_solver->SetInterpolation(AMG.interp_type);
            amg_solver->SetRelaxType(AMG.relax_type);
        } 
        
        // Update member parameters
        B_params[index].dt = dt;
        B_params[index].gamma = gamma;
        B_prec[index] = amg_solver;
        B_mats[index] = B;
    }
}

/** Get error against exact PDE solution if available. Also output if num 
    solution is output */
bool AdvDif::GetError(int save, const char * out, double t, const Vector &u, double &eL1, double &eL2, double &eLinf) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    bool soln_implemented = false;
    Vector * u_exact = NULL;
    if (example == 1 || example == 2) {
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