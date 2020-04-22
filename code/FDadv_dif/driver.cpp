#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "FDSpaceDisc.hpp"

using namespace mfem;
using namespace std;

#define PI 3.14159265358979323846


// Solve scalar advection-diffusion equation,
//     u_t + alpha.grad(f(u)) = div( mu*grad(u) ) + s(x,t),
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
// TIME DISCRETIZATION: Both explicit and implicit RK options are available.
//
// SPATIAL DISCRETIZATION: Arbitrarily high-order FD is are available. 
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
int dim, problem, fluxID;
bool GENERAL_FLUX;

enum Linearity { LINEAR = 0, NONLINEAR = 1 };
class BEOper;
class JacPrec;

/* f(u) = u; fluxID == 0 */
double LinearFlux(double u) { return u; }; 
double GradientLinearFlux(double u) { return 1.0; };
/* f(u) = u^2; fluxID == 1 */
double NonlinearFlux(double u) { return u*u; }; 
double GradientNonlinearFlux(double u) { return 2.*u; };

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
ScalarFun GradientFlux(int fluxID) {
    switch (fluxID) {
        case 0:
            return LinearFlux;
        case 1:
            return NonlinearFlux;
        default:
            return NULL;
    }
}

/* Functions definining some parts of PDE */
double InitialCondition(const Vector &x);
double Source(const Vector &x, double t);
double PDESolution(const Vector &x, double t);


struct NEWTON_params {
    double reltol = 1e-6;
    int maxiter = 10;
    int printlevel = 1; 
    int rebuildJacobian = 0; // TOOD: Frequency at which Jacobian is updated
    int rebuildJacobianPreconditioner = 0; // TODO: Frequency at which preconditioner for Jacobian is rebuilt
};

struct GMRES_params {
    double abstol = 1e-6;
    double reltol = 1e-6;
    int maxiter = 100;
    int printlevel = 2;
    int kdim = 30;
};

struct AMG_params {
    bool use_AIR = true;
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
};

/** Provides the time-dependent RHS of the ODEs after spatially discretizing the 
    PDE,
        du/dt = f(u,t) == - A(u) + D*u + s(t).
    where:
        A: The advection discretization,
        D: The diffusion discretization,
        s: The solution-independent source term discretization. */
class AdvDif : public TimeDependentOperator
{
private:    
    Linearity op_type;          // (assumed) Linearity of flux function
    
    int dim;                    // Spatial dimension
    FDMesh &Mesh;               // Mesh on which the PDE is discretized
    
    Vector alpha;               // Advection coefficients
    Vector mu;                  // Diffusion coefficients
    FDLinearOp * D;             // Diffusion operator
    FDLinearOp * AL;            // Linear advection operator
    FDNonlinearOp * AN;         // Nonlinear advection operator
    FDSpaceDisc * A;            // `The` advection operator
        
    mutable Vector source, temp;// Solution independent source term & auxillary vector

    // Require linear/nonlinear solves for implicit time stepping
    bool solversInit;
    BEOper * be;                    // Operator describing the BE equation to be solved
    IterativeSolver * be_solver;    // `The` solver for BE equation
    NewtonSolver * nlbe_solver;     // Solver for nonlinear BE equation
    GMRESSolver * lbe_solver;       // Solver for `linearized` BE equation
    JacPrec * lbe_prec;             // Preconditioner for linear solver
    NEWTON_params NEWTON;           // Newton solver parameters
    GMRES_params GMRES;             // GMRES solver params
    AMG_params AMG;                 // AMG solver params
    
    
    /// Set solution independent source term at current time.
    inline void SetSource() const { Mesh.EvalFunction(&Source, GetTime(), source); };

    /// Initialize solvers for implicit time-stepping.
    void InitSolvers();
    
public:
    
    AdvDif(FDMesh &Mesh_, int fluxID, Vector alpha_, Vector mu_, 
            int order, FDBias advection_bias);
    ~AdvDif();
    
    /// Allocate memory to and assign initial condition.
    inline void GetU0(Vector * &u) const { Mesh.EvalFunction(&InitialCondition, u); };
    
    /// Compute the right-hand side of the ODE system.
    void Mult(const Vector &u, Vector &du_dt) const;
    
    /// Get error w.r.t. exact PDE solution (if available)
    bool GetError(int save, const char * out, double t, const Vector &u, 
                    double &eL1, double &eL2, double &eLinf);
    
    /// Solve Backward-Euler equation for unknown k for implicit time-steping,
    ///     k + A(x + dt*k) - D*(x + dt*k) = s(t)
    void ImplicitSolve(const double dt, const Vector &u, Vector &k); 
    
    /// Set solver parameters fot implicit time-stepping; MUST be called before InitSolvers()
    inline void SetAMGParams(AMG_params params) {
        if (!solversInit) AMG = params; else mfem_error("SetAMGParams:: Can only be called before InitSolvers::"); };
    inline void SetGMRESParams(GMRES_params params) { 
        if (!solversInit) GMRES = params; else mfem_error("SetGMRESParams:: Can only be called before InitSolvers::"); };
    inline void SetNewtonParams(NEWTON_params params) { 
        if (!solversInit) NEWTON = params; else mfem_error("SetNewtonParams:: Can only be called before InitSolvers::"); };
};



/* SAMPLE RUNS:

*/

/* NOTES:
 - Cannot simultaneously specify CFL AND dt. By default, dt is set by CFL, but setting dt takes presidence!
 
 - Cannot simultaneously specify nt AND tf. Bu default, tf set by nt, but setting tf takes presidence!
*/

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
    problem = 1; // Problem ID: responsible for generating source and exact solution
    GENERAL_FLUX = false; // Setting GENERAL_FLUX==TRUE allows for the "nonlinear treatment" of linear problem. 
    int GENERAL_FLUX_temp = 0;
    FDBias advection_bias = CENTRAL;
    int advection_bias_temp = static_cast<int>(advection_bias);
    double ax=1.0, ay=1.0; // Advection coefficients
    double mx=1.0, my=1.0; // Diffusion coefficients
    
    /* --- Time integration --- */
    int nt     = 10;  // Number of time steps
    double dt  = -1.0;// Time step size
    double CFL = 5.0; // CFL number == dt/dx
    double tf  = -1.0; // Final integration time
    int ode_solver_type = 1;  // RK scheme

    /* --- Spatial discretization --- */
    int order     = 2; // Approximation order
    dim           = 1; // Spatial dimension
    int refLevels = 4; // nx == 2^refLevels in every dimension
    int npx       = -1; // # procs in x-direction
    int npy       = -1; // # procs in y-direction
    
    // Solver parameters
    AMG_params AMG;
    GMRES_params GMRES;
    NEWTON_params NEWTON;
    int use_AIR_temp = (int) AMG.use_AIR;
    
    
    // Output paramaters
    int save         = 0;  // Save solution vector
    const char * out = "data/U"; // Filename of data to be saved...
    
    OptionsParser args(argc, argv);
    args.AddOption(&fluxID, "-f", "--flux-function",
                  "0==Linear==u, 1==Nonlinear==u^2.");
    args.AddOption(&GENERAL_FLUX_temp, "-gf", "--general-flux-function",
                  "Ignores linearity of flux if linear.");              
    args.AddOption(&problem, "-ex", "--example-problem",
                  "1 (and 2 for linear problem)."); 
    args.AddOption(&ax, "-ax", "--alpha-x",
                  "Advection in x-direction."); 
    args.AddOption(&ay, "-ay", "--alpha-y",
                  "Advection in y-direction."); 
    args.AddOption(&mx, "-mx", "--mu-x",
                  "Diffusion in x-direction."); 
    args.AddOption(&my, "-my", "--mu-y",
                  "Diffusion in y-direction."); 
    
    args.AddOption(&ode_solver_type, "-t", "--RK-disc",
                  "Time discretization.");
    args.AddOption(&nt, "-nt", "--num-time-steps", "Number of time steps.");    
                  
    /* Spatial discretization */
    args.AddOption(&order, "-o", "--order",
                  "FD: Approximation order."); 
    args.AddOption(&advection_bias_temp, "-b", "--bias",
                  "Advection bias: 0==CENTRAL, 1==UPWIND"); 
    args.AddOption(&refLevels, "-l", "--level",
                  "FD: Mesh refinement; 2^refLevels DOFs in each dimension.");
    args.AddOption(&dim, "-d", "--dim",
                  "Spatial problem dimension.");
    args.AddOption(&npx, "-px", "--nprocs-in-x",
                  "FD: Number of procs in x-direction.");
    args.AddOption(&npy, "-py", "--nprocs-in-y",
                  "FD: Number of procs in y-direction.");                          
       
    /* CFL parameters */
    args.AddOption(&CFL, "-cfl", "--advection-like-cfl", "cfl==dt/dx");
    args.AddOption(&tf, "-tf", "--final-integration-time", "0 <= t <= tf");
    args.AddOption(&dt, "-dt", "--time-step-size", "t_j = j*dt");

    /* AMG parameters */
    args.AddOption(&use_AIR_temp, "-air", "--use-air", "0==standard AMG, 1==AIR");
    /* Krylov parameters */
    args.AddOption(&GMRES.reltol, "-krtol", "--gmres-rel-tol", "GMRES: Relative stopping tolerance");
    args.AddOption(&GMRES.abstol, "-katol", "--gmres-abs-tol", "GMRES: Absolute stopping tolerance");
    args.AddOption(&GMRES.maxiter, "-kmaxit", "--gmres-max-iterations", "GMRES: Maximum iterations");
    args.AddOption(&GMRES.kdim, "-kdim", "--gmres-dimension", "GMRES: Maximum subspace dimension");
    args.AddOption(&GMRES.printlevel, "-kp", "--gmres-print", "GMRES: Print level");
    /* Newton parameters */
    args.AddOption(&NEWTON.reltol, "-nrtol", "--newton-rel-tol", "Newton: Relative stopping tolerance");
    args.AddOption(&NEWTON.maxiter, "-nmaxit", "--newton-max-iterations", "Newton: Maximum iterations");
    args.AddOption(&NEWTON.printlevel, "-np", "--newton-print", "Newton: Print level");
    
    
    /* --- Text output of solution etc --- */              
    args.AddOption(&out, "-out", "--out-directory", "Name of output file."); 
    args.AddOption(&save, "-save", "--save-sol-data",
                  "save>0 will save solution info, save>1 also saves solution (and exact solution, if implemented).");              
    args.Parse();
    if (myid == 0) {
    //    args.PrintOptions(std::cout); 
    }
    // Set final forms of remaing params
    GENERAL_FLUX = (bool) GENERAL_FLUX_temp;
    advection_bias = static_cast<FDBias>(advection_bias_temp);
    AMG.use_AIR = (bool) use_AIR_temp;
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
    
    
    ///////////////////////////////////////
    // Assemble mesh on which we discretize
    ///////////////////////////////////////
    FDMesh Mesh(MPI_COMM_WORLD, dim, refLevels, np);
    
    
    ///////////////////////////////////////////////////
    // Define the ODE solver used for time integration.
    ///////////////////////////////////////////////////
    ODESolver *ode_solver;
    switch (ode_solver_type)
    {
        // Explicit methods
        case 1: ode_solver = new ForwardEulerSolver; break;
        case 2: ode_solver = new RK2Solver(0.5); break; // midpoint method
        case 3: ode_solver = new RK3SSPSolver; break;
        case 4: ode_solver = new RK4Solver; break;
    
        // Implicit L-stable methods
        case 11: ode_solver = new BackwardEulerSolver; break;
        case 12: ode_solver = new SDIRK23Solver(2); break;
        case 13: ode_solver = new SDIRK33Solver; break;
        case 14: ode_solver = new SDIRK34Solver; break;
        default:
            if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
            MPI_Finalize();
            return 3;
    }

    
    ///////////////////////////////////////////
    /* --- Set up spatial discretization --- */
    ///////////////////////////////////////////
    AdvDif SpaceDisc(Mesh, fluxID, alpha, mu, order, advection_bias);
        
    // Initialize solvers for implicit time integration
    if (ode_solver_type > 10) {
        SpaceDisc.SetAMGParams(AMG);
        SpaceDisc.SetGMRESParams(GMRES);
        SpaceDisc.SetNewtonParams(NEWTON);
    }    
        
    // Get initial condition
    Vector * u = NULL;
    SpaceDisc.GetU0(u);
    
    // Get mesh info
    double dx = Mesh.Get_dx();
    int    nx = Mesh.Get_nx();
    
    // If user hasn't set dt, time step so that we run at prescribed CFL
    if (dt == -1.0) dt = dx * CFL;
    
    // If user hasn't set tf, set it according to nt and dt
    bool set_tf = true;
    if (tf == -1.0) {
        tf = nt * dt;
        set_tf = false;
    // Update nt otherwise
    } else {
        nt = ceil(tf/dt);
    }
    

    
    ///////////////
    /* Time-step */
    ///////////////
    ode_solver->Init(SpaceDisc);
    double t = 0.0;
    SpaceDisc.SetTime(t);
    int step = 0;
    int numsteps = ceil((tf-t)/dt);
    while (t < tf) {
        step++;
        if (myid == 0) std::cout << "Time-step " << step << " of " << numsteps << '\n';
    
        // Step from t to t+dt
        ode_solver->Step(*u, t, dt);
    }
    
    if (set_tf && (fabs(t-tf)>1e-15)) {
        if (myid == 0) std::cout << "WARNING: Requested tf of " << tf << " adjusted to " << t << '\n';
    }
    tf = t; // Update final time


    /* ----------------------- */
    /* --- Save solve data --- */
    /* ----------------------- */
    if (save > 0) {
        ostringstream outt;
        outt << out << "." << myid;
        ofstream sol;
        sol.open(outt.str().c_str());
        if (save > 1) u->Print_HYPRE(sol);
        sol.close();
        
        /* Get error against exact PDE solution if available */
        double eL1, eL2, eLinf = 0.0; 
        bool got_error = SpaceDisc.GetError(save, out, tf, *u, eL1, eL2, eLinf);
        
        // Save data to file enabling easier inspection of solution            
        if (myid == 0) {
            OutDict solinfo;
            solinfo.open(out);
            solinfo << scientific;
            
            /* Temporal info */
            solinfo.Print("RK", ode_solver_type);
            solinfo.Print("dt", dt);
            solinfo.Print("nt", nt);
            solinfo.Print("tf", tf);
            
            /* Spatial info */
            solinfo.Print("dx", dx);
            solinfo.Print("nx", nx);
            solinfo.Print("space_dim", dim);
            solinfo.Print("space_refine", refLevels);
            solinfo.Print("problemID", problem); 
            
            // /* Linear system/solve statistics */
            // solinfo.Print("krtol", reltol);
            // solinfo.Print("katol", abstol);
            // solinfo.Print("kdim", kdim);
            // std::vector<int> avg_iter;
            // std::vector<int> type;
            // std::vector<double> eig_ratio;
            // MyIRK.GetSolveStats(avg_iter, type, eig_ratio);
            // solinfo.Print("nsys", avg_iter.size());
            // for (int system = 0; system < avg_iter.size(); system++) {
            //     solinfo.Print("sys" + to_string(system+1) + "_iters", avg_iter[system]);
            //     solinfo.Print("sys" + to_string(system+1) + "_type", type[system]);
            //     solinfo.Print("sys" + to_string(system+1) + "_eig_ratio", eig_ratio[system]);
            // }
            
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
    delete ode_solver;
    delete u;

    MPI_Finalize();
    return 0;
}




/* Initial condition of PDE */
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


/* Solution-independent source term in PDE */
double Source(const Vector &x, double t) {
    switch (problem) {
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


/* Manufactured PDE solution */
double PDESolution(const Vector &x, double t)
{    
    switch (problem) {
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
            if (fluxID == 1) cout <<  "PDESolution not implemented for NONLINEAR problem " << problem << "\n";
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
            cout <<  "PDESolution:: not implemented for problem " << problem << "\n";
            return 0.0;
    }
}




/** The operator to be inverted solved during implicit time stepping. The form 
    of the operator is dependent on the linearity of the problem we're solving,
        LINEAR:    BEOper(k; dt)    == k + dt*(A - D)*k 
        NONLINEAR: BEOper(k; dt, u) == k + A(u + dt*k) - D*(u + dt*k)

    k -- the unknown, 
    A -- advection operator, 
    D -- diffusion oprator,
    dt -- a constant, 
    u -- current state (only appears in the nonlinear problem) */    
class BEOper : public Operator 
{
private:
    
    friend class JacPrec; // This needs to access members

    Linearity op_type;
    FDSpaceDisc * A;        // `The` advection  operator
    FDNonlinearOp * AN;     // Nonlinear advection operator
    FDLinearOp * AL, * D;   // Linear operators
    
    mutable HypreParMatrix * Jacobian; // Declare mutable so can be updated
    HypreParMatrix * I;
    
    const Vector * u;               // Current state (nonlinear only)
    mutable Vector temp1, temp2;    // Auxillary vectors
    
    double dt;                      // Current time step used in Mult, and in assembling new Jacobians
    mutable bool dt_current;        // Does Jacobian use current value of dt? 
    
    int updateJacobian; // TODO: Need to implement, but if we don't want to update every Newton iter, can use this to just keep same Jacobian... 
public:

    // Linear operator
    BEOper(int height, FDLinearOp * AL_, FDLinearOp * D_) : Operator(height), 
        op_type{LINEAR}, A(AL_), AL(AL_), D(D_), 
        Jacobian(NULL), u(NULL), temp1(height), temp2(height), 
        dt{0.0}, dt_current(false)
        {
            I = (A) ? A->GetHypreParIdentityMatrix() : D->GetHypreParIdentityMatrix();
        }

    // Nonlinear operator. Must be called with a valid nonlinear advection operator!
    BEOper(int height, FDNonlinearOp * AN_, FDLinearOp * D_) : Operator(height),
        op_type{NONLINEAR}, A(AN_), AN(AN_), D(D_), 
        Jacobian(NULL), I(AN_->GetHypreParIdentityMatrix()),
        u(NULL), temp1(height), temp2(height), 
        dt{0.0}, dt_current(false) { };
    
    ~BEOper() { delete I; if (Jacobian) delete Jacobian; };
    
    /// LINEAR: Set current dt
    inline void SetParameters(double dt_) { dt_current = (dt == dt_) ? true : false; dt = dt_; };
    
    /// NONLINEAR: Set current dt and u values
    inline void SetParameters(double dt_, const Vector * u_) { 
        dt_current = (dt == dt_) ? true : false;
        dt = dt_;
        u = u_;
    };

    /** Compute action of operator */ 
    virtual void Mult(const Vector &k, Vector &y) const;

    /// LINEAR: Compute Jacobian. Will be called by GMRES's preconditioner at will.
    HypreParMatrix &GetLinearGradient() const;

    /// NONLINEAR: Compute Jacobian. Is called by Newton every iteration and 
    /// passed into GMRES's SetOperator(), which then passes it to its 
    /// preconditioner through its SetOperator().
    virtual Operator &GetGradient(const Vector &k) const;
};


/** Wrapper for an AMG preconditioner for GMRES when solving a Jacobian systems(s) */
class JacPrec : public Solver
{
    
private:
    HypreBoomerAMG * J_prec; 
    BEOper &be;   
    
public:
    
    JacPrec(BEOper &be_, HypreBoomerAMG * J_prec_) : Solver(J_prec_->Height()), 
        be(be_), J_prec(J_prec_) {};
    
    ~JacPrec() { delete J_prec; };
    
    /// Solve using J_prec.
    inline void Mult(const Vector &x, Vector &y) const { J_prec->Mult(x, y); };
    
    /** Always called by GMRES
        -LINEAR: Passed the BEOper that GMRES is applied to. GMRES will use the 
            associated BEOper::Mult() function to compute the action of the BEOper, 
            and we'll need to build the Jacobian matrix here. 
        -NONLINEAR: Will be passed the HypreParMatrix that GMRES retrieves from
            the associated BEOper::GetGradient(x). This is the HypreParMatrix
            that GMRES will use for the current Newton iteration. */
    inline void SetOperator(const Operator &op) {
        
        // Called for a linear problem
        if (be.op_type == LINEAR)
        {
            // Only build Jacobian if not previously built.
            if (!(be.Jacobian)) {
                J_prec->SetOperator(be.GetLinearGradient());
                
            // Not interested in variable time-stepping, but leave this here.    
            } else if (!(be.dt_current)) {
                mfem_error("JacPrec::SetOperator() You're doing variable "
                    "time-stepping, when would you like to update the linear "
                    "operator used by GMRES's preconditioner?");
            }
        
        // Called for a nonlinear problem
        }
        else if (be.op_type == NONLINEAR) 
        {
            // if update Jacobian preconditioner at every Newton iteration.
            // TODO: Probably unecessary to update as often as possible. Maybe
            // GMRES doesn't even get it updated as frequently as possible. Note
            // that if we're doing Jacobian-free GMRES, we most def. don't want 
            // to update this every Newton iter.    
            J_prec->SetOperator(*(be.Jacobian)); // Note not calling be.GetGradient() since it's just been called by GMRES.
        }
    };
};



/* Compute action of operator:
LINEAR:    BEOper(k; dt)    == k + dt*(A - D)*k 
NONLINEAR: BEOper(k; dt, u) == k + A(u + dt*k) - D*(u + dt*k) */
void BEOper::Mult(const Vector &k, Vector &y) const
{
    if (op_type == LINEAR) {
        if (D && A) {
            A->Mult(k, y);
            D->Mult(k, temp1);
            y.Add(-1., temp1);
        } else if (D) {
            D->Mult(k, y);
            y.Neg();
        } else {
            A->Mult(k, y);
        }
        y *= dt;
        y += k;
    } else {
        add(*u, dt, k, temp1); // temp1 = u + dt*k
        if (D && A) {
            A->Mult(temp1, y);
            D->Mult(temp1, temp2);
            y.Add(-1., temp2);
        } else {
            A->Mult(temp1, y);
        }
        y += k;
    }
}

/* Compute Jacobian for nonlinear operator,
NONLINEAR: BEOper(k; dt, u) == k + A(u + dt*k) - D*(u + dt*k) 
dBEOper/dk = I + dt*Gradient(A(u + dt*k)) - dt*D */
Operator &BEOper::GetGradient(const Vector &k) const {
    
    if (op_type == LINEAR) mfem_error("BEOper::GetGradient() N/A for linear function.");
    
    // TODO: Set flag here as to whether we actually want to build a new Jacobian
    if (Jacobian) delete Jacobian;
    
    dt_current = true; // New Jacobian uses current dt.
    add(*u, dt, k, temp1); // temp1 = u + dt*k
    if (D && A) {
        Jacobian = HypreParMatrixAdd(dt, AN->GetGradient(temp1), -dt, D->Get()); 
        *Jacobian += *I; 
    } else {
        Jacobian = HypreParMatrixAdd(dt, AL->GetGradient(temp1), 1.0, *I); 
    }
    return *Jacobian;
}

/* Compute Jacobian for linear operator,
LINEAR: BEOper(k; dt) == k + dt*A*k - dt*D*k 
dBEOper/dk = I + dt*A - dt*D */
HypreParMatrix &BEOper::GetLinearGradient() const {
    
    if (op_type == NONLINEAR) mfem_error("BEOper::GetLinearGradient() N/A for nonlinear function.");
    
    // TODO: Set flag here as to whether we actually want to build a new Jacobian
    if (Jacobian) delete Jacobian;
    
    dt_current = true; // New Jacobian uses current dt.
    if (D && A) {
        /* Having problems here... 
        This seems to work for the moment. But Add nor ParAdd work consistently 
        on 1 and multiple procs.
        Actually, I think the problem is they don't have the same `col_map_offd`
        arrays... But I don't really know what's going on... Anyway, this next line
        prevents me from freeing the AMG solver (get a seg fault when I try).
        Actually, it works for 1D, but not 2D; what gives?! The problems are set up
        in the exact same way...
        */
        Jacobian = HypreParMatrixAdd(dt, AL->Get(), -dt, D->Get()); 
        *Jacobian += *I; 
    } else if (D) {
    // TODO: Even if I use `Add` here I'm getting seg faults... Why?!
    // But seem to be able to free the AMG solver associated with this K...
        Jacobian = HypreParMatrixAdd(-dt, D->Get(), 1.0, *I); // K <- I -dt*D
    } else {
        Jacobian = HypreParMatrixAdd(dt, AL->Get(), 1.0, *I); // J <- I + dt*A
    }
    // // These owndership flags are identical for all the different cases...
    // std::cout << "data = " << static_cast<signed>(K->OwnsDiag()) << "\n";
    // std::cout << "offd = " << static_cast<signed>(K->OwnsOffd()) << "\n";
    // std::cout << "cols = " << static_cast<signed>(K->OwnsColMap()) << "\n";
    //std::cout << "got J" << '\n';
    return *Jacobian;
}

/* Initialize solvers for inversion of BEOper during implicit time integration. */
void AdvDif::InitSolvers() 
{
    if (solversInit) mfem_error("AdvDif::InitSolvers() Can only initialize solvers once.");
    solversInit = true;
    
    // Initialize BE operator
    if (op_type == LINEAR) {
        be = new BEOper(this->Height(), AL, D);
    } else {
        be = new BEOper(this->Height(), AN, D);
    }
    
    
    // AMG-based preconditioner for GMRES solution of Jacobian system(s)
    HypreBoomerAMG * amg_solver = new HypreBoomerAMG;
    amg_solver->SetPrintLevel(0); 
    amg_solver->SetMaxIter(1); 
    amg_solver->SetTol(0.0);
    amg_solver->SetMaxLevels(50); 
    //amg_solver->iterative_mode = false;
    if (AMG.use_AIR) {                        
        amg_solver->SetLAIROptions(AMG.distance, 
                                    AMG.prerelax, AMG.postrelax,
                                    AMG.strength_tolC, AMG.strength_tolR, 
                                    AMG.filter_tolR, AMG.interp_type, 
                                    AMG.relax_type, AMG.filterA_tol,
                                    AMG.coarsening);                                       
    } else {
        amg_solver->SetInterpolation(0);
        amg_solver->SetCoarsening(AMG.coarsening);
        amg_solver->SetAggressiveCoarsening(1);
    }
    lbe_prec = new JacPrec(*be, amg_solver); 


    // GMRES solver for Jacobian system(s)
    lbe_solver = new GMRESSolver(Mesh.GetComm());
    lbe_solver->iterative_mode = false;
    lbe_solver->SetAbsTol(GMRES.abstol);
    lbe_solver->SetRelTol(GMRES.reltol);
    lbe_solver->SetMaxIter(GMRES.maxiter);
    lbe_solver->SetPrintLevel(GMRES.printlevel);
    lbe_solver->SetKDim(GMRES.kdim);
    lbe_solver->SetPreconditioner(*lbe_prec); // JacPrec is preconditioner for GMRES


    // `THE` solver for the BE equation
    // LINEAR: Use GMRES 
    if (op_type == LINEAR) { 
        be_solver = lbe_solver;
        
    // NONLINEAR: Use Newton
    } else {
        nlbe_solver = new NewtonSolver(Mesh.GetComm());
        nlbe_solver->iterative_mode = false;
        nlbe_solver->SetMaxIter(NEWTON.maxiter);
        nlbe_solver->SetRelTol(NEWTON.reltol);
        nlbe_solver->SetPrintLevel(NEWTON.printlevel);
        if (NEWTON.printlevel == 2) nlbe_solver->SetPrintLevel(-1);
        nlbe_solver->SetSolver(*lbe_solver); // GMRES is linear solver for Newton
        be_solver = nlbe_solver;
    }
}




AdvDif::AdvDif(FDMesh &Mesh_, int fluxID, Vector alpha_, Vector mu_, 
        int order, FDBias advection_bias) 
    : TimeDependentOperator(Mesh_.GetLocalSize()),
        op_type{LINEAR},
        Mesh{Mesh_},
        alpha(alpha_), mu(mu_),
        dim(Mesh_.m_dim),
        D(NULL), AL(NULL), AN(NULL), A(NULL),
        solversInit(false), be(NULL), nlbe_solver(NULL), lbe_solver(NULL), lbe_prec(NULL),
        NEWTON(), GMRES(), AMG(),
        source(Mesh_.GetLocalSize()), temp(Mesh_.GetLocalSize())
{
    // Assemble diffusion operator if non-zero
    if (mu.Normlp(infinity()) > 1e-15) {
        D = new FDLinearOp(Mesh, 2, mu, order, CENTRAL);
        D->Assemble();
    }
    
    // Assemble advection operator if non-zero
    if (alpha.Normlp(infinity()) > 1e-15) {
        if (fluxID == 1 || GENERAL_FLUX) 
        {
            AN = new FDNonlinearOp(Mesh, 1, alpha, Flux(fluxID), GradientFlux(fluxID), 
                        order, advection_bias);
            A = AN; 
            op_type = NONLINEAR; // Will require nonlinear treatment
        } else if (fluxID == 0) {
            AL = new FDLinearOp(Mesh, 1, alpha, order, advection_bias);
            AL->Assemble();
            A = AL; 
        } else {
            mfem_error("Must use fluxID==1 or fluxID==0");
        }   
    }
    if (!A && !D) mfem_error("AdvDif::AdvDif() Require at least one non-zero PDE coefficient.");
};


/* Evaluate RHS of ODEs: du_dt = -A(u) + D*u + s(t) */
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

/** Solve the BE equation for k, given current state u. Depending on the linearity 
of the problem, we solve a different system:
    LINEAR:    k + dt*(A - D)*k = dt*(-A + D)*u + s(t) 
    NONLINEAR: k + A(u + dt*k) - D*(u + dt*k) = s(t) */
void AdvDif::ImplicitSolve(const double dt, const Vector &u, Vector &k)
{
    // Initialize solvers on first pass.
    if (!solversInit) InitSolvers();

    if (op_type == LINEAR) 
    {
        be->SetParameters(dt); // Set time step for be's solve
        be_solver->SetOperator(*be); // GMRES will be applied to current BEOper
        Mult(u, temp); // Set RHS of linear system
        be_solver->Mult(temp, k); 
    } 
    else 
    {
        be->SetParameters(dt, &u); // Set time step and current state for be's solve
        SetSource(); // Set RHS of nonlinear system
        be_solver->SetOperator(*be); // Newton will be applied to current BEOper
        be_solver->Mult(source, k); 
    }

    MFEM_VERIFY(be_solver->GetConverged(), "BE solver did not converge.");
    
    // Facilitate different printing from Newton.
    if (op_type == NONLINEAR && NEWTON.printlevel == 2) {
        if (Mesh.m_rank == 0) {
            std::cout << "Newton: Number of iterations: " << be_solver->GetNumIterations() 
                << ", ||r|| = " << be_solver->GetFinalNorm() << '\n';
        }
    }
}

/* Get error against exact PDE solution if available. Also output if num solution is output */
bool AdvDif::GetError(int save, const char * out, double t, const Vector &u, double &eL1, double &eL2, double &eLinf) {
    
    bool soln_implemented = false;
    Vector * u_exact = NULL;
    if (problem == 1 || problem == 2) {
        soln_implemented = true;
        Mesh.EvalFunction(&PDESolution, t, u_exact); 
    }
    
    if (soln_implemented) {
        int myid;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        ostringstream outt;
        outt << out << "_exact" << "." << myid;
        ofstream sol;
        sol.open(outt.str().c_str());
        if (save > 1) u_exact->Print_HYPRE(sol);
        sol.close();
        
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
    }
    return soln_implemented;
}

AdvDif::~AdvDif()
{
    //if (gmres_solver) delete gmres_solver;
    /* TODO: WHY can't I free amg_solver. Actually, this seems related to 
        *** The MPI_Comm_free() function was called after MPI_FINALIZE was invoked.
        *** This is disallowed by the MPI standard.
    if (amg_solver) delete amg_solver;
    */
    if (be) delete be;
    if (nlbe_solver) delete nlbe_solver;
    if (lbe_solver) delete lbe_solver;
    //if (lbe_prec) delete lbe_prec;
    if (D) delete D;
    if (AL) delete AL;
    if (AN) delete AN;
}