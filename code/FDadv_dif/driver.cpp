#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "FDSpaceDisc.hpp"

using namespace mfem;
using namespace std;

#define PI 3.14159265358979323846


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

struct GMRES_params {
    double abstol = 1e-6;
    double reltol = 1e-6;
    int maxiter = 100;
    int printlevel = 2;
    int kdim = 30;
};

// Solve scalar advection-diffusion equation,
//     u_t + alpha.grad(f(u)) = div( mu*grad(u) ) + s(x,t),
// for constant vectors alpha, and mu.
// 
// An initial condition is specified, and periodic spatial boundaries are used.
// 
// Options for advection flux f are,
//     f(u) = u   -- linear advection,
//     f(u) = u^2 -- nonlinear advection (Burgers' like).
// 
// Implemented in 1 and 2 spatial dimensions.
// 
// Test problems with manufactured solutions are implemented to facillitate 
// convergence studies.
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

enum AdvectionType {
   LINEAR = 0, NONLINEAR = 1 
};

// Problem parameters global functions depend on
Vector alpha, mu;
AdvectionType advection;
int dim, problem;

/* f(u) = u */
double LinearFlux(double u) { return u; };
double GradientLinearFlux(double u) { return 1.0; };
/* f(u) = u^2 */
double NonlinearFlux(double u) { return u*u; };
double GradientNonlinearFlux(double u) { return u; };

/* Functions definining some parts of PDE */
double InitialCondition(const Vector &x);
double Source(const Vector &x, double t);
double PDESolution(const Vector &x, double t);


/** Provides the time-dependent RHS of the ODEs after spatially discretizing the 
    PDE,
        du/dt = -A(u) + D*u + s(t).
    where:
        A: The advection discretization,
        D: The diffusion discretization,
        s: The solution-independent source term discretization.
*/
class AdvDif : public TimeDependentOperator
{
private:    
    AdvectionType advection;    // Type of advection to use
    int problem;                // ID of problem. problem==1 has exact solution implemented
    
    int dim;                    // Spatial dimension
    FDMesh &Mesh;               // Mesh on which the PDE is discretized
    
    FDLinearOp * Dif;           // Diffusion operator
    FDLinearOp * AdvLin;        // Linear advection operator
    //FDNonlinearOp * AdvNonlin;  // Nonlinear advection operator
    
    Vector alpha;               // Advection coefficients
    Vector mu;                  // Diffusion coefficients
    
    // For solution of linear systems when implicit time stepping
    HypreParMatrix * I;         // Identity operator
    HypreParMatrix * K;         // I - dt_K*[A - D]
    double dt_K; 
    GMRESSolver * GMRES_solver;
    HypreBoomerAMG * AMG_solver;
    GMRES_params GMRES;
    AMG_params AMG;
    
    Vector * s;                 // Solution independent source term
    Vector * z;                 // Auxillary vector

    /// Set solution independent source term at current time
    inline void SetSource() const { Mesh.EvalFunction(&Source, GetTime(), *s); };


    void SetupGMRES();

public:
    
    AdvDif(FDMesh &Mesh_, AdvectionType advection_, Vector alpha_, Vector mu_, 
            int order, int problem_ = 0);
    ~AdvDif();
    
    /// Allocate memory to and assign initial condition.
    inline void GetU0(Vector * &u) const { Mesh.EvalFunction(&InitialCondition, u); };
    
    /// Compute the right-hand side of the ODE system.
    void Mult(const Vector &u, Vector &du_dt) const;
    
    /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    void ImplicitSolve(const double dt, const Vector &u, Vector &k);
    
    /// Get error w.r.t. exact PDE solution (if available)
    bool GetError(int save, const char * out, double t, const Vector &u, 
                    double &eL1, double &eL2, double &eLinf);
                    
    inline void SetGMRES_params(GMRES_params params) { GMRES = params; SetupGMRES(); };
    inline void SetAMG_params(AMG_params params) { AMG = params; };                
};


AdvDif::AdvDif(FDMesh &Mesh_, AdvectionType advection_, Vector alpha_, Vector mu_, 
        int order, int problem_) 
    : TimeDependentOperator(Mesh_.m_nxLocalTotal),
        Mesh{Mesh_}, advection{advection_}, 
        alpha(alpha_), mu(mu_),
        problem{problem_}, dim(Mesh_.m_dim),
        Dif(NULL), AdvLin(NULL),
        K(NULL), I(NULL), dt_K{0.0}, GMRES_solver(NULL), AMG_solver(NULL),
        GMRES(), AMG(),
        s(NULL), z(NULL)
{
    s = new Vector(Mesh_.m_nxLocalTotal);
    z = new Vector(Mesh_.m_nxLocalTotal);
    
    if (mu.Normlp(infinity()) > 1e-15)
    {
        Dif = new FDLinearOp(Mesh, 2, mu, order, CENTRAL);
        Dif->Assemble();
    }
    
    if (alpha.Normlp(infinity()) > 1e-15)
    {
        if (advection == LINEAR) 
        {
            AdvLin = new FDLinearOp(Mesh, 1, alpha, order, CENTRAL);
            AdvLin->Assemble();
        } else {
            mfem_error("Nonlinear mult not implemented...");
        }    
    }
    
    
    // Initialize parameters for GMRES
    SetupGMRES();
};


/* Solve the linear system
    K*k = [I + dt*(A - D)]*k = (-A+D)*u + s(t) == Mult(u)
*/
void AdvDif::ImplicitSolve(const double dt, const Vector &u, Vector &k)
{
   if ((fabs(dt_K - dt) > 1e-4 * dt) || !K)
   {
        // Set I once only
        if (!I) {
           if (Dif) I = Dif->GetHypreParIdentityMatrix(); else I = AdvLin->GetHypreParIdentityMatrix();
        }
             
        // Delete existing K 
        if (K) { 
            //delete AMG_solver; // TODO: See other comments about this...
            delete K;
        }

        // Build K
        dt_K = dt;
        if (Dif && AdvLin) {
            /* Having problems here... 
            This seems to work for the moment. But Add nor ParAdd work consistently 
            on 1 and multiple procs.
            Actually, I think the problem is they don't have the same `col_map_offd`
            arrays... But I don't really know what's going on... Anyway, this next line
            prevents me from freeing the AMG solver (get a seg fault when I try).
            Actually, it works for 1D, but not 2D; what gives?! The problems are set up
            in the exact same way...
            */
            K = HypreParMatrixAdd(dt, *(AdvLin->GetOp()), -dt, *(Dif->GetOp())); // K <- dt(A-D)
            *K += *I; // K <- I + dt*(A-D)
            
        } else if (Dif) {
            // TODO: Even if I use `Add` here I'm getting seg faults... Why?!
            // But seem to be able to free the AMG solver associated with this K...
            K = HypreParMatrixAdd(-dt, *(Dif->GetOp()), 1.0, *I); // K <- I -dt*D
        } else if (AdvLin) {
            K = HypreParMatrixAdd(dt, *(AdvLin->GetOp()), 1.0, *I); // K <- I + dt*A
        }
        // // These owndership flags are identical for all the different cases...
        // std::cout << "data = " << static_cast<signed>(K->OwnsDiag()) << "\n";
        // std::cout << "offd = " << static_cast<signed>(K->OwnsOffd()) << "\n";
        // std::cout << "cols = " << static_cast<signed>(K->OwnsColMap()) << "\n";
       
        // Build AMG preconditioner for GMRES
        AMG_solver = new HypreBoomerAMG(*K);
        AMG_solver->SetPrintLevel(0); 
        AMG_solver->SetMaxIter(1); 
        AMG_solver->SetMaxLevels(50); 
        if (AMG.use_AIR) {                        
            AMG_solver->SetLAIROptions(AMG.distance, 
                                        AMG.prerelax, AMG.postrelax,
                                        AMG.strength_tolC, AMG.strength_tolR, 
                                        AMG.filter_tolR, AMG.interp_type, 
                                        AMG.relax_type, AMG.filterA_tol,
                                        AMG.coarsening);                                       
        }
        else {
            AMG_solver->SetInterpolation(0);
            AMG_solver->SetCoarsening(AMG.coarsening);
            AMG_solver->SetAggressiveCoarsening(1);
        }

        // Reset GMRES preconditioner and operator
        GMRES_solver->SetPreconditioner(*AMG_solver);
        GMRES_solver->SetOperator(*K);
    }
    this->Mult(u, *z);
    GMRES_solver->Mult(*z, k);
}


// Build GMRES using parameters 
void AdvDif::SetupGMRES() {
    if (GMRES_solver) delete GMRES_solver;
        
    GMRES_solver = new GMRESSolver(Mesh.m_comm);
    GMRES_solver->SetAbsTol(GMRES.abstol);
    GMRES_solver->SetRelTol(GMRES.reltol);
    GMRES_solver->SetMaxIter(GMRES.maxiter);
    GMRES_solver->SetPrintLevel(GMRES.printlevel);
    GMRES_solver->SetKDim(GMRES.kdim);
    GMRES_solver->iterative_mode = false;
}

/* Evaluate RHS of ODEs: du_dt = -A(u) + D*u + s(t) */
void AdvDif::Mult(const Vector &u, Vector &du_dt) const
{
    if (advection == LINEAR) {
        if (Dif && AdvLin) {
            Dif->Mult(u, du_dt);
            AdvLin->Mult(u, *s); // Use s as auxillary vector, is reset below anyways
            du_dt -= *s;
        } else if (Dif) {
            Dif->Mult(u, du_dt);
        } else if (AdvLin) {
            AdvLin->Mult(u, du_dt);
            du_dt.Neg();
        }
    } else {
        mfem_error("Nonlinear mult not implemented...");
    }
    
    // Add solution-independent source term
    SetSource();
    du_dt += *s;
}


AdvDif::~AdvDif()
{
    delete s;
    delete z;
    if (GMRES_solver) delete GMRES_solver;
    /* TODO: WHY can't I free AMG_solver. Actually, this seems related to 
        *** The MPI_Comm_free() function was called after MPI_FINALIZE was invoked.
        *** This is disallowed by the MPI standard.
    if (AMG_solver) delete AMG_solver;
    */
    //if (AMG_solver) delete AMG_solver;
    if (K) delete K;
    if (I) delete I;
    if (Dif) delete Dif;
    if (AdvLin) delete AdvLin;
    //if (AdvNonlin) delete AdvNonlin;
}

/* SAMPLE RUNS:

*/

/* NOTES:
 - Cannot simultaneously specify CFL AND dt. By default, dt is set by CFL, but setting dt takes presidence!
 
 - Cannot simultaneously specify nt AND tf. Bu default, tf set by nt, but setting tf takes presidence!

 - Adding more AIR iterations seems like it does accelerate the convergence of GMRES for type 2 systems,
    but it does not pay off! E.g., maybe doing 10 AIR iterations instead of 1 halves the number of GMRES
    iterations
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
    problem = 1;
    advection = LINEAR;
    int advection_temp = static_cast<int>(advection);
    double ax=1.0, ay=1.0; // Advection coefficients
    double mx=1.0, my=1.0; // Diffusion coefficients
    
    /* --- Time integration --- */
    int nt     = 10;  // Number of time steps
    double dt  = -1.0;// Time step size
    double CFL = 5.0; // CFL number == dt/dx
    double tf  = -1.0; // Final integration time
    int ode_solver_type = 1;  // IRK scheme

    /* --- Spatial discretization --- */
    int order     = 2; // Approximation order
    dim           = 1; // Spatial dimension
    int refLevels = 4; // nx == 2^refLevels in every dimension
    int npx       = -1; // # procs in x-direction
    int npy       = -1; // # procs in y-direction
    
    // AMG and Krylov parameters
    AMG_params AMG;
    GMRES_params GMRES;
    int use_AIR_temp = (int) AMG.use_AIR;
    
    
    // Output paramaters
    int save         = 0;  // Save solution vector
    const char * out = "data/U"; // Filename of data to be saved...
    
    
    OptionsParser args(argc, argv);
    args.AddOption(&advection_temp, "-adv", "--advection-type",
                  "adv=0 is linear, adv=1 is nonlinear.");
    args.AddOption(&problem, "-ex", "--example-problem",
                  "Problem ID."); 
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
    args.AddOption(&use_AIR_temp, "-air", "--use-air", 
                        "0==standard AMG, 1==AIR");
    
    /* Krylov parameters */
    args.AddOption(&GMRES.reltol, "-rtol", "--rel-tol", "Krlov: Relative stopping tolerance");
    args.AddOption(&GMRES.abstol, "-atol", "--abs-tol", "Krlov: Absolute stopping tolerance");
    args.AddOption(&GMRES.maxiter, "-maxit", "--max-iterations", "Krlov: Maximum iterations");
    args.AddOption(&GMRES.kdim, "-kdim", "--Krylov-dimension", "Krlov: Maximum subspace dimension");
    args.AddOption(&GMRES.printlevel, "-kp", "--Krylov-print", "Krlov: Print level");
    
    /* --- Text output of solution etc --- */              
    args.AddOption(&out, "-out", "--out-directory", "Name of output file."); 
    args.AddOption(&save, "-save", "--save-sol-data",
                  "save>0 will save solution info, save>1 also saves solution (and exact solution, if implemented).");              
    args.Parse();
    if (myid == 0) {
        args.PrintOptions(std::cout); 
    }
    // Set up remaing params
    advection = static_cast<AdvectionType>(advection_temp);
    
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
    
    /* --- Set up spatial discretization --- */
    // Build mesh on which we discretize
    FDMesh Mesh(MPI_COMM_WORLD, dim, refLevels, np);
    
    // Initialize the spatial discretization operator
    AdvDif SpaceDisc(Mesh, advection, alpha, mu, order, problem);
        
    // Get initial condition
    Vector * u = NULL;
    SpaceDisc.GetU0(u);
    
    // Get mesh info
    double dx = Mesh.Get_dx();
    int    nx = Mesh.Get_nx();
    
    // If user hasn't set dt, time step so that we run at prescribed CFL
    if (dt == -1.0) {
        if (dim == 1) {
            dt = dx * CFL; // assume max|a| = 1
        // Time step so that we run with CFL == dt*[max|a_x|/dx + max|a_y|/dy] in 2D
        } else if (dim == 2) {
            double dy = dx;
            dt = CFL/(1/dx + 1/dy); // Assume max|a_x| = max|a_y| = 1
        }
    }
    
    // If user hasn't set tf, set it according to nt and dt
    bool set_tf = true;
    if (tf == -1.0) {
        tf = nt * dt;
        set_tf = false;
    // Update nt otherwise
    } else {
        nt = ceil(tf/dt);
    }
    
    
    // Define the ODE solver used for time integration.
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
        default:
            if (myid == 0) cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
            MPI_Finalize();
            return 3;
    }


    // Set solver parameters for implicit time integration
    if (ode_solver_type > 10) {
        AMG.use_AIR = (bool) use_AIR_temp;
        SpaceDisc.SetAMG_params(AMG);
        SpaceDisc.SetGMRES_params(GMRES);
    }

    ode_solver->Init(SpaceDisc);
    
    /* Main time-stepping loop */
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
            switch (advection) {
                // Linear advection term
                case LINEAR:
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
                // Nonlinear advection term
                case NONLINEAR:
                    switch (x.Size()) {
                        case 1: {
                            double u = 0.5*PI*(x(0)-alpha(0)*t);
                            double z = t*mu(0);
                            return 0.5*exp(-2.*z)*(
                                    -4.*PI*alpha(0)*pow(cos(u),6.)*sin(2.*u)
                                    + exp(z)*pow(cos(u),2.)
                                    * (-mu(0)*(-1.-2.*PI*PI+(4.*PI*PI-1.)*cos(2.*u)) 
                                        + 2.*PI*alpha(0)*sin(2*u)));
                        }
                        case 2:
                        {
                            double u = 0.5*PI*(x(0)-alpha(0)*t-1.);
                            double v = 0.5*PI*(x(1)-alpha(1)*t-1.);
                            double z = (mu(0)+mu(1))*t;
                            return -0.5*exp(-2.*z)*pow(sin(u)*sin(v), 2.)
                                    *(  2.*PI*alpha(0)*sin(2.*u)*pow(v, 2.)*( exp(z)-2.*pow(sin(u)*sin(v), 4.) )
                                      + 2.*PI*alpha(1)*sin(2.*v)*pow(u, 2.)*( exp(z)-2.*pow(sin(u)*sin(v), 4.) )
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
        case 2:
            if (advection == NONLINEAR) cout <<  "PDESolution:: not implemented for NONLINEAR problem " << problem << "\n";
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