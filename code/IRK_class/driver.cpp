#include <iostream>
#include <fstream>

#include "mfem.hpp"

#include "IRK.hpp"
#include "FDadvection.hpp"

using namespace mfem;
using namespace std;


bool GetError(const FDadvection &SpaceDisc, double tf, const HypreParVector &u, double &eL1, double &eL2, double &eLinf);
void SaveDict(string filename, map<string, string> mySolInfo);

/* SAMPLE RUNS:

4th-order in space & time
--- Constant coefficient advection ---
mpirun -np 4 --oversubscribe ./driver -l 7 -d 2 -t 14 -o 4 -FD 1 -nt 50 -save 1

--- Spatially variable coefficient advection ---
mpirun -np 4 --oversubscribe ./driver -l 7 -d 2 -t 14 -o 4 -FD 4 -nt 50 -save 1

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
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);


    /* ------------------------------------------------------ */
    /* --- Set default values for command-line parameters --- */
    /* ------------------------------------------------------ */
    /* --- Time integration --- */
    int nt     = 10;  // Number of time steps
    double dt  = -1.0;// Time step size
    double CFL = 5.0; // CFL number
    double tf  = -1.0; // Final integration time
    int IRK_ID = 01;  // IRK scheme

    /* --- Spatial discretization --- */
    int FD_ProblemID = 1;
    int order     = 1; // Approximation order
    int dim       = 1; // Spatial dimension
    int refLevels = 4; // nx == 2^refLevels in every dimension
    int ndis       = 2; // Degree of numerical dissipation (2 or 4)
    double ndis_c0 = 0.0; 
    int ndis_c1    = 0;  
    int px         = -1; // # procs in x-direction
    int py         = -1; // # procs in y-direction
    
    // AMG parameters
    AMG_parameters AMG_params;
    int AMG_maxit_type2 = AMG_params.maxiter;
    
    // Krylov parameters (just use GMRES)
    double reltol  = 1e-10;
    double abstol  = 1e-10;
    int maxiter    = 200;
    int kdim       = 30;
    int printlevel = 2;
    
    // Output paramaters
    int save         = 0;  // Save solution vector
    const char * out = "data/U"; // Filename of data to be saved...
    
    
    OptionsParser args(argc, argv);
    args.AddOption(&IRK_ID, "-t", "--RK-disc",
                  "Time discretization (see RK IDs).");
    args.AddOption(&nt, "-nt", "--num-time-steps", "Number of time steps.");    
                  
    /* Spatial discretization */
    args.AddOption(&order, "-o", "--order",
                  "FD: Approximation order."); 
    args.AddOption(&refLevels, "-l", "--level",
                  "FD: Mesh refinement; 2^refLevels DOFs in each dimension.");
    args.AddOption(&dim, "-d", "--dim",
                  "Spatial problem dimension.");
    args.AddOption(&px, "-px", "--procx",
                  "FD: Number of procs in x-direction.");
    args.AddOption(&py, "-py", "--procy",
                  "FD: Number of procs in y-direction.");                          
    args.AddOption(&ndis, "-ndis", "--num-dissipation-degree", 
                  "FD: Degree of numerical dissipation");
    args.AddOption(&ndis_c0, "-ndis_c0", "--num-dissipation-size0", 
                  "FD: Size of numerical dissipation is c0*dx^c1");
    args.AddOption(&ndis_c1, "-ndis_c1", "--num-dissipation-size1", 
                  "FD: Size of numerical dissipation is c0*dx^c1");

    /* FD-specific options */          
    args.AddOption(&FD_ProblemID, "-FD", "--FD-prob-ID",
                  "FD: Problem ID.");  
    
    /* CFL parameters */
    args.AddOption(&CFL, "-cfl", "--advection-CFL-number", "CFL==dt*max|a|/dx");
    args.AddOption(&tf, "-tf", "--final-integration-time", "0 <= t <= tf");
    args.AddOption(&dt, "-dt", "--time-step-size", "t_j = j*dt");

    /* AMG parameters */
    args.AddOption(&AMG_maxit_type2, "-maxit_AMG", "--max-AMG-iterations-type2", 
                        "Max AMG iterations for type 2 systems");
    args.AddOption(&AMG_params.relax_type, "-relax", "--AMG-relax-type", 
                        "Type of relaxation for AMG: Jacobi == 0");
                        
    /* Krylov parameters */
    args.AddOption(&reltol, "-rtol", "--rel-tol", "Krlov: Relative stopping tolerance");
    args.AddOption(&abstol, "-atol", "--abs-tol", "Krlov: Absolute stopping tolerance");
    args.AddOption(&maxiter, "-maxit", "--max-iterations", "Krlov: Maximum iterations");
    args.AddOption(&kdim, "-kdim", "--Krylov-dimension", "Krlov: Maximum subspace dimension");
    args.AddOption(&printlevel, "-kp", "--Krylov-print", "Krlov: Print level");
    
    /* --- Text output of solution etc --- */              
    args.AddOption(&out, "-out", "--out-directory", "Name of output file."); 
    args.AddOption(&save, "-save", "--save-sol-data",
                  "save>0 will save solution info, save>1 also saves solution.");              
    args.Parse();
    if (rank == 0) {
        args.PrintOptions(std::cout); 
    }
    
    
    /* --- Get FDadvection object --- */
    std::vector<int> n_px = {};
    if (px != -1) {
        if (dim >= 1) {
            n_px.push_back(px);
        }
        if (dim >= 2) {
            n_px.push_back(py);
        }
    }
    
    // Build Spatial discretization 
    FDadvection SpaceDisc(MPI_COMM_WORLD, dim, refLevels, order, FD_ProblemID, n_px); 
    
    // Add numerical dissipation into FD-advection discretization if meaningful parameters passed */
    if (ndis > 0 && ndis_c0 > 0.0) {
        Num_dissipation dissipation = {ndis, ndis_c0, ndis_c1};
        SpaceDisc.SetNumDissipation(dissipation);
    }
    
    // Set AMG options
    SpaceDisc.SetAMG_parameters(AMG_params, 1);
    AMG_params.maxiter = AMG_maxit_type2; // Update maxiter before passing
    SpaceDisc.SetAMG_parameters(AMG_params, 2); 
    
    // Get initial condition
    HypreParVector * u = NULL;
    SpaceDisc.GetU0(u);
    
    // Get mesh info
    double dx = SpaceDisc.Get_dx();
    int    nx = SpaceDisc.Get_nx();
    
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

    // Build IRK object using spatial discretization 
    IRK MyIRK(&SpaceDisc, static_cast<IRK::Type>(IRK_ID), MPI_COMM_WORLD);        
    
    // Initialize solver
    MyIRK.Init(SpaceDisc);
    
    // Set GMRES settings
    MyIRK.SetSolve(IRK::GMRES, reltol, maxiter, abstol, kdim, printlevel);

    // Time step 
    double t = 0.0;
    MyIRK.Run(*u, t, dt, tf);

    if (set_tf && (fabs(t-tf)>1e-15)) {
        if (rank == 0) std::cout << "WARNING: Requested tf of " << tf << " adjusted to " << t << '\n';
    }
    tf = t; // Update final time


    /* ----------------------- */
    /* --- Save solve data --- */
    /* ----------------------- */
    if (save > 0) {
        if (save > 1) u->Print(out);
        
        double eL1, eL2, eLinf = 0.0; 
        
        /* Get error against exact PDE solution if available */
        bool got_error = GetError(SpaceDisc, tf, *u, eL1, eL2, eLinf);
        
        // Save data to file enabling easier inspection of solution            
        if (rank == 0) {
            std::map<std::string, std::string> info;
    
            info["IRK"]             = to_string(IRK_ID);
            info["tf"]              = to_string(tf);
            info["dt"].resize(16);
            info["dt"].resize(snprintf(&info["dt"][0], 16, "%.6e", dt));
            info["nt"]              = to_string(nt);
    
            info["space_order"]     = to_string(order);
            info["nx"]              = to_string(nx);
            info["space_dim"]       = to_string(dim);
            info["space_refine"]    = to_string(refLevels);
            info["problemID"]       = to_string(FD_ProblemID);
            info["P"]               = to_string(numProcess);
            
            for (int d = 0; d < n_px.size(); d++) {
                info[string("p_x") + to_string(d)] = to_string(n_px[d]);
            }
    
            if (dx != -1.0) {
                info["dx"].resize(16);
                info["dx"].resize(snprintf(&info["dx"][0], 16, "%.6e", dx));
            }
            
            if (got_error) {
                info["eL1"].resize(16);
                info["eL2"].resize(16);
                info["eLinf"].resize(16);
                info["eL1"].resize(snprintf(&info["eL1"][0], 16, "%.6e", eL1));
                info["eL2"].resize(snprintf(&info["eL2"][0], 16, "%.6e", eL2));
                info["eLinf"].resize(snprintf(&info["eLinf"][0], 16, "%.6e", eLinf));
            } 
            
            
            /* Linear system/solve statistics */
            std::vector<int> avg_iter;
            std::vector<int> type;
            std::vector<double> eig_ratio;
            MyIRK.GetSolveStats(avg_iter, type, eig_ratio);
            for (int system = 0; system < avg_iter.size(); system++) {
                info[string("sys") + to_string(system+1) + "_iters"] = to_string(avg_iter[system]);
                info[string("sys") + to_string(system+1) + "_type"] = to_string(type[system]);
                info[string("sys") + to_string(system+1) + "_eig_ratio"] = to_string(eig_ratio[system]);
            }
            SaveDict(out, info);
        }
    }
    
    delete u;

    MPI_Finalize();
    return 0;
}


/* Get error against exact PDE solution if available */
bool GetError(const FDadvection &SpaceDisc, double tf, const HypreParVector &u, 
                double &eL1, double &eL2, double &eLinf) {
    HypreParVector * u_exact = NULL;
    
    if (SpaceDisc.GetUExact(tf, u_exact)) {
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
        double dx = SpaceDisc.Get_dx(0);
        eL1 *= dx;
        eL2 *= sqrt(dx);
        int dim = SpaceDisc.Get_dim();
        if (dim > 1) {
            double dy = SpaceDisc.Get_dx(1);
            eL1 *= dy;
            eL2 *= sqrt(dy);
        }
        
        return true;
    } else {
        return false;
    }
}

/* Print dictionary to file */
void SaveDict(string filename, map<string, string> mySolInfo) 
{
    ofstream solinfo;
    solinfo.open(filename);
    solinfo << scientific; // This means parameters will be printed with enough significant digits

    // Print out contents from additionalInfo to file too
    map<string, string>::iterator it;
    for (it = mySolInfo.begin(); it != mySolInfo.end(); it++) {
        solinfo << it->first << " " << it->second << "\n";
    }
    solinfo.close();
}