#include <iostream>
#include <fstream>

#include "mfem.hpp"

#include "IRK.hpp"
#include "FDadvection.hpp"

using namespace mfem;
using namespace std;


/* SAMPLE RUNS:

4th-order in space & time
--- Constant coefficient advection ---
mpirun -np 4 --oversubscribe ./driver -l 7 -d 2 -t 14 -o 4 -FD 1 -nt 50 -save 1

--- Spatially variable coefficient advection ---
mpirun -np 4 --oversubscribe ./driver -l 7 -d 2 -t 14 -o 4 -FD 4 -nt 50 -save 1

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
    int nt           = 10;
    int save         = 0;  // Save solution vector
    string out       = ""; // Filename of data to be saved...
    
    int IRK_ID       = 01;
    double dt        = -1;

    /* --- Spatial discretization parameters --- */
    int order        = 1;
    int dim          = 1;
    int refLevels    = 4;
    bool M_exists    = false;

    // Finite-difference specific parameters
    int FD_ProblemID = 1;
    int px           = -1;
    int py           = -1;
    int ndis         = 2;
    double ndis_c0   = 0.0;
    int ndis_c1      = 0;

    // CFL parameters
    double CFL = 5.0;
    
    OptionsParser args(argc, argv);
    args.AddOption(&IRK_ID, "-t", "--RK-disc",
                  "Time discretization (see RK IDs).");
    args.AddOption(&nt, "-nt", "--num-time-steps", "Number of time steps.");    
                  
    /* Spatial discretization */
    args.AddOption(&order, "-o", "--order",
                  "Spatial discretization order."); 
    args.AddOption(&refLevels, "-l", "--level",
                  "Number levels mesh refinement; FD uses 2^refLevels DOFs.");
    args.AddOption(&dim, "-d", "--dim",
                  "Problem dimension.");
    args.AddOption(&px, "-px", "--procx",
                  "FD: Number of procs in x-direction.");
    args.AddOption(&py, "-py", "--procy",
                  "FD: Number of procs in y-direction.");                          
    
    /* FD-specific options */          
    args.AddOption(&FD_ProblemID, "-FD", "--FD-prob-ID",
                  "FD: Problem ID.");  
    args.AddOption(&ndis, "-ndis", "--num-dissipation-degree", 
                  "Degree of numerical dissipation");
    args.AddOption(&ndis_c0, "-ndis_c0", "--num-dissipation-size0", 
                  "Size of numerical dissipation is c0*dx^c1");
    args.AddOption(&ndis_c1, "-ndis_c1", "--num-dissipation-size1", 
                  "Size of numerical dissipation is c0*dx^c1");

    /* CFL parameters */
    args.AddOption(&CFL, "-cfl", "...", "CFL==dt*max|a|/dx");

    /* --- Text output of solution etc --- */              
    //args.AddOption(&out, "-out", "--out",
    //              "Name of output file."); 
    args.AddOption(&save, "-save", "--save-sol-data",
                  "Level of information to save.");              
    args.Parse();
    if (rank == 0) {
        args.PrintOptions(std::cout); 
    }
    
    double dx, dy = -1.0;
    
    // Time step so that we run with dt=CFL*dx/max|a|
    if (dim == 1) {
        dx = 2.0 / pow(2.0, refLevels); // Assumes nx = 2^refLevels, and x \in [-1,1] 
        dt = dx * CFL; // assume max|a| = 1
    } else if (dim == 2) {
        dx = 2.0 / pow(2.0, refLevels); // Assumes nx = 2^refLevels, and x \in [-1,1] 
        dy = dx;
        dt = CFL/(1/dx + 1/dy);
    }
    
    
    // // Manually set time to integrate to (just comment or uncomment this...)
    // double T = 2.0; // For 1D in space...
    // //double T = 2.0  * dt; // For 1D in space...
    // if (dim == 2) T = 0.5; // For 2D in space...
    // 
    // // Time step so that we run at approximately CFL_fraction of CFL limit, but integrate exactly up to T
    // nt = ceil(T / dt);
    // //nt = 6;
    // //dt = T / (nt - 1);
    
    /* --- Get SPACETIMEMATRIX object --- */
    std::vector<int> n_px = {};
    if (px != -1) {
        if (dim >= 1) {
            n_px.push_back(px);
        }
        if (dim >= 2) {
            n_px.push_back(py);
        }
    }
    
    // Build Spatial discretization object
    FDadvection SpaceDisc(MPI_COMM_WORLD, dim, refLevels, order, FD_ProblemID, n_px); 
    
    // Add numerical dissipation into FD-advection discretization if meaningful parameters passed */
    if (ndis > 0 && ndis_c0 > 0.0) {
        Num_dissipation dissipation = {ndis, ndis_c0, ndis_c1};
        SpaceDisc.SetNumDissipation(dissipation);
    }
    
    // Get initial condition
    HypreParVector * u = NULL;
    SpaceDisc.GetU0(u);
    
    // Build IRK object using spatial discretization object
    IRK MyIRK(&SpaceDisc, static_cast<IRK::Type>(IRK_ID), MPI_COMM_WORLD);
    
    // Time step
    double t0 = 0.0;
    double tf = dt*nt;
    
    MyIRK.Init(SpaceDisc);
    MyIRK.Run(*u, t0, dt, tf);
    
    if (save > 0) {
        // char * filename;
        // if (std::string(out) == "") {
        //     char * filename = "data/U"; // Default file name
        // } else {
        //     filename = out;
        // }
    
        const char * fname = "data/U";
    
        //SpaceDisc.SaveU(fname); 
    
        u->Print(fname);
    
        // Save data to file enabling easier inspection of solution            
        if (rank == 0) {
            int nx = pow(2, refLevels);
            std::map<std::string, std::string> space_info;
    
            space_info["nt"]              = std::to_string(nt);
    
            space_info["space_order"]     = std::to_string(order);
            space_info["nx"]              = std::to_string(nx);
            space_info["space_dim"]       = std::to_string(dim);
            space_info["space_refine"]    = std::to_string(refLevels);
            space_info["problemID"]       = std::to_string(FD_ProblemID);
            for (int d = 0; d < n_px.size(); d++) {
                space_info[std::string("p_x") + std::to_string(d)] = std::to_string(n_px[d]);
            }
    
            // // Not sure how else to ensure disc error is cast to a string in scientific format...
            // if (gotdiscerror) {
            //     space_info["discerror"].resize(16);
            //     space_info["discerror"].resize(std::snprintf(&space_info["discerror"][0], 16, "%.6e", discerror));
            // } 
    
            space_info["P"]     = std::to_string(numProcess);
    
            if (dx != -1.0) {
                space_info["dx"].resize(16);
                space_info["dx"].resize(std::snprintf(&space_info["dx"][0], 16, "%.6e", dx));
            }
            
            space_info["dt"].resize(16);
            space_info["dt"].resize(std::snprintf(&space_info["dt"][0], 16, "%.6e", dt));
    
            MyIRK.SaveSolInfo(fname, space_info); // TODO write me
        }
    }
    

    MPI_Finalize();
    return 0;
}