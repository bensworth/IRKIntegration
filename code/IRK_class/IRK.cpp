#include "IRK.hpp"
#include <iostream>
#include <fstream>
#include <map>
#include <iomanip> 
#include <cmath> 


/* TODO:
 - Neaten up FD code...
 - Probable memory leak w/ J in FDadvection. Search TODO[seg] in FDadvection.cpp.
 
OK questions for BS (and general thoughts):
    - Do we want to add other Butcher tableaux than are already there?
        + I think we have plenty right now, Gauss and Radau were the important
        ones, and L-stable SDIRK to compare.

    - You mentioned previously computing the eigenvalues of A using 
    MFEM, but Will said this would require compiling MFEM with LAPACK enabled.
    Did you still want to move in this direction, or just do what we have at the
    moment? I'm just wondering because at the moment we've implemented A0
    for every method, but we don't actually need to use A0 since we precompute
    inv(A0) and d0 == A0^{-T}*b0. (Also, can we compute inv(A0) w/ MFEM?)
        + inv(A0) can in principle be computed in MFEM w/o LAPACK I believe.
        Eigenvalues require lapack. The reason I precomputed everything was
        if we want to get 8th or 10th order accuracy, I think it is really
        important those coefficients are really accurate. So in Mathematica,
        I used like 40 digits of precision for the inverse, eigenvalues, etc.,
        then truncated to 17 or 18 to put in here in decimal form.
            I think for now, let's stick with this precomputation. If and when
        we try to get it in MFEM master we will defer to how they want it
        done. I'm finding right now with a PR for AIR to MFEM they are kind
        of picky and also seem to prefer having stuff builtin rather than too
        many options for the user, so probably easiest to wait on that. For
        our paper purposes, we have everything we need. 
        
    - The coefficients of the polynomials defined by (d0 \otimes I) * adj(M_s)
    are very lengthy for moderate s (e.g., see how long they are for s = 5 in IRK::SetXCoeffs()).
    I don't really know if this is an issue or not, e.g., I doubt it actually
    takes much time to execute these expressions (which is done once only once at the 
    start of the run), but depending on what direction you want to move in, there's 
    the option of precomputing and storing these for each method (then we don't have 
    to store A, A^{-1} and d0)
        + It's an interesting idea. The computation is of fairly trivial expense and
        memory use, so it's not really a big deal. It might be more accurate if we
        compute those in very high precision in mathematica a priori though. And I
        guess right now we're in between - half the code computes at run time and
        could be provided an arbitrary tableaux (here) and the other half uses
        built in precomputed values (storing A0 and d). I am thinking we should keep
        the analytic form somewhere (e.g., in a branch), but maybe supply the
        coefficientes precomputed in our main development? 
        
    - Just a thought relating to the above in terms of the direction of/generalizing the code. 
    It'd be nice if we have a bunch of Butcher tableaux implemented, but also  
    allow the user to provide their own, so maybe we can just have a Python script
    (or MATLAB, but maybe not since it's not open source), where the user can provide 
    their own tableau, then the script computes the relevant data from it 
    (eigenvalues and coefficients of {X_j}), then we just read this into the C++
    code at run time?
        + Yea, I think that's a good idea. Easy to add a funciton that just takes
        those coefficients. Wonder if we can compute in very high precision in python?
        As I mentioned before, I think for high order schemes, we want inverses and
        eigenvalues as accurate as possible. I would say put this lower on the 
        priority list though, more of something to provide w/ a paper when we
        get to seriously writing.  
*/


IRK::IRK(IRKOperator *S, IRK::Type RK_ID, MPI_Comm comm)
        : m_S(S), m_CharPolyPrec(*S), m_CharPolyOps(), m_comm{comm},
        m_krylov(NULL)
{
    m_RK_ID = static_cast<int>(RK_ID);

    // Get proc IDs
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numProcess);
    
    /* Set up basic information */
    SetButcherData();  // Set Butcher tableau and associated data
    SetXCoeffs();      // Set coefficients of polynomials {X_j}
    
    /* --- Construct object for every REAL factor in char. poly --- */
    m_CharPolyOps.SetSize(m_zeta.Size() + m_eta.Size()); 
    // Linear factors. I.e., those of type 1
    double dt_dummy = -1.0; // Use dummy dt, will be set properly before inverting factor.
    int count = 0;
    for (int i = 0; i < m_zeta.Size(); i++) {
        m_CharPolyOps[count] = new CharPolyOp(dt_dummy, m_zeta(i), *m_S);
        count++;
    }
    // Quadratic factors. I.e., those of type 2
    for (int i = 0; i < m_eta.Size(); i++) {
        m_CharPolyOps[count] = new CharPolyOp(dt_dummy, m_eta(i), m_beta(i), *m_S);
        count++;
    } 
}

IRK::~IRK() {
    for (int i = 0; i < m_CharPolyOps.Size(); i++) delete m_CharPolyOps[i];    
}


void IRK::SetSolve(IRK::Solve solveID, double reltol, int maxiter,
                   double abstol, int kdim, int printlevel)
{
    m_solveID = static_cast<int>(solveID);
    // CG
    if (m_solveID == 0) {
        m_krylov = new CGSolver(m_comm);
    }
    // MINRES
    else if (m_solveID == 1) {
        m_krylov = new MINRESSolver(m_comm);
    }
    // GMRES
    else if (m_solveID == 2) {
        m_krylov = new GMRESSolver(m_comm);
        static_cast<GMRESSolver*>(m_krylov)->SetKDim(kdim);
    }
    // BiCGStab
    else if (m_solveID == 3) { 
        m_krylov = new BiCGSTABSolver(m_comm);
    }
    // GGMRES
    else if (m_solveID == 4) {
        m_krylov = new FGMRESSolver(m_comm);
        static_cast<FGMRESSolver*>(m_krylov)->SetKDim(kdim);
    }
    else {
        mfem_error("IRK::Invalid solve type.\n");   
    }

    m_krylov->SetRelTol(reltol);
    m_krylov->SetAbsTol(abstol);
    m_krylov->SetMaxIter(maxiter);
    m_krylov->SetPrintLevel(printlevel);
    m_krylov->iterative_mode = false;
}


// Call base class' init and size member vectors
void IRK::Init(TimeDependentOperator &F)
{
    ODESolver::Init(F);
    m_y.SetSize(F.Height(), mem_type);
    m_z.SetSize(F.Height(), mem_type);
    m_w.SetSize(F.Height(), mem_type);
    m_f.SetSize(F.Height(), mem_type);
}

void IRK::Step(Vector &x, double &t, double &dt)
{
    ConstructRHS(x, t, dt, m_z); /* Set m_z */
    // Scale RHS of system by M
    if (m_S->m_M_exists) {
        m_S->ImplicitMult(m_w, m_z);
        m_z = m_w;
    }   
    
    /* Sequentially invert factors in characteristic polynomial */
    for (int i = 0; i < m_CharPolyOps.Size(); i++) {
        if (m_rank == 0) {
            std::cout << "System " << i << " of " << m_CharPolyOps.Size()-1 <<
            ";\t type = " << m_CharPolyOps[i]->Type() << "\n";
        }
        
        // Ensure that correct time step is used in factored polynomial
        m_CharPolyOps[i]->Setdt(dt);

        // Set operator and preconditioner for ith polynomial term
        m_CharPolyPrec.SetType(m_CharPolyOps[i]->Type());
        m_S->SetSystem(i, t, dt, m_CharPolyOps[i]->Gamma(), m_CharPolyOps[i]->Type());
        m_krylov->SetPreconditioner(m_CharPolyPrec);
        m_krylov->SetOperator(*(m_CharPolyOps[i]));    
                
        // Use preconditioned Krylov to invert ith factor in polynomial 
        m_krylov->Mult(m_z, m_y); // y <- FACTOR(i)^{-1} * z
        
        // Solution becomes the RHS for the next factor
        if (i < m_CharPolyOps.Size()-1) {
            // Scale RHS of new system by M        
            if (m_S->m_M_exists) {
                m_S->ImplicitMult(m_y, m_z);
            } else {
                m_z = m_y; 
            }
        }
    }

    // Update solution vector with weighted sum of stage vectors        
    x.Add(dt, m_y);  // x <- x + dt*y
    t += dt;         // Time that x is evaulated at
}

void IRK::Run(Vector &x, double &t, double &dt, double tf) 
{    
    // Set Krylov settings if not already set
    if (!m_krylov) SetSolve();

    /* Main time-stepping loop */
    int step = 0;
    int numsteps = int((tf-t)/dt);
    while (t < tf) {
        step++;
        if (m_rank == 0) std::cout << "RK time-step " << step << " of " << numsteps << '\n';

        // Step from t to t+dt
        Step(x, t, dt);
    }
}



/* Construct z, the RHS of the linear system for integration from t to t+dt */
void IRK::ConstructRHS(const Vector &x, double t, double dt, Vector &z) {
    m_z = 0.0; /* z <- 0 */
    
    for (int i = 0; i < m_s; i++) {
        m_S->SetTime(t + dt*m_c0(i)); // Set time of S to t+dt*c(i)
        m_S->Mult(x, m_f); // f <- M^{-1}*[L*x + g(t)] _OR_ f <- L*x + g(t)
        m_S->PolynomialMult(m_XCoeffs[i], dt, m_f, m_w); /* w <- X_i(dt*M^{-1}*L)*f _OR_ w <- X_i(dt*L)*f */
        z += m_w;
    }
}



/* Set dimensions of Butcher arrays after setting m_s. */
void IRK::SizeButcherData(int nRealEigs, int nCCEigs) {
    m_A0.SetSize(m_s);   
    m_invA0.SetSize(m_s);
    m_b0.SetSize(m_s);
    m_c0.SetSize(m_s);
    m_d0.SetSize(m_s);    
    m_zeta.SetSize(nRealEigs);
    m_beta.SetSize(nCCEigs);
    m_eta.SetSize(nCCEigs);
}



/* Print data to file that allows one to extract the 
    relevant data for plotting, etc. from the saved solution. Pass
    in dictionary with information also to be saved to file that isn't a member 
    variable (e.g. space disc info)
*/
void IRK::SaveSolInfo(string filename, map<string, string> additionalInfo) 
{
    ofstream solinfo;
    solinfo.open(filename);
    solinfo << scientific; // This means parameters will be printed with enough significant digits
    // solinfo << "nt " << m_nt << "\n";
    // solinfo << "dt " << m_dt << "\n";
    
    // Time-discretization-specific information
    solinfo << "timeDisc " << m_RK_ID << "\n";
    
    // Print out contents from additionalInfo to file too
    map<string, string>::iterator it;
    for (it=additionalInfo.begin(); it!=additionalInfo.end(); it++) {
        solinfo << it->first << " " << it->second << "\n";
    }
    solinfo.close();
}


/* Set Butcher tableau and associated parameters */
    // // Implicit Runge Kutta type. Enumeration:
    // //  First digit: group of schemes
    // //  + 0 = L-stable SDIRK
    // //  + 1 = Gauss-Legendre
    // //  + 2 = RadauIIA
    // //  + 3 = Lobatto IIIC
    // //  Second digit: order of scheme
    // enum Type { 
    //     SDIRK1 = 01, SDIRK2 = 02, SDIRK3 = 03, SDIRK4 = 04,
    //     Gauss4 = 14, Gauss6 = 16, Gauss8 = 18, Gauss10 = 110,
    //     RadauIIA3 = 23, RadauIIA5 = 25, RadauIIA7 = 27, RadauIIA9 = 29,
    //     LobIIIC2 = 32, LobIIIC4 = 34, LobIIIC6 = 36, LobIIIC8 = 38
    // };
void IRK::SetButcherData() {
    // 1-stage 1st-order L-stable SDIRK 
    if (m_RK_ID == 1) {
        /* ID: SDIRK1 */
        m_s = 1;
        SizeButcherData(1, 0);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +1.000000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +1.000000000000000;
        /* --- b --- */
        m_b0(0) = +1.000000000000000;
        /* --- c --- */
        m_c0(0) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +1.000000000000000;
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */        
    }
    // 2stage 2nd-order L-stable SDIRK
    else if (m_RK_ID == 2) {
        /* ID: SDIRK2 */
        m_s = 2;
        SizeButcherData(2, 0);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.292893218813453;
        m_A0(0, 1) = +0.000000000000000;
        m_A0(1, 0) = +0.414213562373095;
        m_A0(1, 1) = +0.292893218813453;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +3.414213562373094;
        m_invA0(0, 1) = +0.000000000000000;
        m_invA0(1, 0) = -4.828427124746186;
        m_invA0(1, 1) = +3.414213562373094;
        /* --- b --- */
        m_b0(0) = +0.500000000000000;
        m_b0(1) = +0.500000000000000;
        /* --- c --- */
        m_c0(0) = +0.292893218813453;
        m_c0(1) = +0.707106781186547;
        /* --- d --- */
        m_d0(0) = -0.707106781186546;
        m_d0(1) = +1.707106781186547;
        /* --- zeta --- */
        m_zeta(0) = +3.414213562373094;
        m_zeta(1) = +3.414213562373094;
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */        
    }
    // 3-stage 3rd-order L-stable SDIRK
    else if (m_RK_ID == 3) {
        /* ID: SDIRK3 */
        m_s = 3;
        SizeButcherData(3, 0);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.435866521508459;
        m_A0(0, 1) = +0.000000000000000;
        m_A0(0, 2) = +0.000000000000000;
        m_A0(1, 0) = +0.282066739245771;
        m_A0(1, 1) = +0.435866521508459;
        m_A0(1, 2) = +0.000000000000000;
        m_A0(2, 0) = +1.208496649176010;
        m_A0(2, 1) = -0.644363170684469;
        m_A0(2, 2) = +0.435866521508459;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +2.294280360279042;
        m_invA0(0, 1) = -0.000000000000000;
        m_invA0(0, 2) = -0.000000000000000;
        m_invA0(1, 0) = -1.484721005641544;
        m_invA0(1, 1) = +2.294280360279042;
        m_invA0(1, 2) = +0.000000000000000;
        m_invA0(2, 0) = -8.556127801552645;
        m_invA0(2, 1) = +3.391748836942547;
        m_invA0(2, 2) = +2.294280360279042;
        /* --- b --- */
        m_b0(0) = +1.208496649176010;
        m_b0(1) = -0.644363170684469;
        m_b0(2) = +0.435866521508459;
        /* --- c --- */
        m_c0(0) = +0.435866521508459;
        m_c0(1) = +0.717933260754229;
        m_c0(2) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +2.294280360279042;
        m_zeta(1) = +2.294280360279042;
        m_zeta(2) = +2.294280360279042;
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */        
    }
    // 5-stage 4th-order L-stable SDIRK
    else if (m_RK_ID == 4) {
        /* ID: SDIRK4 */
        m_s = 5;
        SizeButcherData(5, 0);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.250000000000000;
        m_A0(0, 1) = +0.000000000000000;
        m_A0(0, 2) = +0.000000000000000;
        m_A0(0, 3) = +0.000000000000000;
        m_A0(0, 4) = +0.000000000000000;
        m_A0(1, 0) = +0.500000000000000;
        m_A0(1, 1) = +0.250000000000000;
        m_A0(1, 2) = +0.000000000000000;
        m_A0(1, 3) = +0.000000000000000;
        m_A0(1, 4) = +0.000000000000000;
        m_A0(2, 0) = +0.340000000000000;
        m_A0(2, 1) = -0.040000000000000;
        m_A0(2, 2) = +0.250000000000000;
        m_A0(2, 3) = +0.000000000000000;
        m_A0(2, 4) = +0.000000000000000;
        m_A0(3, 0) = +0.272794117647059;
        m_A0(3, 1) = -0.050367647058824;
        m_A0(3, 2) = +0.027573529411765;
        m_A0(3, 3) = +0.250000000000000;
        m_A0(3, 4) = +0.000000000000000;
        m_A0(4, 0) = +1.041666666666667;
        m_A0(4, 1) = -1.020833333333333;
        m_A0(4, 2) = +7.812500000000000;
        m_A0(4, 3) = -7.083333333333333;
        m_A0(4, 4) = +0.250000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +4.000000000000000;
        m_invA0(0, 1) = +0.000000000000000;
        m_invA0(0, 2) = +0.000000000000000;
        m_invA0(0, 3) = +0.000000000000000;
        m_invA0(0, 4) = +0.000000000000000;
        m_invA0(1, 0) = -7.999999999999995;
        m_invA0(1, 1) = +3.999999999999998;
        m_invA0(1, 2) = +0.000000000000001;
        m_invA0(1, 3) = -0.000000000000003;
        m_invA0(1, 4) = +0.000000000000000;
        m_invA0(2, 0) = -6.719999999999998;
        m_invA0(2, 1) = +0.639999999999999;
        m_invA0(2, 2) = +4.000000000000001;
        m_invA0(2, 3) = -0.000000000000002;
        m_invA0(2, 4) = -0.000000000000000;
        m_invA0(3, 0) = -5.235294117647047;
        m_invA0(3, 1) = +0.735294117647057;
        m_invA0(3, 2) = -0.441176470588240;
        m_invA0(3, 3) = +3.999999999999998;
        m_invA0(3, 4) = +0.000000000000000;
        m_invA0(4, 0) = +12.333333333333645;
        m_invA0(4, 1) = +17.166666666666643;
        m_invA0(4, 2) = -137.500000000000171;
        m_invA0(4, 3) = +113.333333333333329;
        m_invA0(4, 4) = +4.000000000000000;
        /* --- b --- */
        m_b0(0) = +1.041666666666667;
        m_b0(1) = -1.020833333333333;
        m_b0(2) = +7.812500000000000;
        m_b0(3) = -7.083333333333333;
        m_b0(4) = +0.250000000000000;
        /* --- c --- */
        m_c0(0) = +0.250000000000000;
        m_c0(1) = +0.750000000000000;
        m_c0(2) = +0.550000000000000;
        m_c0(3) = +0.500000000000000;
        m_c0(4) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +0.000000000000000;
        m_d0(3) = +0.000000000000000;
        m_d0(4) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +4.000000000000000;
        m_zeta(1) = +4.000000000000000;
        m_zeta(2) = +4.000000000000000;
        m_zeta(3) = +4.000000000000000;
        m_zeta(4) = +4.000000000000000;
        /* --- eta --- */
        /* --- beta --- */
        /* -------------------------- */        
    }
    // 2-stage 4th-order Gauss--Legendre
    else if (m_RK_ID == 14) {
        /* ID: Gauss4 */
        m_s = 2;
        SizeButcherData(0, 1);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.250000000000000;
        m_A0(0, 1) = -0.038675134594813;
        m_A0(1, 0) = +0.538675134594813;
        m_A0(1, 1) = +0.250000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +3.000000000000000;
        m_invA0(0, 1) = +0.464101615137754;
        m_invA0(1, 0) = -6.464101615137755;
        m_invA0(1, 1) = +3.000000000000000;
        /* --- b --- */
        m_b0(0) = +0.500000000000000;
        m_b0(1) = +0.500000000000000;
        /* --- c --- */
        m_c0(0) = +0.211324865405187;
        m_c0(1) = +0.788675134594813;
        /* --- d --- */
        m_d0(0) = -1.732050807568877;
        m_d0(1) = +1.732050807568877;
        /* --- zeta --- */
        /* --- eta --- */
        m_eta(0) = +3.000000000000000;
        /* --- beta --- */
        m_beta(0) = +1.732050807568877;
        /* -------------------------- */        
    }
    // 3-stage 6th-order Gauss--Legendre
    else if (m_RK_ID == 16) {
        /* ID: Gauss6 */
        m_s = 3;
        SizeButcherData(1, 1);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.138888888888889;
        m_A0(0, 1) = -0.035976667524939;
        m_A0(0, 2) = +0.009789444015308;
        m_A0(1, 0) = +0.300263194980865;
        m_A0(1, 1) = +0.222222222222222;
        m_A0(1, 2) = -0.022485417203087;
        m_A0(2, 0) = +0.267988333762469;
        m_A0(2, 1) = +0.480421111969383;
        m_A0(2, 2) = +0.138888888888889;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +4.999999999999997;
        m_invA0(0, 1) = +1.163977794943223;
        m_invA0(0, 2) = -0.163977794943222;
        m_invA0(1, 0) = -5.727486121839513;
        m_invA0(1, 1) = +2.000000000000000;
        m_invA0(1, 2) = +0.727486121839514;
        m_invA0(2, 0) = +10.163977794943225;
        m_invA0(2, 1) = -9.163977794943223;
        m_invA0(2, 2) = +5.000000000000001;
        /* --- b --- */
        m_b0(0) = +0.277777777777778;
        m_b0(1) = +0.444444444444444;
        m_b0(2) = +0.277777777777778;
        /* --- c --- */
        m_c0(0) = +0.112701665379258;
        m_c0(1) = +0.500000000000000;
        m_c0(2) = +0.887298334620742;
        /* --- d --- */
        m_d0(0) = +1.666666666666668;
        m_d0(1) = -1.333333333333334;
        m_d0(2) = +1.666666666666667;
        /* --- zeta --- */
        m_zeta(0) = +4.644370709252176;
        /* --- eta --- */
        m_eta(0) = +3.677814645373916;
        /* --- beta --- */
        m_beta(0) = +3.508761919567443;
        /* -------------------------- */  
    }    
    // 4-stage 8th-order Gauss--Legendre
    else if (m_RK_ID == 18) {
        /* ID: Gauss8 */
        m_s = 4;
        SizeButcherData(0, 2);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.086963711284363;
        m_A0(0, 1) = -0.026604180084999;
        m_A0(0, 2) = +0.012627462689405;
        m_A0(0, 3) = -0.003555149685796;
        m_A0(1, 0) = +0.188118117499868;
        m_A0(1, 1) = +0.163036288715637;
        m_A0(1, 2) = -0.027880428602471;
        m_A0(1, 3) = +0.006735500594538;
        m_A0(2, 0) = +0.167191921974189;
        m_A0(2, 1) = +0.353953006033744;
        m_A0(2, 2) = +0.163036288715637;
        m_A0(2, 3) = -0.014190694931141;
        m_A0(3, 0) = +0.177482572254523;
        m_A0(3, 1) = +0.313445114741868;
        m_A0(3, 2) = +0.352676757516272;
        m_A0(3, 3) = +0.086963711284363;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +7.738612787525829;
        m_invA0(0, 1) = +2.045089650303909;
        m_invA0(0, 2) = -0.437070802395799;
        m_invA0(0, 3) = +0.086644023503262;
        m_invA0(1, 0) = -7.201340999706891;
        m_invA0(1, 1) = +2.261387212474169;
        m_invA0(1, 2) = +1.448782034533682;
        m_invA0(1, 3) = -0.233133981212417;
        m_invA0(2, 0) = +6.343622218624971;
        m_invA0(2, 1) = -5.971556459482020;
        m_invA0(2, 2) = +2.261387212474169;
        m_invA0(2, 3) = +1.090852762294336;
        m_invA0(3, 0) = -15.563869598554923;
        m_invA0(3, 1) = +11.892783878056845;
        m_invA0(3, 2) = -13.500802725964952;
        m_invA0(3, 3) = +7.738612787525829;
        /* --- b --- */
        m_b0(0) = +0.173927422568727;
        m_b0(1) = +0.326072577431273;
        m_b0(2) = +0.326072577431273;
        m_b0(3) = +0.173927422568727;
        /* --- c --- */
        m_c0(0) = +0.069431844202974;
        m_c0(1) = +0.330009478207572;
        m_c0(2) = +0.669990521792428;
        m_c0(3) = +0.930568155797026;
        /* --- d --- */
        m_d0(0) = -1.640705321739257;
        m_d0(1) = +1.214393969798579;
        m_d0(2) = -1.214393969798578;
        m_d0(3) = +1.640705321739257;
        /* --- zeta --- */
        /* --- eta --- */
        m_eta(0) = +4.207578794359256;
        m_eta(1) = +5.792421205640745;
        /* --- beta --- */
        m_beta(0) = +5.314836083713509;
        m_beta(1) = +1.734468257869003;
        /* -------------------------- */        
    }    
    // 5-stage 10th-order Gauss--Legendre
    else if (m_RK_ID == 110) {
        /* ID: Gauss10 */
        m_s = 5;
        SizeButcherData(1, 2);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.059231721264047;
        m_A0(0, 1) = -0.019570364359076;
        m_A0(0, 2) = +0.011254400818643;
        m_A0(0, 3) = -0.005593793660812;
        m_A0(0, 4) = +0.001588112967866;
        m_A0(1, 0) = +0.128151005670045;
        m_A0(1, 1) = +0.119657167624842;
        m_A0(1, 2) = -0.024592114619642;
        m_A0(1, 3) = +0.010318280670683;
        m_A0(1, 4) = -0.002768994398770;
        m_A0(2, 0) = +0.113776288004225;
        m_A0(2, 1) = +0.260004651680642;
        m_A0(2, 2) = +0.142222222222222;
        m_A0(2, 3) = -0.020690316430958;
        m_A0(2, 4) = +0.004687154523870;
        m_A0(3, 0) = +0.121232436926864;
        m_A0(3, 1) = +0.228996054579000;
        m_A0(3, 2) = +0.309036559064087;
        m_A0(3, 3) = +0.119657167624842;
        m_A0(3, 4) = -0.009687563141951;
        m_A0(4, 0) = +0.116875329560229;
        m_A0(4, 1) = +0.244908128910495;
        m_A0(4, 2) = +0.273190043625802;
        m_A0(4, 3) = +0.258884699608759;
        m_A0(4, 4) = +0.059231721264047;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +11.183300132670379;
        m_invA0(0, 1) = +3.131312162011809;
        m_invA0(0, 2) = -0.758731795980806;
        m_invA0(0, 3) = +0.239101223353686;
        m_invA0(0, 4) = -0.054314760565339;
        m_invA0(1, 0) = -9.447599601516151;
        m_invA0(1, 1) = +2.816699867329624;
        m_invA0(1, 2) = +2.217886332274817;
        m_invA0(1, 3) = -0.557122620293796;
        m_invA0(1, 4) = +0.118357949604667;
        m_invA0(2, 0) = +6.420116503559337;
        m_invA0(2, 1) = -6.220120454669752;
        m_invA0(2, 2) = +2.000000000000000;
        m_invA0(2, 3) = +1.865995288831779;
        m_invA0(2, 4) = -0.315991337721364;
        m_invA0(3, 0) = -8.015920784810973;
        m_invA0(3, 1) = +6.190522354953043;
        m_invA0(3, 2) = -7.393116276382382;
        m_invA0(3, 3) = +2.816699867329624;
        m_invA0(3, 4) = +1.550036766309846;
        m_invA0(4, 0) = +22.420915025906101;
        m_invA0(4, 1) = -16.193390239999243;
        m_invA0(4, 2) = +15.415443221569852;
        m_invA0(4, 3) = -19.085601178657356;
        m_invA0(4, 4) = +11.183300132670379;
        /* --- b --- */
        m_b0(0) = +0.118463442528095;
        m_b0(1) = +0.239314335249683;
        m_b0(2) = +0.284444444444444;
        m_b0(3) = +0.239314335249683;
        m_b0(4) = +0.118463442528095;
        /* --- c --- */
        m_c0(0) = +0.046910077030668;
        m_c0(1) = +0.230765344947158;
        m_c0(2) = +0.500000000000000;
        m_c0(3) = +0.769234655052841;
        m_c0(4) = +0.953089922969332;
        /* --- d --- */
        m_d0(0) = +1.627766710890126;
        m_d0(1) = -1.161100044223459;
        m_d0(2) = +1.066666666666666;
        m_d0(3) = -1.161100044223459;
        m_d0(4) = +1.627766710890126;
        /* --- zeta --- */
        m_zeta(0) = +7.293477190659289;
        /* --- eta --- */
        m_eta(0) = +4.649348606363296;
        m_eta(1) = +6.703912798307074;
        /* --- beta --- */
        m_beta(0) = +7.142045840675958;
        m_beta(1) = +3.485322832366397;
        /* -------------------------- */        
    }
    // 2-stage 3rd-order Radau IIA
    else if (m_RK_ID == 23) {
        /* ID: RadauIIA3 */
        m_s = 2;
        SizeButcherData(0, 1);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.416666666666667;
        m_A0(0, 1) = -0.083333333333333;
        m_A0(1, 0) = +0.750000000000000;
        m_A0(1, 1) = +0.250000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +1.500000000000000;
        m_invA0(0, 1) = +0.500000000000000;
        m_invA0(1, 0) = -4.500000000000002;
        m_invA0(1, 1) = +2.500000000000000;
        /* --- b --- */
        m_b0(0) = +0.750000000000000;
        m_b0(1) = +0.250000000000000;
        /* --- c --- */
        m_c0(0) = +0.333333333333333;
        m_c0(1) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = -0.000000000000000;
        m_d0(1) = +1.000000000000000;
        /* --- zeta --- */
        /* --- eta --- */
        m_eta(0) = +2.000000000000000;
        /* --- beta --- */
        m_beta(0) = +1.414213562373095;
        /* -------------------------- */        
    }
    // 3-stage 5th-order Radau IIA
    else if (m_RK_ID == 25) {
        /* ID: RadauIIA5 */
        m_s = 3;
        SizeButcherData(1, 1);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.196815477223660;
        m_A0(0, 1) = -0.065535425850198;
        m_A0(0, 2) = +0.023770974348220;
        m_A0(1, 0) = +0.394424314739087;
        m_A0(1, 1) = +0.292073411665228;
        m_A0(1, 2) = -0.041548752125998;
        m_A0(2, 0) = +0.376403062700467;
        m_A0(2, 1) = +0.512485826188422;
        m_A0(2, 2) = +0.111111111111111;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +3.224744871391591;
        m_invA0(0, 1) = +1.167840084690405;
        m_invA0(0, 2) = -0.253197264742181;
        m_invA0(1, 0) = -3.567840084690407;
        m_invA0(1, 1) = +0.775255128608411;
        m_invA0(1, 2) = +1.053197264742181;
        m_invA0(2, 0) = +5.531972647421806;
        m_invA0(2, 1) = -7.531972647421810;
        m_invA0(2, 2) = +5.000000000000001;
        /* --- b --- */
        m_b0(0) = +0.376403062700467;
        m_b0(1) = +0.512485826188422;
        m_b0(2) = +0.111111111111111;
        /* --- c --- */
        m_c0(0) = +0.155051025721682;
        m_c0(1) = +0.644948974278318;
        m_c0(2) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +3.637834252744494;
        /* --- eta --- */
        m_eta(0) = +2.681082873627752;
        /* --- beta --- */
        m_beta(0) = +3.050430199247414;
        /* -------------------------- */        
    }
    // 4-stage 7th-order Radau IIA
    else if (m_RK_ID == 27) {
        /* ID: RadauIIA7 */
        m_s = 4;
        SizeButcherData(0, 2);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.112999479323156;
        m_A0(0, 1) = -0.040309220723522;
        m_A0(0, 2) = +0.025802377420336;
        m_A0(0, 3) = -0.009904676507266;
        m_A0(1, 0) = +0.234383995747400;
        m_A0(1, 1) = +0.206892573935359;
        m_A0(1, 2) = -0.047857128048541;
        m_A0(1, 3) = +0.016047422806516;
        m_A0(2, 0) = +0.216681784623250;
        m_A0(2, 1) = +0.406123263867373;
        m_A0(2, 2) = +0.189036518170056;
        m_A0(2, 3) = -0.024182104899833;
        m_A0(3, 0) = +0.220462211176768;
        m_A0(3, 1) = +0.388193468843172;
        m_A0(3, 2) = +0.328844319980060;
        m_A0(3, 3) = +0.062500000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +5.644107875950090;
        m_invA0(0, 1) = +1.923507277054713;
        m_invA0(0, 2) = -0.585901482103817;
        m_invA0(0, 3) = +0.173878352574246;
        m_invA0(1, 0) = -5.049214638391410;
        m_invA0(1, 1) = +1.221100028894693;
        m_invA0(1, 2) = +1.754680988760837;
        m_invA0(1, 3) = -0.434791461212582;
        m_invA0(2, 0) = +3.492466158625437;
        m_invA0(2, 1) = -3.984517895782496;
        m_invA0(2, 2) = +0.634792095155218;
        m_invA0(2, 3) = +1.822137598434254;
        m_invA0(3, 0) = -6.923488256445451;
        m_invA0(3, 1) = +6.595237669628144;
        m_invA0(3, 2) = -12.171749413182692;
        m_invA0(3, 3) = +8.500000000000002;
        /* --- b --- */
        m_b0(0) = +0.220462211176768;
        m_b0(1) = +0.388193468843172;
        m_b0(2) = +0.328844319980060;
        m_b0(3) = +0.062500000000000;
        /* --- c --- */
        m_c0(0) = +0.088587959512704;
        m_c0(1) = +0.409466864440735;
        m_c0(2) = +0.787659461760847;
        m_c0(3) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +0.000000000000000;
        m_d0(3) = +1.000000000000000;
        /* --- zeta --- */
        /* --- eta --- */
        m_eta(0) = +3.212806896871534;
        m_eta(1) = +4.787193103128462;
        /* --- beta --- */
        m_beta(0) = +4.773087433276642;
        m_beta(1) = +1.567476416895212;
        /* -------------------------- */        
    }
    // 5-stage 9th-order Radau IIA
    else if (m_RK_ID == 29) {        
        /* ID: RadauIIA9 */
        m_s = 5;
        SizeButcherData(1, 2);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.072998864317903;
        m_A0(0, 1) = -0.026735331107946;
        m_A0(0, 2) = +0.018676929763984;
        m_A0(0, 3) = -0.012879106093306;
        m_A0(0, 4) = +0.005042839233882;
        m_A0(1, 0) = +0.153775231479182;
        m_A0(1, 1) = +0.146214867847494;
        m_A0(1, 2) = -0.036444568905128;
        m_A0(1, 3) = +0.021233063119305;
        m_A0(1, 4) = -0.007935579902729;
        m_A0(2, 0) = +0.140063045684810;
        m_A0(2, 1) = +0.298967129491283;
        m_A0(2, 2) = +0.167585070135249;
        m_A0(2, 3) = -0.033969101686618;
        m_A0(2, 4) = +0.010944288744192;
        m_A0(3, 0) = +0.144894308109535;
        m_A0(3, 1) = +0.276500068760159;
        m_A0(3, 2) = +0.325797922910421;
        m_A0(3, 3) = +0.128756753254910;
        m_A0(3, 4) = -0.015708917378805;
        m_A0(4, 0) = +0.143713560791226;
        m_A0(4, 1) = +0.281356015149462;
        m_A0(4, 2) = +0.311826522975741;
        m_A0(4, 3) = +0.223103901083571;
        m_A0(4, 4) = +0.040000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +8.755923977938361;
        m_invA0(0, 1) = +2.891942615380119;
        m_invA0(0, 2) = -0.875186396200266;
        m_invA0(0, 3) = +0.399705207939967;
        m_invA0(0, 4) = -0.133706163849216;
        m_invA0(1, 0) = -7.161380720145386;
        m_invA0(1, 1) = +1.806077724083645;
        m_invA0(1, 2) = +2.363797176068608;
        m_invA0(1, 3) = -0.865900780283135;
        m_invA0(1, 4) = +0.274338077775194;
        m_invA0(2, 0) = +4.122165246243370;
        m_invA0(2, 1) = -4.496017125813395;
        m_invA0(2, 2) = +0.856765245397177;
        m_invA0(2, 3) = +2.518320949211065;
        m_invA0(2, 4) = -0.657062757134360;
        m_invA0(3, 0) = -3.878663219724007;
        m_invA0(3, 1) = +3.393151918064955;
        m_invA0(3, 2) = -5.188340906407188;
        m_invA0(3, 3) = +0.581233052580817;
        m_invA0(3, 4) = +2.809983655279712;
        m_invA0(4, 0) = +8.412424223594297;
        m_invA0(4, 1) = -6.970256116656662;
        m_invA0(4, 2) = +8.777114204150472;
        m_invA0(4, 3) = -18.219282311088101;
        m_invA0(4, 4) = +13.000000000000000;
        /* --- b --- */
        m_b0(0) = +0.143713560791226;
        m_b0(1) = +0.281356015149462;
        m_b0(2) = +0.311826522975741;
        m_b0(3) = +0.223103901083571;
        m_b0(4) = +0.040000000000000;
        /* --- c --- */
        m_c0(0) = +0.057104196114518;
        m_c0(1) = +0.276843013638124;
        m_c0(2) = +0.583590432368917;
        m_c0(3) = +0.860240135656219;
        m_c0(4) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +0.000000000000000;
        m_d0(3) = +0.000000000000000;
        m_d0(4) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +6.286704751729272;
        /* --- eta --- */
        m_eta(0) = +3.655694325463570;
        m_eta(1) = +5.700953298671794;
        /* --- beta --- */
        m_beta(0) = +6.543736899360070;
        m_beta(1) = +3.210265600308550;
        /* -------------------------- */        
    }
    // 2-stage 2nd-order Lobatto IIIC
    else if (m_RK_ID == 32) {
        /* ID: LobIIIC2 */
        m_s = 2;
        SizeButcherData(0, 1);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.500000000000000;
        m_A0(0, 1) = -0.500000000000000;
        m_A0(1, 0) = +0.500000000000000;
        m_A0(1, 1) = +0.500000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +1.000000000000000;
        m_invA0(0, 1) = +1.000000000000000;
        m_invA0(1, 0) = -1.000000000000000;
        m_invA0(1, 1) = +1.000000000000000;
        /* --- b --- */
        m_b0(0) = +0.500000000000000;
        m_b0(1) = +0.500000000000000;
        /* --- c --- */
        m_c0(0) = +0.000000000000000;
        m_c0(1) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +1.000000000000000;
        /* --- zeta --- */
        /* --- eta --- */
        m_eta(0) = +1.000000000000000;
        /* --- beta --- */
        m_beta(0) = +1.000000000000000;
        /* -------------------------- */                
    }
    // 3-stage 4th-order Lobatto IIIC
    else if (m_RK_ID == 34) {
        /* ID: LobIIIC4 */
        m_s = 3;
        SizeButcherData(1, 1);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.166666666666667;
        m_A0(0, 1) = -0.333333333333333;
        m_A0(0, 2) = +0.166666666666667;
        m_A0(1, 0) = +0.166666666666667;
        m_A0(1, 1) = +0.416666666666667;
        m_A0(1, 2) = -0.083333333333333;
        m_A0(2, 0) = +0.166666666666667;
        m_A0(2, 1) = +0.666666666666667;
        m_A0(2, 2) = +0.166666666666667;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +2.999999999999998;
        m_invA0(0, 1) = +4.000000000000000;
        m_invA0(0, 2) = -1.000000000000000;
        m_invA0(1, 0) = -1.000000000000000;
        m_invA0(1, 1) = +0.000000000000000;
        m_invA0(1, 2) = +1.000000000000000;
        m_invA0(2, 0) = +1.000000000000000;
        m_invA0(2, 1) = -4.000000000000000;
        m_invA0(2, 2) = +3.000000000000000;
        /* --- b --- */
        m_b0(0) = +0.166666666666667;
        m_b0(1) = +0.666666666666667;
        m_b0(2) = +0.166666666666667;
        /* --- c --- */
        m_c0(0) = +0.000000000000000;
        m_c0(1) = +0.500000000000000;
        m_c0(2) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +2.625816818958464;
        /* --- eta --- */
        m_eta(0) = +1.687091590520766;
        /* --- beta --- */
        m_beta(0) = +2.508731754924882;
        /* -------------------------- */        
    }
    // 4-stage 6th-order Lobatto IIIC
    else if (m_RK_ID == 36) {
        /* ID: LobIIIC6 */
        m_s = 4;
        SizeButcherData(0, 2);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.083333333333333;
        m_A0(0, 1) = -0.186338998124983;
        m_A0(0, 2) = +0.186338998124983;
        m_A0(0, 3) = -0.083333333333333;
        m_A0(1, 0) = +0.083333333333333;
        m_A0(1, 1) = +0.250000000000000;
        m_A0(1, 2) = -0.094207930708309;
        m_A0(1, 3) = +0.037267799624996;
        m_A0(2, 0) = +0.083333333333333;
        m_A0(2, 1) = +0.427541264041642;
        m_A0(2, 2) = +0.250000000000000;
        m_A0(2, 3) = -0.037267799624996;
        m_A0(3, 0) = +0.083333333333333;
        m_A0(3, 1) = +0.416666666666667;
        m_A0(3, 2) = +0.416666666666667;
        m_A0(3, 3) = +0.083333333333333;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +6.000000000000002;
        m_invA0(0, 1) = +8.090169943749478;
        m_invA0(0, 2) = -3.090169943749475;
        m_invA0(0, 3) = +1.000000000000000;
        m_invA0(1, 0) = -1.618033988749895;
        m_invA0(1, 1) = +0.000000000000000;
        m_invA0(1, 2) = +2.236067977499789;
        m_invA0(1, 3) = -0.618033988749895;
        m_invA0(2, 0) = +0.618033988749894;
        m_invA0(2, 1) = -2.236067977499790;
        m_invA0(2, 2) = +0.000000000000001;
        m_invA0(2, 3) = +1.618033988749895;
        m_invA0(3, 0) = -1.000000000000001;
        m_invA0(3, 1) = +3.090169943749475;
        m_invA0(3, 2) = -8.090169943749476;
        m_invA0(3, 3) = +6.000000000000003;
        /* --- b --- */
        m_b0(0) = +0.083333333333333;
        m_b0(1) = +0.416666666666667;
        m_b0(2) = +0.416666666666667;
        m_b0(3) = +0.083333333333333;
        /* --- c --- */
        m_c0(0) = +0.000000000000000;
        m_c0(1) = +0.276393202250021;
        m_c0(2) = +0.723606797749979;
        m_c0(3) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +0.000000000000000;
        m_d0(3) = +1.000000000000000;
        /* --- zeta --- */
        /* --- eta --- */
        m_eta(0) = +2.220980032989809;
        m_eta(1) = +3.779019967010197;
        /* --- beta --- */
        m_beta(0) = +4.160391445506934;
        m_beta(1) = +1.380176524272843;
        /* -------------------------- */        
    }
    // 5-stage 8th-order Lobatto IIIC
    else if (m_RK_ID == 38) {
        /* ID: LobIIIC8 */
        m_s = 5;
        SizeButcherData(1, 2);

        /* --- Tableau constants --- */
        /* --- A --- */
        m_A0(0, 0) = +0.050000000000000;
        m_A0(0, 1) = -0.116666666666667;
        m_A0(0, 2) = +0.133333333333333;
        m_A0(0, 3) = -0.116666666666667;
        m_A0(0, 4) = +0.050000000000000;
        m_A0(1, 0) = +0.050000000000000;
        m_A0(1, 1) = +0.161111111111111;
        m_A0(1, 2) = -0.069011541029643;
        m_A0(1, 3) = +0.052002165993115;
        m_A0(1, 4) = -0.021428571428571;
        m_A0(2, 0) = +0.050000000000000;
        m_A0(2, 1) = +0.281309183323043;
        m_A0(2, 2) = +0.202777777777778;
        m_A0(2, 3) = -0.052836961100821;
        m_A0(2, 4) = +0.018750000000000;
        m_A0(3, 0) = +0.050000000000000;
        m_A0(3, 1) = +0.270220056229107;
        m_A0(3, 2) = +0.367424239442342;
        m_A0(3, 3) = +0.161111111111111;
        m_A0(3, 4) = -0.021428571428571;
        m_A0(4, 0) = +0.050000000000000;
        m_A0(4, 1) = +0.272222222222222;
        m_A0(4, 2) = +0.355555555555556;
        m_A0(4, 3) = +0.272222222222222;
        m_A0(4, 4) = +0.050000000000000;
        /* --- inv(A) --- */
        m_invA0(0, 0) = +10.000000000000002;
        m_invA0(0, 1) = +13.513004977448478;
        m_invA0(0, 2) = -5.333333333333334;
        m_invA0(0, 3) = +2.820328355884856;
        m_invA0(0, 4) = -1.000000000000001;
        m_invA0(1, 0) = -2.481980506061966;
        m_invA0(1, 1) = +0.000000000000000;
        m_invA0(1, 2) = +3.491486243775877;
        m_invA0(1, 3) = -1.527525231651947;
        m_invA0(1, 4) = +0.518019493938034;
        m_invA0(2, 0) = +0.750000000000000;
        m_invA0(2, 1) = -2.673169155390907;
        m_invA0(2, 2) = +0.000000000000001;
        m_invA0(2, 3) = +2.673169155390906;
        m_invA0(2, 4) = -0.750000000000000;
        m_invA0(3, 0) = -0.518019493938035;
        m_invA0(3, 1) = +1.527525231651947;
        m_invA0(3, 2) = -3.491486243775877;
        m_invA0(3, 3) = -0.000000000000001;
        m_invA0(3, 4) = +2.481980506061966;
        m_invA0(4, 0) = +0.999999999999998;
        m_invA0(4, 1) = -2.820328355884851;
        m_invA0(4, 2) = +5.333333333333331;
        m_invA0(4, 3) = -13.513004977448482;
        m_invA0(4, 4) = +10.000000000000004;
        /* --- b --- */
        m_b0(0) = +0.050000000000000;
        m_b0(1) = +0.272222222222222;
        m_b0(2) = +0.355555555555556;
        m_b0(3) = +0.272222222222222;
        m_b0(4) = +0.050000000000000;
        /* --- c --- */
        m_c0(0) = +0.000000000000000;
        m_c0(1) = +0.172673164646011;
        m_c0(2) = +0.500000000000000;
        m_c0(3) = +0.827326835353989;
        m_c0(4) = +1.000000000000000;
        /* --- d --- */
        m_d0(0) = +0.000000000000000;
        m_d0(1) = +0.000000000000000;
        m_d0(2) = +0.000000000000000;
        m_d0(3) = +0.000000000000000;
        m_d0(4) = +1.000000000000000;
        /* --- zeta --- */
        m_zeta(0) = +5.277122810196466;
        /* --- eta --- */
        m_eta(0) = +2.664731518065844;
        m_eta(1) = +4.696707076835913;
        /* --- beta --- */
        m_beta(0) = +5.884022927615485;
        m_beta(1) = +2.908975454213619;
        /* -------------------------- */    
    }
    else {
        mfem_error("IRK::Invalid Runge Kutta type.\n");
    }
}

/* Given the precomputed vector m_d0, and the matrix m_invA0, compute and 
    store coefficients of the polynomials {X_j}_{j=1}^s */
void IRK::SetXCoeffs() {
        
    // Size data. There are s degree s-1 polynomials X_j (they each have s coefficients)
    m_XCoeffs.resize(m_s, Vector(m_s));
    
    // TODO: If using MFEM::Array rather than std::vector for m_XCoeffs (which is what I'd rather use), 
    // this should be the following, but it gives compiler warning that i cannot seem to get rid of (the code still works tho)...
    //m_XCoeffs.SetSize(m_s, Vector(m_s));
        
    /* Shallow copy inv(A0) and d0 so the following formulae appear shorter */
    DenseMatrix Y = m_invA0;
    Vector      z = m_d0;
    
    if (m_s == 1) {
        /* s=1: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0] = +z(0);    
    }
    else if (m_s == 2) {
        /* s=2: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0] = +Y(1,1)*z(0)-Y(1,0)*z(1);
        m_XCoeffs[0][1] = -z(0);
        
        /* s=2: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0] = +Y(0,0)*z(1)-Y(0,1)*z(0);
        m_XCoeffs[1][1] = -z(1);
    }
    else if (m_s == 3) {    
        /* s=3: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0] = +z(2)*(Y(1,0)*Y(2,1)-Y(1,1)*Y(2,0))-z(1)*(Y(1,0)*Y(2,2)
                            -Y(1,2)*Y(2,0))+z(0)*(Y(1,1)*Y(2,2)-Y(1,2)*Y(2,1));
        m_XCoeffs[0][1] = +Y(1,0)*z(1)+Y(2,0)*z(2)-z(0)*(Y(1,1)+Y(2,2));
        m_XCoeffs[0][2] = +z(0);
        
        /* s=3: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0] = +z(1)*(Y(0,0)*Y(2,2)-Y(0,2)*Y(2,0))-z(2)*(Y(0,0)*Y(2,1)
                            -Y(0,1)*Y(2,0))-z(0)*(Y(0,1)*Y(2,2)-Y(0,2)*Y(2,1));
        m_XCoeffs[1][1] = +Y(0,1)*z(0)+Y(2,1)*z(2)-z(1)*(Y(0,0)+Y(2,2));
        m_XCoeffs[1][2] = +z(1);
        
        /* s=3: Coefficients for polynomial X_{3}(z) */
        m_XCoeffs[2][0] = +z(2)*(Y(0,0)*Y(1,1)-Y(0,1)*Y(1,0))-z(1)*(Y(0,0)*Y(1,2)
                            -Y(0,2)*Y(1,0))+z(0)*(Y(0,1)*Y(1,2)-Y(0,2)*Y(1,1));
        m_XCoeffs[2][1] = +Y(0,2)*z(0)+Y(1,2)*z(1)-z(2)*(Y(0,0)+Y(1,1));
        m_XCoeffs[2][2] = +z(2);    
    }
    else if (m_s == 4) {    
        /* s=4: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0] = +z(2)*(Y(1,0)*Y(2,1)*Y(3,3)-Y(1,0)*Y(2,3)*Y(3,1)-Y(1,1)*Y(2,0)*Y(3,3)+Y(1,1)*Y(2,3)*Y(3,0)
                            +Y(1,3)*Y(2,0)*Y(3,1)-Y(1,3)*Y(2,1)*Y(3,0))-z(3)*(Y(1,0)*Y(2,1)*Y(3,2)-Y(1,0)*Y(2,2)
                            *Y(3,1)-Y(1,1)*Y(2,0)*Y(3,2)+Y(1,1)*Y(2,2)*Y(3,0)+Y(1,2)*Y(2,0)*Y(3,1)-Y(1,2)*Y(2,1)
                            *Y(3,0))-z(1)*(Y(1,0)*Y(2,2)*Y(3,3)-Y(1,0)*Y(2,3)*Y(3,2)-Y(1,2)*Y(2,0)*Y(3,3)+Y(1,2)
                            *Y(2,3)*Y(3,0)+Y(1,3)*Y(2,0)*Y(3,2)-Y(1,3)*Y(2,2)*Y(3,0))+z(0)*(Y(1,1)*Y(2,2)*Y(3,3)
                            -Y(1,1)*Y(2,3)*Y(3,2)-Y(1,2)*Y(2,1)*Y(3,3)+Y(1,2)*Y(2,3)*Y(3,1)+Y(1,3)*Y(2,1)*Y(3,2)
                            -Y(1,3)*Y(2,2)*Y(3,1));
        m_XCoeffs[0][1] = +z(1)*(Y(1,0)*Y(2,2)-Y(1,2)*Y(2,0)+Y(1,0)*Y(3,3)-Y(1,3)*Y(3,0))-z(0)*(Y(1,1)*Y(2,2)
                            -Y(1,2)*Y(2,1)+Y(1,1)*Y(3,3)-Y(1,3)*Y(3,1)+Y(2,2)*Y(3,3)-Y(2,3)*Y(3,2))-z(2)
                            *(Y(1,0)*Y(2,1)-Y(1,1)*Y(2,0)-Y(2,0)*Y(3,3)+Y(2,3)*Y(3,0))-z(3)*(Y(1,0)*Y(3,1)
                            -Y(1,1)*Y(3,0)+Y(2,0)*Y(3,2)-Y(2,2)*Y(3,0));
        m_XCoeffs[0][2] = +z(0)*(Y(1,1)+Y(2,2)+Y(3,3))-Y(2,0)*z(2)-Y(3,0)*z(3)-Y(1,0)*z(1);
        m_XCoeffs[0][3] = -z(0);
        
        /* s=4: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0] = +z(3)*(Y(0,0)*Y(2,1)*Y(3,2)-Y(0,0)*Y(2,2)*Y(3,1)-Y(0,1)*Y(2,0)*Y(3,2)+Y(0,1)*Y(2,2)*Y(3,0)
                            +Y(0,2)*Y(2,0)*Y(3,1)-Y(0,2)*Y(2,1)*Y(3,0))-z(2)*(Y(0,0)*Y(2,1)*Y(3,3)-Y(0,0)*Y(2,3)
                            *Y(3,1)-Y(0,1)*Y(2,0)*Y(3,3)+Y(0,1)*Y(2,3)*Y(3,0)+Y(0,3)*Y(2,0)*Y(3,1)-Y(0,3)*Y(2,1)
                            *Y(3,0))+z(1)*(Y(0,0)*Y(2,2)*Y(3,3)-Y(0,0)*Y(2,3)*Y(3,2)-Y(0,2)*Y(2,0)*Y(3,3)+Y(0,2)
                            *Y(2,3)*Y(3,0)+Y(0,3)*Y(2,0)*Y(3,2)-Y(0,3)*Y(2,2)*Y(3,0))-z(0)*(Y(0,1)*Y(2,2)*Y(3,3)
                            -Y(0,1)*Y(2,3)*Y(3,2)-Y(0,2)*Y(2,1)*Y(3,3)+Y(0,2)*Y(2,3)*Y(3,1)+Y(0,3)*Y(2,1)*Y(3,2)
                            -Y(0,3)*Y(2,2)*Y(3,1));
        m_XCoeffs[1][1] = +z(0)*(Y(0,1)*Y(2,2)-Y(0,2)*Y(2,1)+Y(0,1)*Y(3,3)-Y(0,3)*Y(3,1))-z(1)*(Y(0,0)*Y(2,2)
                            -Y(0,2)*Y(2,0)+Y(0,0)*Y(3,3)-Y(0,3)*Y(3,0)+Y(2,2)*Y(3,3)-Y(2,3)*Y(3,2))+z(2)
                            *(Y(0,0)*Y(2,1)-Y(0,1)*Y(2,0)+Y(2,1)*Y(3,3)-Y(2,3)*Y(3,1))+z(3)*(Y(0,0)*Y(3,1)
                            -Y(0,1)*Y(3,0)-Y(2,1)*Y(3,2)+Y(2,2)*Y(3,1));
        m_XCoeffs[1][2] = +z(1)*(Y(0,0)+Y(2,2)+Y(3,3))-Y(2,1)*z(2)-Y(3,1)*z(3)-Y(0,1)*z(0);
        m_XCoeffs[1][3] = -z(1);
        
        /* s=4: Coefficients for polynomial X_{3}(z) */
        m_XCoeffs[2][0] = +z(2)*(Y(0,0)*Y(1,1)*Y(3,3)-Y(0,0)*Y(1,3)*Y(3,1)-Y(0,1)*Y(1,0)*Y(3,3)+Y(0,1)*Y(1,3)*Y(3,0)
                            +Y(0,3)*Y(1,0)*Y(3,1)-Y(0,3)*Y(1,1)*Y(3,0))-z(3)*(Y(0,0)*Y(1,1)*Y(3,2)-Y(0,0)*Y(1,2)
                            *Y(3,1)-Y(0,1)*Y(1,0)*Y(3,2)+Y(0,1)*Y(1,2)*Y(3,0)+Y(0,2)*Y(1,0)*Y(3,1)-Y(0,2)*Y(1,1)
                            *Y(3,0))-z(1)*(Y(0,0)*Y(1,2)*Y(3,3)-Y(0,0)*Y(1,3)*Y(3,2)-Y(0,2)*Y(1,0)*Y(3,3)+Y(0,2)
                            *Y(1,3)*Y(3,0)+Y(0,3)*Y(1,0)*Y(3,2)-Y(0,3)*Y(1,2)*Y(3,0))+z(0)*(Y(0,1)*Y(1,2)*Y(3,3)
                            -Y(0,1)*Y(1,3)*Y(3,2)-Y(0,2)*Y(1,1)*Y(3,3)+Y(0,2)*Y(1,3)*Y(3,1)+Y(0,3)*Y(1,1)*Y(3,2)
                            -Y(0,3)*Y(1,2)*Y(3,1));
        m_XCoeffs[2][1] = +z(1)*(Y(0,0)*Y(1,2)-Y(0,2)*Y(1,0)+Y(1,2)*Y(3,3)-Y(1,3)*Y(3,2))-z(0)*(Y(0,1)*Y(1,2)-Y(0,2)*Y(1,1)
                            -Y(0,2)*Y(3,3)+Y(0,3)*Y(3,2))-z(2)*(Y(0,0)*Y(1,1)-Y(0,1)*Y(1,0)+Y(0,0)*Y(3,3)-Y(0,3)*Y(3,0)
                            +Y(1,1)*Y(3,3)-Y(1,3)*Y(3,1))+z(3)*(Y(0,0)*Y(3,2)-Y(0,2)*Y(3,0)+Y(1,1)*Y(3,2)-Y(1,2)*Y(3,1));
        m_XCoeffs[2][2] = +z(2)*(Y(0,0)+Y(1,1)+Y(3,3))-Y(1,2)*z(1)-Y(3,2)*z(3)-Y(0,2)*z(0);
        m_XCoeffs[2][3] = -z(2);
        
        /* s=4: Coefficients for polynomial X_{4}(z) */
        m_XCoeffs[3][0] = +z(3)*(Y(0,0)*Y(1,1)*Y(2,2)-Y(0,0)*Y(1,2)*Y(2,1)-Y(0,1)*Y(1,0)*Y(2,2)+Y(0,1)*Y(1,2)*Y(2,0)
                            +Y(0,2)*Y(1,0)*Y(2,1)-Y(0,2)*Y(1,1)*Y(2,0))-z(2)*(Y(0,0)*Y(1,1)*Y(2,3)-Y(0,0)*Y(1,3)
                            *Y(2,1)-Y(0,1)*Y(1,0)*Y(2,3)+Y(0,1)*Y(1,3)*Y(2,0)+Y(0,3)*Y(1,0)*Y(2,1)-Y(0,3)*Y(1,1)
                            *Y(2,0))+z(1)*(Y(0,0)*Y(1,2)*Y(2,3)-Y(0,0)*Y(1,3)*Y(2,2)-Y(0,2)*Y(1,0)*Y(2,3)+Y(0,2)
                            *Y(1,3)*Y(2,0)+Y(0,3)*Y(1,0)*Y(2,2)-Y(0,3)*Y(1,2)*Y(2,0))-z(0)*(Y(0,1)*Y(1,2)*Y(2,3)
                            -Y(0,1)*Y(1,3)*Y(2,2)-Y(0,2)*Y(1,1)*Y(2,3)+Y(0,2)*Y(1,3)*Y(2,1)+Y(0,3)*Y(1,1)*Y(2,2)
                            -Y(0,3)*Y(1,2)*Y(2,1));
        m_XCoeffs[3][1] = +z(1)*(Y(0,0)*Y(1,3)-Y(0,3)*Y(1,0)-Y(1,2)*Y(2,3)+Y(1,3)*Y(2,2))-z(0)*(Y(0,1)*Y(1,3)-Y(0,3)*Y(1,1)
                            +Y(0,2)*Y(2,3)-Y(0,3)*Y(2,2))-z(3)*(Y(0,0)*Y(1,1)-Y(0,1)*Y(1,0)+Y(0,0)*Y(2,2)-Y(0,2)*Y(2,0)
                            +Y(1,1)*Y(2,2)-Y(1,2)*Y(2,1))+z(2)*(Y(0,0)*Y(2,3)-Y(0,3)*Y(2,0)+Y(1,1)*Y(2,3)-Y(1,3)*Y(2,1));
        m_XCoeffs[3][2] = +z(3)*(Y(0,0)+Y(1,1)+Y(2,2))-Y(1,3)*z(1)-Y(2,3)*z(2)-Y(0,3)*z(0);
        m_XCoeffs[3][3] = -z(3);   
    }
    else if (m_s == 5) {    
        /* s=5: Coefficients for polynomial X_{1}(z) */
        m_XCoeffs[0][0] = +z(4)*(Y(1,0)*Y(2,1)*Y(3,2)*Y(4,3)-Y(1,0)*Y(2,1)*Y(3,3)*Y(4,2)-Y(1,0)*Y(2,2)*Y(3,1)*Y(4,3)
                            +Y(1,0)*Y(2,2)*Y(3,3)*Y(4,1)+Y(1,0)*Y(2,3)*Y(3,1)*Y(4,2)-Y(1,0)*Y(2,3)*Y(3,2)*Y(4,1)-Y(1,1)
                            *Y(2,0)*Y(3,2)*Y(4,3)+Y(1,1)*Y(2,0)*Y(3,3)*Y(4,2)+Y(1,1)*Y(2,2)*Y(3,0)*Y(4,3)-Y(1,1)*Y(2,2)
                            *Y(3,3)*Y(4,0)-Y(1,1)*Y(2,3)*Y(3,0)*Y(4,2)+Y(1,1)*Y(2,3)*Y(3,2)*Y(4,0)+Y(1,2)*Y(2,0)*Y(3,1)
                            *Y(4,3)-Y(1,2)*Y(2,0)*Y(3,3)*Y(4,1)-Y(1,2)*Y(2,1)*Y(3,0)*Y(4,3)+Y(1,2)*Y(2,1)*Y(3,3)*Y(4,0)
                            +Y(1,2)*Y(2,3)*Y(3,0)*Y(4,1)-Y(1,2)*Y(2,3)*Y(3,1)*Y(4,0)-Y(1,3)*Y(2,0)*Y(3,1)*Y(4,2)+Y(1,3)
                            *Y(2,0)*Y(3,2)*Y(4,1)+Y(1,3)*Y(2,1)*Y(3,0)*Y(4,2)-Y(1,3)*Y(2,1)*Y(3,2)*Y(4,0)-Y(1,3)*Y(2,2)
                            *Y(3,0)*Y(4,1)+Y(1,3)*Y(2,2)*Y(3,1)*Y(4,0))-z(3)*(Y(1,0)*Y(2,1)*Y(3,2)*Y(4,4)-Y(1,0)*Y(2,1)
                            *Y(3,4)*Y(4,2)-Y(1,0)*Y(2,2)*Y(3,1)*Y(4,4)+Y(1,0)*Y(2,2)*Y(3,4)*Y(4,1)+Y(1,0)*Y(2,4)*Y(3,1)
                            *Y(4,2)-Y(1,0)*Y(2,4)*Y(3,2)*Y(4,1)-Y(1,1)*Y(2,0)*Y(3,2)*Y(4,4)+Y(1,1)*Y(2,0)*Y(3,4)*Y(4,2)
                            +Y(1,1)*Y(2,2)*Y(3,0)*Y(4,4)-Y(1,1)*Y(2,2)*Y(3,4)*Y(4,0)-Y(1,1)*Y(2,4)*Y(3,0)*Y(4,2)+Y(1,1)
                            *Y(2,4)*Y(3,2)*Y(4,0)+Y(1,2)*Y(2,0)*Y(3,1)*Y(4,4)-Y(1,2)*Y(2,0)*Y(3,4)*Y(4,1)-Y(1,2)*Y(2,1)
                            *Y(3,0)*Y(4,4)+Y(1,2)*Y(2,1)*Y(3,4)*Y(4,0)+Y(1,2)*Y(2,4)*Y(3,0)*Y(4,1)-Y(1,2)*Y(2,4)*Y(3,1)
                            *Y(4,0)-Y(1,4)*Y(2,0)*Y(3,1)*Y(4,2)+Y(1,4)*Y(2,0)*Y(3,2)*Y(4,1)+Y(1,4)*Y(2,1)*Y(3,0)*Y(4,2)
                            -Y(1,4)*Y(2,1)*Y(3,2)*Y(4,0)-Y(1,4)*Y(2,2)*Y(3,0)*Y(4,1)+Y(1,4)*Y(2,2)*Y(3,1)*Y(4,0))+z(2)
                            *(Y(1,0)*Y(2,1)*Y(3,3)*Y(4,4)-Y(1,0)*Y(2,1)*Y(3,4)*Y(4,3)-Y(1,0)*Y(2,3)*Y(3,1)*Y(4,4)+Y(1,0)
                            *Y(2,3)*Y(3,4)*Y(4,1)+Y(1,0)*Y(2,4)*Y(3,1)*Y(4,3)-Y(1,0)*Y(2,4)*Y(3,3)*Y(4,1)-Y(1,1)*Y(2,0)
                            *Y(3,3)*Y(4,4)+Y(1,1)*Y(2,0)*Y(3,4)*Y(4,3)+Y(1,1)*Y(2,3)*Y(3,0)*Y(4,4)-Y(1,1)*Y(2,3)*Y(3,4)
                            *Y(4,0)-Y(1,1)*Y(2,4)*Y(3,0)*Y(4,3)+Y(1,1)*Y(2,4)*Y(3,3)*Y(4,0)+Y(1,3)*Y(2,0)*Y(3,1)*Y(4,4)
                            -Y(1,3)*Y(2,0)*Y(3,4)*Y(4,1)-Y(1,3)*Y(2,1)*Y(3,0)*Y(4,4)+Y(1,3)*Y(2,1)*Y(3,4)*Y(4,0)+Y(1,3)
                            *Y(2,4)*Y(3,0)*Y(4,1)-Y(1,3)*Y(2,4)*Y(3,1)*Y(4,0)-Y(1,4)*Y(2,0)*Y(3,1)*Y(4,3)+Y(1,4)*Y(2,0)
                            *Y(3,3)*Y(4,1)+Y(1,4)*Y(2,1)*Y(3,0)*Y(4,3)-Y(1,4)*Y(2,1)*Y(3,3)*Y(4,0)-Y(1,4)*Y(2,3)*Y(3,0)
                            *Y(4,1)+Y(1,4)*Y(2,3)*Y(3,1)*Y(4,0))-z(1)*(Y(1,0)*Y(2,2)*Y(3,3)*Y(4,4)-Y(1,0)*Y(2,2)*Y(3,4)
                            *Y(4,3)-Y(1,0)*Y(2,3)*Y(3,2)*Y(4,4)+Y(1,0)*Y(2,3)*Y(3,4)*Y(4,2)+Y(1,0)*Y(2,4)*Y(3,2)*Y(4,3)
                            -Y(1,0)*Y(2,4)*Y(3,3)*Y(4,2)-Y(1,2)*Y(2,0)*Y(3,3)*Y(4,4)+Y(1,2)*Y(2,0)*Y(3,4)*Y(4,3)+Y(1,2)
                            *Y(2,3)*Y(3,0)*Y(4,4)-Y(1,2)*Y(2,3)*Y(3,4)*Y(4,0)-Y(1,2)*Y(2,4)*Y(3,0)*Y(4,3)+Y(1,2)*Y(2,4)
                            *Y(3,3)*Y(4,0)+Y(1,3)*Y(2,0)*Y(3,2)*Y(4,4)-Y(1,3)*Y(2,0)*Y(3,4)*Y(4,2)-Y(1,3)*Y(2,2)*Y(3,0)
                            *Y(4,4)+Y(1,3)*Y(2,2)*Y(3,4)*Y(4,0)+Y(1,3)*Y(2,4)*Y(3,0)*Y(4,2)-Y(1,3)*Y(2,4)*Y(3,2)*Y(4,0)
                            -Y(1,4)*Y(2,0)*Y(3,2)*Y(4,3)+Y(1,4)*Y(2,0)*Y(3,3)*Y(4,2)+Y(1,4)*Y(2,2)*Y(3,0)*Y(4,3)-Y(1,4)
                            *Y(2,2)*Y(3,3)*Y(4,0)-Y(1,4)*Y(2,3)*Y(3,0)*Y(4,2)+Y(1,4)*Y(2,3)*Y(3,2)*Y(4,0))+z(0)*(Y(1,1)
                            *Y(2,2)*Y(3,3)*Y(4,4)-Y(1,1)*Y(2,2)*Y(3,4)*Y(4,3)-Y(1,1)*Y(2,3)*Y(3,2)*Y(4,4)+Y(1,1)*Y(2,3)
                            *Y(3,4)*Y(4,2)+Y(1,1)*Y(2,4)*Y(3,2)*Y(4,3)-Y(1,1)*Y(2,4)*Y(3,3)*Y(4,2)-Y(1,2)*Y(2,1)*Y(3,3)
                            *Y(4,4)+Y(1,2)*Y(2,1)*Y(3,4)*Y(4,3)+Y(1,2)*Y(2,3)*Y(3,1)*Y(4,4)-Y(1,2)*Y(2,3)*Y(3,4)*Y(4,1)
                            -Y(1,2)*Y(2,4)*Y(3,1)*Y(4,3)+Y(1,2)*Y(2,4)*Y(3,3)*Y(4,1)+Y(1,3)*Y(2,1)*Y(3,2)*Y(4,4)-Y(1,3)
                            *Y(2,1)*Y(3,4)*Y(4,2)-Y(1,3)*Y(2,2)*Y(3,1)*Y(4,4)+Y(1,3)*Y(2,2)*Y(3,4)*Y(4,1)+Y(1,3)*Y(2,4)
                            *Y(3,1)*Y(4,2)-Y(1,3)*Y(2,4)*Y(3,2)*Y(4,1)-Y(1,4)*Y(2,1)*Y(3,2)*Y(4,3)+Y(1,4)*Y(2,1)*Y(3,3)
                            *Y(4,2)+Y(1,4)*Y(2,2)*Y(3,1)*Y(4,3)-Y(1,4)*Y(2,2)*Y(3,3)*Y(4,1)-Y(1,4)*Y(2,3)*Y(3,1)*Y(4,2)
                            +Y(1,4)*Y(2,3)*Y(3,2)*Y(4,1));
        m_XCoeffs[0][1] = +z(1)*(Y(1,0)*Y(2,2)*Y(3,3)-Y(1,0)*Y(2,3)*Y(3,2)-Y(1,2)*Y(2,0)*Y(3,3)+Y(1,2)*Y(2,3)*Y(3,0)
                            +Y(1,3)*Y(2,0)*Y(3,2)-Y(1,3)*Y(2,2)*Y(3,0)+Y(1,0)*Y(2,2)*Y(4,4)-Y(1,0)*Y(2,4)*Y(4,2)-Y(1,2)
                            *Y(2,0)*Y(4,4)+Y(1,2)*Y(2,4)*Y(4,0)+Y(1,4)*Y(2,0)*Y(4,2)-Y(1,4)*Y(2,2)*Y(4,0)+Y(1,0)*Y(3,3)
                            *Y(4,4)-Y(1,0)*Y(3,4)*Y(4,3)-Y(1,3)*Y(3,0)*Y(4,4)+Y(1,3)*Y(3,4)*Y(4,0)+Y(1,4)*Y(3,0)*Y(4,3)
                            -Y(1,4)*Y(3,3)*Y(4,0))-z(0)*(Y(1,1)*Y(2,2)*Y(3,3)-Y(1,1)*Y(2,3)*Y(3,2)-Y(1,2)*Y(2,1)*Y(3,3)
                            +Y(1,2)*Y(2,3)*Y(3,1)+Y(1,3)*Y(2,1)*Y(3,2)-Y(1,3)*Y(2,2)*Y(3,1)+Y(1,1)*Y(2,2)*Y(4,4)-Y(1,1)
                            *Y(2,4)*Y(4,2)-Y(1,2)*Y(2,1)*Y(4,4)+Y(1,2)*Y(2,4)*Y(4,1)+Y(1,4)*Y(2,1)*Y(4,2)-Y(1,4)*Y(2,2)
                            *Y(4,1)+Y(1,1)*Y(3,3)*Y(4,4)-Y(1,1)*Y(3,4)*Y(4,3)-Y(1,3)*Y(3,1)*Y(4,4)+Y(1,3)*Y(3,4)*Y(4,1)
                            +Y(1,4)*Y(3,1)*Y(4,3)-Y(1,4)*Y(3,3)*Y(4,1)+Y(2,2)*Y(3,3)*Y(4,4)-Y(2,2)*Y(3,4)*Y(4,3)-Y(2,3)
                            *Y(3,2)*Y(4,4)+Y(2,3)*Y(3,4)*Y(4,2)+Y(2,4)*Y(3,2)*Y(4,3)-Y(2,4)*Y(3,3)*Y(4,2))-z(2)*(Y(1,0)
                            *Y(2,1)*Y(3,3)-Y(1,0)*Y(2,3)*Y(3,1)-Y(1,1)*Y(2,0)*Y(3,3)+Y(1,1)*Y(2,3)*Y(3,0)+Y(1,3)*Y(2,0)
                            *Y(3,1)-Y(1,3)*Y(2,1)*Y(3,0)+Y(1,0)*Y(2,1)*Y(4,4)-Y(1,0)*Y(2,4)*Y(4,1)-Y(1,1)*Y(2,0)*Y(4,4)
                            +Y(1,1)*Y(2,4)*Y(4,0)+Y(1,4)*Y(2,0)*Y(4,1)-Y(1,4)*Y(2,1)*Y(4,0)-Y(2,0)*Y(3,3)*Y(4,4)+Y(2,0)
                            *Y(3,4)*Y(4,3)+Y(2,3)*Y(3,0)*Y(4,4)-Y(2,3)*Y(3,4)*Y(4,0)-Y(2,4)*Y(3,0)*Y(4,3)+Y(2,4)*Y(3,3)
                            *Y(4,0))+z(3)*(Y(1,0)*Y(2,1)*Y(3,2)-Y(1,0)*Y(2,2)*Y(3,1)-Y(1,1)*Y(2,0)*Y(3,2)+Y(1,1)*Y(2,2)
                            *Y(3,0)+Y(1,2)*Y(2,0)*Y(3,1)-Y(1,2)*Y(2,1)*Y(3,0)-Y(1,0)*Y(3,1)*Y(4,4)+Y(1,0)*Y(3,4)*Y(4,1)
                            +Y(1,1)*Y(3,0)*Y(4,4)-Y(1,1)*Y(3,4)*Y(4,0)-Y(1,4)*Y(3,0)*Y(4,1)+Y(1,4)*Y(3,1)*Y(4,0)-Y(2,0)
                            *Y(3,2)*Y(4,4)+Y(2,0)*Y(3,4)*Y(4,2)+Y(2,2)*Y(3,0)*Y(4,4)-Y(2,2)*Y(3,4)*Y(4,0)-Y(2,4)*Y(3,0)
                            *Y(4,2)+Y(2,4)*Y(3,2)*Y(4,0))+z(4)*(Y(1,0)*Y(2,1)*Y(4,2)-Y(1,0)*Y(2,2)*Y(4,1)-Y(1,1)*Y(2,0)
                            *Y(4,2)+Y(1,1)*Y(2,2)*Y(4,0)+Y(1,2)*Y(2,0)*Y(4,1)-Y(1,2)*Y(2,1)*Y(4,0)+Y(1,0)*Y(3,1)*Y(4,3)
                            -Y(1,0)*Y(3,3)*Y(4,1)-Y(1,1)*Y(3,0)*Y(4,3)+Y(1,1)*Y(3,3)*Y(4,0)+Y(1,3)*Y(3,0)*Y(4,1)-Y(1,3)
                            *Y(3,1)*Y(4,0)+Y(2,0)*Y(3,2)*Y(4,3)-Y(2,0)*Y(3,3)*Y(4,2)-Y(2,2)*Y(3,0)*Y(4,3)+Y(2,2)*Y(3,3)
                            *Y(4,0)+Y(2,3)*Y(3,0)*Y(4,2)-Y(2,3)*Y(3,2)*Y(4,0));
        m_XCoeffs[0][2] = +z(2)*(Y(1,0)*Y(2,1)-Y(1,1)*Y(2,0)-Y(2,0)*Y(3,3)+Y(2,3)*Y(3,0)-Y(2,0)*Y(4,4)+Y(2,4)*Y(4,0))-z(1)
                            *(Y(1,0)*Y(2,2)-Y(1,2)*Y(2,0)+Y(1,0)*Y(3,3)-Y(1,3)*Y(3,0)+Y(1,0)*Y(4,4)-Y(1,4)*Y(4,0))+z(3)
                            *(Y(1,0)*Y(3,1)-Y(1,1)*Y(3,0)+Y(2,0)*Y(3,2)-Y(2,2)*Y(3,0)-Y(3,0)*Y(4,4)+Y(3,4)*Y(4,0))+z(4)
                            *(Y(1,0)*Y(4,1)-Y(1,1)*Y(4,0)+Y(2,0)*Y(4,2)-Y(2,2)*Y(4,0)+Y(3,0)*Y(4,3)-Y(3,3)*Y(4,0))+z(0)
                            *(Y(1,1)*Y(2,2)-Y(1,2)*Y(2,1)+Y(1,1)*Y(3,3)-Y(1,3)*Y(3,1)+Y(1,1)*Y(4,4)-Y(1,4)*Y(4,1)+Y(2,2)
                            *Y(3,3)-Y(2,3)*Y(3,2)+Y(2,2)*Y(4,4)-Y(2,4)*Y(4,2)+Y(3,3)*Y(4,4)-Y(3,4)*Y(4,3));
        m_XCoeffs[0][3] = +Y(1,0)*z(1)+Y(2,0)*z(2)+Y(3,0)*z(3)+Y(4,0)*z(4)-z(0)*(Y(1,1)+Y(2,2)+Y(3,3)+Y(4,4));
        m_XCoeffs[0][4] = +z(0);
        
        /* s=5: Coefficients for polynomial X_{2}(z) */
        m_XCoeffs[1][0] = +z(3)*(Y(0,0)*Y(2,1)*Y(3,2)*Y(4,4)-Y(0,0)*Y(2,1)*Y(3,4)*Y(4,2)-Y(0,0)*Y(2,2)*Y(3,1)*Y(4,4)+
                            Y(0,0)*Y(2,2)*Y(3,4)*Y(4,1)+Y(0,0)*Y(2,4)*Y(3,1)*Y(4,2)-Y(0,0)*Y(2,4)*Y(3,2)*Y(4,1)-Y(0,1)
                            *Y(2,0)*Y(3,2)*Y(4,4)+Y(0,1)*Y(2,0)*Y(3,4)*Y(4,2)+Y(0,1)*Y(2,2)*Y(3,0)*Y(4,4)-Y(0,1)*Y(2,2)
                            *Y(3,4)*Y(4,0)-Y(0,1)*Y(2,4)*Y(3,0)*Y(4,2)+Y(0,1)*Y(2,4)*Y(3,2)*Y(4,0)+Y(0,2)*Y(2,0)*Y(3,1)
                            *Y(4,4)-Y(0,2)*Y(2,0)*Y(3,4)*Y(4,1)-Y(0,2)*Y(2,1)*Y(3,0)*Y(4,4)+Y(0,2)*Y(2,1)*Y(3,4)*Y(4,0)
                            +Y(0,2)*Y(2,4)*Y(3,0)*Y(4,1)-Y(0,2)*Y(2,4)*Y(3,1)*Y(4,0)-Y(0,4)*Y(2,0)*Y(3,1)*Y(4,2)+Y(0,4)
                            *Y(2,0)*Y(3,2)*Y(4,1)+Y(0,4)*Y(2,1)*Y(3,0)*Y(4,2)-Y(0,4)*Y(2,1)*Y(3,2)*Y(4,0)-Y(0,4)*Y(2,2)
                            *Y(3,0)*Y(4,1)+Y(0,4)*Y(2,2)*Y(3,1)*Y(4,0))-z(4)*(Y(0,0)*Y(2,1)*Y(3,2)*Y(4,3)-Y(0,0)*Y(2,1)
                            *Y(3,3)*Y(4,2)-Y(0,0)*Y(2,2)*Y(3,1)*Y(4,3)+Y(0,0)*Y(2,2)*Y(3,3)*Y(4,1)+Y(0,0)*Y(2,3)*Y(3,1)
                            *Y(4,2)-Y(0,0)*Y(2,3)*Y(3,2)*Y(4,1)-Y(0,1)*Y(2,0)*Y(3,2)*Y(4,3)+Y(0,1)*Y(2,0)*Y(3,3)*Y(4,2)
                            +Y(0,1)*Y(2,2)*Y(3,0)*Y(4,3)-Y(0,1)*Y(2,2)*Y(3,3)*Y(4,0)-Y(0,1)*Y(2,3)*Y(3,0)*Y(4,2)+Y(0,1)
                            *Y(2,3)*Y(3,2)*Y(4,0)+Y(0,2)*Y(2,0)*Y(3,1)*Y(4,3)-Y(0,2)*Y(2,0)*Y(3,3)*Y(4,1)-Y(0,2)*Y(2,1)
                            *Y(3,0)*Y(4,3)+Y(0,2)*Y(2,1)*Y(3,3)*Y(4,0)+Y(0,2)*Y(2,3)*Y(3,0)*Y(4,1)-Y(0,2)*Y(2,3)*Y(3,1)
                            *Y(4,0)-Y(0,3)*Y(2,0)*Y(3,1)*Y(4,2)+Y(0,3)*Y(2,0)*Y(3,2)*Y(4,1)+Y(0,3)*Y(2,1)*Y(3,0)*Y(4,2)
                            -Y(0,3)*Y(2,1)*Y(3,2)*Y(4,0)-Y(0,3)*Y(2,2)*Y(3,0)*Y(4,1)+Y(0,3)*Y(2,2)*Y(3,1)*Y(4,0))-z(2)
                            *(Y(0,0)*Y(2,1)*Y(3,3)*Y(4,4)-Y(0,0)*Y(2,1)*Y(3,4)*Y(4,3)-Y(0,0)*Y(2,3)*Y(3,1)*Y(4,4)+Y(0,0)
                            *Y(2,3)*Y(3,4)*Y(4,1)+Y(0,0)*Y(2,4)*Y(3,1)*Y(4,3)-Y(0,0)*Y(2,4)*Y(3,3)*Y(4,1)-Y(0,1)*Y(2,0)
                            *Y(3,3)*Y(4,4)+Y(0,1)*Y(2,0)*Y(3,4)*Y(4,3)+Y(0,1)*Y(2,3)*Y(3,0)*Y(4,4)-Y(0,1)*Y(2,3)*Y(3,4)
                            *Y(4,0)-Y(0,1)*Y(2,4)*Y(3,0)*Y(4,3)+Y(0,1)*Y(2,4)*Y(3,3)*Y(4,0)+Y(0,3)*Y(2,0)*Y(3,1)*Y(4,4)
                            -Y(0,3)*Y(2,0)*Y(3,4)*Y(4,1)-Y(0,3)*Y(2,1)*Y(3,0)*Y(4,4)+Y(0,3)*Y(2,1)*Y(3,4)*Y(4,0)+Y(0,3)
                            *Y(2,4)*Y(3,0)*Y(4,1)-Y(0,3)*Y(2,4)*Y(3,1)*Y(4,0)-Y(0,4)*Y(2,0)*Y(3,1)*Y(4,3)+Y(0,4)*Y(2,0)
                            *Y(3,3)*Y(4,1)+Y(0,4)*Y(2,1)*Y(3,0)*Y(4,3)-Y(0,4)*Y(2,1)*Y(3,3)*Y(4,0)-Y(0,4)*Y(2,3)*Y(3,0)
                            *Y(4,1)+Y(0,4)*Y(2,3)*Y(3,1)*Y(4,0))+z(1)*(Y(0,0)*Y(2,2)*Y(3,3)*Y(4,4)-Y(0,0)*Y(2,2)*Y(3,4)
                            *Y(4,3)-Y(0,0)*Y(2,3)*Y(3,2)*Y(4,4)+Y(0,0)*Y(2,3)*Y(3,4)*Y(4,2)+Y(0,0)*Y(2,4)*Y(3,2)*Y(4,3)
                            -Y(0,0)*Y(2,4)*Y(3,3)*Y(4,2)-Y(0,2)*Y(2,0)*Y(3,3)*Y(4,4)+Y(0,2)*Y(2,0)*Y(3,4)*Y(4,3)+Y(0,2)
                            *Y(2,3)*Y(3,0)*Y(4,4)-Y(0,2)*Y(2,3)*Y(3,4)*Y(4,0)-Y(0,2)*Y(2,4)*Y(3,0)*Y(4,3)+Y(0,2)*Y(2,4)
                            *Y(3,3)*Y(4,0)+Y(0,3)*Y(2,0)*Y(3,2)*Y(4,4)-Y(0,3)*Y(2,0)*Y(3,4)*Y(4,2)-Y(0,3)*Y(2,2)*Y(3,0)
                            *Y(4,4)+Y(0,3)*Y(2,2)*Y(3,4)*Y(4,0)+Y(0,3)*Y(2,4)*Y(3,0)*Y(4,2)-Y(0,3)*Y(2,4)*Y(3,2)*Y(4,0)
                            -Y(0,4)*Y(2,0)*Y(3,2)*Y(4,3)+Y(0,4)*Y(2,0)*Y(3,3)*Y(4,2)+Y(0,4)*Y(2,2)*Y(3,0)*Y(4,3)-Y(0,4)
                            *Y(2,2)*Y(3,3)*Y(4,0)-Y(0,4)*Y(2,3)*Y(3,0)*Y(4,2)+Y(0,4)*Y(2,3)*Y(3,2)*Y(4,0))-z(0)*(Y(0,1)
                            *Y(2,2)*Y(3,3)*Y(4,4)-Y(0,1)*Y(2,2)*Y(3,4)*Y(4,3)-Y(0,1)*Y(2,3)*Y(3,2)*Y(4,4)+Y(0,1)*Y(2,3)
                            *Y(3,4)*Y(4,2)+Y(0,1)*Y(2,4)*Y(3,2)*Y(4,3)-Y(0,1)*Y(2,4)*Y(3,3)*Y(4,2)-Y(0,2)*Y(2,1)*Y(3,3)
                            *Y(4,4)+Y(0,2)*Y(2,1)*Y(3,4)*Y(4,3)+Y(0,2)*Y(2,3)*Y(3,1)*Y(4,4)-Y(0,2)*Y(2,3)*Y(3,4)*Y(4,1)
                            -Y(0,2)*Y(2,4)*Y(3,1)*Y(4,3)+Y(0,2)*Y(2,4)*Y(3,3)*Y(4,1)+Y(0,3)*Y(2,1)*Y(3,2)*Y(4,4)-Y(0,3)
                            *Y(2,1)*Y(3,4)*Y(4,2)-Y(0,3)*Y(2,2)*Y(3,1)*Y(4,4)+Y(0,3)*Y(2,2)*Y(3,4)*Y(4,1)+Y(0,3)*Y(2,4)
                            *Y(3,1)*Y(4,2)-Y(0,3)*Y(2,4)*Y(3,2)*Y(4,1)-Y(0,4)*Y(2,1)*Y(3,2)*Y(4,3)+Y(0,4)*Y(2,1)*Y(3,3)
                            *Y(4,2)+Y(0,4)*Y(2,2)*Y(3,1)*Y(4,3)-Y(0,4)*Y(2,2)*Y(3,3)*Y(4,1)-Y(0,4)*Y(2,3)*Y(3,1)*Y(4,2)
                            +Y(0,4)*Y(2,3)*Y(3,2)*Y(4,1));
        m_XCoeffs[1][1] = +z(0)*(Y(0,1)*Y(2,2)*Y(3,3)-Y(0,1)*Y(2,3)*Y(3,2)-Y(0,2)*Y(2,1)*Y(3,3)+Y(0,2)*Y(2,3)*Y(3,1)
                            +Y(0,3)*Y(2,1)*Y(3,2)-Y(0,3)*Y(2,2)*Y(3,1)+Y(0,1)*Y(2,2)*Y(4,4)-Y(0,1)*Y(2,4)*Y(4,2)-Y(0,2)
                            *Y(2,1)*Y(4,4)+Y(0,2)*Y(2,4)*Y(4,1)+Y(0,4)*Y(2,1)*Y(4,2)-Y(0,4)*Y(2,2)*Y(4,1)+Y(0,1)*Y(3,3)
                            *Y(4,4)-Y(0,1)*Y(3,4)*Y(4,3)-Y(0,3)*Y(3,1)*Y(4,4)+Y(0,3)*Y(3,4)*Y(4,1)+Y(0,4)*Y(3,1)*Y(4,3)
                            -Y(0,4)*Y(3,3)*Y(4,1))-z(1)*(Y(0,0)*Y(2,2)*Y(3,3)-Y(0,0)*Y(2,3)*Y(3,2)-Y(0,2)*Y(2,0)*Y(3,3)
                            +Y(0,2)*Y(2,3)*Y(3,0)+Y(0,3)*Y(2,0)*Y(3,2)-Y(0,3)*Y(2,2)*Y(3,0)+Y(0,0)*Y(2,2)*Y(4,4)-Y(0,0)
                            *Y(2,4)*Y(4,2)-Y(0,2)*Y(2,0)*Y(4,4)+Y(0,2)*Y(2,4)*Y(4,0)+Y(0,4)*Y(2,0)*Y(4,2)-Y(0,4)*Y(2,2)
                            *Y(4,0)+Y(0,0)*Y(3,3)*Y(4,4)-Y(0,0)*Y(3,4)*Y(4,3)-Y(0,3)*Y(3,0)*Y(4,4)+Y(0,3)*Y(3,4)*Y(4,0)
                            +Y(0,4)*Y(3,0)*Y(4,3)-Y(0,4)*Y(3,3)*Y(4,0)+Y(2,2)*Y(3,3)*Y(4,4)-Y(2,2)*Y(3,4)*Y(4,3)-Y(2,3)
                            *Y(3,2)*Y(4,4)+Y(2,3)*Y(3,4)*Y(4,2)+Y(2,4)*Y(3,2)*Y(4,3)-Y(2,4)*Y(3,3)*Y(4,2))+z(2)*(Y(0,0)
                            *Y(2,1)*Y(3,3)-Y(0,0)*Y(2,3)*Y(3,1)-Y(0,1)*Y(2,0)*Y(3,3)+Y(0,1)*Y(2,3)*Y(3,0)+Y(0,3)*Y(2,0)
                            *Y(3,1)-Y(0,3)*Y(2,1)*Y(3,0)+Y(0,0)*Y(2,1)*Y(4,4)-Y(0,0)*Y(2,4)*Y(4,1)-Y(0,1)*Y(2,0)*Y(4,4)
                            +Y(0,1)*Y(2,4)*Y(4,0)+Y(0,4)*Y(2,0)*Y(4,1)-Y(0,4)*Y(2,1)*Y(4,0)+Y(2,1)*Y(3,3)*Y(4,4)-Y(2,1)
                            *Y(3,4)*Y(4,3)-Y(2,3)*Y(3,1)*Y(4,4)+Y(2,3)*Y(3,4)*Y(4,1)+Y(2,4)*Y(3,1)*Y(4,3)-Y(2,4)*Y(3,3)
                            *Y(4,1))-z(3)*(Y(0,0)*Y(2,1)*Y(3,2)-Y(0,0)*Y(2,2)*Y(3,1)-Y(0,1)*Y(2,0)*Y(3,2)+Y(0,1)*Y(2,2)
                            *Y(3,0)+Y(0,2)*Y(2,0)*Y(3,1)-Y(0,2)*Y(2,1)*Y(3,0)-Y(0,0)*Y(3,1)*Y(4,4)+Y(0,0)*Y(3,4)*Y(4,1)
                            +Y(0,1)*Y(3,0)*Y(4,4)-Y(0,1)*Y(3,4)*Y(4,0)-Y(0,4)*Y(3,0)*Y(4,1)+Y(0,4)*Y(3,1)*Y(4,0)+Y(2,1)
                            *Y(3,2)*Y(4,4)-Y(2,1)*Y(3,4)*Y(4,2)-Y(2,2)*Y(3,1)*Y(4,4)+Y(2,2)*Y(3,4)*Y(4,1)+Y(2,4)*Y(3,1)
                            *Y(4,2)-Y(2,4)*Y(3,2)*Y(4,1))-z(4)*(Y(0,0)*Y(2,1)*Y(4,2)-Y(0,0)*Y(2,2)*Y(4,1)-Y(0,1)*Y(2,0)
                            *Y(4,2)+Y(0,1)*Y(2,2)*Y(4,0)+Y(0,2)*Y(2,0)*Y(4,1)-Y(0,2)*Y(2,1)*Y(4,0)+Y(0,0)*Y(3,1)*Y(4,3)
                            -Y(0,0)*Y(3,3)*Y(4,1)-Y(0,1)*Y(3,0)*Y(4,3)+Y(0,1)*Y(3,3)*Y(4,0)+Y(0,3)*Y(3,0)*Y(4,1)-Y(0,3)
                            *Y(3,1)*Y(4,0)-Y(2,1)*Y(3,2)*Y(4,3)+Y(2,1)*Y(3,3)*Y(4,2)+Y(2,2)*Y(3,1)*Y(4,3)-Y(2,2)*Y(3,3)
                            *Y(4,1)-Y(2,3)*Y(3,1)*Y(4,2)+Y(2,3)*Y(3,2)*Y(4,1));
        m_XCoeffs[1][2] = +z(1)*(Y(0,0)*Y(2,2)-Y(0,2)*Y(2,0)+Y(0,0)*Y(3,3)-Y(0,3)*Y(3,0)+Y(0,0)*Y(4,4)-Y(0,4)*Y(4,0)
                            +Y(2,2)*Y(3,3)-Y(2,3)*Y(3,2)+Y(2,2)*Y(4,4)-Y(2,4)*Y(4,2)+Y(3,3)*Y(4,4)-Y(3,4)*Y(4,3))-z(2)
                            *(Y(0,0)*Y(2,1)-Y(0,1)*Y(2,0)+Y(2,1)*Y(3,3)-Y(2,3)*Y(3,1)+Y(2,1)*Y(4,4)-Y(2,4)*Y(4,1))-z(3)
                            *(Y(0,0)*Y(3,1)-Y(0,1)*Y(3,0)-Y(2,1)*Y(3,2)+Y(2,2)*Y(3,1)+Y(3,1)*Y(4,4)-Y(3,4)*Y(4,1))-z(4)
                            *(Y(0,0)*Y(4,1)-Y(0,1)*Y(4,0)-Y(2,1)*Y(4,2)+Y(2,2)*Y(4,1)-Y(3,1)*Y(4,3)+Y(3,3)*Y(4,1))-z(0)
                            *(Y(0,1)*Y(2,2)-Y(0,2)*Y(2,1)+Y(0,1)*Y(3,3)-Y(0,3)*Y(3,1)+Y(0,1)*Y(4,4)-Y(0,4)*Y(4,1));
        m_XCoeffs[1][3] = +Y(0,1)*z(0)+Y(2,1)*z(2)+Y(3,1)*z(3)+Y(4,1)*z(4)-z(1)*(Y(0,0)+Y(2,2)+Y(3,3)+Y(4,4));
        m_XCoeffs[1][4] = +z(1);
        
        /* s=5: Coefficients for polynomial X_{3}(z) */
        m_XCoeffs[2][0] = +z(4)*(Y(0,0)*Y(1,1)*Y(3,2)*Y(4,3)-Y(0,0)*Y(1,1)*Y(3,3)*Y(4,2)-Y(0,0)*Y(1,2)*Y(3,1)*Y(4,3)
                            +Y(0,0)*Y(1,2)*Y(3,3)*Y(4,1)+Y(0,0)*Y(1,3)*Y(3,1)*Y(4,2)-Y(0,0)*Y(1,3)*Y(3,2)*Y(4,1)-Y(0,1)
                            *Y(1,0)*Y(3,2)*Y(4,3)+Y(0,1)*Y(1,0)*Y(3,3)*Y(4,2)+Y(0,1)*Y(1,2)*Y(3,0)*Y(4,3)-Y(0,1)*Y(1,2)
                            *Y(3,3)*Y(4,0)-Y(0,1)*Y(1,3)*Y(3,0)*Y(4,2)+Y(0,1)*Y(1,3)*Y(3,2)*Y(4,0)+Y(0,2)*Y(1,0)*Y(3,1)
                            *Y(4,3)-Y(0,2)*Y(1,0)*Y(3,3)*Y(4,1)-Y(0,2)*Y(1,1)*Y(3,0)*Y(4,3)+Y(0,2)*Y(1,1)*Y(3,3)*Y(4,0)
                            +Y(0,2)*Y(1,3)*Y(3,0)*Y(4,1)-Y(0,2)*Y(1,3)*Y(3,1)*Y(4,0)-Y(0,3)*Y(1,0)*Y(3,1)*Y(4,2)+Y(0,3)
                            *Y(1,0)*Y(3,2)*Y(4,1)+Y(0,3)*Y(1,1)*Y(3,0)*Y(4,2)-Y(0,3)*Y(1,1)*Y(3,2)*Y(4,0)-Y(0,3)*Y(1,2)
                            *Y(3,0)*Y(4,1)+Y(0,3)*Y(1,2)*Y(3,1)*Y(4,0))-z(3)*(Y(0,0)*Y(1,1)*Y(3,2)*Y(4,4)-Y(0,0)*Y(1,1)
                            *Y(3,4)*Y(4,2)-Y(0,0)*Y(1,2)*Y(3,1)*Y(4,4)+Y(0,0)*Y(1,2)*Y(3,4)*Y(4,1)+Y(0,0)*Y(1,4)*Y(3,1)
                            *Y(4,2)-Y(0,0)*Y(1,4)*Y(3,2)*Y(4,1)-Y(0,1)*Y(1,0)*Y(3,2)*Y(4,4)+Y(0,1)*Y(1,0)*Y(3,4)*Y(4,2)
                            +Y(0,1)*Y(1,2)*Y(3,0)*Y(4,4)-Y(0,1)*Y(1,2)*Y(3,4)*Y(4,0)-Y(0,1)*Y(1,4)*Y(3,0)*Y(4,2)+Y(0,1)
                            *Y(1,4)*Y(3,2)*Y(4,0)+Y(0,2)*Y(1,0)*Y(3,1)*Y(4,4)-Y(0,2)*Y(1,0)*Y(3,4)*Y(4,1)-Y(0,2)*Y(1,1)
                            *Y(3,0)*Y(4,4)+Y(0,2)*Y(1,1)*Y(3,4)*Y(4,0)+Y(0,2)*Y(1,4)*Y(3,0)*Y(4,1)-Y(0,2)*Y(1,4)*Y(3,1)
                            *Y(4,0)-Y(0,4)*Y(1,0)*Y(3,1)*Y(4,2)+Y(0,4)*Y(1,0)*Y(3,2)*Y(4,1)+Y(0,4)*Y(1,1)*Y(3,0)*Y(4,2)
                            -Y(0,4)*Y(1,1)*Y(3,2)*Y(4,0)-Y(0,4)*Y(1,2)*Y(3,0)*Y(4,1)+Y(0,4)*Y(1,2)*Y(3,1)*Y(4,0))+z(2)
                            *(Y(0,0)*Y(1,1)*Y(3,3)*Y(4,4)-Y(0,0)*Y(1,1)*Y(3,4)*Y(4,3)-Y(0,0)*Y(1,3)*Y(3,1)*Y(4,4)+Y(0,0)
                            *Y(1,3)*Y(3,4)*Y(4,1)+Y(0,0)*Y(1,4)*Y(3,1)*Y(4,3)-Y(0,0)*Y(1,4)*Y(3,3)*Y(4,1)-Y(0,1)*Y(1,0)
                            *Y(3,3)*Y(4,4)+Y(0,1)*Y(1,0)*Y(3,4)*Y(4,3)+Y(0,1)*Y(1,3)*Y(3,0)*Y(4,4)-Y(0,1)*Y(1,3)*Y(3,4)
                            *Y(4,0)-Y(0,1)*Y(1,4)*Y(3,0)*Y(4,3)+Y(0,1)*Y(1,4)*Y(3,3)*Y(4,0)+Y(0,3)*Y(1,0)*Y(3,1)*Y(4,4)
                            -Y(0,3)*Y(1,0)*Y(3,4)*Y(4,1)-Y(0,3)*Y(1,1)*Y(3,0)*Y(4,4)+Y(0,3)*Y(1,1)*Y(3,4)*Y(4,0)+Y(0,3)
                            *Y(1,4)*Y(3,0)*Y(4,1)-Y(0,3)*Y(1,4)*Y(3,1)*Y(4,0)-Y(0,4)*Y(1,0)*Y(3,1)*Y(4,3)+Y(0,4)*Y(1,0)
                            *Y(3,3)*Y(4,1)+Y(0,4)*Y(1,1)*Y(3,0)*Y(4,3)-Y(0,4)*Y(1,1)*Y(3,3)*Y(4,0)-Y(0,4)*Y(1,3)*Y(3,0)
                            *Y(4,1)+Y(0,4)*Y(1,3)*Y(3,1)*Y(4,0))-z(1)*(Y(0,0)*Y(1,2)*Y(3,3)*Y(4,4)-Y(0,0)*Y(1,2)*Y(3,4)
                            *Y(4,3)-Y(0,0)*Y(1,3)*Y(3,2)*Y(4,4)+Y(0,0)*Y(1,3)*Y(3,4)*Y(4,2)+Y(0,0)*Y(1,4)*Y(3,2)*Y(4,3)
                            -Y(0,0)*Y(1,4)*Y(3,3)*Y(4,2)-Y(0,2)*Y(1,0)*Y(3,3)*Y(4,4)+Y(0,2)*Y(1,0)*Y(3,4)*Y(4,3)+Y(0,2)
                            *Y(1,3)*Y(3,0)*Y(4,4)-Y(0,2)*Y(1,3)*Y(3,4)*Y(4,0)-Y(0,2)*Y(1,4)*Y(3,0)*Y(4,3)+Y(0,2)*Y(1,4)
                            *Y(3,3)*Y(4,0)+Y(0,3)*Y(1,0)*Y(3,2)*Y(4,4)-Y(0,3)*Y(1,0)*Y(3,4)*Y(4,2)-Y(0,3)*Y(1,2)*Y(3,0)
                            *Y(4,4)+Y(0,3)*Y(1,2)*Y(3,4)*Y(4,0)+Y(0,3)*Y(1,4)*Y(3,0)*Y(4,2)-Y(0,3)*Y(1,4)*Y(3,2)*Y(4,0)
                            -Y(0,4)*Y(1,0)*Y(3,2)*Y(4,3)+Y(0,4)*Y(1,0)*Y(3,3)*Y(4,2)+Y(0,4)*Y(1,2)*Y(3,0)*Y(4,3)-Y(0,4)
                            *Y(1,2)*Y(3,3)*Y(4,0)-Y(0,4)*Y(1,3)*Y(3,0)*Y(4,2)+Y(0,4)*Y(1,3)*Y(3,2)*Y(4,0))+z(0)*(Y(0,1)
                            *Y(1,2)*Y(3,3)*Y(4,4)-Y(0,1)*Y(1,2)*Y(3,4)*Y(4,3)-Y(0,1)*Y(1,3)*Y(3,2)*Y(4,4)+Y(0,1)*Y(1,3)
                            *Y(3,4)*Y(4,2)+Y(0,1)*Y(1,4)*Y(3,2)*Y(4,3)-Y(0,1)*Y(1,4)*Y(3,3)*Y(4,2)-Y(0,2)*Y(1,1)*Y(3,3)
                            *Y(4,4)+Y(0,2)*Y(1,1)*Y(3,4)*Y(4,3)+Y(0,2)*Y(1,3)*Y(3,1)*Y(4,4)-Y(0,2)*Y(1,3)*Y(3,4)*Y(4,1)
                            -Y(0,2)*Y(1,4)*Y(3,1)*Y(4,3)+Y(0,2)*Y(1,4)*Y(3,3)*Y(4,1)+Y(0,3)*Y(1,1)*Y(3,2)*Y(4,4)-Y(0,3)
                            *Y(1,1)*Y(3,4)*Y(4,2)-Y(0,3)*Y(1,2)*Y(3,1)*Y(4,4)+Y(0,3)*Y(1,2)*Y(3,4)*Y(4,1)+Y(0,3)*Y(1,4)
                            *Y(3,1)*Y(4,2)-Y(0,3)*Y(1,4)*Y(3,2)*Y(4,1)-Y(0,4)*Y(1,1)*Y(3,2)*Y(4,3)+Y(0,4)*Y(1,1)*Y(3,3)
                            *Y(4,2)+Y(0,4)*Y(1,2)*Y(3,1)*Y(4,3)-Y(0,4)*Y(1,2)*Y(3,3)*Y(4,1)-Y(0,4)*Y(1,3)*Y(3,1)*Y(4,2)
                            +Y(0,4)*Y(1,3)*Y(3,2)*Y(4,1));
        m_XCoeffs[2][1] = +z(1)*(Y(0,0)*Y(1,2)*Y(3,3)-Y(0,0)*Y(1,3)*Y(3,2)-Y(0,2)*Y(1,0)*Y(3,3)+Y(0,2)*Y(1,3)*Y(3,0)
                            +Y(0,3)*Y(1,0)*Y(3,2)-Y(0,3)*Y(1,2)*Y(3,0)+Y(0,0)*Y(1,2)*Y(4,4)-Y(0,0)*Y(1,4)*Y(4,2)-Y(0,2)
                            *Y(1,0)*Y(4,4)+Y(0,2)*Y(1,4)*Y(4,0)+Y(0,4)*Y(1,0)*Y(4,2)-Y(0,4)*Y(1,2)*Y(4,0)+Y(1,2)*Y(3,3)
                            *Y(4,4)-Y(1,2)*Y(3,4)*Y(4,3)-Y(1,3)*Y(3,2)*Y(4,4)+Y(1,3)*Y(3,4)*Y(4,2)+Y(1,4)*Y(3,2)*Y(4,3)
                            -Y(1,4)*Y(3,3)*Y(4,2))-z(0)*(Y(0,1)*Y(1,2)*Y(3,3)-Y(0,1)*Y(1,3)*Y(3,2)-Y(0,2)*Y(1,1)*Y(3,3)
                            +Y(0,2)*Y(1,3)*Y(3,1)+Y(0,3)*Y(1,1)*Y(3,2)-Y(0,3)*Y(1,2)*Y(3,1)+Y(0,1)*Y(1,2)*Y(4,4)-Y(0,1)
                            *Y(1,4)*Y(4,2)-Y(0,2)*Y(1,1)*Y(4,4)+Y(0,2)*Y(1,4)*Y(4,1)+Y(0,4)*Y(1,1)*Y(4,2)-Y(0,4)*Y(1,2)
                            *Y(4,1)-Y(0,2)*Y(3,3)*Y(4,4)+Y(0,2)*Y(3,4)*Y(4,3)+Y(0,3)*Y(3,2)*Y(4,4)-Y(0,3)*Y(3,4)*Y(4,2)
                            -Y(0,4)*Y(3,2)*Y(4,3)+Y(0,4)*Y(3,3)*Y(4,2))-z(2)*(Y(0,0)*Y(1,1)*Y(3,3)-Y(0,0)*Y(1,3)*Y(3,1)
                            -Y(0,1)*Y(1,0)*Y(3,3)+Y(0,1)*Y(1,3)*Y(3,0)+Y(0,3)*Y(1,0)*Y(3,1)-Y(0,3)*Y(1,1)*Y(3,0)+Y(0,0)
                            *Y(1,1)*Y(4,4)-Y(0,0)*Y(1,4)*Y(4,1)-Y(0,1)*Y(1,0)*Y(4,4)+Y(0,1)*Y(1,4)*Y(4,0)+Y(0,4)*Y(1,0)
                            *Y(4,1)-Y(0,4)*Y(1,1)*Y(4,0)+Y(0,0)*Y(3,3)*Y(4,4)-Y(0,0)*Y(3,4)*Y(4,3)-Y(0,3)*Y(3,0)*Y(4,4)
                            +Y(0,3)*Y(3,4)*Y(4,0)+Y(0,4)*Y(3,0)*Y(4,3)-Y(0,4)*Y(3,3)*Y(4,0)+Y(1,1)*Y(3,3)*Y(4,4)-Y(1,1)
                            *Y(3,4)*Y(4,3)-Y(1,3)*Y(3,1)*Y(4,4)+Y(1,3)*Y(3,4)*Y(4,1)+Y(1,4)*Y(3,1)*Y(4,3)-Y(1,4)*Y(3,3)
                            *Y(4,1))+z(3)*(Y(0,0)*Y(1,1)*Y(3,2)-Y(0,0)*Y(1,2)*Y(3,1)-Y(0,1)*Y(1,0)*Y(3,2)+Y(0,1)*Y(1,2)
                            *Y(3,0)+Y(0,2)*Y(1,0)*Y(3,1)-Y(0,2)*Y(1,1)*Y(3,0)+Y(0,0)*Y(3,2)*Y(4,4)-Y(0,0)*Y(3,4)*Y(4,2)
                            -Y(0,2)*Y(3,0)*Y(4,4)+Y(0,2)*Y(3,4)*Y(4,0)+Y(0,4)*Y(3,0)*Y(4,2)-Y(0,4)*Y(3,2)*Y(4,0)+Y(1,1)
                            *Y(3,2)*Y(4,4)-Y(1,1)*Y(3,4)*Y(4,2)-Y(1,2)*Y(3,1)*Y(4,4)+Y(1,2)*Y(3,4)*Y(4,1)+Y(1,4)*Y(3,1)
                            *Y(4,2)-Y(1,4)*Y(3,2)*Y(4,1))+z(4)*(Y(0,0)*Y(1,1)*Y(4,2)-Y(0,0)*Y(1,2)*Y(4,1)-Y(0,1)*Y(1,0)
                            *Y(4,2)+Y(0,1)*Y(1,2)*Y(4,0)+Y(0,2)*Y(1,0)*Y(4,1)-Y(0,2)*Y(1,1)*Y(4,0)-Y(0,0)*Y(3,2)*Y(4,3)
                            +Y(0,0)*Y(3,3)*Y(4,2)+Y(0,2)*Y(3,0)*Y(4,3)-Y(0,2)*Y(3,3)*Y(4,0)-Y(0,3)*Y(3,0)*Y(4,2)+Y(0,3)
                            *Y(3,2)*Y(4,0)-Y(1,1)*Y(3,2)*Y(4,3)+Y(1,1)*Y(3,3)*Y(4,2)+Y(1,2)*Y(3,1)*Y(4,3)-Y(1,2)*Y(3,3)
                            *Y(4,1)-Y(1,3)*Y(3,1)*Y(4,2)+Y(1,3)*Y(3,2)*Y(4,1));
        m_XCoeffs[2][2] = +z(0)*(Y(0,1)*Y(1,2)-Y(0,2)*Y(1,1)-Y(0,2)*Y(3,3)+Y(0,3)*Y(3,2)-Y(0,2)*Y(4,4)+Y(0,4)*Y(4,2))
                            -z(1)*(Y(0,0)*Y(1,2)-Y(0,2)*Y(1,0)+Y(1,2)*Y(3,3)-Y(1,3)*Y(3,2)+Y(1,2)*Y(4,4)-Y(1,4)*Y(4,2))
                            -z(3)*(Y(0,0)*Y(3,2)-Y(0,2)*Y(3,0)+Y(1,1)*Y(3,2)-Y(1,2)*Y(3,1)+Y(3,2)*Y(4,4)-Y(3,4)*Y(4,2))
                            -z(4)*(Y(0,0)*Y(4,2)-Y(0,2)*Y(4,0)+Y(1,1)*Y(4,2)-Y(1,2)*Y(4,1)-Y(3,2)*Y(4,3)+Y(3,3)*Y(4,2))
                            +z(2)*(Y(0,0)*Y(1,1)-Y(0,1)*Y(1,0)+Y(0,0)*Y(3,3)-Y(0,3)*Y(3,0)+Y(0,0)*Y(4,4)-Y(0,4)*Y(4,0)
                            +Y(1,1)*Y(3,3)-Y(1,3)*Y(3,1)+Y(1,1)*Y(4,4)-Y(1,4)*Y(4,1)+Y(3,3)*Y(4,4)-Y(3,4)*Y(4,3));
        m_XCoeffs[2][3] = +Y(0,2)*z(0)+Y(1,2)*z(1)+Y(3,2)*z(3)+Y(4,2)*z(4)-z(2)*(Y(0,0)+Y(1,1)+Y(3,3)+Y(4,4));
        m_XCoeffs[2][4] = +z(2);
        
        /* s=5: Coefficients for polynomial X_{4}(z) */
        m_XCoeffs[3][0] = +z(3)*(Y(0,0)*Y(1,1)*Y(2,2)*Y(4,4)-Y(0,0)*Y(1,1)*Y(2,4)*Y(4,2)-Y(0,0)*Y(1,2)*Y(2,1)*Y(4,4)
                            +Y(0,0)*Y(1,2)*Y(2,4)*Y(4,1)+Y(0,0)*Y(1,4)*Y(2,1)*Y(4,2)-Y(0,0)*Y(1,4)*Y(2,2)*Y(4,1)-Y(0,1)
                            *Y(1,0)*Y(2,2)*Y(4,4)+Y(0,1)*Y(1,0)*Y(2,4)*Y(4,2)+Y(0,1)*Y(1,2)*Y(2,0)*Y(4,4)-Y(0,1)*Y(1,2)
                            *Y(2,4)*Y(4,0)-Y(0,1)*Y(1,4)*Y(2,0)*Y(4,2)+Y(0,1)*Y(1,4)*Y(2,2)*Y(4,0)+Y(0,2)*Y(1,0)*Y(2,1)
                            *Y(4,4)-Y(0,2)*Y(1,0)*Y(2,4)*Y(4,1)-Y(0,2)*Y(1,1)*Y(2,0)*Y(4,4)+Y(0,2)*Y(1,1)*Y(2,4)*Y(4,0)
                            +Y(0,2)*Y(1,4)*Y(2,0)*Y(4,1)-Y(0,2)*Y(1,4)*Y(2,1)*Y(4,0)-Y(0,4)*Y(1,0)*Y(2,1)*Y(4,2)+Y(0,4)
                            *Y(1,0)*Y(2,2)*Y(4,1)+Y(0,4)*Y(1,1)*Y(2,0)*Y(4,2)-Y(0,4)*Y(1,1)*Y(2,2)*Y(4,0)-Y(0,4)*Y(1,2)
                            *Y(2,0)*Y(4,1)+Y(0,4)*Y(1,2)*Y(2,1)*Y(4,0))-z(4)*(Y(0,0)*Y(1,1)*Y(2,2)*Y(4,3)-Y(0,0)*Y(1,1)
                            *Y(2,3)*Y(4,2)-Y(0,0)*Y(1,2)*Y(2,1)*Y(4,3)+Y(0,0)*Y(1,2)*Y(2,3)*Y(4,1)+Y(0,0)*Y(1,3)*Y(2,1)
                            *Y(4,2)-Y(0,0)*Y(1,3)*Y(2,2)*Y(4,1)-Y(0,1)*Y(1,0)*Y(2,2)*Y(4,3)+Y(0,1)*Y(1,0)*Y(2,3)*Y(4,2)
                            +Y(0,1)*Y(1,2)*Y(2,0)*Y(4,3)-Y(0,1)*Y(1,2)*Y(2,3)*Y(4,0)-Y(0,1)*Y(1,3)*Y(2,0)*Y(4,2)+Y(0,1)
                            *Y(1,3)*Y(2,2)*Y(4,0)+Y(0,2)*Y(1,0)*Y(2,1)*Y(4,3)-Y(0,2)*Y(1,0)*Y(2,3)*Y(4,1)-Y(0,2)*Y(1,1)
                            *Y(2,0)*Y(4,3)+Y(0,2)*Y(1,1)*Y(2,3)*Y(4,0)+Y(0,2)*Y(1,3)*Y(2,0)*Y(4,1)-Y(0,2)*Y(1,3)*Y(2,1)
                            *Y(4,0)-Y(0,3)*Y(1,0)*Y(2,1)*Y(4,2)+Y(0,3)*Y(1,0)*Y(2,2)*Y(4,1)+Y(0,3)*Y(1,1)*Y(2,0)*Y(4,2)
                            -Y(0,3)*Y(1,1)*Y(2,2)*Y(4,0)-Y(0,3)*Y(1,2)*Y(2,0)*Y(4,1)+Y(0,3)*Y(1,2)*Y(2,1)*Y(4,0))-z(2)
                            *(Y(0,0)*Y(1,1)*Y(2,3)*Y(4,4)-Y(0,0)*Y(1,1)*Y(2,4)*Y(4,3)-Y(0,0)*Y(1,3)*Y(2,1)*Y(4,4)+Y(0,0)
                            *Y(1,3)*Y(2,4)*Y(4,1)+Y(0,0)*Y(1,4)*Y(2,1)*Y(4,3)-Y(0,0)*Y(1,4)*Y(2,3)*Y(4,1)-Y(0,1)*Y(1,0)
                            *Y(2,3)*Y(4,4)+Y(0,1)*Y(1,0)*Y(2,4)*Y(4,3)+Y(0,1)*Y(1,3)*Y(2,0)*Y(4,4)-Y(0,1)*Y(1,3)*Y(2,4)
                            *Y(4,0)-Y(0,1)*Y(1,4)*Y(2,0)*Y(4,3)+Y(0,1)*Y(1,4)*Y(2,3)*Y(4,0)+Y(0,3)*Y(1,0)*Y(2,1)*Y(4,4)
                            -Y(0,3)*Y(1,0)*Y(2,4)*Y(4,1)-Y(0,3)*Y(1,1)*Y(2,0)*Y(4,4)+Y(0,3)*Y(1,1)*Y(2,4)*Y(4,0)+Y(0,3)
                            *Y(1,4)*Y(2,0)*Y(4,1)-Y(0,3)*Y(1,4)*Y(2,1)*Y(4,0)-Y(0,4)*Y(1,0)*Y(2,1)*Y(4,3)+Y(0,4)*Y(1,0)
                            *Y(2,3)*Y(4,1)+Y(0,4)*Y(1,1)*Y(2,0)*Y(4,3)-Y(0,4)*Y(1,1)*Y(2,3)*Y(4,0)-Y(0,4)*Y(1,3)*Y(2,0)
                            *Y(4,1)+Y(0,4)*Y(1,3)*Y(2,1)*Y(4,0))+z(1)*(Y(0,0)*Y(1,2)*Y(2,3)*Y(4,4)-Y(0,0)*Y(1,2)*Y(2,4)
                            *Y(4,3)-Y(0,0)*Y(1,3)*Y(2,2)*Y(4,4)+Y(0,0)*Y(1,3)*Y(2,4)*Y(4,2)+Y(0,0)*Y(1,4)*Y(2,2)*Y(4,3)
                            -Y(0,0)*Y(1,4)*Y(2,3)*Y(4,2)-Y(0,2)*Y(1,0)*Y(2,3)*Y(4,4)+Y(0,2)*Y(1,0)*Y(2,4)*Y(4,3)+Y(0,2)
                            *Y(1,3)*Y(2,0)*Y(4,4)-Y(0,2)*Y(1,3)*Y(2,4)*Y(4,0)-Y(0,2)*Y(1,4)*Y(2,0)*Y(4,3)+Y(0,2)*Y(1,4)
                            *Y(2,3)*Y(4,0)+Y(0,3)*Y(1,0)*Y(2,2)*Y(4,4)-Y(0,3)*Y(1,0)*Y(2,4)*Y(4,2)-Y(0,3)*Y(1,2)*Y(2,0)
                            *Y(4,4)+Y(0,3)*Y(1,2)*Y(2,4)*Y(4,0)+Y(0,3)*Y(1,4)*Y(2,0)*Y(4,2)-Y(0,3)*Y(1,4)*Y(2,2)*Y(4,0)
                            -Y(0,4)*Y(1,0)*Y(2,2)*Y(4,3)+Y(0,4)*Y(1,0)*Y(2,3)*Y(4,2)+Y(0,4)*Y(1,2)*Y(2,0)*Y(4,3)-Y(0,4)
                            *Y(1,2)*Y(2,3)*Y(4,0)-Y(0,4)*Y(1,3)*Y(2,0)*Y(4,2)+Y(0,4)*Y(1,3)*Y(2,2)*Y(4,0))-z(0)*(Y(0,1)
                            *Y(1,2)*Y(2,3)*Y(4,4)-Y(0,1)*Y(1,2)*Y(2,4)*Y(4,3)-Y(0,1)*Y(1,3)*Y(2,2)*Y(4,4)+Y(0,1)*Y(1,3)
                            *Y(2,4)*Y(4,2)+Y(0,1)*Y(1,4)*Y(2,2)*Y(4,3)-Y(0,1)*Y(1,4)*Y(2,3)*Y(4,2)-Y(0,2)*Y(1,1)*Y(2,3)
                            *Y(4,4)+Y(0,2)*Y(1,1)*Y(2,4)*Y(4,3)+Y(0,2)*Y(1,3)*Y(2,1)*Y(4,4)-Y(0,2)*Y(1,3)*Y(2,4)*Y(4,1)
                            -Y(0,2)*Y(1,4)*Y(2,1)*Y(4,3)+Y(0,2)*Y(1,4)*Y(2,3)*Y(4,1)+Y(0,3)*Y(1,1)*Y(2,2)*Y(4,4)-Y(0,3)
                            *Y(1,1)*Y(2,4)*Y(4,2)-Y(0,3)*Y(1,2)*Y(2,1)*Y(4,4)+Y(0,3)*Y(1,2)*Y(2,4)*Y(4,1)+Y(0,3)*Y(1,4)
                            *Y(2,1)*Y(4,2)-Y(0,3)*Y(1,4)*Y(2,2)*Y(4,1)-Y(0,4)*Y(1,1)*Y(2,2)*Y(4,3)+Y(0,4)*Y(1,1)*Y(2,3)
                            *Y(4,2)+Y(0,4)*Y(1,2)*Y(2,1)*Y(4,3)-Y(0,4)*Y(1,2)*Y(2,3)*Y(4,1)-Y(0,4)*Y(1,3)*Y(2,1)*Y(4,2)
                            +Y(0,4)*Y(1,3)*Y(2,2)*Y(4,1));
        m_XCoeffs[3][1] = +z(0)*(Y(0,1)*Y(1,2)*Y(2,3)-Y(0,1)*Y(1,3)*Y(2,2)-Y(0,2)*Y(1,1)*Y(2,3)+Y(0,2)*Y(1,3)*Y(2,1)
                            +Y(0,3)*Y(1,1)*Y(2,2)-Y(0,3)*Y(1,2)*Y(2,1)-Y(0,1)*Y(1,3)*Y(4,4)+Y(0,1)*Y(1,4)*Y(4,3)+Y(0,3)
                            *Y(1,1)*Y(4,4)-Y(0,3)*Y(1,4)*Y(4,1)-Y(0,4)*Y(1,1)*Y(4,3)+Y(0,4)*Y(1,3)*Y(4,1)-Y(0,2)*Y(2,3)
                            *Y(4,4)+Y(0,2)*Y(2,4)*Y(4,3)+Y(0,3)*Y(2,2)*Y(4,4)-Y(0,3)*Y(2,4)*Y(4,2)-Y(0,4)*Y(2,2)*Y(4,3)
                            +Y(0,4)*Y(2,3)*Y(4,2))-z(3)*(Y(0,0)*Y(1,1)*Y(2,2)-Y(0,0)*Y(1,2)*Y(2,1)-Y(0,1)*Y(1,0)*Y(2,2)
                            +Y(0,1)*Y(1,2)*Y(2,0)+Y(0,2)*Y(1,0)*Y(2,1)-Y(0,2)*Y(1,1)*Y(2,0)+Y(0,0)*Y(1,1)*Y(4,4)-Y(0,0)
                            *Y(1,4)*Y(4,1)-Y(0,1)*Y(1,0)*Y(4,4)+Y(0,1)*Y(1,4)*Y(4,0)+Y(0,4)*Y(1,0)*Y(4,1)-Y(0,4)*Y(1,1)
                            *Y(4,0)+Y(0,0)*Y(2,2)*Y(4,4)-Y(0,0)*Y(2,4)*Y(4,2)-Y(0,2)*Y(2,0)*Y(4,4)+Y(0,2)*Y(2,4)*Y(4,0)
                            +Y(0,4)*Y(2,0)*Y(4,2)-Y(0,4)*Y(2,2)*Y(4,0)+Y(1,1)*Y(2,2)*Y(4,4)-Y(1,1)*Y(2,4)*Y(4,2)-Y(1,2)
                            *Y(2,1)*Y(4,4)+Y(1,2)*Y(2,4)*Y(4,1)+Y(1,4)*Y(2,1)*Y(4,2)-Y(1,4)*Y(2,2)*Y(4,1))-z(1)*(Y(0,0)
                            *Y(1,2)*Y(2,3)-Y(0,0)*Y(1,3)*Y(2,2)-Y(0,2)*Y(1,0)*Y(2,3)+Y(0,2)*Y(1,3)*Y(2,0)+Y(0,3)*Y(1,0)
                            *Y(2,2)-Y(0,3)*Y(1,2)*Y(2,0)-Y(0,0)*Y(1,3)*Y(4,4)+Y(0,0)*Y(1,4)*Y(4,3)+Y(0,3)*Y(1,0)*Y(4,4)
                            -Y(0,3)*Y(1,4)*Y(4,0)-Y(0,4)*Y(1,0)*Y(4,3)+Y(0,4)*Y(1,3)*Y(4,0)+Y(1,2)*Y(2,3)*Y(4,4)-Y(1,2)
                            *Y(2,4)*Y(4,3)-Y(1,3)*Y(2,2)*Y(4,4)+Y(1,3)*Y(2,4)*Y(4,2)+Y(1,4)*Y(2,2)*Y(4,3)-Y(1,4)*Y(2,3)
                            *Y(4,2))+z(2)*(Y(0,0)*Y(1,1)*Y(2,3)-Y(0,0)*Y(1,3)*Y(2,1)-Y(0,1)*Y(1,0)*Y(2,3)+Y(0,1)*Y(1,3)
                            *Y(2,0)+Y(0,3)*Y(1,0)*Y(2,1)-Y(0,3)*Y(1,1)*Y(2,0)+Y(0,0)*Y(2,3)*Y(4,4)-Y(0,0)*Y(2,4)*Y(4,3)
                            -Y(0,3)*Y(2,0)*Y(4,4)+Y(0,3)*Y(2,4)*Y(4,0)+Y(0,4)*Y(2,0)*Y(4,3)-Y(0,4)*Y(2,3)*Y(4,0)+Y(1,1)
                            *Y(2,3)*Y(4,4)-Y(1,1)*Y(2,4)*Y(4,3)-Y(1,3)*Y(2,1)*Y(4,4)+Y(1,3)*Y(2,4)*Y(4,1)+Y(1,4)*Y(2,1)
                            *Y(4,3)-Y(1,4)*Y(2,3)*Y(4,1))+z(4)*(Y(0,0)*Y(1,1)*Y(4,3)-Y(0,0)*Y(1,3)*Y(4,1)-Y(0,1)*Y(1,0)
                            *Y(4,3)+Y(0,1)*Y(1,3)*Y(4,0)+Y(0,3)*Y(1,0)*Y(4,1)-Y(0,3)*Y(1,1)*Y(4,0)+Y(0,0)*Y(2,2)*Y(4,3)
                            -Y(0,0)*Y(2,3)*Y(4,2)-Y(0,2)*Y(2,0)*Y(4,3)+Y(0,2)*Y(2,3)*Y(4,0)+Y(0,3)*Y(2,0)*Y(4,2)-Y(0,3)
                            *Y(2,2)*Y(4,0)+Y(1,1)*Y(2,2)*Y(4,3)-Y(1,1)*Y(2,3)*Y(4,2)-Y(1,2)*Y(2,1)*Y(4,3)+Y(1,2)*Y(2,3)
                            *Y(4,1)+Y(1,3)*Y(2,1)*Y(4,2)-Y(1,3)*Y(2,2)*Y(4,1));
        m_XCoeffs[3][2] = +z(0)*(Y(0,1)*Y(1,3)-Y(0,3)*Y(1,1)+Y(0,2)*Y(2,3)-Y(0,3)*Y(2,2)-Y(0,3)*Y(4,4)+Y(0,4)*Y(4,3))
                            -z(1)*(Y(0,0)*Y(1,3)-Y(0,3)*Y(1,0)-Y(1,2)*Y(2,3)+Y(1,3)*Y(2,2)+Y(1,3)*Y(4,4)-Y(1,4)*Y(4,3))
                            -z(2)*(Y(0,0)*Y(2,3)-Y(0,3)*Y(2,0)+Y(1,1)*Y(2,3)-Y(1,3)*Y(2,1)+Y(2,3)*Y(4,4)-Y(2,4)*Y(4,3))
                            -z(4)*(Y(0,0)*Y(4,3)-Y(0,3)*Y(4,0)+Y(1,1)*Y(4,3)-Y(1,3)*Y(4,1)+Y(2,2)*Y(4,3)-Y(2,3)*Y(4,2))
                            +z(3)*(Y(0,0)*Y(1,1)-Y(0,1)*Y(1,0)+Y(0,0)*Y(2,2)-Y(0,2)*Y(2,0)+Y(1,1)*Y(2,2)-Y(1,2)*Y(2,1)
                            +Y(0,0)*Y(4,4)-Y(0,4)*Y(4,0)+Y(1,1)*Y(4,4)-Y(1,4)*Y(4,1)+Y(2,2)*Y(4,4)-Y(2,4)*Y(4,2));
        m_XCoeffs[3][3] = +Y(0,3)*z(0)+Y(1,3)*z(1)+Y(2,3)*z(2)+Y(4,3)*z(4)-z(3)*(Y(0,0)+Y(1,1)+Y(2,2)+Y(4,4));
        m_XCoeffs[3][4] = +z(3);
        
        /* s=5: Coefficients for polynomial X_{5}(z) */
        m_XCoeffs[4][0] = +z(4)*(Y(0,0)*Y(1,1)*Y(2,2)*Y(3,3)-Y(0,0)*Y(1,1)*Y(2,3)*Y(3,2)-Y(0,0)*Y(1,2)*Y(2,1)*Y(3,3)
                            +Y(0,0)*Y(1,2)*Y(2,3)*Y(3,1)+Y(0,0)*Y(1,3)*Y(2,1)*Y(3,2)-Y(0,0)*Y(1,3)*Y(2,2)*Y(3,1)-Y(0,1)
                            *Y(1,0)*Y(2,2)*Y(3,3)+Y(0,1)*Y(1,0)*Y(2,3)*Y(3,2)+Y(0,1)*Y(1,2)*Y(2,0)*Y(3,3)-Y(0,1)*Y(1,2)
                            *Y(2,3)*Y(3,0)-Y(0,1)*Y(1,3)*Y(2,0)*Y(3,2)+Y(0,1)*Y(1,3)*Y(2,2)*Y(3,0)+Y(0,2)*Y(1,0)*Y(2,1)
                            *Y(3,3)-Y(0,2)*Y(1,0)*Y(2,3)*Y(3,1)-Y(0,2)*Y(1,1)*Y(2,0)*Y(3,3)+Y(0,2)*Y(1,1)*Y(2,3)*Y(3,0)
                            +Y(0,2)*Y(1,3)*Y(2,0)*Y(3,1)-Y(0,2)*Y(1,3)*Y(2,1)*Y(3,0)-Y(0,3)*Y(1,0)*Y(2,1)*Y(3,2)+Y(0,3)
                            *Y(1,0)*Y(2,2)*Y(3,1)+Y(0,3)*Y(1,1)*Y(2,0)*Y(3,2)-Y(0,3)*Y(1,1)*Y(2,2)*Y(3,0)-Y(0,3)*Y(1,2)
                            *Y(2,0)*Y(3,1)+Y(0,3)*Y(1,2)*Y(2,1)*Y(3,0))-z(3)*(Y(0,0)*Y(1,1)*Y(2,2)*Y(3,4)-Y(0,0)*Y(1,1)
                            *Y(2,4)*Y(3,2)-Y(0,0)*Y(1,2)*Y(2,1)*Y(3,4)+Y(0,0)*Y(1,2)*Y(2,4)*Y(3,1)+Y(0,0)*Y(1,4)*Y(2,1)
                            *Y(3,2)-Y(0,0)*Y(1,4)*Y(2,2)*Y(3,1)-Y(0,1)*Y(1,0)*Y(2,2)*Y(3,4)+Y(0,1)*Y(1,0)*Y(2,4)*Y(3,2)
                            +Y(0,1)*Y(1,2)*Y(2,0)*Y(3,4)-Y(0,1)*Y(1,2)*Y(2,4)*Y(3,0)-Y(0,1)*Y(1,4)*Y(2,0)*Y(3,2)+Y(0,1)
                            *Y(1,4)*Y(2,2)*Y(3,0)+Y(0,2)*Y(1,0)*Y(2,1)*Y(3,4)-Y(0,2)*Y(1,0)*Y(2,4)*Y(3,1)-Y(0,2)*Y(1,1)
                            *Y(2,0)*Y(3,4)+Y(0,2)*Y(1,1)*Y(2,4)*Y(3,0)+Y(0,2)*Y(1,4)*Y(2,0)*Y(3,1)-Y(0,2)*Y(1,4)*Y(2,1)
                            *Y(3,0)-Y(0,4)*Y(1,0)*Y(2,1)*Y(3,2)+Y(0,4)*Y(1,0)*Y(2,2)*Y(3,1)+Y(0,4)*Y(1,1)*Y(2,0)*Y(3,2)
                            -Y(0,4)*Y(1,1)*Y(2,2)*Y(3,0)-Y(0,4)*Y(1,2)*Y(2,0)*Y(3,1)+Y(0,4)*Y(1,2)*Y(2,1)*Y(3,0))+z(2)
                            *(Y(0,0)*Y(1,1)*Y(2,3)*Y(3,4)-Y(0,0)*Y(1,1)*Y(2,4)*Y(3,3)-Y(0,0)*Y(1,3)*Y(2,1)*Y(3,4)+Y(0,0)
                            *Y(1,3)*Y(2,4)*Y(3,1)+Y(0,0)*Y(1,4)*Y(2,1)*Y(3,3)-Y(0,0)*Y(1,4)*Y(2,3)*Y(3,1)-Y(0,1)*Y(1,0)
                            *Y(2,3)*Y(3,4)+Y(0,1)*Y(1,0)*Y(2,4)*Y(3,3)+Y(0,1)*Y(1,3)*Y(2,0)*Y(3,4)-Y(0,1)*Y(1,3)*Y(2,4)
                            *Y(3,0)-Y(0,1)*Y(1,4)*Y(2,0)*Y(3,3)+Y(0,1)*Y(1,4)*Y(2,3)*Y(3,0)+Y(0,3)*Y(1,0)*Y(2,1)*Y(3,4)
                            -Y(0,3)*Y(1,0)*Y(2,4)*Y(3,1)-Y(0,3)*Y(1,1)*Y(2,0)*Y(3,4)+Y(0,3)*Y(1,1)*Y(2,4)*Y(3,0)+Y(0,3)
                            *Y(1,4)*Y(2,0)*Y(3,1)-Y(0,3)*Y(1,4)*Y(2,1)*Y(3,0)-Y(0,4)*Y(1,0)*Y(2,1)*Y(3,3)+Y(0,4)*Y(1,0)
                            *Y(2,3)*Y(3,1)+Y(0,4)*Y(1,1)*Y(2,0)*Y(3,3)-Y(0,4)*Y(1,1)*Y(2,3)*Y(3,0)-Y(0,4)*Y(1,3)*Y(2,0)
                            *Y(3,1)+Y(0,4)*Y(1,3)*Y(2,1)*Y(3,0))-z(1)*(Y(0,0)*Y(1,2)*Y(2,3)*Y(3,4)-Y(0,0)*Y(1,2)*Y(2,4)
                            *Y(3,3)-Y(0,0)*Y(1,3)*Y(2,2)*Y(3,4)+Y(0,0)*Y(1,3)*Y(2,4)*Y(3,2)+Y(0,0)*Y(1,4)*Y(2,2)*Y(3,3)
                            -Y(0,0)*Y(1,4)*Y(2,3)*Y(3,2)-Y(0,2)*Y(1,0)*Y(2,3)*Y(3,4)+Y(0,2)*Y(1,0)*Y(2,4)*Y(3,3)+Y(0,2)
                            *Y(1,3)*Y(2,0)*Y(3,4)-Y(0,2)*Y(1,3)*Y(2,4)*Y(3,0)-Y(0,2)*Y(1,4)*Y(2,0)*Y(3,3)+Y(0,2)*Y(1,4)
                            *Y(2,3)*Y(3,0)+Y(0,3)*Y(1,0)*Y(2,2)*Y(3,4)-Y(0,3)*Y(1,0)*Y(2,4)*Y(3,2)-Y(0,3)*Y(1,2)*Y(2,0)
                            *Y(3,4)+Y(0,3)*Y(1,2)*Y(2,4)*Y(3,0)+Y(0,3)*Y(1,4)*Y(2,0)*Y(3,2)-Y(0,3)*Y(1,4)*Y(2,2)*Y(3,0)
                            -Y(0,4)*Y(1,0)*Y(2,2)*Y(3,3)+Y(0,4)*Y(1,0)*Y(2,3)*Y(3,2)+Y(0,4)*Y(1,2)*Y(2,0)*Y(3,3)-Y(0,4)
                            *Y(1,2)*Y(2,3)*Y(3,0)-Y(0,4)*Y(1,3)*Y(2,0)*Y(3,2)+Y(0,4)*Y(1,3)*Y(2,2)*Y(3,0))+z(0)*(Y(0,1)
                            *Y(1,2)*Y(2,3)*Y(3,4)-Y(0,1)*Y(1,2)*Y(2,4)*Y(3,3)-Y(0,1)*Y(1,3)*Y(2,2)*Y(3,4)+Y(0,1)*Y(1,3)
                            *Y(2,4)*Y(3,2)+Y(0,1)*Y(1,4)*Y(2,2)*Y(3,3)-Y(0,1)*Y(1,4)*Y(2,3)*Y(3,2)-Y(0,2)*Y(1,1)*Y(2,3)
                            *Y(3,4)+Y(0,2)*Y(1,1)*Y(2,4)*Y(3,3)+Y(0,2)*Y(1,3)*Y(2,1)*Y(3,4)-Y(0,2)*Y(1,3)*Y(2,4)*Y(3,1)
                            -Y(0,2)*Y(1,4)*Y(2,1)*Y(3,3)+Y(0,2)*Y(1,4)*Y(2,3)*Y(3,1)+Y(0,3)*Y(1,1)*Y(2,2)*Y(3,4)-Y(0,3)
                            *Y(1,1)*Y(2,4)*Y(3,2)-Y(0,3)*Y(1,2)*Y(2,1)*Y(3,4)+Y(0,3)*Y(1,2)*Y(2,4)*Y(3,1)+Y(0,3)*Y(1,4)
                            *Y(2,1)*Y(3,2)-Y(0,3)*Y(1,4)*Y(2,2)*Y(3,1)-Y(0,4)*Y(1,1)*Y(2,2)*Y(3,3)+Y(0,4)*Y(1,1)*Y(2,3)
                            *Y(3,2)+Y(0,4)*Y(1,2)*Y(2,1)*Y(3,3)-Y(0,4)*Y(1,2)*Y(2,3)*Y(3,1)-Y(0,4)*Y(1,3)*Y(2,1)*Y(3,2)
                            +Y(0,4)*Y(1,3)*Y(2,2)*Y(3,1));
        m_XCoeffs[4][1] = +z(0)*(Y(0,1)*Y(1,2)*Y(2,4)-Y(0,1)*Y(1,4)*Y(2,2)-Y(0,2)*Y(1,1)*Y(2,4)+Y(0,2)*Y(1,4)*Y(2,1)
                            +Y(0,4)*Y(1,1)*Y(2,2)-Y(0,4)*Y(1,2)*Y(2,1)+Y(0,1)*Y(1,3)*Y(3,4)-Y(0,1)*Y(1,4)*Y(3,3)-Y(0,3)
                            *Y(1,1)*Y(3,4)+Y(0,3)*Y(1,4)*Y(3,1)+Y(0,4)*Y(1,1)*Y(3,3)-Y(0,4)*Y(1,3)*Y(3,1)+Y(0,2)*Y(2,3)
                            *Y(3,4)-Y(0,2)*Y(2,4)*Y(3,3)-Y(0,3)*Y(2,2)*Y(3,4)+Y(0,3)*Y(2,4)*Y(3,2)+Y(0,4)*Y(2,2)*Y(3,3)
                            -Y(0,4)*Y(2,3)*Y(3,2))-z(4)*(Y(0,0)*Y(1,1)*Y(2,2)-Y(0,0)*Y(1,2)*Y(2,1)-Y(0,1)*Y(1,0)*Y(2,2)
                            +Y(0,1)*Y(1,2)*Y(2,0)+Y(0,2)*Y(1,0)*Y(2,1)-Y(0,2)*Y(1,1)*Y(2,0)+Y(0,0)*Y(1,1)*Y(3,3)-Y(0,0)
                            *Y(1,3)*Y(3,1)-Y(0,1)*Y(1,0)*Y(3,3)+Y(0,1)*Y(1,3)*Y(3,0)+Y(0,3)*Y(1,0)*Y(3,1)-Y(0,3)*Y(1,1)
                            *Y(3,0)+Y(0,0)*Y(2,2)*Y(3,3)-Y(0,0)*Y(2,3)*Y(3,2)-Y(0,2)*Y(2,0)*Y(3,3)+Y(0,2)*Y(2,3)*Y(3,0)
                            +Y(0,3)*Y(2,0)*Y(3,2)-Y(0,3)*Y(2,2)*Y(3,0)+Y(1,1)*Y(2,2)*Y(3,3)-Y(1,1)*Y(2,3)*Y(3,2)-Y(1,2)
                            *Y(2,1)*Y(3,3)+Y(1,2)*Y(2,3)*Y(3,1)+Y(1,3)*Y(2,1)*Y(3,2)-Y(1,3)*Y(2,2)*Y(3,1))-z(1)*(Y(0,0)
                            *Y(1,2)*Y(2,4)-Y(0,0)*Y(1,4)*Y(2,2)-Y(0,2)*Y(1,0)*Y(2,4)+Y(0,2)*Y(1,4)*Y(2,0)+Y(0,4)*Y(1,0)
                            *Y(2,2)-Y(0,4)*Y(1,2)*Y(2,0)+Y(0,0)*Y(1,3)*Y(3,4)-Y(0,0)*Y(1,4)*Y(3,3)-Y(0,3)*Y(1,0)*Y(3,4)
                            +Y(0,3)*Y(1,4)*Y(3,0)+Y(0,4)*Y(1,0)*Y(3,3)-Y(0,4)*Y(1,3)*Y(3,0)-Y(1,2)*Y(2,3)*Y(3,4)+Y(1,2)
                            *Y(2,4)*Y(3,3)+Y(1,3)*Y(2,2)*Y(3,4)-Y(1,3)*Y(2,4)*Y(3,2)-Y(1,4)*Y(2,2)*Y(3,3)+Y(1,4)*Y(2,3)
                            *Y(3,2))+z(2)*(Y(0,0)*Y(1,1)*Y(2,4)-Y(0,0)*Y(1,4)*Y(2,1)-Y(0,1)*Y(1,0)*Y(2,4)+Y(0,1)*Y(1,4)
                            *Y(2,0)+Y(0,4)*Y(1,0)*Y(2,1)-Y(0,4)*Y(1,1)*Y(2,0)-Y(0,0)*Y(2,3)*Y(3,4)+Y(0,0)*Y(2,4)*Y(3,3)
                            +Y(0,3)*Y(2,0)*Y(3,4)-Y(0,3)*Y(2,4)*Y(3,0)-Y(0,4)*Y(2,0)*Y(3,3)+Y(0,4)*Y(2,3)*Y(3,0)-Y(1,1)
                            *Y(2,3)*Y(3,4)+Y(1,1)*Y(2,4)*Y(3,3)+Y(1,3)*Y(2,1)*Y(3,4)-Y(1,3)*Y(2,4)*Y(3,1)-Y(1,4)*Y(2,1)
                            *Y(3,3)+Y(1,4)*Y(2,3)*Y(3,1))+z(3)*(Y(0,0)*Y(1,1)*Y(3,4)-Y(0,0)*Y(1,4)*Y(3,1)-Y(0,1)*Y(1,0)
                            *Y(3,4)+Y(0,1)*Y(1,4)*Y(3,0)+Y(0,4)*Y(1,0)*Y(3,1)-Y(0,4)*Y(1,1)*Y(3,0)+Y(0,0)*Y(2,2)*Y(3,4)
                            -Y(0,0)*Y(2,4)*Y(3,2)-Y(0,2)*Y(2,0)*Y(3,4)+Y(0,2)*Y(2,4)*Y(3,0)+Y(0,4)*Y(2,0)*Y(3,2)-Y(0,4)
                            *Y(2,2)*Y(3,0)+Y(1,1)*Y(2,2)*Y(3,4)-Y(1,1)*Y(2,4)*Y(3,2)-Y(1,2)*Y(2,1)*Y(3,4)+Y(1,2)*Y(2,4)
                            *Y(3,1)+Y(1,4)*Y(2,1)*Y(3,2)-Y(1,4)*Y(2,2)*Y(3,1));
        m_XCoeffs[4][2] = +z(0)*(Y(0,1)*Y(1,4)-Y(0,4)*Y(1,1)+Y(0,2)*Y(2,4)-Y(0,4)*Y(2,2)+Y(0,3)*Y(3,4)-Y(0,4)*Y(3,3))
                            -z(1)*(Y(0,0)*Y(1,4)-Y(0,4)*Y(1,0)-Y(1,2)*Y(2,4)+Y(1,4)*Y(2,2)-Y(1,3)*Y(3,4)+Y(1,4)*Y(3,3))
                            -z(2)*(Y(0,0)*Y(2,4)-Y(0,4)*Y(2,0)+Y(1,1)*Y(2,4)-Y(1,4)*Y(2,1)-Y(2,3)*Y(3,4)+Y(2,4)*Y(3,3))
                            -z(3)*(Y(0,0)*Y(3,4)-Y(0,4)*Y(3,0)+Y(1,1)*Y(3,4)-Y(1,4)*Y(3,1)+Y(2,2)*Y(3,4)-Y(2,4)*Y(3,2))
                            +z(4)*(Y(0,0)*Y(1,1)-Y(0,1)*Y(1,0)+Y(0,0)*Y(2,2)-Y(0,2)*Y(2,0)+Y(0,0)*Y(3,3)-Y(0,3)*Y(3,0)
                            +Y(1,1)*Y(2,2)-Y(1,2)*Y(2,1)+Y(1,1)*Y(3,3)-Y(1,3)*Y(3,1)+Y(2,2)*Y(3,3)-Y(2,3)*Y(3,2));
        m_XCoeffs[4][3] = +Y(0,4)*z(0)+Y(1,4)*z(1)+Y(2,4)*z(2)+Y(3,4)*z(3)-z(4)*(Y(0,0)+Y(1,1)+Y(2,2)+Y(3,3));
        m_XCoeffs[4][4] = +z(4);               
        
    }
    else {
        string msg = "IRK::SetButcherData() Coefficients of polynomials {X_j} not implemented for s = " + to_string(m_s) + "\n";
        mfem_error(msg.c_str());
    }
    
    // Remove zero coefficients that occur if using stiffly accurate IRK scheme
    StiffAccSimplify();
}


/* Modify XCoeffs in instance of a stiffly accurate IRK scheme.

This implementation relies on m_d0 == (0.0,0.0,...,1.0) for stiffly accurate 
schemes, so there can be no rounding errors in m_d0!
*/
void IRK::StiffAccSimplify() {
    // Leave if first s-1 entries are not 0.0
    for (int i = 0; i < m_s-1; i++) if (m_d0(i) != 0.0) return; 
        
    // Leave if last entry is not 1.0
    if (m_d0(m_s-1) != 1.0) return; 
    
    // Truncate last entry from coefficient Vectors of first s-1 polynomials
    for (int i = 0; i < m_s-1; i++) m_XCoeffs[i].SetSize(m_s-1);
}