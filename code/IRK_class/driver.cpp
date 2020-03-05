#include <iostream>
#include <fstream>
#include "mfem.hpp"

#include "IRK.hpp"
#include "SpatialDiscretization.hpp"
#include "FDadvection.hpp"

using namespace mfem;
using namespace std;

/* NOTES:

Solver is a subclass of Operator (see operator.hpp)
GMRESSolver, CGSolver are subclasses of IterativeSolver is a subclass of Solver (see solvers.hpp)
*/

// class FDadvection : SpatialDisretizationOp
// {
// 
// };
// 
// SpatialDisretizationOp::SpatialDisretizationOp(HypreParMatrix &M, HypreParMatrix &L, Vector &g) 
//     : m_M(M), m_L(L), m_g(g) { };
// 
// void SpatialDisretizationOp::GetSolIndepComponent(Vector y, double t)  const {
// }
// 
// // Get y <- P(m_dt*M^{-1}*L)*x
// void SpatialDisretizationOp::PolyMult(const Vector coefficients, const Vector &x, Vector &y) const {    
// 
//     int n = coefficients.size();
//     // 
//     // y = 
//     // 
//     // for (int i = 0; i < n; i++) {
//     // 
//     //     y += coefficients(i)*x
//     // 
//     //     m_z = copy(x);
//     //     L.Mult(x, m_z);
//     // 
//     // 
//     // }
// 
// 
// }
// 
// void SpatialDisretization::Mult(const Vector &x, Vector &y) const 
// {
//     // // y = M^{-1} (K x + b)
//     // K.Mult(x, z);
//     // z += b;
//     // M_solver.Mult(z, y);
// }
// 
// 
// 
// /* Class describing the operator to be inverted. GMRES/CG will compute action using the Mult function here */
// class ConjPairOp : public Operator
// {
// 
// protected:
//     SpatialDisretization m_SD;
// 
// public:
//     ConjPairOp(SpatialDisretization SD);
// 
//     virtual void Mult(const double beta, const double eta, const Vector &x, Vector &y) const;
// };
// /* */
// ConjPairOp::ConjPairOp(SpatialDisretization SD) : Operator(), m_SD(SD) {};
// 
// /* Compute action of the operator to be inverted:
//     If beta != 0: y <- [(eta^2 + beta^2)I - 2*eta*L + L^2]*x 
//     If beta == 0: y <- (eta*I - L)*x
//  */
// void ConjPairOp::Mult(const double beta, const double eta, const Vector &x, Vector &y) const {
// 
//     if (fabs(beta) < 1e-14) {
//         Vector polyCoefficients(2);
//         polyCoefficients(0) = eta;
//         polyCoefficients(1) = -1.0;
//         m_SD.PolyMult(polyCoefficients, x, y);
//     } else {
//         Vector polyCoefficients(3);
//         polyCoefficients(0) = eta*eta + beta*beta;
//         polyCoefficients(1) = -2.0*eta;
//         polyCoefficients(2) = 1.0;
//         m_SD.PolyMult(polyCoefficients, x, y);
//     }
// }


int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    std::cout << "Hello. Test test test: Process " << rank << " of " << numProcess-1 << '\n';

    bool M_exists = false;
    int FD_ID = 3;
    FDadvection SpaceDisc(MPI_COMM_WORLD, M_exists, 1, 4, 1, FD_ID); 

    // SpaceDisc.Test(0.134);
    // SpaceDisc.SaveL();
    // SpaceDisc.SaveM();
    // //SpaceDisc.PrintG();
    // SpaceDisc.SaveU();

    double dt = 0.1;
    int RK_ID = 3;    
    int nt = 10;
    IRK MyIRK(MPI_COMM_WORLD, RK_ID, &SpaceDisc, 0.1, 10);
    
    SpaceDisc.SaveU();
    
    MyIRK.TimeStep();
    
    SpaceDisc.SaveU();
    
    //Test->Test();


    
    
    // SpaceDisc.Test(1.6765);
    // SpaceDisc.PrintL();
    // SpaceDisc.PrintM();
    // SpaceDisc.PrintG();
    // SpaceDisc.PrintU0();
    
    //std::cout << "\n" << '\n';

    MPI_Finalize();
    return 0;
}