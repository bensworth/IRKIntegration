#include <iostream>
#include <fstream>
#include "mfem.hpp"
#include "IRK.hpp"
using namespace mfem;

int main(int argc, char *argv[])
{
    // Initialize parallel
    int rank, numProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcess);

    std::cout << "Hello. Test test test: Process " << rank << " of " << numProcess-1 << '\n';

    int RK_ID = 3;
    IRK * Test = new IRK(MPI_COMM_WORLD, RK_ID);

    MPI_Finalize();
    return 0;
}