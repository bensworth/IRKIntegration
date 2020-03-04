#ifndef IRK_H
#define IRK_H

#include <mpi.h>
#include "HYPRE.h"
#include "mfem.hpp"
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;


class IRK
{
    
private:    
    
    /* Runge-Kutta Butcher tableaux variables */
    int                              m_RK_ID;
    int                              m_s;       /* Number of stages in RK scheme */
    vector<vector<double>> m_A0;      /* Coefficients in matrix A0 */
    vector<double>              m_b0;      /* Weights in vector b */
    vector<double>              m_c0;      /* Coefficients node vector c */
    
    vector<vector<double>> m_invA0;   /* Coefficients in inverse of matrix A0 */
    vector<double>              m_b0tilde; /* Weights in vector b^\top * invA0 */
    vector<double>              m_beta;    /* Imaginary components of eigenvalues of invA0 */
    vector<double>              m_eta;     /* Real components of eigenvalues of invA0 */
    
    vector<vector<double>> m_XCoeffs;                 /* Coefficients for polynomials in block rectangular matrix X  */
    
    /* --- Relating to HYPRE solution of linear systems --- */
    MPI_Comm            m_globComm;            /* Global communicator */
    
protected:    
    
    void SetButcherCoeffs();
    void SetXCoeffs();
    void PolyAction(); /* Compute action of a polynomial on a vector */
    
public:
    IRK(MPI_Comm globComm, int RK_ID);
    ~IRK();
};

#endif