#ifndef IRKOperator_H
#define IRKOperator_H

//#include "IRK.hpp"

#include <mpi.h>
#include "HYPRE.h"
#include "mfem.hpp"
#include <map>
#include <vector>
#include <string>
#include <iostream>

using namespace mfem;
using namespace std;


/* 
Abstract base class for linear spatial discretizations of a PDE resulting in the 
time-dependent ODE M*du/dt = L(t)*u + g(t), u(0) = u0. 
*/
class IRKOperator : public TimeDependentOperator
{    
protected:
    MPI_Comm m_globComm;

public:
    IRKOperator(MPI_Comm comm) : m_globComm{comm} { };
    ~IRKOperator() { };

    // Get y <- M^{-1}*L(t)*x
    virtual void ExplicitMult(const Vector &x, Vector &y) = 0 const;
    
    // Precondition (\gamma*M - dt*L)
    virtual void ImplicitPrec(const Vector &x, Vector &y) = 0 const;

    // Apply mass matrix y = M*x; default implementation assumes M = I
    virtual void ApplyM(const Vector &x, Vector &y) const;

    // Function to ensure that ImplicitPrec preconditions (\gamma*M - dt*L)
    // with gamma and dt as passed to this function.
    //      + index -> index of eigenvalue (pair) in IRK storage
    //      + type -> eigenvalue type, 1 = real, 2 = complex pair
    //      + t -> time.
    // These additional parameters are to provide ways to track when
    // (\gamma*M - dt*L) must be reconstructed or not to minimize setup.
    virtual void SetSystem(int index, double t, double dt,
                           double gamma, int type) = 0;

    // Get y <- P(alpha*M^{-1}*L)*x for P a polynomial defined by coefficients.
    // Coefficients must be provided for all monomial terms (even if they're 0) and 
    // in increasing order (from 0th to nth)
    void PolynomialMult(Vector coefficients, double alpha, const Vector &x, Vector &y) const
    {
        int n = coefficients.Size() - 1;
        y.Set(coefficients[n], x); // y <- coefficients[n]*x
        Vector z(y.Size()); // An auxillary vector
        for (int ell = n-1; ell >= 0; ell--) {
            this->ExplicitMult(y, z); // z <- M^{-1}*L*y       
            add(coefficients[ell], x, alpha, z, y); // y <- coefficients[ell]*x + alpha*z
        } 
    };
};
                            
#endif                            