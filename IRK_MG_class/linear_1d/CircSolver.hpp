#ifndef CIRCSOLVER_H
#define CIRCSOLVER_H

#include <fftw3.h>


/** Abstract class for solving real circulant linear systems of the form
        A*x = b
    With circulant matrix A.
*/
class CircSolver {    
    
private:    
    int N;                      // Dimension of system
    double rec_N;               // Reciprocal of N: 1/N
    fftw_complex * eigenvalues; // Eigenvalues of A^{-1}
    
    // Vectors we apply FFT's to which serve as RHS, solution, and auxillary vectors
    mutable fftw_complex * x_fftw;      
    mutable fftw_complex * b_fftw;
    
    // Plans for applying FFT's to b.
    fftw_plan b_fft_plan;
    fftw_plan b_ifft_plan;
    
    
    // Compute element-wise product between x and eigenvalues and store in b
    // Do p <- p*q, for p=c+id, q=y+iz
    // p*q=(c+id)*(y+iz)=(cy-dz) + i(cz+dy)
    void EigMult() const {
        for (int i = 0; i < N; i++) {
            b_fftw[i][0] = x_fftw[i][0]*eigenvalues[i][0]-x_fftw[i][1]*eigenvalues[i][1];
            b_fftw[i][1] = x_fftw[i][0]*eigenvalues[i][1]+x_fftw[i][1]*eigenvalues[i][0];
        }
    }


public:
    
    // Stencil of A is non-zeros in either its first column (COL), or first row (ROW).
    enum StencilOrientation { 
        COL = 0, ROW = 1
    };
    
    /// Set sysyem dimension N, and compute eigenvalues using non-zeros in stencil of A.
    CircSolver(int N_, int stencil_nnz, const int * stencil_inds, const double * stencil_data, 
                StencilOrientation orientation) : N{N_}, rec_N{1.0/N}
    {
        ////////////////////////////////////////////////////
        // --- Initilize things needed in solve phase --- //
        ////////////////////////////////////////////////////
        
        // Allocate memory for memeber vectors
        eigenvalues = (fftw_complex *) fftw_malloc( N * sizeof(fftw_complex) );
        x_fftw      = (fftw_complex *) fftw_malloc( N * sizeof(fftw_complex) );
        b_fftw      = (fftw_complex *) fftw_malloc( N * sizeof(fftw_complex) );
        
        // Create plans for applying FFT's
        // x_fftw <- F*b_fftw
        b_fft_plan = fftw_plan_dft_1d(N, b_fftw, x_fftw, FFTW_FORWARD, FFTW_ESTIMATE);
        // x_fftw <- F^-1*b_fftw
        b_ifft_plan = fftw_plan_dft_1d(N, b_fftw, x_fftw, FFTW_BACKWARD, FFTW_ESTIMATE);
        
        
        
        ///////////////////////////////////////////////////
        // --- Compute and store eigenvalues of A^-1 --- //
        ///////////////////////////////////////////////////
        
        // Do in-place FFT to compute eigenvalues from first column of A.
        fftw_plan eig_fft_plan = fftw_plan_dft_1d(N, eigenvalues, eigenvalues, FFTW_FORWARD, FFTW_ESTIMATE);
        
        // Zero out column of A
        for (int i = 0; i < N; i++) {
            eigenvalues[i][0] = 0.; // Real part
            eigenvalues[i][1] = 0.; // Imag part
        }
        // Add in non-zero real components
        // If have first row of A, need to reorder to get first column for eigenvalue computation
        // 0 -> 0
        // 1 -> n-1
        // 2 -> n-2
        //   ...
        // n-3 -> 3 
        // n-2 -> 2 
        // n-1 -> 1 
        // stencil_inds[i] -> (-stencil_inds[i] + N) % N;
        if (orientation == StencilOrientation::ROW) {
            for (int i = 0; i < stencil_nnz; i++) {
                eigenvalues[(-stencil_inds[i] + N) % N][0] = stencil_data[i]; 
            }
        } else {
            for (int i = 0; i < stencil_nnz; i++) {
                eigenvalues[stencil_inds[i]][0] = stencil_data[i]; 
            }
        }
        
        
        // Compute eigenvalues of A: eigenvalue <- F*eigenvalue
        fftw_execute(eig_fft_plan);
        
        // No longer need this plan since eigenvalues computed once only
        fftw_destroy_plan(eig_fft_plan);
        
        // Invert eigenvalues: eigenvalues <- 1/eigenvalues
        // z <- 1/z, z = c+id
        // 1/z = 1/(c+id)*(c-id)/(c-id) = (c-id)/(c^2 + d^2)
        double mag = 0.;
        for (int i = 0; i < N; i++) {
            mag = eigenvalues[i][0]*eigenvalues[i][0] + eigenvalues[i][1]*eigenvalues[i][1];
            eigenvalues[i][0] /= mag;
            eigenvalues[i][1] /= -mag;
        }
    } 
    
    
    /// Clean up
    ~CircSolver() {
        fftw_destroy_plan(b_fft_plan);
        fftw_destroy_plan(b_ifft_plan);
        fftw_free(eigenvalues);
        fftw_free(b_fftw);
        fftw_free(x_fftw);
    }
    
    /// Solve linear system A*x = b
    void Mult(const double * b, double * x) const {
        
        // Copy b into b_fftw and zero out complex part of b_fftw from past use
        for (int i = 0; i < N; i++) {
            b_fftw[i][0] = b[i]; // Real part
            b_fftw[i][1] = 0.;   // Imag part
        }
        
        // x_fftw <- F*b_fftw
        fftw_execute(b_fft_plan);
        
        // b_fftw <- x_fftw.*eigenvalues
        EigMult();
        
        // x_fftw <- F^-1*b_fftw
        fftw_execute(b_ifft_plan);
        
        // Copy real component of x_fftw into x and scale by 1/N since FFTW routines miss this out.
        for (int i = 0; i < N; i++) {
            x[i] = x_fftw[i][0] * rec_N;
        }
    }
};


#endif