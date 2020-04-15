#include "FDadvection.hpp"

/* -------------------------------------------------------------------------- */
/* --------------------------- Utility functions ---------------------------- */
/* -------------------------------------------------------------------------- */

// Integer ceiling division
int div_ceil(int numerator, int denominator)
{
        std::div_t res = std::div(numerator, denominator);
        return res.rem ? (res.quot + 1) : res.quot;
}

// Factorial
int factorial(int n) { return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n; };


/* -------------------------------------------------------------------------- */
/* ------------------------- FDApproximation class -------------------------- */
/* -------------------------------------------------------------------------- */

// Operator for which stencil bias is not applicable (i.e. for even-order derivatives)
FDApprox::FDApprox(int derivative, int order, double dx)
{
    FDApprox(derivative, order, dx, CENTRAL);
}

// Operator for which bias is applicable (i.e., odd-order derivatives)
FDApprox::FDApprox(int derivative, int order, double dx, FDBias bias) : 
    m_derivative{derivative}, m_order{order}, m_dx{dx}, m_bias{bias}, 
    m_c_set(false),
    m_c_current(false),
    m_conservative(false),
    m_x{0.0},
    m_x_current(false),
    m_nodes{NULL}, m_plusNodes{NULL}, m_minusNodes{NULL}, 
    m_weights{NULL}, m_localWeights{NULL}, m_plusWeights{NULL}, m_minusWeights{NULL},
    m_delete_nodes(true)
{
    if (bias == UPWIND && m_derivative % 2 == 0) {
        mfem_error("FDApprox::FDApprox() Even-order derivatives cannot use UPWIND bias");
    }
    
    if (m_derivative > 2) {
        mfem_error("FDApprox::FDApprox() Only 1st and 2nd derivatives implememted");
    }
    
    m_size = m_order + 1;
}

FDApprox::~FDApprox() 
{
    if (m_nodes && m_delete_nodes) delete m_nodes; m_nodes = NULL;
    if (m_weights) delete m_weights; m_weights = NULL;
    if (m_localWeights) delete m_localWeights; m_localWeights = NULL;
    if (m_plusNodes) delete m_plusNodes; m_plusNodes = NULL;
    if (m_plusWeights) delete m_plusWeights; m_plusWeights = NULL;
    if (m_minusNodes) delete m_minusNodes; m_minusNodes = NULL;
    if (m_minusWeights) delete m_minusWeights; m_minusWeights = NULL;
}

/* Local approximation to derivative at point x. 
NOTE: User does not assume owndership of nodes and weights, they're simply 
pointed to  */
void FDApprox::GetApprox(int * &nodes, double * &weights)
{
    if (!m_c_set) {
        mfem_error("FDApprox::GetApprox() coefficient not set");
    }
    
    // If neither coefficient nor x updated, nodes & weights are still valid. 
    if (m_c_current && m_x_current) {
        weights = m_weights;
        nodes   = m_nodes;
        return;
    }
    
    // Constant-constant operator
    if (!m_variable) {
        // Free member stencils if previously allocated memory
        if (m_nodes) delete m_nodes; m_nodes = NULL;
        if (m_weights) delete m_weights; m_weights = NULL;
        // D1 
        if (m_derivative == 1) {
            // Upwind D1
            if (m_bias == UPWIND) {
                // "Wind blows" left --> right
                if (m_constant_c >= 0) {
                    GetD1UpwindPlus(m_nodes, m_weights);
                } else {
                    GetD1UpwindMinus(m_nodes, m_weights);
                }
            // Central D1
            } else {
                GetD1Central(m_nodes, m_weights);
            }
        // D2 (central)
        } else if (m_derivative == 2) {
            GetD2Central(m_nodes, m_weights);
        }
        
        // Scale weights by coefficient
        for (int i = 0; i < m_size; i++) {
            m_weights[i] *= m_constant_c;
        }
        
        weights = m_weights;
        nodes   = m_nodes;
        
    // Variable-coefficient operator 
    // Requires computing new weights (and possibly getting local nodes)
    } else if (m_variable_c) {
        
        // This block will only ever be called on the 1st call to this function,
        // it sets appropriate member stencils.
        if (!m_localWeights) {
            // Upwind D1 AND conservative D2 use upwind D1 stencils in both directions
            if ((m_derivative == 1 && m_bias == UPWIND) || (m_derivative == 2 && m_conservative)) {
                GetD1UpwindPlus(m_plusNodes, m_plusWeights);
                GetD1UpwindMinus(m_minusNodes, m_minusWeights);
                // m_nodes will just point to m_plusNodes OR m_minusNodes,
                // which will be used to free the associated data
                m_delete_nodes = false; 
            
            // Central D1 requires central stencil
            } else if (m_derivative == 1 && m_bias == CENTRAL) {
                GetD1Central(m_nodes, m_weights);
            
            // Non-conservative D2 requires central stencil
            } else if (m_derivative == 2 && !m_conservative) {
                GetD2Central(m_nodes, m_weights);
            }
            
            // Allocate memory for local weights 
            m_localWeights = new double[m_size];
        }
        
        // D1
        if (m_derivative == 1) {
            if (m_bias == UPWIND) {
                GetVariableD1Upwind(m_nodes, m_localWeights);
            } else if (m_bias == CENTRAL) {
                GetVariableD1Central(m_localWeights);
            }
        }
        // D2
        else if (m_derivative == 2) {
            GetVariableD2Central(m_localWeights);
        }
        
        weights = m_localWeights;
        nodes = m_nodes;
        
    }
    
    // Member stencils are w.r.t. current x and coefficient
    m_x_current = true;
    m_c_current = true;
}

/* Stencils for upwind discretizations of D1 == d/dx for stencil biased to the left. */
void FDApprox::GetD1UpwindPlus(int * &nodes, double * &weights) const
{    
    nodes   = new int[m_size];
    weights = new double[m_size];
    
    switch (m_order) {
        /* --- Order 1 discretization --- */
        case 1: 
            nodes[0] = -1;
            nodes[1] =  0;
            weights[0] = -1.0;
            weights[1] =  1.0;
            break;
        /* --- Order 2 discretization --- */    
        case 2:
            nodes[0] = -2;
            nodes[1] = -1;
            nodes[2] =  0;
            weights[0] =  1.0/2.0;
            weights[1] = -4.0/2.0;
            weights[2] =  3.0/2.0;
            break;
        /* --- Order 3 discretization --- */    
        case 3:
            nodes[0] = -2;
            nodes[1] = -1;
            nodes[2] =  0;
            nodes[3] =  1;
            weights[0] =  1.0/6.0;
            weights[1] = -6.0/6.0;
            weights[2] =  3.0/6.0;
            weights[3] =  2.0/6.0;
            break;
        /* --- Order 4 discretization --- */    
        case 4:
            nodes[0] = -3;
            nodes[1] = -2;
            nodes[2] = -1;
            nodes[3] =  0;
            nodes[4] =  1;
            weights[0] = -1.0/12.0;
            weights[1] =  6.0/12.0;
            weights[2] = -18.0/12.0;
            weights[3] =  10.0/12.0;
            weights[4] =  3.0/12.0;
            break;
        /* --- Order 5 discretization --- */    
        case 5:   
            nodes[0] = -3;
            nodes[1] = -2;
            nodes[2] = -1;
            nodes[3] =  0;
            nodes[4] =  1;
            nodes[5] =  2;
            weights[0] = -2.0/60.0;
            weights[1] =  15.0/60.0;
            weights[2] = -60.0/60.0;
            weights[3] =  20.0/60.0;
            weights[4] =  30.0/60.0;
            weights[5] = -3.0/60.0;
            break;
        /* --- Order 6 discretization --- */    
        case 6:   
            nodes[0] = -4;
            nodes[1] = -3;
            nodes[2] = -2;
            nodes[3] = -1;
            nodes[4] = +0;
            nodes[5] = +1;
            nodes[6] = +2;
            weights[0] = +1.0/60.0;
            weights[1] = -2.0/15.0;
            weights[2] = +1.0/2.0;
            weights[3] = -4.0/3.0;
            weights[4] = +7.0/12.0;
            weights[5] = +2.0/5.0;
            weights[6] = -1.0/30.0;
            break;
        /* --- Order 7 discretization --- */    
        case 7:
            nodes[0] = -4;
            nodes[1] = -3;
            nodes[2] = -2;
            nodes[3] = -1;
            nodes[4] = +0;
            nodes[5] = +1;
            nodes[6] = +2;
            nodes[7] = +3;
            weights[0] = +1.0/140.0;
            weights[1] = -1.0/15.0;
            weights[2] = +3.0/10.0;
            weights[3] = -1.0/1.0;
            weights[4] = +1.0/4.0;
            weights[5] = +3.0/5.0;
            weights[6] = -1.0/10.0;
            weights[7] = +1.0/105.0;
            break;
        /* --- Order 8 discretization --- */    
        case 8:   
            nodes[0] = -5;
            nodes[1] = -4;
            nodes[2] = -3;
            nodes[3] = -2;
            nodes[4] = -1;
            nodes[5] = +0;
            nodes[6] = +1;
            nodes[7] = +2;
            nodes[8] = +3;
            weights[0] = -1.0/280.0;
            weights[1] = +1.0/28.0;
            weights[2] = -1.0/6.0;
            weights[3] = +1.0/2.0;
            weights[4] = -5.0/4.0;
            weights[5] = +9.0/20.0;
            weights[6] = +1.0/2.0;
            weights[7] = -1.0/14.0;
            weights[8] = +1.0/168.0;
            break;
        /* --- Order 9 discretization --- */    
        case 9:  
            nodes[0] = -5;
            nodes[1] = -4;
            nodes[2] = -3;
            nodes[3] = -2;
            nodes[4] = -1;
            nodes[5] = +0;
            nodes[6] = +1;
            nodes[7] = +2;
            nodes[8] = +3;
            nodes[9] = +4;
            weights[0] = -1.0/630.0;
            weights[1] = +1.0/56.0;
            weights[2] = -2.0/21.0;
            weights[3] = +1.0/3.0;
            weights[4] = -1.0/1.0;
            weights[5] = +1.0/5.0;
            weights[6] = +2.0/3.0;
            weights[7] = -1.0/7.0;
            weights[8] = +1.0/42.0;
            weights[9] = -1.0/504.0;
            break;
        /* --- Order 10 discretization --- */    
        case 10:   
            nodes[0] = -6;
            nodes[1] = -5;
            nodes[2] = -4;
            nodes[3] = -3;
            nodes[4] = -2;
            nodes[5] = -1;
            nodes[6] = +0;
            nodes[7] = +1;
            nodes[8] = +2;
            nodes[9] = +3;
            nodes[10] = +4;
            weights[0] = +1.0/1260.0;
            weights[1] = -1.0/105.0;
            weights[2] = +3.0/56.0;
            weights[3] = -4.0/21.0;
            weights[4] = +1.0/2.0;
            weights[5] = -6.0/5.0;
            weights[6] = +11.0/30.0;
            weights[7] = +4.0/7.0;
            weights[8] = -3.0/28.0;
            weights[9] = +1.0/63.0;
            weights[10] = -1.0/840.0;
            break;
        default:
            std::cout << "FDApprox::GetD1UpwindPlus() invalid discretization. Orders 1--10 implemented.\n";
            MPI_Finalize();
            exit(1);
    }
    
    for (int i = 0; i < m_size; i++) {
        weights[i] /= m_dx;
    }
}

/* Stencils for upwind discretizations of D1 == d/dx for stencil biased to the right. */
void FDApprox::GetD1UpwindMinus(int * &nodes, double * &weights) const
{
    // Get left-bias stencils 
    int    * plusNodes   = NULL;
    double * plusWeights = NULL;
    GetD1UpwindPlus(plusNodes, plusWeights);
    
    // Reverse left-biased stencils and flip sign of weights
    nodes   = new int[m_size];
    weights = new double[m_size];
    for (int i = 0; i < m_size; i++) {
        nodes[i]   = -plusNodes[m_size-1-i];
        weights[i] = -plusWeights[m_size-1-i];
    } 
    delete plusNodes;
    delete plusWeights;
}


/* Stencils for upwind discretizations of variable-coefficient D1 == D1(x)
    D1 == d/dx(c(x) * ) -- conservative == true
    D1 == c(x)*d/dx     -- conservative == false

NOTES:
 - localNodes just returns a pointer to plusNodes/minusNodes depending on wind direction
 - Memory must be already allocated for localWeights   
 */ 
void FDApprox::GetVariableD1Upwind(int * &localNodes, double * &localWeights) const
{   
     
    // Coefficient at point in question; the sign of this determines the upwind direction
    double coefficient0 = m_variable_c(m_x); 
    
    // Wind blows from minus to plus
    if (coefficient0 >= 0.0) {
        localNodes = m_plusNodes;
    
        // Conservative: Need to discretize d/dx(coefficient(x) * )
        if (m_conservative) {
            for (int i = 0; i < m_size; i++) {
                localWeights[i] = m_variable_c(m_x + m_dx*localNodes[i]) * m_plusWeights[i];
            }
    
        // Non-conservative: Need to discretize coefficient(x0)*d/dx    
        } else {
            for (int i = 0; i < m_size; i++) {
                localWeights[i] = coefficient0 * m_plusWeights[i];
            }
        }
    
    // Wind blows from plus to minus
    } else {        
        localNodes = m_minusNodes;
        
        // Conservative: Need to discretize d/dx(coefficient(x) * )
        if (m_conservative) {
            for (int i = 0; i < m_size; i++) {
                localWeights[i] = m_variable_c(m_x + m_dx*localNodes[i]) * m_minusWeights[i];
            }
    
        // Non-conservative: Need to discretize coefficient(x0)*d/dx    
        } else {
            for (int i = 0; i < m_size; i++) {
                localWeights[i] = coefficient0 * m_minusWeights[i];
            }
        }
    }    
}
                            

/* Stencils for central discretizations of D1 == d/dx. 

NOTES:
 -A 0 coefficient for the 0th node is always included so the stencil is contiguous  
*/
void FDApprox::GetD1Central(int * &nodes, double * &weights) const
{    
    nodes   = new int[m_size];
    weights = new double[m_size];

    switch (m_order) {
        /* --- Order 2 discretization --- */
        case 2:
            nodes[0] = -1;
            nodes[1] = +0;
            nodes[2] = +1;
            weights[0] = -1.0/2.0;
            weights[1] = +0.0/1.0;
            weights[2] = +1.0/2.0;
            break;
        /* --- Order 4 discretization --- */
        case 4:
            nodes[0] = -2;
            nodes[1] = -1;
            nodes[2] = +0;
            nodes[3] = +1;
            nodes[4] = +2;
            weights[0] = +1.0/12.0;
            weights[1] = -2.0/3.0;
            weights[2] = +0.0/1.0;
            weights[3] = +2.0/3.0;
            weights[4] = -1.0/12.0;
            break;
        /* --- Order 6 discretization --- */
        case 6:
            nodes[0] = -3;
            nodes[1] = -2;
            nodes[2] = -1;
            nodes[3] = +0;
            nodes[4] = +1;
            nodes[5] = +2;
            nodes[6] = +3;
            weights[0] = -1.0/60.0;
            weights[1] = +3.0/20.0;
            weights[2] = -3.0/4.0;
            weights[3] = +0.0/1.0;
            weights[4] = +3.0/4.0;
            weights[5] = -3.0/20.0;
            weights[6] = +1.0/60.0;
            break;                    
        /* --- Order 8 discretization --- */
        case 8:
            nodes[0] = -4;
            nodes[1] = -3;
            nodes[2] = -2;
            nodes[3] = -1;
            nodes[4] = +0;
            nodes[5] = +1;
            nodes[6] = +2;
            nodes[7] = +3;
            nodes[8] = +4;
            weights[0] = +1.0/280.0;
            weights[1] = -4.0/105.0;
            weights[2] = +1.0/5.0;
            weights[3] = -4.0/5.0;
            weights[4] = +0.0/1.0;
            weights[5] = +4.0/5.0;
            weights[6] = -1.0/5.0;
            weights[7] = +4.0/105.0;
            weights[8] = -1.0/280.0;
            break;        
        /* --- Order 10 discretization --- */
        case 10:
            nodes[0] = -5;
            nodes[1] = -4;
            nodes[2] = -3;
            nodes[3] = -2;
            nodes[4] = -1;
            nodes[5] = +0;
            nodes[6] = +1;
            nodes[7] = +2;
            nodes[8] = +3;
            nodes[9] = +4;
            nodes[10] = +5;
            weights[0] = -1.0/1260.0;
            weights[1] = +5.0/504.0;
            weights[2] = -5.0/84.0;
            weights[3] = +5.0/21.0;
            weights[4] = -5.0/6.0;
            weights[5] = +0.0/1.0;
            weights[6] = +5.0/6.0;
            weights[7] = -5.0/21.0;
            weights[8] = +5.0/84.0;
            weights[9] = -5.0/504.0;
            weights[10] = +1.0/1260.0;
            break;  
        default:
            std::cout << "FDApprox::GetD1Central() invalid discretization. Orders 2,4,6,8,10 implemented.\n";
            MPI_Finalize();
            exit(1);
    }
    
    for (int i = 0; i < m_size; i++) {
        weights[i] /= m_dx;
    }
}

/* Stencils for central discretizations of variable-coefficient D1 == D1(x)
    D1 == d/dx(c(x) * ) -- conservative == true
    D1 == c(x)*d/dx     -- conservative == false

NOTES:
 - Memory must be already allocated for localWeights   
 */ 
void FDApprox::GetVariableD1Central(double * &localWeights) const
{
    
    // Conservative: Need to discretize d/dx(coefficient(x) * )
    if (m_conservative) {
        for (int i = 0; i < m_size; i++) {
            localWeights[i] = m_variable_c(m_x + m_dx*m_nodes[i]) * m_weights[i];
        }

    // Non-conservative: Need to discretize coefficient(x0)*d/dx    
    } else {
        double coefficient0 = m_variable_c(m_x);
        for (int i = 0; i < m_size; i++) {
            localWeights[i] = coefficient0 * m_weights[i];
        }
    }
} 

/* Stencils for central discretizations of D2 == d^2/dx^2. */
void FDApprox::GetD2Central(int * &nodes, double * &weights) const
{   
    nodes   = new int[m_size];
    weights = new double[m_size];

    switch (m_order) {
        /* --- Order 2 discretization --- */
        case 2:
            nodes[0] = -1;
            nodes[1] = +0;
            nodes[2] = +1;
            weights[0] = +1.0/1.0;
            weights[1] = -2.0/1.0;
            weights[2] = +1.0/1.0;
            break;          
        /* --- Order 4 discretization --- */
        case 4:
            nodes[0] = -2;
            nodes[1] = -1;
            nodes[2] = +0;
            nodes[3] = +1;
            nodes[4] = +2;
            weights[0] = -1.0/12.0;
            weights[1] = +4.0/3.0;
            weights[2] = -5.0/2.0;
            weights[3] = +4.0/3.0;
            weights[4] = -1.0/12.0;
            break;          
        /* --- Order 6 discretization --- */
        case 6:
            nodes[0] = -3;
            nodes[1] = -2;
            nodes[2] = -1;
            nodes[3] = +0;
            nodes[4] = +1;
            nodes[5] = +2;
            nodes[6] = +3;
            weights[0] = +1.0/90.0;
            weights[1] = -3.0/20.0;
            weights[2] = +3.0/2.0;
            weights[3] = -49.0/18.0;
            weights[4] = +3.0/2.0;
            weights[5] = -3.0/20.0;
            weights[6] = +1.0/90.0;
            break;
        /* --- Order 8 discretization --- */
        case 8:
            nodes[0] = -4;
            nodes[1] = -3;
            nodes[2] = -2;
            nodes[3] = -1;
            nodes[4] = +0;
            nodes[5] = +1;
            nodes[6] = +2;
            nodes[7] = +3;
            nodes[8] = +4;
            weights[0] = -1.0/560.0;
            weights[1] = +8.0/315.0;
            weights[2] = -1.0/5.0;
            weights[3] = +8.0/5.0;
            weights[4] = -205.0/72.0;
            weights[5] = +8.0/5.0;
            weights[6] = -1.0/5.0;
            weights[7] = +8.0/315.0;
            weights[8] = -1.0/560.0;
            break;
        /* --- Order 10 discretization --- */
        case 10:
            nodes[0] = -5;
            nodes[1] = -4;
            nodes[2] = -3;
            nodes[3] = -2;
            nodes[4] = -1;
            nodes[5] = +0;
            nodes[6] = +1;
            nodes[7] = +2;
            nodes[8] = +3;
            nodes[9] = +4;
            nodes[10] = +5;
            weights[0] = +1.0/3150.0;
            weights[1] = -5.0/1008.0;
            weights[2] = +5.0/126.0;
            weights[3] = -5.0/21.0;
            weights[4] = +5.0/3.0;
            weights[5] = -5269.0/1800.0;
            weights[6] = +5.0/3.0;
            weights[7] = -5.0/21.0;
            weights[8] = +5.0/126.0;
            weights[9] = -5.0/1008.0;
            weights[10] = +1.0/3150.0;
            break;                                  
        default:
            std::cout << "FDApprox::GetD2Central() invalid discretization. Orders 2,4,6,8,10 implemented.\n";
            MPI_Finalize();
            exit(1);
    }
    
    for (int i = 0; i < m_size; i++) {
        weights[i] /= m_dx * m_dx;
    }
}

/* Stencils for central discretizations of variable-coefficient D2 == D2(x)
    D2 == d/dx(coefficient(x)*d/dx() ) -- conservative == true
    D2 == coefficient(x0)*d^2/dx^2     -- conservative == false

NOTES:
 - localCoefficient(index) returns the value of coefficient(x0 + index*dx),
        where x=x0 is the point being discretized at
 - Memory must be already allocated for localWeights   
 */ 
void FDApprox::GetVariableD2Central(double * &localWeights) const
{

    // Conservative: Need to discretize d/dx(coefficient(x)*d/dx )
    if (m_conservative) {
        
        // Need to apply d/dx plus, then d/dx minus or vice versa
        mfem_error("Oops. I'm not yet implememted");

        for (int i = 0; i < m_size; i++) {
            localWeights[i] = 0.0;
        }
        
        

    // Non-conservative: Need to discretize coefficient(x)*d^2/dx^2    
    } else {
        double coefficient0 = m_variable_c(m_x);
        for (int i = 0; i < m_size; i++) {
            localWeights[i] = coefficient0 * m_weights[i];
        }
    }
}

/* -------------------------------------------------------------------------- */
/* ----------------------------- FDMesh class ------------------------------- */
/* -------------------------------------------------------------------------- */

/* Constructor */
FDMesh::FDMesh(MPI_Comm comm, int dim, int refLevels, std::vector<int> np)
    : m_comm{comm}, m_dim{dim}, m_refLevels{refLevels}, m_np{np}, 
    m_nx(dim), m_dx(dim), m_parallel(false)
{    
    
    // Get MPI information
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_commSize);
    
    /* ------------------------------ */
    /* --- Setup grid information --- */
    /* ------------------------------ */
    double nx = pow(2, refLevels);
    double dx = 2.0 / nx; 
    double xboundary0 = -1.0; // Assume x \in [-1,1].
    
    // Size vectors
    m_nx.resize(dim);
    m_dx.resize(dim);
    m_boundary0.resize(dim);
    
    if (dim >= 1) {
        m_nx[0] = nx;
        m_dx[0] = dx;
        m_boundary0[0] = xboundary0;
        m_nxTotal = m_nx[0]; 
    }
    
    // Just make domain in y-direction the same as in x-direction
    if (dim >= 2) {
        int ny = nx;
        double dy = 2.0 / ny; 
        
        m_nx[1] = ny;
        m_dx[1] = dy;
        m_boundary0[1] = xboundary0;
        
        m_nxTotal *= m_nx[1]; 
    }
    
    // These will be updated below if using spatial parallelism.
    m_globOffset = 0; 
    m_nxOnProcTotal = m_nxTotal; 
    
    /* -------------------------------------------------------- */
    /* -------------- Set up spatial parallelism -------------- */
    /* -------------------------------------------------------- */
    /* Ensure spatial parallelism setup is permissible and decide which 
        mesh points current process and its neighbours own, etc */
    if (m_commSize > 1) m_parallel = true;
    
    if (m_parallel) {
        if (m_commSize > m_nxTotal) mfem_error("Number of processors cannot be less than number of mesh points!");
        
        if (!m_np.empty()) {        
            // Ensure passed # of processors is feasible
            int pTotal = 1;
            for (const int &element: m_np) pTotal *= element;
            if (pTotal != m_commSize) mfem_error("MPI communicator has different # of procs than prescribed");
        }
        
        /* --- One spatial dimension --- */
        // x_WEST--> |p[0]|...|p[N-1]| <--x_EAST
        if (m_dim == 1) 
        {
            // Size vectors
            m_np.resize(1);
            m_pIdx.resize(1);
            m_nxOnProc.resize(1);
            m_nxOnProcInt.resize(1);
            
            m_pIdx[0] = m_rank;
            m_np[0]   = m_commSize;
            m_nxOnProcInt[0] = m_nx[0]/m_np[0];
            
            // All procs in interior have same number of points
            if (m_rank < m_np[0]-1)  {
                m_nxOnProc[0] = m_nxOnProcInt[0];  
            // Proc on EAST boundary takes the remainder of points
            } else {
                m_nxOnProc[0] = (m_nx[0] - (m_np[0]-1)*m_nxOnProcInt[0]); 
            }
            m_nxOnProcTotal = m_nxOnProc[0];
            
            // Global index of first point on proc
            m_globOffset = m_pIdx[0] * m_nxOnProcInt[0]; 
            
        /* --- Two spatial dimensions --- */
        } 
        else if (m_dim == 2) 
        {    
            
            /* Setup default, square process grid (user must manually pass dimensions of proc grid if not square) */
            if (m_np.empty()) {
                int temp = sqrt(m_commSize);
                if (temp * temp != m_commSize) {
                    mfem_error("Total # of processors not square; must specify # for each dimension");
                }
                m_np.resize(2);
                m_np[0] = temp; 
                m_np[1] = temp; 
            }
            
            // Size vectors
            m_pIdx.resize(2);
            m_nxOnProc.resize(2);
            m_nxOnProcInt.resize(2);
            m_nxOnProcBnd.resize(2);
            
            // Get indices on proc grid
            m_pIdx[0] = m_rank % m_np[0]; // Proc grid, x index
            m_pIdx[1] = m_rank / m_np[0]; // Proc grid, y index
            
            // Number of DOFs on procs in interior of domain
            m_nxOnProcInt[0] = m_nx[0] / m_np[0];
            m_nxOnProcInt[1] = m_nx[1] / m_np[1];
            
            // Number of DOFs on procs on boundary of proc domain 
            m_nxOnProcBnd[0] = m_nx[0] - (m_np[0]-1)*m_nxOnProcInt[0]; // East boundary
            m_nxOnProcBnd[1] = m_nx[1] - (m_np[1]-1)*m_nxOnProcInt[1]; // North boundary
            
            // Compute number of DOFs on proc
            if (m_pIdx[0] < m_np[0] - 1) {
                m_nxOnProc[0] = m_nxOnProcInt[0]; // All procs in interior have same number of DOFS
            } else {
                m_nxOnProc[0] = m_nxOnProcBnd[0]; // Procs on EAST boundary take the remainder of DOFs
            }
            if (m_pIdx[1] < m_np[1] - 1) {
                m_nxOnProc[1] = m_nxOnProcInt[1]; // All procs in interior have same number of DOFS
            } else {
                m_nxOnProc[1] = m_nxOnProcBnd[1]; // All procs in interior have same number of DOFS
            }
            m_nxOnProcTotal = m_nxOnProc[0] * m_nxOnProc[1]; 
            
            // Compute global index of first DOF on proc
            m_globOffset = m_pIdx[0]*m_nxOnProcInt[0]*m_nxOnProc[1] + m_pIdx[1]*m_nx[0]*m_nxOnProcInt[1];
                        
            /* --- Communicate size information to my four nearest neighbours --- */
            // Assumes we do not have to communicate further than our nearest neighbouring procs...
            // Note: we could work this information out just using the grid setup but it's more fun to send/retrieve from other procs
            // Global proc indices of my nearest neighbours; the processor grid is assumed periodic here to enforce periodic BCs
            int pNInd = m_pIdx[0]  + ((m_pIdx[1]+1) % m_np[1]) * m_np[0];
            int pSInd = m_pIdx[0]  + ((m_pIdx[1]-1 + m_np[1]) % m_np[1]) * m_np[0];
            int pEInd = (m_pIdx[0] + 1) % m_np[0] + m_pIdx[1] * m_np[0];
            int pWInd = (m_pIdx[0] - 1 + m_np[0]) % m_np[0] + m_pIdx[1] * m_np[0];
            
            // Send out index of first DOF I own to my nearest neighbours
            MPI_Send(&m_globOffset, 1, MPI_INT, pNInd, 0, m_comm);
            MPI_Send(&m_globOffset, 1, MPI_INT, pSInd, 0, m_comm);
            MPI_Send(&m_globOffset, 1, MPI_INT, pEInd, 0, m_comm);
            MPI_Send(&m_globOffset, 1, MPI_INT, pWInd, 0, m_comm);
            
            // Recieve index of first DOF owned by my nearest neighbours
            m_nborGlobOffset.resize(4); // Neighbours are ordered as NORTH, SOUTH, EAST, WEST
            MPI_Recv(&m_nborGlobOffset[0], 1, MPI_INT, pNInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborGlobOffset[1], 1, MPI_INT, pSInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborGlobOffset[2], 1, MPI_INT, pEInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborGlobOffset[3], 1, MPI_INT, pWInd, 0, m_comm, MPI_STATUS_IGNORE);
            
            // Send out dimensions of DOFs I own to nearest neighbours
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pNInd, 0, m_comm);
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pSInd, 0, m_comm);
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pEInd, 0, m_comm);
            MPI_Send(&m_nxOnProc[0], 2, MPI_INT, pWInd, 0, m_comm);
            
            // Receive dimensions of DOFs that my nearest neighbours own
            m_nborNxOnProc.resize(8); // Just stack the nx and ny on top of one another in pairs in same vector
            MPI_Recv(&m_nborNxOnProc[0], 2, MPI_INT, pNInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborNxOnProc[2], 2, MPI_INT, pSInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborNxOnProc[4], 2, MPI_INT, pEInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborNxOnProc[6], 2, MPI_INT, pWInd, 0, m_comm, MPI_STATUS_IGNORE);
        }
    }
    std::cout << "Passed FDMesh constructor...\n";
}

// Map grid index to grid point in specified dimension
double FDMesh::MeshIndToPoint(int meshInd, int dim) const
{
    return m_boundary0[dim] + meshInd * m_dx[dim];
}





// Initialize IRKOperator with local problem size, which is defined by the size of the mesh
FDadvection::FDadvection(MPI_Comm comm, const FDMesh &mesh, int order, int problemID)
    : IRKOperator(comm, mesh.m_nxOnProcTotal),
    m_comm{comm},
    m_mesh{mesh},
    m_dim{mesh.m_dim}, 
    m_order{order},
    m_spatialDOFs{mesh.m_nxTotal},
    m_onProcSize{mesh.m_nxOnProcTotal},
    m_problemID{problemID},
    m_periodic(false), m_inflow(false), m_PDE_soln_implemented(false), m_dissipation(false),
    m_I(NULL), m_L(NULL),
    m_L_isTimedependent(true), 
    m_parallel(mesh.m_parallel), m_A_precs(),
    m_AMG_params_type1(), m_AMG_params_type2()
{    
    
    // Seed random number generator so results are consistent!
    srand(0);

    // Get MPI communicator information
    m_rank = m_mesh.m_rank;
    m_commSize = m_mesh.m_commSize;

    /* Set variables based on form of PDE */
    /* Test problems with periodic boundaries */
    if (m_problemID == 1) { /* Constant-coefficient */
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_G_isTimedependent = false;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 2) { /* Variable-coefficient in convervative form */
        m_conservativeForm  = true; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 3) { /* Variable-coefficient in non-convervative form */
        m_conservativeForm  = false; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 4) { /* Spatially variable-coefficient in convervative form */
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
        
    /* Test problems with inflow/outflow boundaries */
    } else if (m_problemID == 101) { /* Constant-coefficient */
        m_conservativeForm  = true; 
        m_L_isTimedependent = false;
        m_G_isTimedependent = true; 
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 102) { /* Variable-coefficient in convervative form */
        m_conservativeForm  = true; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;
    } else if (m_problemID == 103) { /* Variable-coefficient in non-convervative form */
        m_conservativeForm  = false; 
        m_L_isTimedependent = true;
        m_G_isTimedependent = true;
        m_PDE_soln_implemented = true;    
        
    // Quit because wave speeds, sources, IC's etc are not implemented for a `general` problem
    } else { 
        if (m_rank == 0) std::cout << "WARNING: FD problemID == " << m_problemID << " not recognised!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    
    // Set BC flag
    if (m_problemID < 100) {
        m_periodic = true;
    } else if (m_problemID >= 100) {
        m_inflow = true;
        if (m_dim > 1) {
            std::cout << "WARNING: FD with inflow BCs not implemented in 2D" << '\n';
            MPI_Finalize();
            exit(1);
        }
    }
    
    // Set m_L at initial time
    double t = this->GetTime();
    // Get L if not previously obtained
    if (!m_L) SetL(t);
    
    std::cout << "Passed FDadvection constructor...\n";
}


// Mapping between global indexing of unknowns and true mesh indices
// TODO: Add in support here for 2D problem both with and without spatial parallel...
int FDadvection::GlobalIndToMeshInd(int globInd) const
{
    if (m_periodic) {
        return globInd;
    } else {
        return globInd+1; // The solution at inflow boundary is eliminated since it's prescribed by the boundary condition
    }
}


/* Set member variables holding parameters for AMG solve. 
Pass type == 0 to set both type 1 and 2 with same parameters */
void FDadvection::SetAMG_parameters(AMG_parameters parameters, int type) {
    if (type == 0) { 
        m_AMG_params_type1 = parameters;
        m_AMG_params_type2 = parameters;
    } else if (type == 1) {
        m_AMG_params_type1 = parameters;
    } else if (type == 2) {
        m_AMG_params_type2 = parameters;
    }
}

/* -------------------------------------------------------------------------- */
/* --------- Implementation of pure virtual functions in base class --------- */
/* -------------------------------------------------------------------------- */

/* Precondition A == (\gamma*I - dt*L). Just invoke solve function of the already assembled preconditioner */
void FDadvection::ImplicitPrec(const Vector &x, Vector &y) const
{
    m_A_precs[m_A_idx]->Mult(x, y);
}
    
/* Ensure that ImplicitPrec preconditions A == (\gamma*I - dt*L)
with gamma and dt as passed to this function.
     + index -> index of real char poly factor, [0,#number of real factors)
     + type -> eigenvalue type, 1 = real, 2 = complex pair
     + t -> time.
These additional parameters are to provide ways to track when
A == (\gamma*I - dt*L) must be reconstructed or not to minimize setup.

NOTES: I'm assuming that t doesn't matter here since the implementation is ignoring it for the time being
*/
void FDadvection::SetSystem(int index, double t, double dt, double gamma, int type) {
    
    // Set index of preconditioner to be called in ImplicitPrec()
    m_A_idx = index;
    
    // Build new preconditioner since one's not already been built
    if (index >= m_A_precs.Size()) {
        
        if (!m_I) SetI(); // Assemble identity matrix if it's not already been assembled
        
        /* Build J, the operator to be inverted */
        HypreParMatrix * A = new HypreParMatrix(*m_L); // A <- deepcopy(L) - TODO[seg]: where is this deleted? 
        *A *= -dt; // A <- -dt*A
        A->Add(gamma, *m_I); // A <- A + gamma*I
        
        /* Build AMG preconditioner for J */
        HypreBoomerAMG * amg = new HypreBoomerAMG(*A);
        
        // Set AMG paramaters dependent on the type of preconditioner
        AMG_parameters AIR; 
        if (type == 1) {
            AIR = m_AMG_params_type1;
        } else if (type == 2) {
            AIR = m_AMG_params_type2;
        }
        
        amg->SetMaxIter(AIR.maxiter);
        amg->SetLAIROptions(AIR.distance, AIR.prerelax, AIR.postrelax,
                               AIR.strength_tolC, AIR.strength_tolR, 
                               AIR.filter_tolR, AIR.interp_type, 
                               AIR.relax_type, AIR.filterA_tol,
                               AIR.coarsening);
        amg->SetPrintLevel(0);
        amg->SetTol(0.0);
        
        // Save parameter info used to generate preconditioner
        A_parameters A_info = {gamma, dt, index};
        
        // Update member variables with newly formed preconditioner
        m_A_precs.Append(amg);
        m_A_info.Append(A_info);
        
    // Preconditioner already exists for this index, check dt hasn't changed!    
    }
    else {
        if (m_A_info[index].dt != dt) {
            if (m_rank == 0) mfem_error("FDadvection::SetSystem() assumes that dt cannot change with time");
        }
    }
}


/* Compute y <- L*x + g(t) */
void FDadvection::Mult(const Vector &x, Vector &y) const
{
    if (m_L_isTimedependent) {
        if (m_rank == 0) mfem_error("FDadvection::Mult() IRK implementation requires L to be time independent!");
    }
    
    m_L->Mult(x, y);
    
    // Get g(t) and add it to y
    HypreParVector * g = NULL;
    double t = this->GetTime();
    GetG(t, g);
    y.Add(1.0, *g); // y <- y + 1.0*g
    delete g;
}


/* Compute y <- L*x */
void FDadvection::ApplyL(const Vector &x, Vector &y) const
{
    if (m_L_isTimedependent) {
        if (m_rank == 0) mfem_error("FDadvection::ApplyL() IRK implementation requires L to be time independent!");
    }
    
    m_L->Mult(x, y);
}



/* -------------------------------------------------------------------------- */
/* These functions essentially just wrap the CSR spatial discretization into  */
/* MFEM-implemented HYPRE matrices and vectors                                */
/* -------------------------------------------------------------------------- */

// Set member variable m_I
void FDadvection::SetI() {
    if (!m_I) GetHypreParMatrixI(m_I);
}

// Set member variable m_L
void FDadvection::SetL(double t) {
    if (m_L) delete m_L; m_L = NULL;
    GetHypreParMatrixL(t, m_L);
}


/* Get identity mass matrix; parallel distribution is based on what's saved in the
associated member variables of this class */
void FDadvection::GetHypreParMatrixI(HypreParMatrix * &M) const
{
    int    * M_rowptr  = new int[m_onProcSize+1];
    int    * M_colinds = new int[m_onProcSize];
    double * M_data    = new double[m_onProcSize];
    M_rowptr[0] = 0;
    for (int i = 0; i < m_onProcSize; i++) {
        M_colinds[i]   = i + m_mesh.m_globOffset;
        M_data[i]      = 1.0;
        M_rowptr[i+1]  = i+1;
    }
    GetHypreParMatrixFromCSRData(m_comm,  
                                    m_mesh.m_globOffset, m_mesh.m_globOffset + m_onProcSize-1, m_spatialDOFs, 
                                    M_rowptr, M_colinds, M_data,
                                    M); 
    
    /* These are copied into M, so can delete */
    delete[] M_rowptr;
    delete[] M_colinds;
    delete[] M_data;
} 

void FDadvection::GetHypreParMatrixL(double t, HypreParMatrix * &L) const {
    
    double * U = NULL;
    bool getU = false;
    int U_ID = -1;
    int m_size = -1;
    
    int      ilower;
    int      iupper;
    int      spatialDOFs;
    int      onProcSize;
    int    * L_rowptr;
    int    * L_colinds;
    double * L_data;
    
    // No parallelism: Spatial discretization on single processor
    if (!m_parallel) {
        GetSpatialDiscretizationL(L_rowptr, L_colinds, L_data, U,  getU, U_ID, spatialDOFs, t, m_size);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator    
    } else {
        GetSpatialDiscretizationL(m_comm, L_rowptr, L_colinds, L_data, 
                                    U,  getU, U_ID, ilower, iupper, spatialDOFs, 
                                    t, m_size);
    }

    // Flip the sign of L since the base class expects L from du/dt = L*u and not du/dt + L*u = 0
    NegateData(L_rowptr[0], L_rowptr[iupper-ilower+1], L_data);

    GetHypreParMatrixFromCSRData(m_comm,  
                                    ilower, iupper, spatialDOFs, 
                                    L_rowptr, L_colinds, L_data,
                                    L); 
    
    /* These are copied into L, so can delete */
    // TODO[seg]: L_data was causing seg fault for me w/ ImplicitSolve, check ownership.
    // delete L_rowptr;
    // delete L_colinds;
    // delete L_data;
} 


void FDadvection::NegateData(int start, int stop, double * &data) const {
    for (int i = start; i < stop; i++) {
        data[i] *= -1.0;
    }
}


/* Get exact solution (if available) as a HYPRE vector.
Note u owns the data */
bool FDadvection::GetUExact(double t, HypreParVector * &u) const {
    
    // Just retun if not implemented
    if (!m_PDE_soln_implemented) return false;
    
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    double * U;

    // No parallelism: Spatial discretization on single processor
    if (!m_parallel) {
        GetExactPDESolution(U, spatialDOFs, t);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute solution across spatial communicator
    } else {
        GetExactPDESolution(m_comm, U, ilower, iupper, spatialDOFs, t);    
    }    
    GetHypreParVectorFromData(m_comm, 
                             ilower, iupper, spatialDOFs, 
                             U, u);
    // Set owndership flag for data                         
    u->MakeDataOwner();
                             
    return true;
}

/* Get initial condition in an MFEM HypreParVector. 
Note u0 owns the data */
void FDadvection::GetU0(HypreParVector * &u0) const {
    
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    double * U;

    // No parallelism: Spatial discretization on single processor
    if (!m_parallel) {
        GetInitialCondition(U, spatialDOFs);
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator
    } else {
        GetInitialCondition(m_comm, U, ilower, iupper, spatialDOFs);    
    }    
    GetHypreParVectorFromData(m_comm, 
                             ilower, iupper, spatialDOFs, 
                             U, u0);
    // Set owndership flag for data
    u0->MakeDataOwner();
}

/* Get solution-independent source term in an MFEM HypreParVector. 
Note g owns the data */
void FDadvection::GetG(double t, HypreParVector * &g) const {
    
    int      spatialDOFs;
    int      ilower;
    int      iupper;
    double * G;
    
    // No parallelism: Spatial discretization on single processor
    if (!m_parallel) {
        // Call when NOT using spatial parallelism                                        
        GetSpatialDiscretizationG(G, spatialDOFs, t); 
        ilower = 0; 
        iupper = spatialDOFs - 1; 
    // Spatial parallelism: Distribute initial condition across spatial communicator
    } else {
        // Call when using spatial parallelism                          
        GetSpatialDiscretizationG(m_comm, G, ilower, iupper, spatialDOFs, t); 
    }    
    
    GetHypreParVectorFromData(m_comm, ilower, iupper, spatialDOFs, G, g);
                             
    // Set owndership flag for data
    g->MakeDataOwner();
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */




/* Copy external numerical dissipation parameters into member variable */
void FDadvection::SetNumDissipation(Num_dissipation dissipation_params) 
{
    m_dissipation = true;
    m_dissipation_params = dissipation_params;
    
    if (m_dim > 1) {
        if (m_rank == 0) std::cout << "WARNING: Numerical dissipation only implemented for 1D problems..." << '\n';
        MPI_Finalize();
        exit(1);
    }
}



FDadvection::~FDadvection()
{
    if (m_I) delete m_I;
    if (m_L) delete m_L;
    
    // TODO: why doesn't this work??
    // for (int i = 0; i < m_CharPolyPrecs.Size(); i++) {
    //     if (m_CharPolyPrecs[i]) delete m_CharPolyPrecs[i];
    // }
    // TODO: does this free the data??
    if (m_A_precs.Size() > 0) m_A_precs.DeleteAll();
}




/* Return data to be used as an initial iterate. Data depends on integer U0ID */
double FDadvection::GetInitialIterate(double x, int U0ID) const {
    if (U0ID == -1) {  // PDE initial condition
        return InitCond(x);
    } else if (U0ID == 0) { // Zero  
        return 0.0; 
    } else { // Random number in [0,1]
        return (double)rand() / (double)RAND_MAX;
    }
}

/* Return data to be used as an initial iterate. Data depends on integer U0ID */
double FDadvection::GetInitialIterate(double x, double y, int U0ID) const {
    if (U0ID == -1) {  // PDE initial condition
        return InitCond(x, y);
    } else if (U0ID == 0) { // Zero  
        return 0.0; 
    } else { // Random number in [0,1]
        return (double)rand() / (double)RAND_MAX;
    }
}


/* Exact solution for model problems. 

This depends on initial conditions, source terms, wave speeds, and  mesh
So if any of these are updated the solutions given here will be wrong...  */
double FDadvection::PDE_Solution(double x, double t) const {
    if (m_problemID == 1 || m_problemID == 101) {
        return InitCond( std::fmod(x + 1 - t, 2)  - 1 );
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4 || m_problemID == 102 || m_problemID == 103) {
        return cos( PI*(x-t) ) * exp( cos( 2*PI*t ) - 1 );     
    } else {
        return 0.0; // Just so we're not given a compilation warning
    }
}


double FDadvection::PDE_Solution(double x, double y, double t) const {
    if (m_problemID == 1) {
        return InitCond( std::fmod(x + 1 - t, 2) - 1, std::fmod(y + 1 - t, 2)  - 1 );
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4) {
        return cos( PI*(x-t) ) * cos( PI*(y-t) ) * exp( cos( 4*PI*t ) - 1 );     
    } else {
        return 0.0; // Just so we're not given a compilation warning
    }
}

 
double FDadvection::InitCond(double x) const
{        
    if (m_problemID == 1 || m_problemID == 101) {
        return pow(sin(PI * x), 4.0);
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4 || m_problemID == 102 || m_problemID == 103) {
        return cos(PI * x);
    } else {
        return 0.0;
    }
}

double FDadvection::InflowBoundary(double t) const
{
    if (m_problemID == 101) {
        return InitCond(1.0 - WaveSpeed(1.0, t)*t); // Set to be the analytical solution at the RHS boundary
    }  else if (m_problemID == 102 || m_problemID == 103) {
        return PDE_Solution(m_mesh.m_boundary0[0], t);     // Just evaluate the analytical PDE soln on the boundary
    } else {
        return 0.0;
    }
}

double FDadvection::InitCond(double x, double y) const
{        
    if (m_problemID == 1) {
        return pow(cos(PI * x), 4.0) * pow(sin(PI * y), 2.0);
        //return ;
        // if ((x >= 0) && (y >= 0)) return 1.0;
        // if ((x < 0) && (y >= 0)) return 2.0;
        // if ((x < 0) && (y < 0)) return 3.0;
        // if ((x >= 0) && (y < 0)) return 4.0;
    } else if (m_problemID == 2 || m_problemID == 3 || m_problemID == 4) {
        return cos(PI * x) * cos(PI * y);
    } else {
        return 0.0;
    }
}


// Wave speed for 1D problem
// For inflow problems, this MUST be positive near and on the inflow and outflow boundaries!
double FDadvection::WaveSpeed(double x, double t) const {
    if (m_problemID == 1 || m_problemID == 101) {
        return 1.0;
    } else if (m_problemID == 2 || m_problemID == 3) {
        return cos( PI*(x-t) ) * exp( -pow(sin(2*PI*t), 2.0) );
    } else if (m_problemID == 4) {
        return 0.5*(1.0 + pow(cos(PI*x), 2.0));
    } else if (m_problemID == 102 || m_problemID == 103) {
        return 0.5*(1.0 + pow(cos(PI*(x-t)), 2.0)) * exp( -pow(sin(2*PI*t), 2.0) ); 
    }  else  {
        return 0.0;
    }
}


// Wave speed for 2D problem; need to choose component as 1 or 2.
double FDadvection::WaveSpeed(double x, double y, double t, int component) const {
    if (m_problemID == 1) {
        return 1.0;
    } else if (m_problemID == 2 || m_problemID == 3) {
        if (component == 0) {
            return cos( PI*(x-t) ) * cos(PI*y) * exp( -pow(sin(2*PI*t), 2.0) );
        } else {
            return sin(PI*x) * cos( PI*(y-t) ) * exp( -pow(sin(2*PI*t), 2.0) );
        }
    } else if (m_problemID == 4) {
        if (component == 0) {
            return 0.5*(1.0 + pow(cos(PI*x), 2.0)) * 0.5*(1.0 + pow(sin(PI*y), 2.0));
        } else {
            return 0.5*(1.0 + pow(sin(PI*x), 2.0)) * 0.5*(1.0 + pow(cos(PI*y), 2.0));
        }
    } else {
        return 0.0;
    }
}






// RHS of PDE 
double FDadvection::PDE_Source(double x, double t) const
{
    if (m_problemID == 1 || m_problemID == 101) {
        return 0.0;
    } else if (m_problemID == 2) {
        return PI * exp( -2*pow(sin(PI*t), 2.0)*(cos(2*PI*t) + 2) ) * ( sin(2*PI*(t-x)) 
                    - exp( pow(sin(2*PI*t), 2.0) )*(  sin(PI*(t-x)) + 2*sin(2*PI*t)*cos(PI*(t-x)) ) );
    } else if (m_problemID == 3) {
        return 0.5 * PI * exp( -2*pow(sin(PI*t), 2.0)*(cos(2*PI*t) + 2) ) * ( sin(2*PI*(t-x)) 
                    - 2*exp( pow(sin(2*PI*t), 2.0) )*(  sin(PI*(t-x)) + 2*sin(2*PI*t)*cos(PI*(t-x)) ) );
    } else if (m_problemID == 4) {
        return 0.5 * exp( -1.0 + cos(2*PI*t) ) * (-PI*cos(PI*(x-t))*(4*sin(2*PI*t) + sin(2*PI*x)) + PI*pow(sin(PI*x), 2.0)*sin(PI*(x-t)) );
    } else if (m_problemID == 102) {
        return 0.5 * exp(-2.0*(2.0 + cos(2.0*PI*t))*pow(sin(PI*t), 2.0))
                    * ( PI*(1.0 + 3.0*pow(cos(PI*(t-x)), 2.0))*sin(PI*(t-x))
                    - 2.0*PI*exp(pow(sin(2*PI*t), 2.0))*(2.0*cos(PI*(t-x))*sin(2.0*PI*t) + sin(PI*(t-x))) );
    
    } else if (m_problemID == 103) {
        return 0.5 * exp(-2.0*(2.0 + cos(2.0*PI*t))*pow(sin(PI*t), 2.0))
                    * ( PI*(1.0 + pow(cos(PI*(t-x)), 2.0))*sin(PI*(t-x))
                    - 2.0*PI*exp(pow(sin(2*PI*t), 2.0))*(2.0*cos(PI*(t-x))*sin(2.0*PI*t) + sin(PI*(t-x))) );
                    
    } else {
        return 0.0;
    }
}

// RHS of PDE 
double FDadvection::PDE_Source(double x, double y, double t) const
{
    if (m_problemID == 1) {
        return 0.0;
    } else if (m_problemID == 2) {
        return PI*exp(-3*pow(sin(2*PI*t), 2.0)) * 
            (
            cos(PI*(t-y))*( -exp(pow(sin(2*PI*t), 2.0)) * sin(PI*(t-x)) + cos(PI*y)*sin(2*PI*(t-x)) ) +
            cos(PI*(t-x))*( -exp(pow(sin(2*PI*t), 2.0)) * (4*cos(PI*(t-y))*sin(4*PI*t) + sin(PI*(t-y))) + sin(PI*x)*sin(2*PI*(t-y)) )
            );
    } else if (m_problemID == 3) {
        return 0.5*PI*exp(-3*pow(sin(2*PI*t), 2.0)) * 
            (
            cos(PI*(t-y))*( -2*exp(pow(sin(2*PI*t), 2.0)) * sin(PI*(t-x)) + cos(PI*y)*sin(2*PI*(t-x)) ) +
            cos(PI*(t-x))*( -2*exp(pow(sin(2*PI*t), 2.0)) * (4*cos(PI*(t-y))*sin(4*PI*t) + sin(PI*(t-y))) + sin(PI*x)*sin(2*PI*(t-y)) )
            );
    } else if (m_problemID == 4) {
        return -0.25*PI*exp(-1.0 + cos(4*PI*t)) * 
            (
            cos(PI*(y-t))*sin(PI*(x-t))*(-3.0 + pow(sin(PI*y),2.0) + pow(cos(PI*x),2.0)*(1.0+pow(sin(PI*y),2.0)) )
            +
            cos(PI*(x-t)) * 
            (
                (-3.0 + pow(sin(PI*x),2.0) + pow(cos(PI*y),2.0)*(1.0+pow(sin(PI*x),2.0)))*sin(PI*(y-t)) 
                +
                0.5*cos(PI*(y-t))*(32.0*sin(4*PI*t) + 3.0*(sin(2*PI*x) + sin(2*PI*y)) - sin(2*PI*(x+y)) )
            )
            ); 
            
    } else {
        return 0.0;
    }
}



// NO SPATIAL PARALLELISM: Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::GetSpatialDiscretizationL(int * &L_rowptr, int * &L_colinds,
                                           double * &L_data, double * &U0, bool getU0, int U0ID,
                                           int &spatialDOFs, double t, int &bsize) const
{
    if (m_dim == 1) {
        // Simply call the same routine as if using spatial parallelism
        int dummy1, dummy2;
        Get1DSpatialDiscretizationL(NULL, L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID, dummy1, dummy2,
                                      spatialDOFs, t, bsize);                          
    } else if (m_dim == 2) {
        // Call a serial implementation of 2D code
        Get2DSpatialDiscretizationL(L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID,
                                      spatialDOFs, t, bsize);
    }
}

// USING SPATIAL PARALLELISM: Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::GetSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) const
{
    if (m_dim == 1) {
        Get1DSpatialDiscretizationL(globComm, L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID, localMinRow, localMaxRow,
                                      spatialDOFs, t, bsize);
    } else if (m_dim == 2) {
        Get2DSpatialDiscretizationL(globComm, L_rowptr,
                                      L_colinds, L_data, U0,
                                      getU0, U0ID, localMinRow, localMaxRow,
                                      spatialDOFs, t, bsize);
    }
}




// USING SPATIAL PARALLELISM: Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::Get2DSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) const
{
    // Unpack variables frequently used
    int nx          = m_mesh.m_nx[0];
    double dx       = m_mesh.m_dx[0];
    int ny          = m_mesh.m_nx[1];
    double dy       = m_mesh.m_dx[1];
    
    // Initialize FD approximations of each term
    FDApprox xadv(1, m_order, dx, UPWIND);
    xadv.SetVarCoefficientType(m_conservativeForm);
    int xStencilNnz = xadv.GetSize();
    FDApprox yadv(1, m_order, dy, UPWIND);
    yadv.SetVarCoefficientType(m_conservativeForm);
    int yStencilNnz = yadv.GetSize();
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    spatialDOFs   = m_spatialDOFs;                      
    localMinRow   = m_mesh.m_globOffset;                   // First row on proc
    localMaxRow   = localMinRow + m_onProcSize - 1;  // Last row on proc 
    int L_nnz     = (xStencilNnz + yStencilNnz - 1) * m_onProcSize; // Nnz on proc. Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    L_rowptr      = new int[m_onProcSize + 1];
    L_colinds     = new int[L_nnz];
    L_data        = new double[L_nnz];
    L_rowptr[0]   = 0;
    if (getU0) U0 = new double[m_onProcSize]; // Initial guesss at solution
    int rowcount  = 0;
    int dataInd   = 0;
    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // These will just point to an pre-allocated arrays, so don't need memory allocated!
    double * xLocalWeights, * yLocalWeights = NULL;
    int    * xLocalNodes, * yLocalNodes = NULL; 
    
    // Components of wavespeed in each direction
    std::function<double(double)> xWaveSpeed;
    std::function<double(double)> yWaveSpeed; 
    
    int      xIndOnProc; 
    int      yIndOnProc; 
    int      xIndGlobal;
    int      yIndGlobal;
    double   x;
    double   y;
    
    // Given local indices on current process return global index
    std::function<int(int, int, int)> MeshIndsOnProcToGlobalInd = [this, localMinRow](int row, int xIndOnProc, int yIndOnProc) { return localMinRow + xIndOnProc + yIndOnProc*m_mesh.m_nxOnProc[0]; };
    
    // Given connection that overflows in some direction onto a neighbouring process, return global index of that connection. OverFlow variables are positive integers.
    std::function<int(int, int)> MeshIndsOnNorthProcToGlobalInd = [this](int xIndOnProc, int yOverFlow)  { return m_mesh.m_nborGlobOffset[0] + xIndOnProc + (yOverFlow-1)*m_mesh.m_nborNxOnProc[0]; };
    std::function<int(int, int)> MeshIndsOnSouthProcToGlobalInd = [this](int xIndOnProc, int yOverFlow)  { return m_mesh.m_nborGlobOffset[1] + xIndOnProc + (m_mesh.m_nborNxOnProc[3]-yOverFlow)*m_mesh.m_nborNxOnProc[2]; };
    std::function<int(int, int)> MeshIndsOnEastProcToGlobalInd  = [this](int xOverFlow,  int yIndOnProc) { return m_mesh.m_nborGlobOffset[2] + xOverFlow-1  + yIndOnProc*m_mesh.m_nborNxOnProc[4]; };
    std::function<int(int, int)> MeshIndsOnWestProcToGlobalInd  = [this](int xOverFlow,  int yIndOnProc) { return m_mesh.m_nborGlobOffset[3] + m_mesh.m_nborNxOnProc[6]-1 - (xOverFlow-1) + yIndOnProc*m_mesh.m_nborNxOnProc[6]; };
    
    
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {                                          
        xIndOnProc = rowcount % m_mesh.m_nxOnProc[0];                      // x-index on proc
        yIndOnProc = rowcount / m_mesh.m_nxOnProc[0];                      // y-index on proc
        xIndGlobal = m_mesh.m_pIdx[0] * m_mesh.m_nxOnProcInt[0] + xIndOnProc; // Global x-index
        yIndGlobal = m_mesh.m_pIdx[1] * m_mesh.m_nxOnProcInt[1] + yIndOnProc; // Global y-index
        y          = m_mesh.MeshIndToPoint(yIndGlobal, 1);              // y-value of current point
        x          = m_mesh.MeshIndToPoint(xIndGlobal, 0);              // x-value of current point

        // Get stencil for discretizing x-derivative at current point 
        xWaveSpeed = [this, y, t](double x) { return WaveSpeed(x, y, t, 0); };
        xadv.SetX(x);
        xadv.SetCoefficient(xWaveSpeed);
        xadv.GetApprox(xLocalNodes, xLocalWeights);
        
        // Get stencil for discretizing y-derivative at current point 
        yWaveSpeed = [this, x, t](double y) { return WaveSpeed(x, y, t, 1); };
        yadv.SetX(y);
        yadv.SetCoefficient(yWaveSpeed);
        yadv.GetApprox(yLocalNodes, yLocalWeights);
        
        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, then continuing to loop over y
        // Actually, periodicity stuffs this up I think...
        for (int yNzInd = 0; yNzInd < yStencilNnz; yNzInd++) {

            // The two stencils will intersect somewhere at this y-point
            if (yLocalNodes[yNzInd] == 0) {
                for (int xNzInd = 0; xNzInd < xStencilNnz; xNzInd++) {
                    int temp = xIndOnProc + xLocalNodes[xNzInd]; // Local x-index of current connection
                    // Connection to process on WEST side
                    if (temp < 0) {
                        L_colinds[dataInd] = MeshIndsOnWestProcToGlobalInd(abs(temp), yIndOnProc);
                    // Connection to process on EAST side
                    } else if (temp > m_mesh.m_nxOnProc[0]-1) {
                        L_colinds[dataInd] = MeshIndsOnEastProcToGlobalInd(temp - (m_mesh.m_nxOnProc[0]-1), yIndOnProc);
                    // Connection is on processor
                    } else {
                        L_colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, temp, yIndOnProc);
                    }
                    L_data[dataInd]    = xLocalWeights[xNzInd];

                    // The two stencils intersect at this point x-y-point, i.e. they share a 
                    // column in L, so add y-derivative information to x-derivative information that exists there
                    if (xLocalNodes[xNzInd] == 0) L_data[dataInd] += yLocalWeights[yNzInd]; 
                    dataInd += 1;
                }
    
            // There is no possible intersection between between x- and y-stencils
            } else {
                int temp = yIndOnProc + yLocalNodes[yNzInd]; // Local y-index of current connection
                // Connection to process on SOUTH side
                if (temp < 0) {
                    L_colinds[dataInd] = MeshIndsOnSouthProcToGlobalInd(xIndOnProc, abs(temp));
                // Connection to process on NORTH side
                } else if (temp > m_mesh.m_nxOnProc[1]-1) {
                    L_colinds[dataInd] = MeshIndsOnNorthProcToGlobalInd(xIndOnProc, temp - (m_mesh.m_nxOnProc[1]-1));
                // Connection is on processor
                } else {
                    L_colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, xIndOnProc, temp);
                }
                
                L_data[dataInd]    = yLocalWeights[yNzInd];
                dataInd += 1;
            }
        }    
    
        // Set initial guess at the solution
        if (getU0) U0[rowcount] = GetInitialIterate(x, y, U0ID); 
    
        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    }    
    
    // Check that sufficient data was allocated
    if (dataInd > L_nnz) {
        std::cout << "WARNING: FD spatial discretization matrix has more nonzeros than allocated.\n";
    }
}
                             


/* Serial implementation of 2D spatial discretization. Is essentially the same as the
parallel version, but the indexing is made simpler */
void FDadvection::Get2DSpatialDiscretizationL(int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID,
                                              int &spatialDOFs, double t, int &bsize) const
{
    // Unpack variables frequently used
    int nx          = m_mesh.m_nx[0];
    double dx       = m_mesh.m_dx[0];
    int ny          = m_mesh.m_nx[1];
    double dy       = m_mesh.m_dx[1];
    
    // Initialize FD approximations of each term
    FDApprox xadv(1, m_order, dx, UPWIND);
    xadv.SetVarCoefficientType(m_conservativeForm);
    int xStencilNnz = xadv.GetSize();
    FDApprox yadv(1, m_order, dy, UPWIND);
    yadv.SetVarCoefficientType(m_conservativeForm);
    int yStencilNnz = yadv.GetSize();

    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    spatialDOFs     =  m_spatialDOFs;
    int localMinRow = 0;
    int localMaxRow = m_spatialDOFs - 1;
    int L_nnz       = (xStencilNnz + yStencilNnz - 1) * m_onProcSize; // Nnz on proc. Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    L_rowptr        = new int[m_onProcSize + 1];
    L_colinds       = new int[L_nnz];
    L_data          = new double[L_nnz];
    int rowcount    = 0;
    int dataInd     = 0;
    L_rowptr[0]     = 0;
    if (getU0) U0   = new double[m_onProcSize]; // Initial guesss at solution

    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // These will just point to an pre-allocated arrays, so don't need memory allocated!
    int    * xLocalNodes, * yLocalNodes = NULL; 
    double * xLocalWeights, * yLocalWeights = NULL;
        
    // Components of wavespeed in each direction
    std::function<double(double)> xWaveSpeed;
    std::function<double(double)> yWaveSpeed; 
    
    int      xInd;
    int      yInd;
    double   x;
    double   y;
    
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {
        xInd = row % nx;                   // x-index of current point
        yInd = row / nx;                   // y-index of current point
        x    = m_mesh.MeshIndToPoint(xInd, 0); // x-value of current point
        y    = m_mesh.MeshIndToPoint(yInd, 1); // y-value of current point

        // Get stencil for discretizing x-derivative at current point 
        xWaveSpeed = [this, y, t](double x) { return WaveSpeed(x, y, t, 0); };
        xadv.SetX(x);
        xadv.SetCoefficient(xWaveSpeed);
        xadv.GetApprox(xLocalNodes, xLocalWeights);
        
        // Get stencil for discretizing y-derivative at current point 
        yWaveSpeed = [this, x, t](double y) { return WaveSpeed(x, y, t, 1); };
        yadv.SetX(y);
        yadv.SetCoefficient(yWaveSpeed);
        yadv.GetApprox(yLocalNodes, yLocalWeights);

        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, then continuing to loop over y
        // Actually, periodicity stuffs this up I think...
        for (int yNzInd = 0; yNzInd < yStencilNnz; yNzInd++) {
            
            // The two stencils will intersect somewhere at this y-point
            if (yLocalNodes[yNzInd] == 0) {

                for (int xNzInd = 0; xNzInd < xStencilNnz; xNzInd++) {
                    // Account for periodicity here. This always puts resulting x-index in range 0,nx-1
                    L_colinds[dataInd] = ((xInd + xLocalNodes[xNzInd] + nx) % nx) + yInd*nx; 
                    L_data[dataInd]    = xLocalWeights[xNzInd];

                    // The two stencils intersect at this point x-y-point, i.e. they share a 
                    // column in L, so add y-derivative information to x-derivative information that exists there
                    if (xLocalNodes[xNzInd] == 0) {
                        L_data[dataInd] += yLocalWeights[yNzInd]; 
                    }
                    dataInd += 1;
                }

            // There is no possible intersection between between x- and y-stencils
            } else {
                // Account for periodicity here. This always puts resulting y-index in range 0,ny-1
                L_colinds[dataInd] = xInd + ((yInd + yLocalNodes[yNzInd] + ny) % ny)*nx;
                L_data[dataInd]    = yLocalWeights[yNzInd];
                dataInd += 1;
            }
        }    
        
        // Set initial guess at the solution
        if (getU0) U0[rowcount] = GetInitialIterate(x, y, U0ID);

        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;
    } 
    
    // Check that sufficient data was allocated
    if (dataInd > L_nnz) {
        std::cout << "WARNING: FD spatial discretization matrix has more nonzeros than allocated.\n";
    }  
}


// Get local CSR structure of FD spatial discretization matrix, L
void FDadvection::Get1DSpatialDiscretizationL(const MPI_Comm &globComm, int *&L_rowptr,
                                              int *&L_colinds, double *&L_data, double *&U0,
                                              bool getU0, int U0ID, int &localMinRow, int &localMaxRow,
                                              int &spatialDOFs, double t, int &bsize) const
{
    
    // Unpack variables frequently used
    // x-related variables
    int nx          = m_mesh.m_nx[0];
    double dx       = m_mesh.m_dx[0];
    
    // Initialize FD approximation
    FDApprox xadv(1, m_order, dx, UPWIND);
    auto waveSpeed = [this, t](double x) { return WaveSpeed(x, t); };
    xadv.SetCoefficient(waveSpeed, m_conservativeForm);
    int xStencilNnz = xadv.GetSize();
    
    int NnzPerRow = xStencilNnz; // Estimate of NNZ per row of L
    // Get stencil information for numerical dissipation term if there is one, so total NNZ estimate can be updated */
    int      dissNnz     = -1;
    int    * dissInds    = NULL;
    double * dissWeights = NULL;
    if (m_dissipation) {
        dissNnz = m_dissipation_params.degree + 1;
        Get1DDissipationStencil(dissInds, dissWeights, dissNnz);
        
        // kth-degree dissipation uses k/2 points in both directions
        // pth-order upwind uses floor[(p+2)/2] DOFs in upwind direction
        NnzPerRow = 2 * std::max( (m_order + 2)/2, m_dissipation_params.degree/2 ) + 1; // So this is a bound on nnz of total stencil
    } 
    
    /* ----------------------------------------------------------------------- */
    /* ------ Initialize variables needed to compute CSR structure of L ------ */
    /* ----------------------------------------------------------------------- */
    localMinRow  = m_mesh.m_globOffset;                    // First row on proc
    localMaxRow  = m_mesh.m_globOffset + m_onProcSize - 1; // Last row on proc
    spatialDOFs  = m_spatialDOFs;
    int L_nnz    = NnzPerRow * m_onProcSize;  // Nnz on proc. This is a bound. Will always be slightly less than this for inflow/outflow boudaries
    L_rowptr     = new int[m_onProcSize + 1];
    L_colinds    = new int[L_nnz];
    L_data       = new double[L_nnz];
    int rowcount = 0;
    int dataInd  = 0;
    L_rowptr[0]  = 0;
    if (getU0) U0 = new double[m_onProcSize]; // Initial guesss at solution    
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // These will just point to an pre-allocated arrays, so don't need memory allocated!
    double * localWeights = NULL;
    int    * localNodes = NULL;
    
    double x;
    int xInd;
    
         
    // Different components of the domain for inflow/outflow boundaries
    double xIntLeftBndry  = m_mesh.MeshIndToPoint(m_order/2 + 2, 0); // For x < this, stencil has some dependence on inflow
    double xIntRightBndry = m_mesh.MeshIndToPoint(m_mesh.m_nx[0] - div_ceil(m_order, 2) + 1, 0); // For x > this, stencil has some dependence on outflow ghost points

         
    /* ------------------------------------------------------------------- */
    /* ------ Get CSR structure of L for all rows on this processor ------ */
    /* ------------------------------------------------------------------- */
    for (int row = localMinRow; row <= localMaxRow; row++) {
        xInd = GlobalIndToMeshInd(row);    // Mesh index of point we're discretizing at
        x    = m_mesh.MeshIndToPoint(xInd, 0);   // Value of point we're discretizing at
        
        // Get stencil for discretizing x-derivative at current point 
        xadv.SetX(x);
        xadv.GetApprox(localNodes, localWeights);        
                                        
        // Periodic BCs simply wrap stencil at both boundaries
        if (m_periodic) {
            if (!m_dissipation) {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = (localNodes[count] + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }
            
            /* Add numerical dissipation term to advection term. 
                Note: It's easiest to just create new arrays and delete them everytime since the nnz 
                and structure of the summed stencil can change depending on the upwinding direction */
            } else {
                std::map<int, double> sum; // The summed/combined stencil
                Merge1DStencilsIntoMap(localNodes, localWeights, xStencilNnz, dissInds, dissWeights, dissNnz, sum);
                std::map<int, double>::iterator it;
                for (it = sum.begin(); it != sum.end(); it++) {
                    L_colinds[dataInd] = (it->first + row + nx) % nx; // Account for periodicity here. This always puts in range 0,nx-1
                    L_data[dataInd]    = it->second;
                    dataInd += 1;
                }
            }  
        
        // Inflow/outflow boundaries; need to adapt each boundary
        } else if (m_inflow) {
            // Ensure no numerical dissipation: Code set up to handle this... Inflow BCs hard enough on their own
            if (m_dissipation) {
                std::cout << "WARNING: Numerical dissipation not implemented for inflow/outlow BCs!" << '\n';
                MPI_Finalize();
                exit(1);
            }
            
            // DOFs stencil is influenced by inflow and potentially ghost points 
            if (x < xIntLeftBndry) {
                
                /* We simply use the stencil as normal but we truncating it to the interior points  
                with the remaining components are picked up in the solution-independent vector */
                for (int count = 0; count < xStencilNnz; count++) {
                    if (localNodes[count] + row >= 0) { // Only allow dependencies on interior points
                        L_colinds[dataInd] = localNodes[count] + row;
                        L_data[dataInd]    = localWeights[count];
                        dataInd += 1;
                    }
                } 
                
            // DOFs stencil is influenced by ghost points at outflow; need to modify stencil based on extrapolation
            } else if (x > xIntRightBndry) {

                // New stencil for discretization at outflow boundary
                int      xOutflowStencilNnz;
                int    * localOutflowInds;
                double * localOutflowWeights;
                GetOutflowDiscretization(xOutflowStencilNnz, localOutflowWeights, localOutflowInds, xStencilNnz, localWeights, localNodes, 0, xInd); 
                
                // Add in stencil after potentially performing extrapolation
                for (int count = 0; count < xOutflowStencilNnz; count++) {
                    L_colinds[dataInd] = localOutflowInds[count] + row;
                    L_data[dataInd]    = localOutflowWeights[count];
                    dataInd += 1;
                }
                
                delete[] localOutflowInds;
                delete[] localOutflowWeights;
                
            // DOFs stencil only depends on interior DOFs (proceed as normal)
            } else {
                for (int count = 0; count < xStencilNnz; count++) {
                    L_colinds[dataInd] = localNodes[count] + row;
                    L_data[dataInd]    = localWeights[count];
                    dataInd += 1;
                }     
            }
        }     
    
        // Set initial guess at the solution
        if (getU0) U0[rowcount] = GetInitialIterate(x, U0ID);
        
        L_rowptr[rowcount+1] = dataInd;
        rowcount += 1;    
    }  
}


/* Merge two 1D stencils */
void FDadvection::Merge1DStencilsIntoMap(int * indsIn1, double * weightsIn1, int nnzIn1, 
                            int * indsIn2, double * weightsIn2, int nnzIn2,
                            std::map<int, double> &out) const
{
    // Insert first stencil into map
    for (int i = 0; i < nnzIn1; i++) {
        out[indsIn1[i]] = weightsIn1[i];
    }
    
    // Add second stencil into map
    for (int i = 0; i < nnzIn2; i++) {
        out[indsIn2[i]] += weightsIn2[i];
    }
}

// Update stencil at outflow boundary by performing extrapolation of the solution from the interior

// Hard-coded to assume that stencil of any point uses wind blowing from left to right.

// DOFInd is the index of x_{DOFInd}. This is the point we're discretizing at
void FDadvection::GetOutflowDiscretization(int &outflowStencilNnz, double * &localOutflowWeights, int * &localOutflowInds, 
                                    int stencilNnz, double * localWeights, int * localNodes, int dim, int xInd) const
{
    
    int p = m_order; // Interpolation polynomial is of degree at most p-1 (interpolates p DOFs closest to boundary)
    outflowStencilNnz = p;
    
    localOutflowInds    = new int[outflowStencilNnz];
    localOutflowWeights = new double[outflowStencilNnz];
    
    std::map<int, double>::iterator it;
    std::map<int, double> entries; 
    
    // Populate dictionary with stencil information depending on interior DOFs
    int count = 0;
    for (int j = xInd - p/2 - 1; j <= m_mesh.m_nx[dim]; j++) {
        entries[localNodes[count]] = localWeights[count];
        count += 1;
    }
    count -= 1; // Note that localNodes[count] == connection to u_[nx]
    
    // Extrapolation leads to additional coupling to the p DOFs closest to the boundary
    for (int k = 0; k <= p-1; k++) {
        // Coefficient for u_{nx-p+1+k} connection from extrapolation
        double delta = 0.0;
        for (int j = 1; j <= xInd + div_ceil(p, 2) - 1 - m_mesh.m_nx[dim]; j++) {
            delta += localWeights[count + j] * LagrangeOutflowCoefficient(j, k, p);
        }
        // Add weighting for this DOF to interior stencil weights
        entries[ m_mesh.m_nx[dim] - xInd - p + 1 + k ] += delta;
    }
    
    // Copy data from dictionary into array to be returned.
    int dataInd = 0;
    for (it = entries.begin(); it != entries.end(); it++) {
        localOutflowInds[dataInd]    = it->first;
        localOutflowWeights[dataInd] = it->second;
        dataInd += 1;
    }
    
}


// Coefficients of DOFs arising from evaluating Lagrange polynomial at a ghost point
double FDadvection::LagrangeOutflowCoefficient(int i, int k, int p) const
{
    double phi = 1.0;
    for (int ell = 0; ell <= p-1; ell++) {
        if (ell != k) {
            phi *= (i+p-1.0-ell)/(k-ell);
        }
    }
    return phi;
}

                    



// Evaluate grid-function when grid is distributed on a single process
void FDadvection::GetGridFunction(void * GridFunction, 
                                    double * &B, 
                                    int &spatialDOFs) const
{
    spatialDOFs = m_spatialDOFs;
    B = new double[m_spatialDOFs];
    
    // One spatial dimension
    if (m_dim == 1) {
        // Cast function to the correct format
        std::function<double(double)> GridFunction1D = *(std::function<double(double)> *) GridFunction;
        
        for (int xInd = 0; xInd < m_mesh.m_nx[0]; xInd++) {
            B[xInd] = GridFunction1D(m_mesh.MeshIndToPoint(GlobalIndToMeshInd(xInd), 0));
        }
        
    // Two spatial dimensions
    } else if (m_dim == 2) {
        // Cast function to the correct format
        std::function<double(double, double)> GridFunction2D = *(std::function<double(double, double)> *) GridFunction;
        
        int rowInd = 0;
        for (int yInd = 0; yInd < m_mesh.m_nx[1]; yInd++) {
            for (int xInd = 0; xInd < m_mesh.m_nx[0]; xInd++) {
                B[rowInd] = GridFunction2D(m_mesh.MeshIndToPoint(xInd, 0), m_mesh.MeshIndToPoint(yInd, 1));
                rowInd += 1;
            }
        }
    }
}


// Evaluate grid-function when grid is distributed across multiple processes
void FDadvection::GetGridFunction(void * GridFunction, 
                                    const MPI_Comm &globComm, 
                                    double * &B, 
                                    int &localMinRow, 
                                    int &localMaxRow, 
                                    int &spatialDOFs) const
{
    spatialDOFs  = m_spatialDOFs;
    localMinRow  = m_mesh.m_globOffset;                    // First row on process
    localMaxRow  = m_mesh.m_globOffset + m_onProcSize - 1; // Last row on process
    int rowcount = 0;
    B            = new double[m_onProcSize]; 

    // One spatial dimension
    if (m_dim == 1) {
        // Cast function to the correct format
        std::function<double(double)> GridFunction1D = *(std::function<double(double)> *) GridFunction;

        for (int row = localMinRow; row <= localMaxRow; row++) {
            B[rowcount] = GridFunction1D(m_mesh.MeshIndToPoint(GlobalIndToMeshInd(row), 0));
            rowcount += 1;
        }
        
    // Two spatial dimensions
    } else if  (m_dim == 2) {
        // Cast function to the correct format
        std::function<double(double, double)> GridFunction2D = *(std::function<double(double, double)> *) GridFunction;

        int xInd, yInd;      
        for (int row = localMinRow; row <= localMaxRow; row++) {
            xInd = m_mesh.m_pIdx[0] * m_mesh.m_nxOnProcInt[0] + rowcount % m_mesh.m_nxOnProc[0]; // x-index of current point
            yInd = m_mesh.m_pIdx[1] * m_mesh.m_nxOnProcInt[1] + rowcount / m_mesh.m_nxOnProc[0]; // y-index of current point
            B[rowcount] = GridFunction2D(m_mesh.MeshIndToPoint(xInd, 0), m_mesh.MeshIndToPoint(yInd, 1));
            rowcount += 1;
        }
    }
}


// Get PDE solution
bool FDadvection::GetExactPDESolution(const MPI_Comm &globComm, 
                                            double * &U, 
                                            int &localMinRow, 
                                            int &localMaxRow, 
                                            int &spatialDOFs, 
                                            double t) const
{
    if (m_PDE_soln_implemented) {
        // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
        if (m_dim == 1) {
            std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Solution(x, t); };
            GetGridFunction((void *) &GridFunction, globComm, U, localMinRow, localMaxRow, spatialDOFs);
        } else if (m_dim == 2) {
            std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Solution(x, y, t); };
            GetGridFunction((void *) &GridFunction, globComm, U, localMinRow, localMaxRow, spatialDOFs);
        }  
        return true;
    } else {
        return false;
    }
}

bool FDadvection::GetExactPDESolution(double * &U, int &spatialDOFs, double t) const
{
    if (m_PDE_soln_implemented) {
        // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
        if (m_dim == 1) {
            std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Solution(x, t); };
            GetGridFunction((void *) &GridFunction, U, spatialDOFs);
        } else if (m_dim == 2) {
            std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Solution(x, y, t); };
            GetGridFunction((void *) &GridFunction, U, spatialDOFs);
        }  
        return true;
    } else {
        return false;
    }
}

// Get solution-independent component of spatial discretization in vector  G
void FDadvection::GetSpatialDiscretizationG(const MPI_Comm &globComm, 
                                            double * &G, 
                                            int &localMinRow, 
                                            int &localMaxRow, 
                                            int &spatialDOFs, 
                                            double t) const
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Source(x, t); };
        GetGridFunction((void *) &GridFunction, globComm, G, localMinRow, localMaxRow, spatialDOFs);
        
        // Update G with inflow boundary information if necessary
        // All DOFs with coupling to inflow boundary are assumed to be on process 0 (there are very few of them)
        if (m_rank == 0 && m_inflow) AppendInflowStencil1D(G, t);
        
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Source(x, y, t); };
        GetGridFunction((void *) &GridFunction, globComm, G, localMinRow, localMaxRow, spatialDOFs);
    }  
}


// Get solution-independent component of spatial discretization in vector  G
void FDadvection::GetSpatialDiscretizationG(double * &G, int &spatialDOFs, double t) const
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this, t](double x) { return PDE_Source(x, t); };
        GetGridFunction((void *) &GridFunction, G, spatialDOFs);
        
        // Update G with inflow boundary information if necessary
        if (m_inflow) AppendInflowStencil1D(G, t);
        
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this, t](double x, double y) { return PDE_Source(x, y, t); };
        GetGridFunction((void *) &GridFunction, G, spatialDOFs);
    }  
}


/* Get the first p-1 derivatives of u at the inflow boundary at time t0

NOTES:
    -All derivatives are approximated via 2nd-order centred finite differences
    
    -Lots of terms here are explicitly written as functions of time so that higher-order
        derivatives that use them can perform numerical differentiation in time
*/
void FDadvection::GetInflowBoundaryDerivatives1D(double * &du, double t0) const
{
    //int p = m_order-1; // Hack for seeing if I really need all the derivatives...
    int p = m_order; // Order of spatial discretization
    du    = new double[p]; 
    du[0] = InflowBoundary(t0); // The boundary itself. i.e, the 0th-derivative 
    
    /* ------------------- */
    /* --- Compute u_x --- */
    /* ------------------- */
    if (p >= 2) {
        double h = 1e-6;            // Spacing used in FD approximations of derivatives
        double x0 = m_mesh.m_boundary0[0]; // Coordinate of inflow boundary
        
        std::function<double(double)> s_x0 = [&, this](double t) { return PDE_Source(x0, t); };
        std::function<double(double)> a_x0 = [&, this](double t) { return WaveSpeed(x0, t); };
        
        std::function<double(double, double)> a = [this](double x, double t) { return WaveSpeed(x, t); };
        
        // x-derivative of wave speed evaluated at point x0 as a function of time
        std::function<double(double)> dadx_x0 = [&, this](double t) 
                { return GetCentralFDApprox([&, this](double x) { return a(x, t); }, x0, 1, h); };
        
        std::function<double(double)> z    = [this](double t) { return InflowBoundary(t); };
        
        // t-derivative of BC as a function of time
        std::function<double(double)> dzdt = [&, this](double t) { return GetCentralFDApprox(z, t, 1, h); };
        
        // x-derivative of u as a function of time
        std::function<double(double)> dudx;
        
        // Get u_x(x0,t) as a function of time depending on form of PDE
        if (m_conservativeForm) {
            dudx = [&, this](double t) { return (s_x0(t) - dadx_x0(t)*z(t) - dzdt(t))/a_x0(t); };
        } else {
            dudx = [&, this](double t) { return (s_x0(t) - dzdt(t))/a_x0(t); };
        }
        
        // Evaluate u_x(x0,t) at time t0
        du[1] = dudx(t0);
        
        /* -------------------- */
        /* --- Compute u_xx --- */
        /* -------------------- */
        if (p >= 3) {
            std::function<double(double, double)> s = [this](double x, double t) { return PDE_Source(x, t); };
            // x-derivative of source evaluated at point x0 as a function of time
            std::function<double(double)> dsdx_x0 = [&, this](double t) 
                    { return GetCentralFDApprox([&, this](double x) { return s(x, t); }, x0, 1, h); };
            
            // xx-derivative of wave speed evaluated at point x0 as a function of time
            std::function<double(double)> d2adx2_x0 = [&, this](double t) 
                    { return GetCentralFDApprox([&, this](double x) { return a(x, t); }, x0, 2, h); };
            
            // x-derivative of reciprocal of wave speed evaluated at point x0 as a function of time
            std::function<double(double)> dradx_x0 = [&, this](double t) 
                    { return GetCentralFDApprox([&, this](double x) { return 1.0/a(x, t); }, x0, 1, h); };
            
            // xt-derivative of u as a function of time
            std::function<double(double)> d2udxdt = [&, this](double t) { return GetCentralFDApprox(dudx, t, 1, h); };
            
            // xx-derivative of u as a function of time
            std::function<double(double)> d2udx2;
            
            // Get u_xx(x0,t) as a function of time depending on form of PDE
            if (m_conservativeForm) {
                d2udx2 = [&, this](double t) { return (dsdx_x0(t) - d2adx2_x0(t)*z(t) - dadx_x0(t)*dudx(t) - d2udxdt(t))/a_x0(t) 
                                                + dradx_x0(t)*(s_x0(t) - dadx_x0(t)*z(t) - dzdt(t)); };
            } else {
                d2udx2 = [&, this](double t) { return (dsdx_x0(t) - d2udxdt(t))/a_x0(t) 
                                                + dradx_x0(t)*(s_x0(t) - dzdt(t)); };
            }
            
            // Evaluate u_xx(x0,t) at time t0
            du[2] = d2udx2(t0);
            //du[0] = 0.0;
            
            if (p >= 4)  {
                std::cout << "WARNING: Inflow derivatives only implemented up to degree 2\n";
                MPI_Finalize();
                exit(1);
            }
        }
    }
}

// Return a central approximation of order-th derivative of f centred at x0
double FDadvection::GetCentralFDApprox(std::function<double(double)> f, double x0, int order, double h) const {
    
    // Just use 2nd-order approximations
    if (order == 1) {
        return (- 0.5*f(x0-h) + 0.5*f(x0+h))/h;
    } else if (order == 2) {
        return (f(x0-h) -2*f(x0) + f(x0+h))/(h*h);
    } else if (order == 3) {
        return (-0.5*f(x0-2*h) + f(x0-h) - f(x0+h) + 0.5*f(x0+2*h))/(h*h*h);
    } else {
        std::cout << "WARNING: FD approximations for derivative of order " << order << " not implemented!" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
}


// Get values at inflow boundary and ghost points associated with it using inverse Lax--Wendroff procedure
void FDadvection::GetInflowValues(std::map<int, double> &uGhost, double t, int dim) const
{
    
    double * du;
    GetInflowBoundaryDerivatives1D(du, t);
    uGhost[0] = du[0]; // The inflow boundary value itself
    
    // Approximate solution at p/2 ghost points using Taylor series based at inflow
    for (int i = -1; i >= -m_order/2; i--) {
        uGhost[i] = du[0]; // 0th-order derivative
        for (int k = 1; k <= m_order-1; k++) { // True value??
        //for (int k = 1; k <= m_order-2; k++) { // Works almost identically???
            uGhost[i] += pow(i*m_mesh.m_dx[dim], k) / factorial(k) * du[k];
        }
    }
}


// Update solution-independent term discretization information pertaining to inflow boundary
// Hard-coded to assume that wind blows left to right for these points near the boundary...
void FDadvection::AppendInflowStencil1D(double * &G, double t) const {
    
    // Unpack variables frequently used
    int nx          = m_mesh.m_nx[0];
    double dx       = m_mesh.m_dx[0];
    int xFD_Order   = m_order;
    int xStencilNnz = xFD_Order + 1; // Width of the FD stencil
    int xDim        = 0;
    
    
    /* --- Get solution at inflow and ghost points --- */
    std::map<int, double> uGhost; // Use dictionary so we can access data via its physical grid index    
    GetInflowValues(uGhost, t, xDim);
    
    /* ---------------------------------------------------------------- */
    /* ------ Get components required to approximate derivatives ------ */
    /* ---------------------------------------------------------------- */
    // These will just point to an pre-allocated arrays, so don't need memory allocated!
    double * localWeights;
    int    * localNodes;
    
    // Initialize FD approximation
    FDApprox adv(1, xFD_Order, dx, UPWIND);
    
    // Width of stencils
    xStencilNnz = adv.GetSize();
    
    // Wavespeed in each direction
    std::function<double(double)> waveSpeed;
    
    double x;
    int xInd;
    
    
    // There are p/2+1 DOFs whose stencil depends on inflow and potentially ghost points
    for (int row = 0; row <= m_order/2; row++) {
        
        xInd = GlobalIndToMeshInd(row);
            
        // Value of grid point we're discretizing at
        x = m_mesh.MeshIndToPoint(xInd, xDim); 
            
        // Get stencil for discretizing x-derivative at current point 
        waveSpeed = [this, t](double x) { return WaveSpeed(x, t); };
        adv.SetX(x);
        adv.SetCoefficient(waveSpeed);
        adv.GetApprox(localNodes, localWeights);      
            
        // Loop over entries in stencil, adding couplings to boundary point or ghost points                            
        // TODO: Why do I have to subtract and not add here??
        for (int count = 0; count < xStencilNnz; count++) {
            if (xInd + localNodes[count] <= 0) G[row] -= localWeights[count] * uGhost[xInd + localNodes[count]]; 
        }                                 
    }
}


// Allocate vector U0 memory and populate it with initial condition.
void FDadvection::GetInitialCondition(const MPI_Comm &globComm, 
                                        double * &U0, 
                                        int &localMinRow, 
                                        int &localMaxRow, 
                                        int &spatialDOFs) const
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this](double x) { return InitCond(x); };
        GetGridFunction((void *) &GridFunction, globComm, U0, localMinRow, localMaxRow, spatialDOFs);
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this](double x, double y) { return InitCond(x, y); };
        GetGridFunction((void *) &GridFunction, globComm, U0, localMinRow, localMaxRow, spatialDOFs);
    }   
}


// Allocate vector U0 memory and populate it with initial condition.
void FDadvection::GetInitialCondition(double * &U0, int &spatialDOFs) const
{
    // Just pass lambdas to GetGrid function; cannot figure out better way to do this...
    if (m_dim == 1) {
        std::function<double(double)> GridFunction = [this](double x) { return InitCond(x); };
        GetGridFunction((void *) &GridFunction, U0, spatialDOFs);
    } else if (m_dim == 2) {
        std::function<double(double, double)> GridFunction = [this](double x, double y) { return InitCond(x, y); };
        GetGridFunction((void *) &GridFunction, U0, spatialDOFs);
    }  
}


/* Stencils for centred discretizations of diffusion operator, d^degree/dx^degree. degree \in {2,4} 

NOTES:
    degree == 2 uses a 2nd-order FD stencil
    degree == 4 uses a 2nd-order FD stencil
    assumes mesh spacing is same in all grid dimensions
    Uses centred differences
*/
void FDadvection::Get1DDissipationStencil(int * &inds, double * &weights, int &nnz) const
{
    // Using a 2nd-order difference means the stencil will use degree+1 nodes
    nnz     = m_dissipation_params.degree + 1;
    inds    = new int[nnz];
    weights = new double[nnz];
    
    if (m_dissipation_params.degree == 2) {
        inds[0] = -1;
        inds[1] = +0;
        inds[2] = +1;
        weights[0] = +1.0;
        weights[1] = -2.0;
        weights[2] = +1.0;
        
    }  else if (m_dissipation_params.degree == 4) {
        inds[0] = -2;
        inds[1] = -1;
        inds[2] = +0;
        inds[3] = +1;
        inds[4] = +2;
        weights[0] = +1.0;
        weights[1] = -4.0;
        weights[2] = +6.0;
        weights[3] = -4.0;
        weights[4] = +1.0;
        
    } else {
        std::cout << "WARNING: FD-advection numerical dissipation must be of degree 2 or 4" << '\n';
        MPI_Finalize();
        exit(1);
    }
    
    // Mesh-dependent weighting of dissipation (c0*dx^c1) \times FD coefficient (dx^{-degree})
    // Note the -1 here means that c0 > 0 corresponds to dissipation, while c0 < 0 means soln will blow up
    double c = -1.0 * m_dissipation_params.c0 * pow(m_mesh.m_dx[0], 1.0*m_dissipation_params.c1) * pow(m_mesh.m_dx[0], -1.0*m_dissipation_params.degree);
    
    /* Scale discretization weights */
    for (int i = 0; i < nnz; i++) {
        weights[i] *= c;
    }
}


// // Ensure there are sufficiently many DOFs to discretize derivative
// if (m_mesh.m_nx[dim] < m_order + 1) {
//     std::cout << "WARNING: FD stencil requires more grid points than are on grid! Increase nx!" << '\n';
//     MPI_Finalize();
//     exit(1);
// }



/* -------------------------------------------------------------------------- */
/* ----- Some utility functions that may be helpful for derived classes ----- */
/* -------------------------------------------------------------------------- */

/** Get parallel (square) matrix A from its local CSR data
NOTES: HypreParMatrix makes copies of the data, so it can be deleted */
void FDadvection::GetHypreParMatrixFromCSRData(MPI_Comm comm,  
                                               int localMinRow, int localMaxRow, HYPRE_Int globalNumRows, 
                                               int * A_rowptr, int * A_colinds, double * A_data,
                                               HypreParMatrix * &A) const
{
    int localNumRows = localMaxRow - localMinRow + 1;
    HYPRE_Int rows[2] = {localMinRow, localMaxRow+1};
    HYPRE_Int * cols = rows;
    // TODO: Maybe lodge issue on MFEM github. It's insane that the next line of 
    // code doesn't lead to MFEM reordering the matrix so that the first entry in
    // every row is the diagonal... I.e., the MFEM constructor checks whether rows
    // and cols point to the same location rather than checking if their values are 
    // the same. Such a simple difference (i.e., the above line c.f. the line below) 
    // shouldn't lead to massively different behaviour (most HYPRE functions
    // assume matrices are ordered like this and so they don't work as expected, 
    // but do not throw an error). I think the rest of the constructor musn't work
    // as expected anyway, because even in cols != rows, it's still meant to re-order
    // so that the 1st entry is the diagonal one. I suspect the issue is that the
    // re-ordering only happens when rows == cols; surely this is not right?
    //HYPRE_Int cols[2] = {localMinRow, localMaxRow+1};
    A = new HypreParMatrix(comm, localNumRows, globalNumRows, globalNumRows, 
                            A_rowptr, A_colinds, A_data, 
                            rows, cols); 
}

/** Get parallel vector x from its local data
NOTES: HypreParVector doesn't make a copy of the data, so it cannot be deleted */
void FDadvection::GetHypreParVectorFromData(MPI_Comm comm, 
                                            int localMinRow, int localMaxRow, HYPRE_Int globalNumRows, 
                                            double * x_data, HypreParVector * &x) const
{
    int rows[2] = {localMinRow, localMaxRow+1}; 
    x = new HypreParVector(comm, globalNumRows, x_data, rows);
}

/* Get identity matrix that's compatible with A */
void FDadvection::GetHypreParIdentityMatrix(const HypreParMatrix &A, HypreParMatrix * &I) const
{
    int globalNumRows = A.GetGlobalNumRows();
    int * row_starts = A.GetRowStarts();
    int localNumRows = row_starts[1] - row_starts[0];
    
    int    * I_rowptr  = new int[localNumRows+1];
    int    * I_colinds = new int[localNumRows];
    double * I_data    = new double[localNumRows];
    I_rowptr[0] = 0;
    for (int i = 0; i < localNumRows; i++) {
        I_colinds[i]   = i + row_starts[0];
        I_data[i]      = 1.0;
        I_rowptr[i+1]  = i+1;
    }
    
    I = new HypreParMatrix(A.GetComm(), 
                            localNumRows, globalNumRows, globalNumRows, 
                            I_rowptr, I_colinds, I_data, 
                            row_starts, row_starts); 
    
    /* These are copied into I, so can delete */
    delete[] I_rowptr;
    delete[] I_colinds;
    delete[] I_data;
} 

// /* Get identity mass matrix operator */
// void FDadvection::GetIRKOperatorM(HypreParMatrix * &M) {
//     if (m_M_exists) {
//         std::cout << "WARNING: If a mass matrix exists (as indicated), the derived class must implement it!" << '\n';
//         MPI_Finalize();
//         exit(1);
//     }
// 
//     if (!m_L) {
//         std::cout << "WARNING: Cannot get mass matrix M w/ out first getting discretization L" << '\n';
//         MPI_Finalize();
//         exit(1);
//     }
//     GetHypreParIdentityMatrix(*m_L, M);
// }

// void FDadvection::Test(double t) {
//     SetU0();
//     SetG(t);
//     SetL(t);
//     SetM();
// 
// 
//     //m_z(m_u->Size());
// 
//     m_z = Vector(m_u->Size());
// 
//     std::cout << "broken1.5" << '\n';
// 
//     //Vector z = Vector(m_u->Size());
//     //m_M->Mult(*m_u, z);
//     //z.Print(cout);
//     //m_z = new HypreParVector(m_u->Size());
// 
// 
//     std::cout << "broken2" << '\n';
// 
//     Array<double> c(2); // = Array({1.0, 3.0});
//     c[0] = 1.0;
//     c[1] = -3.0;
// 
//     Vector d = Vector(m_z.Size()); 
//     PolyMult(c, 1.0, *m_u, d);
//     d.Print(cout);
// }




