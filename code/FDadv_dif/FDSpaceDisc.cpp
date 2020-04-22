#include "FDSpaceDisc.hpp"

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
    m_nxLocalTotal = m_nxTotal; 
    
    /* -------------------------------------------------------- */
    /* -------------- Set up spatial parallelism -------------- */
    /* -------------------------------------------------------- */
    /* Ensure spatial parallelism setup is permissible and decide which 
        mesh points current process and its neighbours own, etc */
    if (m_commSize > 1) m_parallel = true;
    
    //if (m_parallel) {
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
            m_nxLocal.resize(1);
            m_nxLocalInt.resize(1);
            
            m_pIdx[0] = m_rank;
            m_np[0]   = m_commSize;
            m_nxLocalInt[0] = m_nx[0]/m_np[0];
            
            // All procs in interior have same number of points
            if (m_rank < m_np[0]-1)  {
                m_nxLocal[0] = m_nxLocalInt[0];  
            // Proc on EAST boundary takes the remainder of points
            } else {
                m_nxLocal[0] = (m_nx[0] - (m_np[0]-1)*m_nxLocalInt[0]); 
            }
            m_nxLocalTotal = m_nxLocal[0];
            
            // Global index of first point on proc
            m_globOffset = m_pIdx[0] * m_nxLocalInt[0]; 
            
            // Set ranks of neighbours on periodic domain
            m_nborpIdxEast.resize(1);         
            m_nborpIdxWest.resize(1);     
            m_nborpIdxWest[0] = (m_pIdx[0] > 0) ? m_pIdx[0]-1 : m_np[0]-1;
            m_nborpIdxEast[0] = (m_pIdx[0] < m_np[0]-1) ? m_pIdx[0]+1 : 0;

            
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
            m_nxLocal.resize(2);
            m_nxLocalInt.resize(2);
            m_nxLocalBnd.resize(2);
            
            // Get indices on proc grid
            m_pIdx[0] = m_rank % m_np[0]; // Proc grid, x index
            m_pIdx[1] = m_rank / m_np[0]; // Proc grid, y index
            
            // Number of DOFs on procs in interior of domain
            m_nxLocalInt[0] = m_nx[0] / m_np[0];
            m_nxLocalInt[1] = m_nx[1] / m_np[1];
            
            // Number of DOFs on procs on boundary of proc domain 
            m_nxLocalBnd[0] = m_nx[0] - (m_np[0]-1)*m_nxLocalInt[0]; // East boundary
            m_nxLocalBnd[1] = m_nx[1] - (m_np[1]-1)*m_nxLocalInt[1]; // North boundary
            
            // Compute number of DOFs on proc
            if (m_pIdx[0] < m_np[0] - 1) {
                m_nxLocal[0] = m_nxLocalInt[0]; // All procs in interior have same number of DOFS
            } else {
                m_nxLocal[0] = m_nxLocalBnd[0]; // Procs on EAST boundary take the remainder of DOFs
            }
            if (m_pIdx[1] < m_np[1] - 1) {
                m_nxLocal[1] = m_nxLocalInt[1]; // All procs in interior have same number of DOFS
            } else {
                m_nxLocal[1] = m_nxLocalBnd[1]; // All procs in interior have same number of DOFS
            }
            m_nxLocalTotal = m_nxLocal[0] * m_nxLocal[1]; 
            
            // Compute global index of first DOF on proc
            m_globOffset = m_pIdx[0]*m_nxLocalInt[0]*m_nxLocal[1] + m_pIdx[1]*m_nx[0]*m_nxLocalInt[1];
                        
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
            MPI_Send(&m_nxLocal[0], 2, MPI_INT, pNInd, 0, m_comm);
            MPI_Send(&m_nxLocal[0], 2, MPI_INT, pSInd, 0, m_comm);
            MPI_Send(&m_nxLocal[0], 2, MPI_INT, pEInd, 0, m_comm);
            MPI_Send(&m_nxLocal[0], 2, MPI_INT, pWInd, 0, m_comm);
            
            // Receive dimensions of DOFs that my nearest neighbours own
            m_nborNxLocal.resize(8); // Just stack the nx and ny on top of one another in pairs in same vector
            MPI_Recv(&m_nborNxLocal[0], 2, MPI_INT, pNInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborNxLocal[2], 2, MPI_INT, pSInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborNxLocal[4], 2, MPI_INT, pEInd, 0, m_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&m_nborNxLocal[6], 2, MPI_INT, pWInd, 0, m_comm, MPI_STATUS_IGNORE);
        }
    //std::cout << "Passed FDMesh constructor...\n";
}

// Map grid index to grid point in specified dimension
double FDMesh::MeshIndToPoint(int meshInd, int dim) const
{
    return m_boundary0[dim] + meshInd * m_dx[dim];
}


/* Evaluate local function values */
void FDMesh::EvalFunction(double (*Function)(const Vector &x), 
                                  Vector * &u) const
{
    double * u_data = new double[m_nxLocalTotal];
    Vector meshPoints = Vector(m_dim);
    
    // Loop over all grid points on process
    for (int i = 0; i < m_nxLocalTotal; i++) {
        LocalDOFIndToMeshPoints(i, meshPoints);
        u_data[i] = (*Function)(meshPoints);
    }
    u = new Vector(u_data, m_nxLocalTotal);
    u->MakeDataOwner();
}

/* Evaluate time-dependent grid function */
void FDMesh::EvalFunction(double (*TDFunction)(const Vector &x, double t),
                                  double t, Vector * &u) const
{
    double * u_data = new double[m_nxLocalTotal];
    Vector meshPoints(m_dim);

    // Loop over all grid points on process
    for (int i = 0; i < m_nxLocalTotal; i++) {
        LocalDOFIndToMeshPoints(i, meshPoints);
        u_data[i] = (*TDFunction)(meshPoints, t);
    }
    u = new Vector(u_data, m_nxLocalTotal);
    u->MakeDataOwner();
}


/* Evaluate local function values */
void FDMesh::EvalFunction(double (*Function)(const Vector &x), 
                                  Vector &u) const
{
    Vector meshPoints(m_dim);
    // Loop over all grid points on process
    for (int i = 0; i < m_nxLocalTotal; i++) {
        LocalDOFIndToMeshPoints(i, meshPoints);
        u(i) = (*Function)(meshPoints);
    }
}

/* Evaluate time-dependent grid function */
void FDMesh::EvalFunction(double (*TDFunction)(const Vector &x, double t),
                                  double t, Vector &u) const
{
    Vector meshPoints(m_dim);
    // Loop over all grid points on process
    for (int i = 0; i < m_nxLocalTotal; i++) {
        LocalDOFIndToMeshPoints(i, meshPoints);
        u(i) = (*TDFunction)(meshPoints, t);
    }
}


/* Get mesh points from local DOF index (i.e., i \in [0,m_localSize)) */
void FDMesh::LocalDOFIndToMeshPoints(int i, Vector &x) const
{
    switch (m_dim) {
        case 1:
            x(0) = MeshIndToPoint(m_pIdx[0]*m_nxLocalInt[0] + i, 0); 
            break;
        case 2:
            x(0) = MeshIndToPoint(m_pIdx[0]*m_nxLocalInt[0] + (i % m_nxLocal[0]), 0); 
            x(1) = MeshIndToPoint(m_pIdx[1]*m_nxLocalInt[1] + (i / m_nxLocal[0]), 1); 
            break;
    }
}


/* Get global index of DOF that's on a neighbouring process.

overflow > 0 is the number the amount we overflow into the neighbour
In 2D, locInd is the local mesh index in the direction perpendicular to the neighbour

 */
int FDMesh::LocalOverflowToGlobalDOFInd(int locInd, int overflow, Direction nbor) const
{
    
    // Ensure this is positive
    overflow = abs(overflow);
    switch (m_dim) {
        case 2:
            switch (nbor) {
                case EAST:
                    return m_nborGlobOffset[2] + overflow-1 + locInd*m_nborNxLocal[4];
                case WEST:
                    return m_nborGlobOffset[3] + m_nborNxLocal[6]-1 - (overflow-1) + locInd*m_nborNxLocal[6];
                case NORTH:
                    return m_nborGlobOffset[0] + locInd + (overflow-1)*m_nborNxLocal[0];
                case SOUTH:
                    return m_nborGlobOffset[1] + locInd + (m_nborNxLocal[3]-overflow)*m_nborNxLocal[2];                 
                default:
                    return 0;
            }
        default:
            return 0;
    }

}

/* Get local Mesh indices from local DOF index */
void FDMesh::LocalDOFIndToLocalMeshInds(int i, Array<int> &meshInds) const
{
    switch (m_dim) {
        case 1:
            meshInds[0] = i;
            break;
        case 2:
            meshInds[0] = i % m_nxLocal[0];
            meshInds[1] = i / m_nxLocal[0];
            break;
    }
}



// Given local indices on current process return global index
//auto MeshIndsOnProcToGlobalInd = [this, localMinRow](int row, int xLocInd, int yLocInd) { return localMinRow + xLocInd + yLocInd*m_mesh.m_nxLocal[0]; };

int FDMesh::LocalMeshIndsToGlobalDOFInd(const Array<int> &meshInds) const
{
    switch (m_dim) {
        case 1:
            return m_globOffset + meshInds[0];
        case 2:
            return m_globOffset + meshInds[0] + meshInds[1]*m_nxLocal[0];
        default:
            return 0;
    }
}




FDSpaceDisc::FDSpaceDisc(const FDMesh &mesh, int derivative, int order, FDBias bias) :
            m_mesh{mesh},
            m_derivative{derivative},
            m_comm{mesh.m_comm},
            m_dim{mesh.m_dim}, 
            m_order{order},
            m_bias{bias},
            m_globSize{mesh.m_nxTotal},
            m_localSize{mesh.m_nxLocalTotal},
            m_parallel(mesh.m_parallel)
{    
    
    // Seed random number generator so results are consistent!
    srand(0);

    // Get MPI communicator information
    m_rank = m_mesh.m_rank;
    m_commSize = m_mesh.m_commSize;
    
    // Get DOF ordering information
    m_globMinRow = m_mesh.m_globOffset;             // First row on proc
    m_globMaxRow = m_globMinRow + m_localSize - 1;  // Last row on proc 
}

FDSpaceDisc::~FDSpaceDisc() 
{
    
}


/* Get CSR structure for discretization of 1D constant-coefficient operator */
void FDSpaceDisc::Get1DConstantOperatorCSR(int derivative, Vector c,
                                         int order, FDBias bias,
                                         int * &rowptr, int * &colinds, double * &data) const
{
    // Initialize FD approximation
    FDApprox dxapprox(derivative, order, m_mesh.m_dx[0], bias);
    dxapprox.SetCoefficient(c(0));
    int dxNnz = dxapprox.GetSize();
    
    // Get FD approximation
    int * dxNodes = NULL; 
    double * dxWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);

    
    /* -------------------------------------- */
    /* ------ Initialize CSR structure ------ */
    /* -------------------------------------- */
    int localNnz = dxNnz * m_localSize; 
    rowptr      = new int[m_localSize + 1];
    colinds     = new int[localNnz];
    data        = new double[localNnz];
    rowptr[0]   = 0;
    int dataInd = 0;


    /* --------------------------------------------- */
    /* ------ Get CSR structure of local rows ------ */
    /* --------------------------------------------- */
    for (int row = 0; row < m_localSize; row++) {
        for (int i = 0; i < dxNnz; i++) {
            // Account for periodicity here. This always puts in range [0,nx-1]
            colinds[dataInd] = (dxNodes[i] + row + m_mesh.m_globOffset + m_mesh.m_nx[0]) % m_mesh.m_nx[0]; 
            data[dataInd]    = dxWeights[i];
            dataInd++;
        }
        rowptr[row+1] = dataInd;
    }  
    
    // Check sufficient data was allocated
    if (dataInd > localNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
}


/* Get CSR structure for discretization of 2D constant-coefficient operator */
void FDSpaceDisc::Get2DConstantOperatorCSR(int derivative, Vector c,
                                         int order, FDBias bias,
                                         int * &rowptr, int * &colinds, double * &data) const
{
    // Initialize FD approximation of each term
    FDApprox dxapprox(derivative, order, m_mesh.m_dx[0], bias);
    dxapprox.SetCoefficient(c(0));
    int dxNnz = dxapprox.GetSize();
    FDApprox dyapprox(derivative, order, m_mesh.m_dx[1], bias);
    dyapprox.SetCoefficient(c(1));
    int dyNnz = dyapprox.GetSize();
    
    // Get FD approximation for each term
    int * dxNodes, * dyNodes = NULL; 
    double * dxWeights, * dyWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);
    dyapprox.GetApprox(dyNodes, dyWeights);

    
    /* -------------------------------------- */
    /* ------ Initialize CSR structure ------ */
    /* -------------------------------------- */
    // Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    int localNnz = (dxNnz + dyNnz - 1) * m_localSize; 
    rowptr      = new int[m_localSize + 1];
    colinds     = new int[localNnz];
    data        = new double[localNnz];
    rowptr[0]   = 0;
    int dataInd = 0;

    Array<int> locInds(2), locIndsMax(2), tempInds(2);
    locIndsMax[0] = m_mesh.m_nxLocal[0]-1;
    locIndsMax[1] = m_mesh.m_nxLocal[1]-1;


    /* --------------------------------------------- */
    /* ------ Get CSR structure of local rows ------ */
    /* --------------------------------------------- */
    for (int row = 0; row < m_localSize; row++) {                                          
        
        m_mesh.LocalDOFIndToLocalMeshInds(row, locInds);
        
        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, 
        // then continuing to loop over y (Actually, periodicity stuffs this up I think...)
        for (int j = 0; j < dyNnz; j++) {

            // The two stencils intersect at (0,0)
            if (dyNodes[j] == 0) {
                for (int i = 0; i < dxNnz; i++) {
                    int con = locInds[0] + dxNodes[i]; // Local x-index of current connection
                    // Connection to process on WEST side
                    if (con < 0) {
                        colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[1], con, WEST);
                        //colinds[dataInd] = MeshIndsOnWestProcToGlobalInd(abs(con), yLocInd);
                    // Connection to process on EAST side
                    } else if (con > locIndsMax[0]) {
                        colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[1], con - locIndsMax[0], EAST);
                        //colinds[dataInd] = MeshIndsOnEastProcToGlobalInd(con - (m_mesh.m_nxLocal[0]-1), yLocInd);
                    // Connection is on processor
                    } else {
                        //colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, con, yLocInd);
                        tempInds[0] = con;
                        tempInds[1] = locInds[1];
                        colinds[dataInd] = m_mesh.LocalMeshIndsToGlobalDOFInd(tempInds);
                    }
                    data[dataInd] = dxWeights[i];

                    // The two stencils intersect at this point, i.e. they share a column in the matrix
                    if (dxNodes[i] == 0) data[dataInd] += dyWeights[j]; 
                    dataInd += 1;
                }

            // No intersection possible between between x- and y-stencils
            } else {
                int con = locInds[1] + dyNodes[j]; // Local y-index of current connection
                // Connection to process on SOUTH side
                if (con < 0) {
                    colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[0], con, SOUTH);
                    //colinds[dataInd] = MeshIndsOnSouthProcToGlobalInd(xLocInd, abs(con));
                // Connection to process on NORTH side
                } else if (con > locIndsMax[1]) {
                    colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[0], con - locIndsMax[1], NORTH);
                    //colinds[dataInd] = MeshIndsOnNorthProcToGlobalInd(xLocInd, con - (m_mesh.m_nxLocal[1]-1));
                // Connection is on processor
                } else {
                    tempInds[0] = locInds[0];
                    tempInds[1] = con;
                    colinds[dataInd] = m_mesh.LocalMeshIndsToGlobalDOFInd(tempInds);
                    //colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, xLocInd, con);
                }

                data[dataInd] = dyWeights[j];
                dataInd++;
            }
        }    
        rowptr[row+1] = dataInd;
    }    
    // Check sufficient data was allocated
    if (dataInd > localNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
}



/* -------------------------------------------------------------------------- */
/* ----- Some utility functions that may be helpful for derived classes ----- */
/* -------------------------------------------------------------------------- */

/** Get parallel (square) matrix A from its local CSR data
NOTES: HypreParMatrix makes copies of the data, so it can be deleted */
void FDSpaceDisc::GetHypreParMatrixFromCSRData(MPI_Comm comm,  
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
    
     // /// Get diag ownership flag
     // std::cout << "data = " << static_cast<signed>(A->OwnsDiag()) << "\n";
     // std::cout << "offd = " << static_cast<signed>(A->OwnsOffd()) << "\n";
     // std::cout << "cols = " << static_cast<signed>(A->OwnsColMap()) << "\n";
     // 
     // A->SetOwnerFlags(1, 1, 1);
     // 
     // std::cout << "data = " << static_cast<signed>(A->OwnsDiag()) << "\n";
     // std::cout << "offd = " << static_cast<signed>(A->OwnsOffd()) << "\n";
     // std::cout << "cols = " << static_cast<signed>(A->OwnsColMap()) << "\n";
     // 
     
    //A->SetOwnerFlags(0, 0, 0);                        
}

/** Get parallel vector x from its local data
NOTES: HypreParVector doesn't make a copy of the data, so it cannot be deleted */
void FDSpaceDisc::GetHypreParVectorFromData(MPI_Comm comm, 
                                            int localMinRow, int localMaxRow, HYPRE_Int globalNumRows, 
                                            double * x_data, HypreParVector * &x) const
{
    int rows[2] = {localMinRow, localMaxRow+1}; 
    x = new HypreParVector(comm, globalNumRows, x_data, rows);
}


 
/* Get identity matrix */
HypreParMatrix * FDSpaceDisc::GetHypreParIdentityMatrix() const
{
    int    * I_rowptr  = new int[m_localSize+1];
    int    * I_colinds = new int[m_localSize];
    double * I_data    = new double[m_localSize];
    I_rowptr[0] = 0;
    for (int i = 0; i < m_localSize; i++) {
        I_colinds[i]   = i + m_globMinRow;
        I_data[i]      = 1.0;
        I_rowptr[i+1]  = i+1;
    }

    HypreParMatrix * I = NULL; 
    GetHypreParMatrixFromCSRData(m_comm, 
                                m_globMinRow, m_globMaxRow, m_globSize,
                                I_rowptr, I_colinds, I_data,
                                I); 

    /* These are copied into I, so can delete */
    delete[] I_rowptr;
    delete[] I_colinds;
    delete[] I_data;
    
    return I;
} 


/* Constructors */
FDLinearOp::FDLinearOp(const FDMesh &mesh, int derivative, Vector c, int order, 
                        FDBias bias) : FDSpaceDisc(mesh, derivative, order, bias),
                        m_c(c), m_Op(NULL) { };                        

FDLinearOp::~FDLinearOp()
{
    if (m_Op) delete m_Op;    
}
    
void FDLinearOp::Assemble() const
{
    // Operator already assembled, cannot have changed...
    if (m_Op) return; 
    
    // Get local CSR structure of matrix
    int * rowptr, * colinds = NULL;
    double * data = NULL;
    if (m_dim == 1) 
    {
        Get1DConstantOperatorCSR(m_derivative, m_c, m_order, m_bias, 
                                 rowptr, colinds, data);
    } 
    else if (m_dim == 2)
    {
        Get2DConstantOperatorCSR(m_derivative, m_c, m_order, m_bias, 
                                 rowptr, colinds, data);
    }
    
    
    // Assemble global matrix
    GetHypreParMatrixFromCSRData(m_mesh.m_comm, 
                                 m_globMinRow, m_globMaxRow, m_globSize,
                                 rowptr, colinds, data, m_Op);
                                 
    delete[] rowptr;
    delete[] colinds;
    delete[] data;                             
};


FDNonlinearOp::FDNonlinearOp(const FDMesh &mesh, int derivative, Vector c, 
                                double (*f)(double), int order, FDBias bias) 
    : FDSpaceDisc(mesh, derivative, order, bias), m_Gradient(NULL),
                                m_c(c), m_f(f), m_df(NULL) 
{ 
if (bias != CENTRAL) mfem_error("Don't have upwind implemented...");
};

FDNonlinearOp::FDNonlinearOp(const FDMesh &mesh, int derivative, Vector c, 
                                double (*f)(double), double (*df)(double), 
                                int order, FDBias bias) 
    : FDSpaceDisc(mesh, derivative, order, bias), m_Gradient(NULL),
                                m_c(c), m_f(f), m_df(df) 
{
if (bias != CENTRAL) mfem_error("Don't have upwind implemented...");
};

void FDNonlinearOp::Mult(const Vector &a, Vector &b) const
{
    if (m_dim == 1) {
        Mult1DParallel(a, b);
        // if (m_mesh.GetCommSize() > 1) {
        //     Mult1DParallel(a, b);
        // } else {
        //     Mult1DSerial(a, b);
        // }
    }
    else if (m_dim == 2) {
        Mult2D(a, b);    
    }
}
HypreParMatrix &FDNonlinearOp::GetGradient(const Vector &u) const
{
    if (m_dim == 1) {
        return GetGradient1D(u);
    } else if (m_dim == 2) {
        return GetGradient2D(u);    
    } else {
        return *m_Gradient;
    }
}

// c.grad(f(u)) in serial
void FDNonlinearOp::Mult1DSerial(const Vector &a, Vector &b) const
{
    FDApprox dxapprox(m_derivative, m_order, m_mesh.m_dx[0], m_bias);
    dxapprox.SetCoefficient(m_c(0));
    int dxNnz = dxapprox.GetSize();
    
    // Get FD approximation
    int * dxNodes = NULL; 
    double * dxWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);

    b = 0.0;
    for (int row = 0; row < m_localSize; row++) {
        for (int i = 0; i < dxNnz; i++) {
            // Account for periodicity here. This always puts in range [0,nx-1]
            int j = (dxNodes[i] + row + m_mesh.m_globOffset + m_mesh.m_nx[0]) % m_mesh.m_nx[0]; 
            b(row) += ((*m_f)(a(j)))*dxWeights[i];
        }
    }  
}

// c.grad(f(u)) in parallel
void FDNonlinearOp::Mult1DParallel(const Vector &a, Vector &b) const
{
    FDApprox dxapprox(m_derivative, m_order, m_mesh.m_dx[0], m_bias);
    dxapprox.SetCoefficient(m_c(0));
    int dxNnz = dxapprox.GetSize();
    
    // Get FD approximation
    int * dxNodes = NULL; 
    double * dxWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);

    // Unpack some variables
    MPI_Comm comm = m_mesh.GetComm();     // The communicator
    int left = m_mesh.m_nborpIdxWest[0];  // Left neighbour
    int right = m_mesh.m_nborpIdxEast[0]; // Right neighbour

    // Number of nodes in stencil to the left and right of us
    int nRecvFromLeft = 0, nRecvFromRight = 0;
    for (int i = 0; i < dxNnz; i++) {
        if (dxNodes[i] < 0) {
            nRecvFromLeft++;
        } else if (dxNodes[i] > 0) {
            nRecvFromRight++;
        }
    }
    MFEM_VERIFY(nRecvFromLeft < m_mesh.m_nxLocal[0] && nRecvFromRight < m_mesh.m_nxLocal[0], 
        "FDNonlinearOp::Mult1DParallel() Stencil cannot extend further than immediate neighbouring processes.");
    
    double * aRecvFromRight = new double[nRecvFromRight];
    double * aRecvFromLeft = new double[nRecvFromLeft];
    
    // Get neighbouring processes boundary data
    // Exploit that the chuncks we need to send are continguous in memory
    if (m_mesh.GetCommSize() > 1) {
        // How many boundary points do I send to my neighbours?
        int nSendLeft = 0, nSendRight = 0; 

        // Tell neighbours how much data I need and they tell me how much they need
        MPI_Send(&nRecvFromLeft, 1, MPI_INT, left, 0, comm);
        MPI_Recv(&nSendRight, 1, MPI_INT, right, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(&nRecvFromRight, 1, MPI_INT, right, 1, comm);
        MPI_Recv(&nSendLeft, 1, MPI_INT, left, 1, comm, MPI_STATUS_IGNORE);
        
        // ---- Send data to the left and recieve from the right
        // Send boundary data to LHS neighbour
        MPI_Send(a.GetData(), nSendLeft, MPI_DOUBLE, left, 2, comm);
        // Receive RHS neighbour's boundary data
        MPI_Recv(aRecvFromRight, nRecvFromRight, MPI_DOUBLE, right, 2, comm, MPI_STATUS_IGNORE);
        
        // ---- Send data to the right and recieve it from the left
        // Send boundary data to RHS neighbour
        MPI_Send(a.GetData()+m_mesh.m_nxLocal[0]-nSendRight, nSendRight, MPI_DOUBLE, right, 3, comm);
        // Receive LHS neighbours boundary data
        MPI_Recv(aRecvFromLeft, nRecvFromLeft, MPI_DOUBLE, left, 3, comm, MPI_STATUS_IGNORE);
        
    // Single process, just periodically wrap the data so it can use the same implementation below
    } else {
        for (int i = 0; i < nRecvFromRight; i++) aRecvFromRight[i] = a(i);
        for (int i = 0; i < nRecvFromLeft; i++) aRecvFromLeft[i] = a(m_mesh.m_nxLocal[0]-nRecvFromLeft+i);
    }
    
    b = 0.0;
    for (int row = 0; row < m_localSize; row++) {
        for (int i = 0; i < dxNnz; i++) {
            int conn = dxNodes[i] + row;
            
            // Connection to left neighbouring process
            if (conn < 0) {
                conn += nRecvFromLeft;
                b(row) += ((*m_f)(aRecvFromLeft[conn]))*dxWeights[i];
            // Connection to right neighbouring process    
            } else if (conn >= m_localSize) {
                conn -= m_localSize;
                b(row) += ((*m_f)(aRecvFromRight[conn]))*dxWeights[i];
            // Connection is on process    
            } else {
                b(row) += ((*m_f)(a(conn)))*dxWeights[i];
            }
        }
    }  
    delete[] aRecvFromLeft;
    delete[] aRecvFromRight;
}


// c.grad(f(u)) = c(0)*df/dx + c(1)*df/dy
void FDNonlinearOp::Mult2D(const Vector &a, Vector &b) const
{
    // Initialize FD approximation of each term
    FDApprox dxapprox(m_derivative, m_order, m_mesh.m_dx[0], m_bias);
    dxapprox.SetCoefficient(m_c(0));
    int dxNnz = dxapprox.GetSize();
    FDApprox dyapprox(m_derivative, m_order, m_mesh.m_dx[1], m_bias);
    dyapprox.SetCoefficient(m_c(1));
    int dyNnz = dyapprox.GetSize();
    
    // Get FD approximation for each term
    int * dxNodes, * dyNodes = NULL; 
    double * dxWeights, * dyWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);
    dyapprox.GetApprox(dyNodes, dyWeights);

    Array<int> locInds(2), conInds(2);
    b = 0.0;
    for (int row = 0; row < m_localSize; row++) {
        m_mesh.LocalDOFIndToLocalMeshInds(row, locInds); // Indices of point we're discretizing at...
        // x-derivative
        conInds[1] = locInds[1];
        for (int i = 0; i < dxNnz; i++) {
            // Account for periodicity here. This always puts in range [0,nx-1]
            conInds[0] = (locInds[0] + dxNodes[i] + m_mesh.m_nx[0]) % m_mesh.m_nx[0]; 
            b(row) += ((*m_f)(a(m_mesh.LocalMeshIndsToGlobalDOFInd(conInds))))*dxWeights[i];
        }
        
        // y-derivative         
        conInds[0] = locInds[0];                                 
        for (int j = 0; j < dyNnz; j++) {
            // Account for periodicity here. This always puts in range [0,nx-1]
            conInds[1] = (locInds[1] + dyNodes[j] + m_mesh.m_nx[1]) % m_mesh.m_nx[1]; 
            b(row) += ((*m_f)(a(m_mesh.LocalMeshIndsToGlobalDOFInd(conInds))))*dyWeights[j];
        }    
    }           
}


/* Get gradient of 2D operator, c.grad(f(u)) */
HypreParMatrix &FDNonlinearOp::GetGradient2D(const Vector &u) const
{
    //std::cout << "get grad..." << '\n';
    if (!m_df) mfem_error("FDNonlinearOp::GetGradient() Must set gradient function (see SetGradientFunction())");

    if (m_Gradient) delete m_Gradient;
    
    // Initialize FD approximation of each term
    FDApprox dxapprox(m_derivative, m_order, m_mesh.m_dx[0], m_bias);
    dxapprox.SetCoefficient(m_c(0));
    int dxNnz = dxapprox.GetSize();
    FDApprox dyapprox(m_derivative, m_order, m_mesh.m_dx[1], m_bias);
    dyapprox.SetCoefficient(m_c(1));
    int dyNnz = dyapprox.GetSize();
    
    // Get FD approximation for each term
    int * dxNodes, * dyNodes = NULL; 
    double * dxWeights, * dyWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);
    dyapprox.GetApprox(dyNodes, dyWeights);

    
    /* -------------------------------------- */
    /* ------ Initialize CSR structure ------ */
    /* -------------------------------------- */
    // Discretization of x- and y-derivatives at point i,j will both use i,j in their stencils (hence the -1)
    int localNnz = (dxNnz + dyNnz - 1) * m_localSize; 
    int * rowptr      = new int[m_localSize + 1];
    int * colinds     = new int[localNnz];
    double * data        = new double[localNnz];
    rowptr[0]   = 0;
    int dataInd = 0;

    Array<int> locInds(2), locIndsMax(2), tempInds(2);
    locIndsMax[0] = m_mesh.m_nxLocal[0]-1;
    locIndsMax[1] = m_mesh.m_nxLocal[1]-1;


    /* --------------------------------------------- */
    /* ------ Get CSR structure of local rows ------ */
    /* --------------------------------------------- */
    for (int row = 0; row < m_localSize; row++) {                                          
        //std::cout << "row = " << row << '\n';
        m_mesh.LocalDOFIndToLocalMeshInds(row, locInds);
        
        // Build so that column indices are in ascending order, this means looping 
        // over y first until we hit the current point, then looping over x, 
        // then continuing to loop over y (Actually, periodicity stuffs this up I think...)
        for (int j = 0; j < dyNnz; j++) {

            // The two stencils intersect at (0,0)
            if (dyNodes[j] == 0) {
                for (int i = 0; i < dxNnz; i++) {
                    int con = locInds[0] + dxNodes[i]; // Local x-index of current connection
                    // Connection to process on WEST side
                    if (con < 0) {
                        colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[1], con, WEST);
                        //colinds[dataInd] = MeshIndsOnWestProcToGlobalInd(abs(con), yLocInd);
                    // Connection to process on EAST side
                    } else if (con > locIndsMax[0]) {
                        colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[1], con - locIndsMax[0], EAST);
                        //colinds[dataInd] = MeshIndsOnEastProcToGlobalInd(con - (m_mesh.m_nxLocal[0]-1), yLocInd);
                    // Connection is on processor
                    } else {
                        //colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, con, yLocInd);
                        tempInds[0] = con;
                        tempInds[1] = locInds[1];
                        colinds[dataInd] = m_mesh.LocalMeshIndsToGlobalDOFInd(tempInds);
                    }
                    data[dataInd] = dxWeights[i] * ((*m_df)(u(colinds[dataInd])));
                    
                    // The two stencils intersect at this point, i.e. they share a column in the matrix
                    if (dxNodes[i] == 0) data[dataInd] += dyWeights[j] * ((*m_df)(u(colinds[dataInd])));; 
                    //std::cout << data[dataInd] << '\n';
                    dataInd += 1;
                }

            // No intersection possible between between x- and y-stencils
            } else {
                int con = locInds[1] + dyNodes[j]; // Local y-index of current connection
                // Connection to process on SOUTH side
                if (con < 0) {
                    colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[0], con, SOUTH);
                    //colinds[dataInd] = MeshIndsOnSouthProcToGlobalInd(xLocInd, abs(con));
                // Connection to process on NORTH side
                } else if (con > locIndsMax[1]) {
                    colinds[dataInd] = m_mesh.LocalOverflowToGlobalDOFInd(locInds[0], con - locIndsMax[1], NORTH);
                    //colinds[dataInd] = MeshIndsOnNorthProcToGlobalInd(xLocInd, con - (m_mesh.m_nxLocal[1]-1));
                // Connection is on processor
                } else {
                    tempInds[0] = locInds[0];
                    tempInds[1] = con;
                    colinds[dataInd] = m_mesh.LocalMeshIndsToGlobalDOFInd(tempInds);
                    //colinds[dataInd] = MeshIndsOnProcToGlobalInd(row, xLocInd, con);
                }

                data[dataInd] = dyWeights[j] * ((*m_df)(u(colinds[dataInd])));
                //std::cout << data[dataInd] << '\n';
                dataInd++;
            }
        }    
        rowptr[row+1] = dataInd;
    }    
    // Check sufficient data was allocated
    if (dataInd > localNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    
    // Assemble global matrix
    GetHypreParMatrixFromCSRData(m_mesh.m_comm, 
                                 m_globMinRow, m_globMaxRow, m_globSize,
                                 rowptr, colinds, data, m_Gradient);
                                 
    delete[] rowptr;
    delete[] colinds;
    delete[] data;   
    
    return *m_Gradient;                           
}


/* Get gradient of 1D operator, c.grad(f(u)) */
HypreParMatrix &FDNonlinearOp::GetGradient1D(const Vector &u) const
{
    if (!m_df) mfem_error("FDNonlinearOp::GetGradient() Must set gradient function (see SetGradientFunction())");

    if (m_Gradient) delete m_Gradient;
    
    // Initialize FD approximation
    FDApprox dxapprox(m_derivative, m_order, m_mesh.m_dx[0], m_bias);
    dxapprox.SetCoefficient(m_c(0));
    int dxNnz = dxapprox.GetSize();

    // Get FD approximation
    int * dxNodes = NULL; 
    double * dxWeights = NULL;
    dxapprox.GetApprox(dxNodes, dxWeights);


    /* -------------------------------------- */
    /* ------ Initialize CSR structure ------ */
    /* -------------------------------------- */
    int localNnz = dxNnz * m_localSize; 
    int * rowptr  = new int[m_localSize + 1];
    int * colinds = new int[localNnz];
    double * data = new double[localNnz];
    rowptr[0]   = 0;
    int dataInd = 0;

    double c = m_c(0);

    /* --------------------------------------------- */
    /* ------ Get CSR structure of local rows ------ */
    /* --------------------------------------------- */
    for (int row = 0; row < m_localSize; row++) {
        for (int i = 0; i < dxNnz; i++) {
            // Account for periodicity here. This always puts in range [0,nx-1]
            int j = (dxNodes[i] + row + m_mesh.m_globOffset + m_mesh.m_nx[0]) % m_mesh.m_nx[0]; 
            colinds[dataInd] = j;
            data[dataInd]    = dxWeights[i] * ((*m_df)(u(j)));
            dataInd++;
        }
        rowptr[row+1] = dataInd;
    }  

    // Check sufficient data was allocated
    if (dataInd > localNnz) {
        std::cout << "WARNING: Matrix has more nonzeros than allocated.\n";
    }
    
    // Assemble global matrix
    GetHypreParMatrixFromCSRData(m_mesh.m_comm, 
                                 m_globMinRow, m_globMaxRow, m_globSize,
                                 rowptr, colinds, data, m_Gradient);
                                 
    delete[] rowptr;
    delete[] colinds;
    delete[] data;   
    
    return *m_Gradient;                           
}