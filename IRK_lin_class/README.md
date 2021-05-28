Information on linear IRK algorithms. 

There are two types of algorithms implemented to apply IRK algorithms to the quasi-time-dependent system of ODEs:
    M*du/dt = L*u + g(t), u(0) = u0,    L : R^N -> R^N.
    
1. The first algorithm is that of Southworth et al. (2021) and is based on first reducing the s*N x s*N linear system into a sequence of N x N systems. 

2. The second algorithms are based on block preconditioning strategies for the block-coupled s*N x s*N system of stage equations. The initial algorithms here were proposed by Staff et al. (2006) and involve diagonal and triangular block splittings of the system. An algorithm of the same type proposed by Rana et al. (2021) that uses a lower triangular preconditioner is also implemented.

Both methods rely on the `IRKOperator` class, of which several functions must be implemented by the user.

1. **Complex-conjugate-preconditioned IRK**

2. **Block-preconditioned IRK**
    The stage equations take the form
              [ M - a_11*dt*L  ...    - a_1s*dt*L ][k1]    [b1]
        F*k = [       ...      ...        ...     ][...] = [...]
              [   - a_s1*dt*L  ...  M - a_ss*dt*L ][ks]    [bs]

  * The available block preconditioners are given by `enum class BlockPreconditioner`
  From our tests, the most robust appears to be `RANALD`, followed by that of `GSL`.

  * The stage equation operator F that must be inverted once per time step is 
  `BlockStageOperator : public BlockOperator`. The `Mult` function of this class computes the action of F on a vector. The implementation only requires computing the action of L s times.

  * The operator F is approximately inverted with a Krylov method that is applicable to non-symmetric linear systems, such as (F)GMRES. The Krylov method is preconditoned with the block preconditioner which is `class BlockStagePreconditioner : public Solver`.

  * The `Mult` function of this class applies the action of the preconditioner which is simply a block diagonal solve, or block upper/lower triangular solve depending on the structure of the preconditioner coefficients. In these solves, the diagonal blocks are approximately inverted by the user-provided function `ImplicitPrec` as implemented through the `IRKOperator` class. 


