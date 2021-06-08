The basic idea here get a heuristic understanding of the impact of the residual stopping tolerances for our complex-conjugate preconditioned IRK algorithm vs. the IRK block preconditioning approach of Staff et al. (and others).

The algorithms both solve for different things, with ours solving for the linear combination of stage vectors via a sequence of nested solves, and the Staff et al. alg. solving for all the stage vectors at once. It’s not clear to me how one should impose a stopping criterion that ultimately leads to the linear combination of stage vectors having a similar absolute accuracy for the two problems (in the ideal case, we’d like to impose an absolute stopping criterion for each that means that we just reach the discretization error of the IRK method). 

This is also further complicated if we used a relative residual-based stopping criterion because the zero initial guess leads to different initial residual norms for the two different approaches. 

So, for example, I’m concerned that imposing a 10^-13 relative residual stopping criterion might lead to much more accurate stage vectors with our algorithm that it does Staff’s (say). In which case, the comparison wouldn’t really be a fair one because we’d be oversolving our linear systems compared to Staff’s. 


Anyway, here what we'll do is fix a (very high-order) spatial discretization, and fix an IRK discretization, and we'll solve the resulting time integration problem with both of the solvers for a set of different GMRES halting tolerances. And we want to see if both of the algorithms require a similar GMRES halting tolerance to reach the discretization error.
(The reason we use high-order accuracy is so that we get a small discretization error such that we have to solve the linear systems accurately and we can therefore cleanly see the effect of a halting tolerance that is not small enough).