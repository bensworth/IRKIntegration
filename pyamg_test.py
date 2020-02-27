import pdb
import time
import pyamg
import numpy as np
from scipy import sparse

# - Important that c1 > 0. If c1 = 0  and c2 > 0, convegence can be bad,
# or diverge for c2 ~> 1.
# - Can we show that \eta, \beta > 1 for RK schemes?
# - Applying adjugate can be done by summing columns (all applied to same
# stage vector). Then each column is a monic polynomial of degree s-1. The
# highest order term L^{s-1} can be applied outside the sum, which requires
# s-1 matvecs. By forming L*k, L*(L*k), ..., and storing, we can limit the
# rest to s(s-2) matvecs and some extra storage. Altogether, we get s(s-1)-1
# matvecs to apply the adjugate. N(2,3,4,5,6) = {2,5,11,19,29}. Gets rather
# expensive for very high order (s=5 or 6), but per stage it's only increasing
# linearly, which is really not so bad. 
# - Numerical precision may be an issue if we are computing L^6*k


accel = 'cg'
maxiter = 250
c1 = 0.1
c2 = 0.0

A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
N = A.shape[0]
A += c1*sparse.identity(N)


residuals = []
start = time.perf_counter()
ml = pyamg.ruge_stuben_solver(A)

end = time.perf_counter()
setup_time = end-start
print("AMG: ",A.shape[0]," DOFs, ",A.nnz," nnzs")
print("\tSetup time          = ",setup_time)

B = A + c2*sparse.identity(N)
ml.levels[0].A = B

b = np.random.rand(N)
start = time.perf_counter()
sol = ml.solve(b, tol=1e-10, residuals=residuals, accel=accel, maxiter=maxiter)
end = time.perf_counter()
solve_time = end-start

OC = ml.operator_complexity()
CC = ml.cycle_complexity()

# Average convergence factor
if residuals[-1] == 0:
    residuals[-1] = 1e-16
CF = (residuals[-1]/residuals[0])**(1.0/(len(residuals)-1))
last_CF = residuals[-1]/residuals[-2]

# All Convergence factors
conv_factors = np.zeros((len(residuals)-1,1))
for i in range(0,len(residuals)-1):
    conv_factors[i] = residuals[i+1]/residuals[i]

print("\tSolve time          = ",solve_time)
print("\tOperator complexity = ",OC)
print("\tCycle complexity    = ",CC)
print("\tWork per digit      = ",-CC / np.log10(CF))
print("\tAverage CF          = ",CF)
print("\tFinal CF            = ",last_CF)
print("\tNum iters           = ",len(residuals))
