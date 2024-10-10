import numpy as np

def AlmostEqual(x,y):
    # Test whether two float numbers are close enough
    
    # Machine epsilon
    eps = np.finfo(np.double).eps
    # Absolute difference
    dif_abs = np.abs(x-y)
    
    if x==0.0 or y==0.0:
        if dif_abs <= 2*eps:
            return True
        else:
            return False
    else:
        if dif_abs<=eps * np.abs(x) or dif_abs<=eps * np.abs(y):
            return True
        else:
            return False
#############################################
def delta(n,m):
    # Kronecker delta
    if n==m:
        return 1
    else:
        return 0
#############################################      
def TridiagonalSolver(N, d, l, u, y):
    # Solving a tridiagonal linear algebraic system
    # N: maximum index
    # Note that there are N+1 unknows, with index from 0 to N
    # y: RHS: y0, y1,...,yN
    # d: diagonal entries: d0, ..., dN
    # l: subdiagonal entries: l0=0, l1, ..., lN
    # Mathematically, only l1, ..., lN appear in the matrix
    # Here, l0 is included for the convenience of indexing
    # u: superdiagonal entries: u0, ..., uN
    
    # Note: without copying d and y, this algorithm changes them.
    d = np.copy(d)
    y = np.copy(y)
    
    # Eliminating the subdiagonal entries
    for i in range(1, N+1):
        d[i] = d[i] - l[i]/d[i-1] * u[i-1]
        y[i] = y[i] - l[i]/d[i-1] * y[i-1]
    
    # Eliminating the super diagonal entries
    y[N] = y[N]/d[N]
    for i in range(N,0,-1):
        y[i-1] = (y[i-1] - u[i-1] * y[i]) / d[i-1]
       
    return y
#############################################      
def TridiagonalMultiplication(d, l, u, x):
    # Matrix multiplication of a tridiagonal matrix to a vector
    # The tridiagonal matrix is given by its three diagonals
    # d:diagonal entries: d_i: i=0,1,...,N
    
    N = len(d)-1
    
    result = np.zeros(N+1)
    
    result[0] = d[0]*x[0] + u[0]*x[1]
    
    for i in range(1, N):
        result[i] = x[i-1]*l[i] + x[i]*d[i] + x[i+1]*u[i]
        
    result[N] = l[N]*x[N-1] + d[N]*x[N]
    
    return result



    
    

    