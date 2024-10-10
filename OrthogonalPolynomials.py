import numpy as np
from scipy.special import eval_legendre
from scipy.optimize import fsolve, root_scalar
from scipy import integrate

import GeneralFunctionality
import importlib
importlib.reload(GeneralFunctionality)
from GeneralFunctionality import *


##########################################################
def LegendreGaussLobattoNodes(N):
    # Finding the Legendre Gauss Lobatto Nodes
    # Two nodes are -1 and 1.
    # The others are the roots of L'_N(x)
    # L'_N(x)=0 -> xL_N(x)-L_{N-1}(x)=0
    
    # If N==1, the nodes only include -1 and 1
    if N==1:
        return np.array([-1, 1])
    
    f = lambda x, N: x * eval_legendre(N, x) - eval_legendre(N-1, x)
    
    # Initial guess
    # David's book P66, Eq (3.7)
    N2 = N//2
    J = np.arange(1, N2+1)
    x0 = -np.cos((J+0.25)*np.pi/N - 3/8/N/np.pi/(J+0.25))
    
    # Finding the zeros for j=1,...,N//2
    zeros = fsolve(f, x0, args=(N,))
    
    # Other zeros can be obtained by symmetry
    # Finally, we add -1, 1 to the two ends of the array
    if N%2!=0: # N odd
        return np.concatenate((np.array([-1.0]), zeros, -zeros[-1::-1], np.array([1.0])))
    else: # N even
        # zeros[-1] is x_{N/2}, which is zero.
        # We force it to be zero here, instead of using the solution by fsolve
        zeros[-1] = 0.0
        return np.concatenate((np.array([-1.0]), zeros, -zeros[-2::-1], np.array([1.0])))
    
######################################################################
def LegendreGaussLobattoWeights(x):
    # Calculating the Legendre Gauss Lobatto weights
    # x: xj: j=0,...,N: Legendre Gauss Lobatto points
    # Formula: w_j = 2/N/(N+1)/(L_N(x_j))^2
    N = len(x) - 1
    LNx = eval_legendre(N,x)
    return 2/N/(N+1)/LNx**2


######################################################################
def ChebyshevGaussRadauNodes(N):
    # Generating Chebyshev-Gauss-Radau nodes
    # The nodes are labeled from 0 to N (N+1 nodes)
    # The right boundary point 1 is included
    
    pi = np.pi
    J = np.arange(0, N+1)
    x = np.cos( 2*pi*J/(2*N+1) )
    
    return np.flip(x)
######################################################################
def ChebyshevGaussRadauWeights(x):
    # Generating Chebyshev-Gauss-Radau weights
    # The corresponding nodes are labeled from 0 to N (N+1 nodes)
    # The right boundary point 1 is included
    
    N = len(x) - 1
    
    pi = np.pi
    w = np.ones(N+1) * pi / (N+1)
    w[0] = pi / (2*N+1)
    return w
######################################################################

def ChebyshevGaussLobattoNodes(N):
    # Finding the Chebyshev Gauss Lobatto Nodes
    # These nodes share an analytic formular
    # In the previous implementation, the Chebyshev nodes are ordred from 1 to -1.
    # Now the order is from -1 to 1
    
    J = np.arange(0, N+1)
    x = -np.cos(np.pi*J/N)
    
    return x

def ChebyshevGaussLobattoWeights(x):
    # Calculating the Legendre Gauss Lobatto weights
    # For inner nodes, the weight is pi/N
    # For end nodes, the weigh is pi/(2N)
    N = len(x) - 1
    w = np.ones(N+1) * np.pi / N
    w[0] *= 0.5
    w[-1] *= 0.5
    return w

######################################################################
def BarycentricWeights(x):
    # Calculating normalized Barycentric Weights
    # x: xj, j=0,1,...,N, interpolation nodes
    # The original weights is normalized against the maximum absolute weight
    # This function calculates the weights directly according to the formula,
    # which only works when the number of interpolating points is small.
    # When N is about 800, this function collapses.
    
    N = len(x) - 1
    
    w = np.ones(N+1)
    
    for j in range(N+1):
        for i in range(N+1):
            if i != j:
                w[j] *= (x[j] - x[i])
    
    w = 1 / w
    return w / np.linalg.norm(w, np.inf)
    
######################################################################
def PolynomialDiffMatrix(x, w):
    # Calculating the differentiation matrix based on the interpolation nodes
    # x: x_j: j=0,1,...,N
    # w: Barycentric weights
    
    N = len(x)-1
    
    # Initialization
    D = np.zeros((N+1, N+1))
    
    # Calculating the entries
    for i in range(N+1):
        for j in range(N+1):
            if j!=i:
                D[i,j]= w[j]/w[i]/(x[i]-x[j])
        D[i,i] = -np.sum(D[i,:]) # Negative sum trick
    
    return D

######################################################################
def ChebyshevDiffMatrix(x):
    # Calculating the Chebyshev Differentiation Matrix
    # Explicit formulas exist for the entries
    # x: chebyshev nodes, xj: j =0,1,...,N
    
    N = len(x) - 1
    
    # c: auxiliary constants: c0=cN=2, cj=1 for j=1,...,N-1
    c = np.ones(N+1)
    c[0] = 2
    c[N] = 2
    
    # Initialize the matrix
    D = np.zeros((N+1, N+1))
    # Calculating off-diagonal entries
    for i in range(N+1):
        for j in range(N+1):
            if i!=j:
                D[i,j] = c[i]/c[j] * (-1)**(i+j) / (x[i] - x[j])
        # Diagonal entries, negative sum trick
        D[i,i] = -np.sum(D[i,:])
    
    return D
######################################################################
def PolynomialDiffMatrix_HighOrder(m, x, w, D):
    # Calculating the mth-order differentiation matrix by recursion
    # m: the order of the derivativem m>=2
    # x: interpolation nodes
    # w: Barycentric weights
    # D: the differentiation matrix for the 1st-order derivative
    
    if m<2:
        print("The order m must be no less than 2")
        return
    
    N = len(x) - 1
    
    # D_old represents D^{k-1}
    D_old = D
    # D_new represents D^k
    D_new = np.zeros((N+1, N+1))
    
    for k in range(2,m+1):
        # Calculating D^k from D^{k-1}
        for i in range(N+1):
            # Set the diagonal entries to be zero 
            # in order to implement the negative sum trick
            D_new[i,i] = 0
            # Calculating the off-diagonal entries
            for j in range(N+1):
                if i!=j:
                    D_new[i,j] = k/(x[i]-x[j]) * (w[j]/w[i]*D_old[i,i]-D_old[i,j])
            # Calculating the diagonal entries with the negative sum trick
            D_new[i,i] = -np.sum(D_new[i,:])
        
        # Update D^{k-1}    
        D_old = np.copy(D_new)
    
    return D_new


######################################################################
def InterpolatingMatrix(x, y, w):
    # Finding the transformation matrix that gives the values 
    # of the interpolating polynomial at a new set of nodes
    # x: the original interpolating nodes: x0,...,xN
    # y: new nodes: y0,...,yM
    # w: Barycentric weights
    
    N = len(x) - 1
    M = len(y) - 1
    
    T = np.zeros((M+1, N+1))
    for k in range(M+1):
        # If y[k] equals one of the original nodes x[j0]
        # Then T[k, j0]=1 and T[k,j]=0 for other j
        identical_nodes = False
        j0 = 0
        for j in range(N+1):
            if AlmostEqual(y[k], x[j]):
                identical_nodes = True
                j0 = j
                break
                
        if identical_nodes==True:
            T[k, j0] = 1
        else: #y[k]!=x[j] for any j
            factors = w / (y[k]-x)
            T[k, :] = factors / np.sum(factors)
    
    return T

###############################################
def LegendreCoefficients(f, N):
    # Calculating the Legendre coefficients F_k for the truncated expansion:
    # f(x) = \sum_{k=0}^{N} F_k L_k(x)
    # F_k = (f, L_k) / (L_k, L_k)
    # (f, g) = \int_{-1}^{1} f(x) g(x) \dif x
    # (L_k, L_k) = 2/(2k+1)
    
    F = np.zeros(N+1)
    
    for k in range(N+1):
        integrand = lambda x, k: f(x) * eval_legendre(k, x)
        F[k] = integrate.quad(integrand, -1, 1, args=(k,))[0]
        F[k] = F[k] * (2*k+1)/2
        
    return F

###############################################
def LegendreTruncation(x, F):
    # Given Legendre coefficients F, calculated the value of 
    # the truncated series at x
    # f(x) = \sum_{k=0}^{N} F_k L_k(x)
    # F: Fk: k=0,1,...,N
    
    N = len(F)-1
    result = 0
    
    for k in range(N+1):
        result += eval_legendre(k, x) * F[k]
    
    return result

#########################################
def LegendreGalerkinBasis(k,x):
    # The basis function in the Legendre Galerkin method
    # phi_k = 1/sqrt(4k+6) * (L_k - L_{k+2})
    return (eval_legendre(k,x) - eval_legendre(k+2,x)) / np.sqrt(4*k+6)

#########################################
def LegendreGalerkinTruncation(x, F):
    # Evaluate the truncated series in the Legendre Galerkin method
    # The truncated series = \sum_{k=0}^{N-2} F_k \phi_k
    # F: coefficients: F_k: k=0,1,...,N-2
    N = len(F) + 1
    result = 0
    for k in range(N-1): # i=0,1,...N-2
        result += F[k] * LegendreGalerkinBasis(k,x)
    return result


#########################################
def LagrangeInterpolationBary(x, x_nodes, f, w):
    # Evaluating the Lagrange interpolant of f at x
    # x_nodes: xj, j=0,1,...,N, interpolating nodes
    # f: function values at the nodes
    # w: wj, j=0,1,...,N, barycentricweights
    
    N = len(x_nodes) - 1
    
    # If x is equal to one of the nodes, return the function value at the nodes
    for j in range(N+1):
        if AlmostEqual(x, x_nodes[j]):
            return f[j]
    
    factors = w / (x-x_nodes)
    return np.sum(f * factors) / np.sum(factors)
    
########################################
def LagrangePolynomialBary(j, x, x_nodes, w):
    # Computing l_j(x) using the barycentric weights
    # x_nodes: xj, j=0,1,...,N, interpolating nodes
    # w: wj, j=0,1,...,N, barycentricweights
    
    N = len(x_nodes) - 1
    
    # If x is equal to one of the nodes, return 1 if x==x[j] or return 0 otherwise
    for i in range(N+1):
        if AlmostEqual(x, x_nodes[i]):
            if i == j:
                return 1
            else:
                return 0
    
    factors = w / (x-x_nodes)
    return factors[j] / np.sum(factors)
    
    # Old implementation, deprecated
    #N = len(x_nodes) - 1
    #f = np.zeros(N+1)
    #f[j] = 1.0
    #return LagrangeInterpolationBary(x, x_nodes, f, w)


#########################################
def LagrangeInterpolationDerivativeBary(x, x_nodes, f, w, D):
    # Evaluating the derivative of the Lagrange interpolant of f at x
    # x_nodes: xj, j=0,1,...,N, interpolating nodes
    # f: function values at the nodes
    # w: wj, j=0,1,...,N, barycentricweights
    # Differentiation matrix
    
    N = len(x_nodes) - 1
    
    # If x is equal to one of the nodes, use the differentiation matrix
    for j in range(N+1):
        if AlmostEqual(x, x_nodes[j]):
            return np.dot(D, f)[j]
    
    
    # Calculating the value of the interpolant at x by the Barycentric formula
    factors = w / (x-x_nodes)
    sum_factors = np.sum(factors)
    p = np.sum(f * factors) / sum_factors
    # Calculating the derivative
    px = np.sum((p - f) / (x-x_nodes) * factors) / sum_factors
    
    return px

###########################################
def LagrangePolynomialDerivativeBary(j, x, x_nodes, w, D):
    # Computing the derivative of the lagrange polynomial of node j wrt x
    # j: the "location" of the lagrange polynomial, lj(x_i) = delta_{ij}
    # x: scalar argument
    # x_nodes: interpolating nodes
    # w: barycentric weights
    # Differentiation matrix
    
    f = np.zeros(x_nodes.shape)
    f[j] = 1.0
    
    return LagrangeInterpolationDerivativeBary(x, x_nodes, f, w, D)
    
    
###########################################
def LagrangePolynomialsBoundaryValues(N, x_nodes, w_bary):
    # Computing l_j(-1), l_j(1) for j=0,1,...,N
    # Used in nodal discontinuous Galerkin method
    # x_nodes: interpolating nodes
    # w_bary: barycentric weights
    
    l_left = np.zeros(N+1)
    l_right = np.zeros(N+1)
    
    for j in range(N+1):
        l_left[j] = LagrangePolynomialBary(j, -1.0, x_nodes, w_bary)
        l_right[j] = LagrangePolynomialBary(j, 1.0, x_nodes, w_bary)
    
    return l_left, l_right




##########################
# Deprecated
##########################
######################################################################
def ChebyshevGaussLobattoNodes_Reversed(N):
    # Finding the Chebyshev Gauss Lobatto Nodes
    # Deprecated, since ChebyshevGaussLobattoNodes gives the same output
    
    J = np.arange(0, N+1)
    x = -np.cos(np.pi*J/N)
    
    return x

    
    
    
    
    
    
    
    
    
    
    