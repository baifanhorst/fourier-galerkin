# Curve class
# Used for curved edges of an element


import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

# Add the parent folder to the search path to import the OrthogonalPolynomials module
import sys
sys.path.append('../')

# Reloading the module
import importlib
import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *

class Curve():
    # Boundary curve class
    
    def __init__(self, N, parameter_point_type, func_x, func_y, func_deri_x, func_deri_y, par_range):
        # N: the largest index of boundary points on the curve
        # The index range is 0,1,...,N
        
        
        # parameter_point_type: Legendre or Chebyshev
        # The parameter of the curve is always in [-1,1]
        
        # func_x, func_y: parameterized functions for the x and y coordinates of the curve
        # The parameter for these functions does not have to be in [-1,1]
        # Its range is given by par_range
        
        # func_deri_x, func_deri_y: parameterized functions for the derivatives of x and y coordinates of the curve
        
        # par_range: the parameter range for func_x, func_y, func_deri_x, func_deri_y
        # In short, func_x, func_y, func_deri_x, func_deri_y represent the parametrized equations for the curve and the corresponding derivatives, with the parameter in par_range
        
        
        # Class attributes
        self.N = N
        
        # Other class attributes created in member functions
        # self.parameter_points:
            # The parameter values corresponding to boundary points
            # Created in set_parameter_points
            # These points range from -1 to 1
            # Usually, Legendre/Chebyshev Gauss Lobatto points are used
            # In the textbook, this is denoted by s
        self.set_parameter_points(parameter_point_type)
        
        # self.w_Bary:
            # Barycentric weights corresponding to self.parameter_points
        self.set_BarycentricWeights()
          
        # self.D:
            # Differentiation matrix wrt self.parameter_points
        # self.D_list:
            # Differentiation matrices up to 4th order
        self.set_DiffMatrix()
        
        # self.x_nodes, self.y_nodes:
            # The coordinates of the points on the boundary that correspond to
            # self.parameter_points
            # Created in reparameterization
        # self.parameter_points_original
            # The original parameter values corresponding to self.parameter_points
            # Created in reparameterization
            # In the textbook, this is denoted by t
        # self.L:
            # Curve length
            # Created in reparameterization    
        self.reparameterization(func_x, func_y, func_deri_x, func_deri_y, par_range)

            
        # self.x_deri_nodes, self.y_deri_nodes:
            # the derivatives of x and y coordinates up to 4th order
            # Created in cal_derivatives_node
        self.cal_derivatives_node()
        
    
    
    def set_parameter_points(self, point_type):
        # Set the parameter values at the boundary points
        # There are two types of parameter points: Legendre, Chebyshev
        # Generate: 
            # self.paramter_points
        
        if point_type=='Legendre':
            self.parameter_points = LegendreGaussLobattoNodes(self.N)
        
        if point_type=='Chebyshev':
            self.parameter_points = ChebyshevGaussLobattoNodes_Reversed(self.N)
        
        
    def set_BarycentricWeights(self):
        # Computing the Barycentric weights corresponding to self.parameter_points
        self.w_Bary = BarycentricWeights(self.parameter_points)
        
    def set_DiffMatrix(self):
        # Find differentiation matrices up to 4th order
        # Generate:
            # self.D: 1st diff matrix
            # self.D_list: a list of all diff matrices up to 4th order
        
        # Old notation for the 1st diff matrix
        self.D = PolynomialDiffMatrix(self.parameter_points, self.w_Bary)
        
        # List of diff matrices
        # e.g. D_list[2] is the 2nd diff matrix
        self.D_list = []
        # 0th diff matrix (identity)
        self.D_list.append(np.identity(self.N + 1))
        # 1st diff matrix (=D)
        self.D_list.append(self.D)
        # 2nd-4th diff matrices
        for i in range(2,4+1):
            self.D_list.append(PolynomialDiffMatrix_HighOrder(i, self.parameter_points, self.w_Bary, self.D)) 
        
    def reparameterization(self, x, y, x_deri, y_deri, par_range):
        # Finding the points on the curve that correspond to parameter_points
        # x,y: the original parameterized functions for the curve
        # x_deri, y_deri: derivatives of x and y
        # par_range: the range for the original parameter, a 1D numpy array of shape (2,)
        # Generate:
            # self.L: curve length
            # self.x_nodes, self.y_nodes: coordinates at self.parameter_points
            # self.parameter_points_original: the values of the original parameter corresponding to self.parameter_points
        
        t_start = par_range[0]
        t_end = par_range[1] 
        
        # The integrand to get the curve length, as well as 
        # for solving the parameter correspondence.
        def integrand(t):
            return np.sqrt(x_deri(t)**2 + y_deri(t)**2)
        
        L = integrate.quad(integrand, t_start, t_end)[0]
        self.L = L
        
        #print("L", L)
        
        
        def func_solve_par(t, s):
            # The rhs of the equation to be solved to get the original parameter t
            # value that corresponds to the new parameter s
            return integrate.quad(integrand, t_start, t)[0] * 2 / L - 1 - s
        
        self.x_nodes = np.zeros(self.parameter_points.shape)
        self.y_nodes = np.zeros(self.parameter_points.shape)
        self.parameter_points_original = np.zeros(self.parameter_points.shape)
        
        # Set the original parameter values for the end nodes
        self.parameter_points_original[0] = t_start
        self.parameter_points_original[-1] = t_end
        
        # Find the x and y coordinates for the inner nodes
        for i in range(1, self.N): # i= 1,...,N-1
            s = self.parameter_points[i]
            t = fsolve(func_solve_par, t_start, args=(s,))[0]
            self.parameter_points_original[i] = t
            self.x_nodes[i] = x(t)
            self.y_nodes[i] = y(t)
         
        # Find the x and y coordinates for the end nodes
        # If computed as above, there would be errors in the end node's coordinates
        self.x_nodes[0] = x(t_start)
        self.y_nodes[0] = y(t_start)
        self.x_nodes[-1] = x(t_end)
        self.y_nodes[-1] = y(t_end)
      
            
    def cal_coordinates(self, s):
        # Given the parameter s, calculate the coordinates of the point on the curve
        # using interpolation
        # Return: np.array([x,y]), a numpy array containing the coordinates
        
        x = LagrangeInterpolationBary(s, self.parameter_points, self.x_nodes, self.w_Bary)
        y = LagrangeInterpolationBary(s, self.parameter_points, self.y_nodes, self.w_Bary)
        
        return np.array([x,y])
    
    
    def cal_derivatives(self, m, s):
        # Given order m and parameter s, calculate the mth derivative (m<=4)
        # of the coordinate functions x(s) and y(s) on the curve
        
        # s in [-1, 1]: parameter
        
        
        # We interpolate nodal derivatives here. However, if the function is approximated
        # by a polynomial, taking derivative reduces its order. Instead of using this reduced
        # polynomial, we created a new polynomial which has the same order as the original
        # The nodal derivative values are those of the reduced polynomial. 
        # So there is an inconsisitency here. 
        # But the result is accurate enough
        
        # Return: np.array([x_deri, y_deri]):
            # x_deri, y_deri: mth derivatives of the x and y component
        
        x_deri = LagrangeInterpolationBary(s, self.parameter_points, self.x_deri_nodes[m], self.w_Bary)
        y_deri = LagrangeInterpolationBary(s, self.parameter_points, self.y_deri_nodes[m], self.w_Bary)
        
        return np.array([x_deri, y_deri])
    
    
    
    def cal_derivatives_node(self):
        # Computing nodal derivatives at self.parameter_points
        # Generate: self.x_deri_nodes, self.y_deri_nodes
            # Both are np.array, dimension: 5 * (N+1)
            # e.g. self.x_deri_nodes[1,:] stores 1st derivative values at self.parameter_points
            
        
        self.x_deri_nodes = np.zeros((5, self.N + 1))
        self.y_deri_nodes = np.zeros((5, self.N + 1))

        self.x_deri_nodes[0, :] = self.x_nodes[:]         
        self.y_deri_nodes[0, :] = self.y_nodes[:] 
        
        for m in range(1, 4+1):
            self.x_deri_nodes[m, :] = np.dot(self.D_list[m], self.x_nodes)[:]
            self.y_deri_nodes[m, :] = np.dot(self.D_list[m], self.y_nodes)[:]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #########################################
    # Deprecated
    #########################################
    def cal_derivatives_old(self, s):
        # Given the parameter s, calculate the derivatives 
        # of the coordinate functions x(s) and y(s) on the curve
        # using interpolation
        
        x_deri = LagrangeInterpolationDerivativeBary(s, self.parameter_points, 
                                                     self.x_nodes, self.w_Bary, self.D)
        y_deri = LagrangeInterpolationDerivativeBary(s, self.parameter_points, 
                                                     self.y_nodes, self.w_Bary, self.D)
        
        return np.array([x_deri, y_deri])
    
    
    def cal_derivatives_node_old(self):
        self.x_deri_nodes = np.zeros(self.parameter_points.shape)
        self.y_deri_nodes = np.zeros(self.parameter_points.shape)
        
        for i, s in enumerate(self.parameter_points): 
            self.x_deri_nodes[i] = LagrangeInterpolationDerivativeBary(s, self.parameter_points, 
                                                     self.x_nodes, self.w_Bary, self.D)
            self.y_deri_nodes[i] = LagrangeInterpolationDerivativeBary(s, self.parameter_points, 
                                                     self.y_nodes, self.w_Bary, self.D)
            
    
   
        
            
        
    