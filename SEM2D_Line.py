# Line class
# Used for straight edges of an element


import numpy as np

# Add the parent folder to the search path to import the OrthogonalPolynomials module
import sys
sys.path.append('../')

# Reloading the module
import importlib
import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *

##########################################################

class Line():
    # Boundary line class
    
    def __init__(self, N, parameter_point_type, point_start, point_end):
        # N: the largest index of boundary points on the curve
        # The index range is 0,1,...,N
        self.N = N
        
        # Starting point and end point
        # Each point is a 1D numpy array containing the x and y coordinates
        self.point_start = point_start
        self.point_end = point_end
        
        # Other parameters created in class functions
        # self.parameter_points:
            # The parameter values corresponding to boundary points (the line is parametrized. In the textbook, the parameter is 's')
            # Created in set_parameter_points
            # These points range from -1 to 1
            # Usually, Legendre/Chebyshev Gauss Lobatto points are used
            # parameter_point_type: 'Legendre', 'Chebyshev'     
        self.set_parameter_points(parameter_point_type)
        
        # self.w_Bary:
            # Barycentric weights corresponding to self.parameter_points
        self.set_BarycentricWeights()
          
        # self.D:
            # Differentiation matrix wrt self.parameter_points
        self.set_DiffMatrix()
        
        # self.x_nodes, self.y_nodes:
            # The coordinates of nodes
        self.cal_coordinates_node()
       
        # self.x_deri_nodes, self.y_deri_nodes:
            # the derivatives of x and y coordinates up to 4th order
        self.cal_derivatives_node()
        
        
    
    
    def set_parameter_points(self, point_type):
        # Set the parameter values at the boundary points
        # which are just Legendre-Gauss-Lobatto or Chebyshev-Gauss-Lobatto points
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
        # Using the Barycentric weights to get the differentiation matrix wrt self.parameter_points
        # Generate:
            # self.D: 1th diff matrix
            
        # For the same-name function in the SEM2D_Curve class, a list of diff matrices is created
        # We don't create such a list here.
            
        self.D = PolynomialDiffMatrix(self.parameter_points, self.w_Bary)
    
    
    def cal_coordinates(self, s):
        # Calculating the coordinates of the point at s
        # Return: np.array([x,y]), a numpy array containing the coordinates
        return ((1-s) * self.point_start + (1+s) * self.point_end) / 2
        
    def cal_coordinates_node(self):
        # Calculating the coordinates of all the nodes
        # Generate: self.x_nodes, self.y_nodes
        
        self.x_nodes = ( (1 - self.parameter_points) * self.point_start[0] 
                        +(1 + self.parameter_points) * self.point_end[0] ) / 2
        self.y_nodes = ( (1 - self.parameter_points) * self.point_start[1] 
                        +(1 + self.parameter_points) * self.point_end[1] ) / 2
        
    def cal_derivatives(self, order, s):
        # Given order and parameter s, calculate the mth derivative (m<=4)
        # of the coordinate functions x(s) and y(s)
        
        # Calculating the derivatives of the coordinates at any point on a straight line
        # s in [-1, 1]: parameter
        
        # Return: np.array([x_deri, y_deri]):
            # x_deri, y_deri: mth derivatives of the x and y component
            
        # The 0th derivative is the map itself
        # All mth derivatives (m >=2) are zeros
        if order == 0:
            return self.cal_coordinates(s)
        elif order == 1:
            deri = (self.point_end - self.point_start) / 2
        else:
            deri = np.zeros(self.point_start.shape)
        
        return deri
    
    def cal_derivatives_node(self):
        # Calculating the derivatives of the coordinates at all nodes
        
        # Generate: self.x_deri_nodes, self.y_deri_nodes
            # Both are np.array, dimension: 5 * (N+1)
            # e.g. self.x_deri_nodes[1,:] stores 1st derivative values at self.parameter_points
        
        
        self.x_deri_nodes = np.zeros((5, self.N + 1))
        self.y_deri_nodes = np.zeros((5, self.N + 1))

        self.x_deri_nodes[0, :] = self.x_nodes[:]         
        self.y_deri_nodes[0, :] = self.y_nodes[:] 
        
        x_deri, y_deri = self.cal_derivatives(1,0)
        
        self.x_deri_nodes[1, :] = np.ones(self.parameter_points.shape) * x_deri
        self.y_deri_nodes[1, :] = np.ones(self.parameter_points.shape) * y_deri
        
        
        
        
            
        
    