import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize


import sys
#sys.path.append('../symbolic')
sys.path.append('../')


import importlib


import SEM2D_Line
importlib.reload(SEM2D_Line)
from SEM2D_Line import *


import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *


# Element class
class Element():
    def __init__(self, idx, corners, label_corners, grid):
        # idx: element index, which is also the index of the element in the element list
        # This is used in mapping element node (i,j) to its global position in the linear system
        self.idx = idx
        
        # shape: shape of the element
        # There are two options: 'quad', 'curved'
        # 'quad': quadrilateral with four straight edges
        # 'curved': curved quadrilateral, at least one edge is curved
        # Currently, we construct only quad element
        #self.shape = shape
        
        # grid: grid class
        # All elements have the same grid, thus 'grid' is only stored once.
        # Note that the grid dimensions must be the same
        self.grid = grid
        
        # Extract dimensions for the following
        # Nx == Ny
        # We keep two variables for the two dimensions just for possible extensions
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        
        # increments in parameters
        # Deprecated, now stored in the grid object
        # dxi: self.grid.dxi
        # deta: self.grid.deta
        #self.dxi = 2 / Nx
        #self.deta = 2 / Ny
        
        # Computing the total number of nodes
        # Used in constructing the whole linear system for all elements
        self.num_nodes = (Nx+1) * (Ny+1)
        
        
        
        # corners: (4,2) array, each row contains the coordinates of a corner
        # The corners are in counterclockwise direction
            # x1 -> x2 -> x3 -> x4
        self.corners = corners
        
        # corner labels
        # Used in set up boundary conditions
        self.label_corners = label_corners
        
       
        # Computational nodes (xi, eta nodes)
        # indexing = 'ij': this guarantees that nodes_comp_x[i,j] = grid.nodes_x[i]
        self.nodes_comp_x, self.nodes_comp_y = np.meshgrid(self.grid.nodes_x,
                                                           self.grid.nodes_y,
                                                           indexing='ij')    
        
        
        # Physical nodes (x,y nodes)
        self.nodes_phy_x = np.zeros((Nx+1, Ny+1))
        self.nodes_phy_y = np.zeros((Nx+1, Ny+1))
        
        # Currently, we only keep the quad map
        self.cal_QuadMap_nodes()
        
        #if self.shape == 'quad':
            #self.cal_QuadMap_nodes()
        
        #elif self.shape == 'curved':
        #    self.cal_Map_nodes()
            
            
        # Derivatives
            # self.X_xi: partial x partial xi
            # self.X_eta = partial x partial eta
            # self.Y_xi: partial y partial xi
            # self.Y_eta = partial y partial eta
   
        # Only compute quad map derivatives
        self.cal_QuadMapDerivatives_nodes()
        
        # Deprecated:
        # Old notations
            # self.X_xi: partial x partial xi
            # self.X_eta = partial x partial eta
            # self.Y_xi: partial y partial xi
            # self.Y_eta = partial y partial eta
        # Map derivatives of high orders
        # self.x_deri, self.y_deri
        # e.g. self.x[1][2]: \partial^3 x \partial xi \partial eta^2
        #if self.shape == 'quad':
        #    self.cal_QuadMapDerivatives_nodes()
        #elif self.shape == 'curved':
        #    self.cal_MapDerivatives_nodes()
        
            
        
        
        # Jacobian 
        # self.J
        self.cal_Jacobian()
        
        
        # The scaling factors and normal vectors on the four boundaries
        # self.scal_lower, self.norm_vect_lower
        # self.scal_upper, self.norm_vect_upper
        # self.scal_left, self.norm_vect_left
        # self.scal_right, self.norm_vect_right
        # self.norm_vect: This is a 3D array with boundary values be the normal derivatives
        # This is used when setting patching conditions or Neumann BC
        # The corner values takes one of the normal derivatives, but this does not matter since
        # we never use the normal derivatives at a corner node
        self.cal_normal_vector_nodes()
        
        
    
        
        
        
        
    ########################################
    # Functions for quadrilateral maps
    ########################################
    
    
    
    def cal_QuadMap_nodes(self):
        # Computing the physical nodes for the quadrilateral map 
        # Generate:
            # self.nodes_phy_x and self.nodes_phy_y

        
        # Notation in the textbook
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]
        
        xi = self.nodes_comp_x
        eta = self.nodes_comp_y
        
        self.nodes_phy_x = 1/4 * ( x1 * (1-xi) * (1-eta)
                                 + x2 * (1+xi) * (1-eta)
                                 + x3 * (1+xi) * (1+eta)
                                 + x4 * (1-xi) * (1+eta) )
                                  
        
        
        self.nodes_phy_y = 1/4 * ( y1 * (1-xi) * (1-eta)
                                 + y2 * (1+xi) * (1-eta)
                                 + y3 * (1+xi) * (1+eta)
                                 + y4 * (1-xi) * (1+eta) )
    
        

    
    
    
        
    def cal_QuadMapDerivatives_nodes(self):
        
        # Currently, we only compute the first derivatives
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]
        
        xi = self.nodes_comp_x
        eta = self.nodes_comp_y
        
        self.X_xi = 0.25 * (1-eta) * (x2 - x1) + 0.25 * (1+eta) * (x3 - x4)
        self.Y_xi = 0.25 * (1-eta) * (y2 - y1) + 0.25 * (1+eta) * (y3 - y4)
                   
        self.X_eta = 0.25 * (1-xi) * (x4 - x1) + 0.25 * (1+xi) * (x3 - x2)
        self.Y_eta = 0.25 * (1-xi) * (y4 - y1) + 0.25 * (1+xi) * (y3 - y2)
    
    
    
    def cal_Jacobian(self):
        # Computing the Jabobian at nodes
        self.J = self.X_xi * self.Y_eta - self.X_eta * self.Y_xi
    
    
    
    def cal_normal_vector_nodes(self):
        # Computing the normal vectors on the boundary
        
        # Cartesian basis
        ex = np.array([1,0])
        ey = np.array([0,1])
        
        Nx = self.grid.Nx
        Ny = self.grid.Ny
        
        
        # Create a norm vector 2D array to store normal vectors
        # This is used when setting patching conditions and Neumann BCs
        # Only boundary inner nodes of these arrays will be used
        self.norm_vect = np.zeros((Nx+1, Ny+1, 2))
        
        
        # 'Lower boundary'
        j = 0
        self.norm_vect_lower = np.zeros((Nx + 1, 2))
        self.scal_lower = np.zeros(Nx + 1)
        for i in range(Nx + 1): #i=0,1,...,Nx
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_lower[i] = np.sqrt(X_xi**2 + Y_xi**2)
            # The normal vector is outward, so a negative sign is present.
            self.norm_vect_lower[i] = -sign_J / self.scal_lower[i] * (X_xi * ey - Y_xi * ex)
            self.norm_vect[i,j] = self.norm_vect_lower[i]
            
        # 'Upper boundary'
        j = Ny
        self.norm_vect_upper = np.zeros((Nx + 1, 2))
        self.scal_upper = np.zeros(Nx + 1)
        for i in range(Nx + 1): #i=0,1,...,Nx
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_upper[i] = np.sqrt(X_xi**2 + Y_xi**2)
            # The normal vector 
            self.norm_vect_upper[i] = sign_J / self.scal_upper[i] * (X_xi * ey - Y_xi * ex)
            self.norm_vect[i,j] = self.norm_vect_upper[i]
            
        # 'Left boundary'
        i = 0
        self.norm_vect_left = np.zeros((Ny + 1, 2))
        self.scal_left = np.zeros(Ny + 1)
        for j in range(Ny + 1): #j=0,1,...,Ny
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_left[j] = np.sqrt(X_eta**2 + Y_eta**2)
            # The normal vector is outward, so a negative sign is present.
            self.norm_vect_left[j] = -sign_J / self.scal_left[j] * (Y_eta * ex - X_eta * ey)
            self.norm_vect[i,j] = self.norm_vect_left[j]
            
            
            
        # 'Right boundary'
        i = Nx
        self.norm_vect_right = np.zeros((Ny + 1, 2))
        self.scal_right = np.zeros(Ny + 1)
        for j in range(Ny + 1): #j=0,1,...,Ny
            # Use simplified variable names
            #x = self.nodes_phy_x[i,j]
            #y = self.nodes_phy_y[i,j]
            X_xi, Y_xi, X_eta, Y_eta = self.X_xi[i,j], self.Y_xi[i,j], self.X_eta[i,j], self.Y_eta[i,j]
            sign_J = np.sign(self.J[i,j])
            # The scaling factor
            self.scal_right[j] = np.sqrt(X_eta**2 + Y_eta**2)
            # The normal vector
            self.norm_vect_right[j] = sign_J / self.scal_right[j] * (Y_eta * ex - X_eta * ey)
            self.norm_vect[i,j] = self.norm_vect_right[j]
    
    
    
    
    ####################################        
    # Visualization           
    ####################################      
    
        
    def visualizing_Grid_NormalVector(self):
        # Visualizing the nodes and the normal vectors
        # The normal vectors are from four 1D arrays
        fig, ax = plt.subplots(nrows=1, ncols=1)
        Nx = self.grid.Nx
        Ny = self.grid.Ny

        ax.scatter(self.nodes_phy_x, self.nodes_phy_y)


        
        j = 0
        ax.quiver(self.nodes_phy_x[:,j], self.nodes_phy_y[:,j], 
                  self.norm_vect_lower[:,0], self.norm_vect_lower[:,1])
        
        j = Ny   
        ax.quiver(self.nodes_phy_x[:,j], self.nodes_phy_y[:,j], 
                  self.norm_vect_upper[:,0], self.norm_vect_upper[:,1])    

        i = 0
        ax.quiver(self.nodes_phy_x[i,:], self.nodes_phy_y[i,:], 
                  self.norm_vect_left[:,0], self.norm_vect_left[:,1])

        i = Nx
        ax.quiver(self.nodes_phy_x[i,:], self.nodes_phy_y[i,:], 
                  self.norm_vect_right[:,0], self.norm_vect_right[:,1])   
        
        ax.grid(True)
        ax.set_title('Grid, Normal Vector')
        ax.set_aspect('equal')
        
        
        
    def cal_QuadMap_SingleNode(self, xi, eta):
        # Computing the physical node for the quadrilateral map at (xi, eta)
        # Generate:
            # self.nodes_phy_x and self.nodes_phy_y

        
        # Notation in the textbook
        x1, y1 = self.corners[0]
        x2, y2 = self.corners[1]
        x3, y3 = self.corners[2]
        x4, y4 = self.corners[3]
        
        
        
        x = 1/4 * ( x1 * (1-xi) * (1-eta)
                                 + x2 * (1+xi) * (1-eta)
                                 + x3 * (1+xi) * (1+eta)
                                 + x4 * (1-xi) * (1+eta) )
                                  
        
        
        y = 1/4 * ( y1 * (1-xi) * (1-eta)
                                 + y2 * (1+xi) * (1-eta)
                                 + y3 * (1+xi) * (1+eta)
                                 + y4 * (1-xi) * (1+eta) )
        
        return np.array([x,y])
    
    
    
    
    
    
    def cal_Coord_XiEta(self, point):
        # Given the coordinates of a point inside element e, find the corresponding xi and eta
        # point: np 1D array, np.array([x,y]), or list [x,y] or tuple [x,y]
        # return: np.array([xi, eta]), the corresponding xi and eta
        
        # Define the equations to be solved
        def LHS(x):
            coords = self.cal_QuadMap_SingleNode(x[0], x[1])
            return [coords[0] - point[0], 
                    coords[1] - point[1]]
        
        return optimize.root(LHS, np.array([0,0])).x
                
    
    
    
    
    
     
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
    
            