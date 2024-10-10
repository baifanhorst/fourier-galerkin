import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reloading the module
import importlib

import Mesh
importlib.reload(Mesh)

import Utility
importlib.reload(Utility)

import Visualization
importlib.reload(Visualization)

import Elasticity
importlib.reload(Elasticity)

import LinearSystem_3DAxisymmetric
importlib.reload(LinearSystem_3DAxisymmetric)




class Solid():
    
    def __init__(self, name):
        
        self.name = name
        
        # Number of unknown displacement components
        self.num_unknown = 3
        
        # Extract info from the mesh file of Abacus format (*.inp)
        # and write the info to csv files.
        Mesh.set_Mesh(name)
        
    def set_FourierMode(self, nmax):
        # Maximum Absolute Fourier Mode
        # The resulting Fourier mode range is [-nmax, nmax]
        self.nmax = nmax
        
        
    def set_AngularGrid(self, Nphi):
        # Create angular grid
        # Maximum angular node label
        # angular node label: 0 to Nphi
        self.Nphi = Nphi
        # Angular values
        # self.phi[0] = self.phi[Nphi], we keep this redundancy for convenience
        self.phi = np.linspace(0, 2*np.pi, Nphi+1)
    
    
    def set_ElasticProperty(self, E=1, nu=0.3):
        self.Eprop = Elasticity.Elasticity(E, nu)  
    
               
    
    def set_Element(self, grid_sizes = (3,3), grid_types = ("Legendre", "Legendre")):
        # Generating elements
        
        # grid_sizes: (N, N), N is the maximum node label in each direction
        # grid_types: grid types for the two directions. 
            # Although the default types are provided, for nodal Galerkin methods they are
            # the only choices. Chebyshev grid won't work.
        
        
        ###########################
        # Read node info
        ###########################
        # Line format in 'nodes.csv': node label, x, y, z
        df = pd.read_csv('./mesh/nodes.csv', header=None)
        # The last column include z coordinates, which are all zeros, not used in the following
        self.mesh_nodes = df.to_numpy()[:, 1:-1] 
        # Number of nodes
        self.num_nodes = self.mesh_nodes.shape[0]
        
        
        ###########################
        # Read element info
        ###########################
        # Line in 'elements.csv': element label, node 1 label, node 2 label, node 3 label 
        df = pd.read_csv('./mesh/elements.csv', header=None)
        # The first column include element labels. No need to store them
        self.mesh_elements = df.to_numpy()[:, 1:] 
        # The original node labels start from 1
        # Reduce the node labels by 1 according to python convention (starting from 0)
        self.mesh_elements = self.mesh_elements - 1 
        # Number of elements
        self.num_elements = self.mesh_elements.shape[0]
        
        
        
        ##########################################
        # Read node labels on the Dirichlet boundary
        ##########################################
        # 'tag_node_BC_Dirichlet.csv' contains a single line, containing the node labels
        df = pd.read_csv('./mesh/tag_node_BC_Dirichlet.csv', header=None)
        # The dataframe has a single line, whose last entry is ' ', which should be dropped
        # 'to_numpy' gives a single-row 2D array, which needs to be converted to 1D, by using [0]
        self.mesh_tag_nodes_BC_Dirichlet = df.iloc[:, :-1].to_numpy()[0] 
        # Reduce the node labels by 1 according to python convention (starting from 0)
        self.mesh_tag_nodes_BC_Dirichlet -= 1
        
    
    
        ##########################################
        # Read node labels on the Neumann boundary
        ##########################################
        # Similar to reading Dirichlet nodes (see above)
        df = pd.read_csv('./mesh/tag_node_BC_Neumann.csv', header=None)
        self.mesh_tag_nodes_BC_Neumann = df.iloc[:, :-1].to_numpy()[0] 
        self.mesh_tag_nodes_BC_Neumann -= 1
        
        
        ##########################################
        # Generating elements
        ##########################################
        self.element_list = Utility.set_Element_Quad(self.mesh_elements, 
                                             self.mesh_nodes, 
                                             grid_sizes, grid_types)
        
        # Show all elements 
        Visualization.visualizing_elements(self.element_list, legend='off')
        
        ##########################################
        # Generating edges
        ##########################################
        self.edge_list = Utility.set_Edge(self.element_list)
        
        ##########################################
        # Generating common edges
        ##########################################
        self.edgecommon_list = Utility.set_EdgeCommon(self.edge_list)
        
        
        ##########################################
        # Generating common corners
        ##########################################
        self.cornercommon_list = Utility.set_CornerCommon(self.mesh_nodes, self.element_list)
    
    
    
    
    
    
    def init_Arrays(self, exact=True):
        for e in self.element_list:
            Utility.init_Solution_RHS(self.num_unknown, e, self.nmax, exact)
            
    
    def init_SingleFourierMode(self, n, f_Fourier, p_Fourier):
        # Set the exact solution and the contribution of the body force to RHS
        # and the contribution of boundary load to RHS
        
        # u_Fourier: function of the known displacement, (3,)
        # f_Fourier: function of the known body force, (3,)
        # p_Fourier: function of the known external load, (3,)
        
        
        Utility.set_RHS_BodyForce(n, self.num_unknown, self.element_list, f_Fourier, self.Eprop)
        Utility.set_RHS_ExternalSress(n, self.num_unknown, self.element_list, p_Fourier, self.Eprop, 
                            self.mesh_tag_nodes_BC_Neumann)
            
            
    
    
    def init_SingleFourierMode_Exact(self, n, u_Fourier, f_Fourier, p_Fourier):
        # Set the exact solution and the contribution of the body force to RHS
        # and the contribution of boundary load to RHS
        
        # u_Fourier: function of the known displacement, (3,)
        # f_Fourier: function of the known body force, (3,)
        # p_Fourier: function of the known external load, (3,)
        
        
        Utility.set_Solution_Exact(n, self.num_unknown, self.element_list, u_Fourier)
        Utility.set_RHS_BodyForce(n, self.num_unknown, self.element_list, f_Fourier, self.Eprop)
        Utility.set_RHS_ExternalSress(n, self.num_unknown, self.element_list, p_Fourier, self.Eprop, 
                            self.mesh_tag_nodes_BC_Neumann)
        
    
        
    
    
    def init_LinearSystem(self):
        C, d = LinearSystem_3DAxisymmetric.init_C_d(self.element_list, self.num_unknown)
        LinearSystem_3DAxisymmetric.set_C(C, self.num_unknown, self.Eprop, self.element_list)
        self.C = C
        self.d = d
    
        
    
    def solve_LinearSystem_SingleFourierMode(self, n, u_Fourier):
        
        # Set up and solve the linear system for a single Fourier mode
        C, d = self.C, self.d
        
        # RHS
        LinearSystem_3DAxisymmetric.set_d(d, 
                self.num_unknown, self.Eprop, n, self.element_list)
        
        # Combine LHS
        C_combined = LinearSystem_3DAxisymmetric.combine_LHS(C, n)
        
        
        
        # Patching
        for edge in self.edgecommon_list:
            LinearSystem_3DAxisymmetric.patch_Edge(C_combined, d, edge, 
                                                   self.num_unknown, self.element_list)
    

        for corner in self.cornercommon_list:
            LinearSystem_3DAxisymmetric.patch_Corner(C_combined, d, corner, 
                                                     self.num_unknown, self.element_list)
            
        
        
        # Dirichlet BC
        for idx_unknown, func in zip(range(self.num_unknown), u_Fourier):
            LinearSystem_3DAxisymmetric.set_BC_Dirichlet(C_combined, d, n, idx_unknown, func, 
                                self.mesh_tag_nodes_BC_Dirichlet, self.element_list)
         
        LinearSystem_3DAxisymmetric.solve_System(C_combined, d, n,
                        self.element_list, self.num_unknown)
        
        
        
    
        
        
        
        
        
        
        
        
        