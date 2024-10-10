import numpy as np

import pydoc

#import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits.axes_grid1 import make_axes_locatable

# RBF interpolator, used to interpolate boundary outer normal vector
#from scipy.interpolate import RBFInterpolator


# Add the parent folder to the search path to import the OrthogonalPolynomials module
import sys
#sys.path.append('../symbolic')
sys.path.append('../')

# Reloading the module
import importlib

#import Coefficients
#importlib.reload(Coefficients)
#from Coefficients import *


import OrthogonalPolynomials
importlib.reload(OrthogonalPolynomials)
from OrthogonalPolynomials import *

import Utility
importlib.reload(Utility)

#########################################
# Location finders for constructing the linear system
#########################################

def set_LocationRange(e, location_type, location):
    # Find the range of indices in the two directions
    # e: element
    # location_type: 'inner', 'edge', 'corner'
    # location: gaps that determine the index range
        # if location_type == 'inner', then
        # location = (gap_i_start, gap_i_end, gap_j_start, gap_j_end)
            # gap_i_start: the number of points skipped at the beginning of the x direction
            # gap_i_end: the number of points skipped at the end of the x direction
            # gap_j_start, gap_j_end are similar
            
        # if location_type == 'edge', then
        # location = (edge_idx, gap_start, gap_end), or list, or 1D array of the three quantities
            # edge_idx: the index of the edge to which the BC is applied
            # gap_start: the number of grid points skipped in the beginning
            # gap_end: the number of grid points skipped at the end
            # e.g. location = (3, 1, 1), the BC is applied to edge 3
            # with the first and the last grid points on that layer skipped. Then the grid
            # index range is i = 1,...,N-1, j=M-1
            
        # if location_type == 'corner', then
        # location = 1,2,3,4, which are corner indices
        
    N = e.grid.Nx
    M = e.grid.Ny
    
    if location_type == 'inner':
        
        gap_i_start, gap_i_end, gap_j_start, gap_j_end = location
        # Starting i index
        imin = gap_i_start
        # End i index
        imax = N - gap_i_end
        # Starting j index
        jmin = gap_j_start
        # End j index
        jmax = M - gap_j_end
        
    
    elif location_type == 'edge':
        
        edge_idx, gap_start, gap_end = location
    
        if edge_idx == 1:
            imin = gap_start
            imax = N - gap_end
            jmin = 0
            jmax = jmin
        elif edge_idx == 2:
            imin = N
            imax = imin
            jmin = gap_start
            jmax = M - gap_end
        elif edge_idx == 3:
            imin = gap_start
            imax = N - gap_end
            jmin = M
            jmax = jmin
        elif edge_idx == 4:
            imin = 0
            imax = imin
            jmin = gap_start
            jmax = M - gap_end
            
    elif location_type == 'corner':
        
        edge_idx = location
        
        if edge_idx == 1:
            imin = 0
            imax = 0
            jmin = 0
            jmax = 0
        elif edge_idx == 2:
            imin = N
            imax = N
            jmin = 0
            jmax = 0
        elif edge_idx == 3:
            imin = N
            imax = N
            jmin = M
            jmax = M
        elif edge_idx == 4:
            imin = 0
            imax = 0
            jmin = M
            jmax = M
        
    return imin, imax, jmin, jmax


def set_LocationCorner(e, idx_corner):
    N = e.grid.Nx
    M = e.grid.Ny 
    
    if idx_corner == 1:
        return 0, 0
    elif idx_corner == 2:
        return N, 0
    elif idx_corner == 3:
        return N, M
    elif idx_corner == 4:
        return 0, M
        
        







############################################
# Index maps
############################################

def map_idx(i, j, idx_unknown, element, element_list):
    # Convert the grid index (i,j) of the input element 
    # to the index n of the whole linear system Cx=d 
    
    # The ordering of the index: for each unknown variable, store its values element by element
    # The resulting array = [values of 0th unknown, values of 1th unknown, ...]
    # where values of ith unknown = values in the 0th element, values in the 1th element, ...
    
    # i,j: the indices of grid of the input element
        # i=0,...,N, j=0,...,M, N,M are the maxium x and y grid labels of the element
    # idx_unknown: the indice of the unknown, starting from 0
        # In 2D displacement solution, there are two unknowns u and v
        # the index of u is 0, the index of v is 1
        # In 3D axisymmetric case, there are 3 unknowns, ux (uz), uy (ur), uphi
        # Their indices are 0, 1, 2
    # element: current element    
    # element_list: list of all elements
        # The (0,0) node of each element corresponds to an index in C
        # which depends on the number of nodes of all previous elements
        # Thus, we need element_list
    
    
    # Create a list of numbers of nodes of all elements
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
    
    # Compute the total number of nodes of all elements
    num_nodes_total = np.sum(num_nodes_list, dtype=np.int64)
    
    #idx_start: the starting index (the index of (0,0))
    # idx_start = the number of total grid nodes of all previous elements + the number of nodes of all previous unknown variables
    idx_start =  np.sum(num_nodes_list[0:element.idx], dtype=np.int64)
    idx_start += idx_unknown * num_nodes_total
   
    # Within each element, the 2D nodal values are stretched into 1D as follows:
    # (u00, u01, ..., u0M,   u10, u11, ...,u1M, ..., uN0, uN1, ..., uNM)
    
    M = element.grid.Ny
    return i*(M+1) + j + idx_start






def map_idx_edgecommon_to_element(i, edge, element_idx):
    # Given a common edge node index i, find the node index pair
    # of the neighboring element labeled by element_idx (=0, 1)
    
    # i: common edge node index
    # edge: common edge class object
    # element_idx: 0, 1, representing two neighboring elements
    
    # Extract the element, the curve label, marker for alignment
    element, curve_idx, mark_align = edge.element_info[element_idx]
    
    Nx = element.grid.Nx
    Ny = element.grid.Ny

    
    if mark_align == 1:
        
        if curve_idx == 1:
            return (i, 0)
        elif curve_idx == 2:
            return (Nx, i)
        elif curve_idx ==3:
            return (i, Ny)
        elif curve_idx ==4:
            return (0, i)
        
    elif mark_align == -1:
        
        if curve_idx == 1:
            return (Nx-i, 0)
        elif curve_idx == 2:
            return (Nx, Ny-i)
        elif curve_idx ==3:
            return (Nx-i, Ny)
        elif curve_idx ==4:
            return (0, Ny-i)
        
        

############################################
# Linear system constructors
############################################
    
##################################################
# Setting up the governing equations
##################################################


def init_C_d(element_list, num_unknown):
    # Initializing the linear system Ax = d
    # element_list: list of all elements
    # num_unknown: number of unknown variables
    
    # Return: C, d
        # C[0]: 1 part 
        # C[1]: n part (only coefficients, without n)
        # C[2]: n^2 part (only coefficients, without n^2)
        # LHS: A = C[0] + n * C[1] + n^2 * C[2]
        # d: RHS
        
    
    
    # Create a list of number of nodes of all elements
    # The number of nodes of each element is its total number of grid points
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
        
    # Get the total number of unknown grid values
    # For each unknown variable, there are np.sum(num_nodes_list, dtype=np.int64) grid values
    
    # Note that np.sum returns float as default
    # We must convert it to int in order to use it as an index
    num_nodes_total = np.sum(num_nodes_list, dtype=np.int64) * num_unknown
    
    # The linear system is Cx = d
    C = np.zeros((3, num_nodes_total, num_nodes_total), dtype=complex)
    d = np.zeros(num_nodes_total, dtype=complex)
    
    return C, d





def cal_Coeff_NG_3DAxisymmetricElasticity_Fourier_1term(e, Eprop):
    # Finding the constant part in the coefficients involved in NG 
    # 3D elasticity problem in axisymmetric domain
    # The coefficients are in the Fourier transformed equations
    
    X_xi = e.X_xi
    X_eta = e.X_eta
    Y_xi = e.Y_xi
    Y_eta = e.Y_eta
    J = e.J
    X = e.nodes_phy_x
    
    
    G = Eprop.G
    l = Eprop.lamb
    
    
    Nx = e.grid.Nx
    Ny = e.grid.Ny
    
    # Initialize coefficient matrices
    # A: coeff_p_phi_p_u
    # Coefficient in front of (\partial phi) * (\partial u)
    # Meaning of the dimensions (idx_eqn, idx_unknown, idx_phi_partial, idx_unknown_partial, n, m)
        # idx_eqn: 0->r, 1->phi, 2->z 
        # idx_unknown: 0->ur, 1->uphi, 2->uz
        # idx_phi_partial: 0->\partial r, 1->\partial z
        # idx_unknown_partial: 0->\partial r, 1->\partial z
        
    # B: coeff_p_phi_u
    # Coefficient in front of (\partial phi) * (u)
    # (idx_eqn, idx_unknown, idx_phi_partial, n, m)
    
    # C: coeff_phi_p_u
    # Coefficient in front of (phi) * (\partial u)
    # (idx_eqn, idx_unknown, idx_unknown_partial, n, m)
    
    # D: coeff_phi_u
    # Coefficient in front of (phi) * (u)
    # (idx_eqn, idx_unknown, n, m)
    
    A = np.zeros((3, 3, 2, 2, Nx+1, Ny+1), dtype=complex)
    B = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    C = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    D = np.zeros((3, 3, Nx+1, Ny+1), dtype=complex)
    
    # Equation 0
    
    A[0,0,0,0] =  G*X*X_eta**2/J + 2*G*X*Y_eta**2/J + X*Y_eta**2*l/J
    A[0,0,0,1] =  -G*X*X_eta*X_xi/J - 2*G*X*Y_eta*Y_xi/J - X*Y_eta*Y_xi*l/J
    A[0,0,1,0] =  -G*X*X_eta*X_xi/J - 2*G*X*Y_eta*Y_xi/J - X*Y_eta*Y_xi*l/J
    A[0,0,1,1] =  G*X*X_xi**2/J + 2*G*X*Y_xi**2/J + X*Y_xi**2*l/J
    A[0,2,0,0] =  -G*X*X_eta*Y_eta/J - X*X_eta*Y_eta*l/J
    A[0,2,0,1] =  G*X*X_eta*Y_xi/J + X*X_xi*Y_eta*l/J
    A[0,2,1,0] =  G*X*X_xi*Y_eta/J + X*X_eta*Y_xi*l/J
    A[0,2,1,1] =  -G*X*X_xi*Y_xi/J - X*X_xi*Y_xi*l/J
    B[0,0,0] =  Y_eta*l
    B[0,0,1] =  -Y_xi*l
    C[0,0,0] =  Y_eta*l
    C[0,0,1] =  -Y_xi*l
    C[0,2,0] =  -X_eta*l
    C[0,2,1] =  X_xi*l
    D[0,0] =  2*G*J/X + J*l/X
    
    # Equation 1
    A[1,1,0,0] =  G*X*X_eta**2/J + G*X*Y_eta**2/J
    A[1,1,0,1] =  -G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J
    A[1,1,1,0] =  -G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J
    A[1,1,1,1] =  G*X*X_xi**2/J + G*X*Y_xi**2/J
    B[1,1,0] =  -G*Y_eta
    B[1,1,1] =  G*Y_xi
    C[1,1,0] =  -G*Y_eta
    C[1,1,1] =  G*Y_xi
    D[1,1] =  G*J/X
    
    # Equation 2
    A[2,0,0,0] =  -G*X*X_eta*Y_eta/J - X*X_eta*Y_eta*l/J
    A[2,0,0,1] =  G*X*X_xi*Y_eta/J + X*X_eta*Y_xi*l/J
    A[2,0,1,0] =  G*X*X_eta*Y_xi/J + X*X_xi*Y_eta*l/J
    A[2,0,1,1] =  -G*X*X_xi*Y_xi/J - X*X_xi*Y_xi*l/J
    A[2,2,0,0] =  2*G*X*X_eta**2/J + G*X*Y_eta**2/J + X*X_eta**2*l/J
    A[2,2,0,1] =  -2*G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J - X*X_eta*X_xi*l/J
    A[2,2,1,0] =  -2*G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J - X*X_eta*X_xi*l/J
    A[2,2,1,1] =  2*G*X*X_xi**2/J + G*X*Y_xi**2/J + X*X_xi**2*l/J
    B[2,0,0] =  -X_eta*l
    B[2,0,1] =  X_xi*l

    e.coeff_p_phi_p_u = A
    e.coeff_p_phi_u = B
    e.coeff_phi_p_u = C
    e.coeff_phi_u = D   
    

def cal_Coeff_NG_3DAxisymmetricElasticity_Fourier_nterm(e, Eprop):
    # Finding the term n coefficients involved in NG 
    # 3D elasticity problem in axisymmetric domain
    # The coefficients are in the Fourier transformed equations
    
    X_xi = e.X_xi
    X_eta = e.X_eta
    Y_xi = e.Y_xi
    Y_eta = e.Y_eta
    J = e.J
    X = e.nodes_phy_x
    
    
    G = Eprop.G
    l = Eprop.lamb
    
    
    Nx = e.grid.Nx
    Ny = e.grid.Ny
    
    # Initialize coefficient matrices
    # A: coeff_p_phi_p_u
    # Coefficient in front of (\partial phi) * (\partial u)
    # Meaning of the dimensions (idx_eqn, idx_unknown, idx_phi_partial, idx_unknown_partial, n, m)
        # idx_eqn: 0->r, 1->phi, 2->z 
        # idx_unknown: 0->ur, 1->uphi, 2->uz
        # idx_phi_partial: 0->\partial r, 1->\partial z
        # idx_unknown_partial: 0->\partial r, 1->\partial z
        
    # B: coeff_p_phi_u
    # Coefficient in front of (\partial phi) * (u)
    # (idx_eqn, idx_unknown, idx_phi_partial, n, m)
    
    # C: coeff_phi_p_u
    # Coefficient in front of (phi) * (\partial u)
    # (idx_eqn, idx_unknown, idx_unknown_partial, n, m)
    
    # D: coeff_phi_u
    # Coefficient in front of (phi) * (u)
    # (idx_eqn, idx_unknown, n, m)
    
    A = np.zeros((3, 3, 2, 2, Nx+1, Ny+1), dtype=complex)
    B = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    C = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    D = np.zeros((3, 3, Nx+1, Ny+1), dtype=complex)
    
    # Equation 0 
    B[0,1,0] =  1j*Y_eta*l
    B[0,1,1] =  -1j*Y_xi*l
    C[0,1,0] =  -1j*G*Y_eta
    C[0,1,1] =  1j*G*Y_xi
    D[0,1] =  3*1j*G*J/X + 1j*J*l/X
    
    # Equation 1
    B[1,0,0] =  1j*G*Y_eta
    B[1,0,1] =  -1j*G*Y_xi
    B[1,2,0] =  -1j*G*X_eta
    B[1,2,1] =  1j*G*X_xi
    C[1,0,0] =  -1j*Y_eta*l
    C[1,0,1] =  1j*Y_xi*l
    C[1,2,0] =  1j*X_eta*l
    C[1,2,1] =  -1j*X_xi*l
    D[1,0] =  -3*1j*G*J/X - 1j*J*l/X

    # Equation 2
    B[2,1,0] =  -1j*X_eta*l
    B[2,1,1] =  1j*X_xi*l
    C[2,1,0] =  1j*G*X_eta
    C[2,1,1] =  -1j*G*X_xi

    e.coeff_p_phi_p_u = A
    e.coeff_p_phi_u = B
    e.coeff_phi_p_u = C
    e.coeff_phi_u = D   
    
    
    
def cal_Coeff_NG_3DAxisymmetricElasticity_Fourier_n2term(e, Eprop):
    # Finding the term n coefficients involved in NG 
    # 3D elasticity problem in axisymmetric domain
    # The coefficients are in the Fourier transformed equations
    
    X_xi = e.X_xi
    X_eta = e.X_eta
    Y_xi = e.Y_xi
    Y_eta = e.Y_eta
    J = e.J
    X = e.nodes_phy_x
    
    
    G = Eprop.G
    l = Eprop.lamb
    
    
    Nx = e.grid.Nx
    Ny = e.grid.Ny
    
    # Initialize coefficient matrices
    # A: coeff_p_phi_p_u
    # Coefficient in front of (\partial phi) * (\partial u)
    # Meaning of the dimensions (idx_eqn, idx_unknown, idx_phi_partial, idx_unknown_partial, n, m)
        # idx_eqn: 0->r, 1->phi, 2->z 
        # idx_unknown: 0->ur, 1->uphi, 2->uz
        # idx_phi_partial: 0->\partial r, 1->\partial z
        # idx_unknown_partial: 0->\partial r, 1->\partial z
        
    # B: coeff_p_phi_u
    # Coefficient in front of (\partial phi) * (u)
    # (idx_eqn, idx_unknown, idx_phi_partial, n, m)
    
    # C: coeff_phi_p_u
    # Coefficient in front of (phi) * (\partial u)
    # (idx_eqn, idx_unknown, idx_unknown_partial, n, m)
    
    # D: coeff_phi_u
    # Coefficient in front of (phi) * (u)
    # (idx_eqn, idx_unknown, n, m)
    
    A = np.zeros((3, 3, 2, 2, Nx+1, Ny+1), dtype=complex)
    B = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    C = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    D = np.zeros((3, 3, Nx+1, Ny+1), dtype=complex)
    
    # Equation 0 
    D[0,0] =  G*J/X
    
    # Equation 1
    D[1,1] =  2*G*J/X + J*l/X
    
    # Equation 2
    D[2,2] =  G*J/X

    e.coeff_p_phi_p_u = A
    e.coeff_p_phi_u = B
    e.coeff_phi_p_u = C
    e.coeff_phi_u = D   
    
    

    
    
def set_LHS_SingleEqn_SingleNode(C, idx_eqn, e, i, j, num_unknown, element_list):
    # Constructing the LHS of a single equation at a given node
    # C: 
        # If the LHS is constructed together, C as in Cx=d
        # If the LHS is splitted into 1, n, n^2 parts, C[0], C[1], C[2] correspond to the coefficients
    
    
    # idx_eqn: 0 (x), 1 (y)
    # e: element
    # i,j: element grid point indices where the equation belongs to
        # e, i, j fully determine the location of the grid
    # num_unknown: number of unknown
    # element_list: list of element, used for index mapping
    
    D_xi = e.grid.Dx
    D_eta = e.grid.Dy
    
    w_xi = e.grid.w_x
    w_eta = e.grid.w_y
    
    
    Ac = e.coeff_p_phi_p_u
    Bc = e.coeff_p_phi_u
    Cc = e.coeff_phi_p_u
    Dc = e.coeff_phi_u
    
    N = e.grid.Nx
    M = e.grid.Ny
    
    # The location of this equation 
    ind_1st = map_idx(i, j, idx_eqn, e, element_list)
    
                
    # A term, \partial phi \partial u 
    idx_phi_p = 0
    for idx_unknown in range(num_unknown):
        for idx_unknown_p in [0,1]:
            for n in range(0, N+1):
                coeff = w_xi[n] * w_eta[j] * D_xi[n,i] 
                coeff *= Ac[idx_eqn, idx_unknown, idx_phi_p, idx_unknown_p, n, j]
                
                if np.abs(coeff)>1e-8:
                
                    if idx_unknown_p == 0:
                        for k in range(0, N+1):
                            ind_2nd = map_idx(k, j, idx_unknown, e, element_list)
                            C[ind_1st, ind_2nd] += coeff * D_xi[n,k]
                    
                    elif idx_unknown_p == 1:
                        for s in range(0, M+1):
                            ind_2nd = map_idx(n, s, idx_unknown, e, element_list)
                            C[ind_1st, ind_2nd] += coeff * D_eta[j,s]
                    
                    
                
    idx_phi_p = 1
    for idx_unknown in range(num_unknown):
        for idx_unknown_p in [0,1]:
            for m in range(0, M+1):
                coeff = w_xi[i] * w_eta[m] * D_eta[m,j] 
                coeff *= Ac[idx_eqn, idx_unknown, idx_phi_p, idx_unknown_p, i, m]
                
                if np.abs(coeff)>1e-8:
                
                    if idx_unknown_p == 0:
                        for k in range(0, N+1):
                            ind_2nd = map_idx(k, m, idx_unknown, e, element_list)
                            C[ind_1st, ind_2nd] += coeff * D_xi[i,k]
                    
                    elif idx_unknown_p == 1:
                        for s in range(0, M+1):
                            ind_2nd = map_idx(i, s, idx_unknown, e, element_list)
                            C[ind_1st, ind_2nd] += coeff * D_eta[m,s]
                        
                        
    # B Term, \partial phi u
    idx_phi_p = 0
    for idx_unknown in range(num_unknown):
        for n in range(0, N+1):
            coeff = w_xi[n] * w_eta[j] * D_xi[n,i]
            coeff *= Bc[idx_eqn, idx_unknown, idx_phi_p, n, j]
            if np.abs(coeff)>1e-8:
                ind_2nd = map_idx(n, j, idx_unknown, e, element_list)
                C[ind_1st, ind_2nd] += coeff
            
    idx_phi_p = 1
    for idx_unknown in range(num_unknown):
        for m in range(0, M+1):
            coeff = w_xi[i] * w_eta[m] * D_eta[m,j] 
            coeff *= Bc[idx_eqn, idx_unknown, idx_phi_p, i, m]
            if np.abs(coeff)>1e-8:
                ind_2nd = map_idx(i, m, idx_unknown, e, element_list)
                C[ind_1st, ind_2nd] += coeff
    
    # C Term phi \partial u
    
    for idx_unknown in range(num_unknown):
        for idx_unknown_p in [0,1]:
            coeff = w_xi[i] * w_eta[j] * Cc[idx_eqn, idx_unknown, idx_unknown_p, i, j]
            if np.abs(coeff)>1e-8:
                if idx_unknown_p == 0:
                    for k in range(0, N+1):
                        ind_2nd = map_idx(k, j, idx_unknown, e, element_list)
                        C[ind_1st, ind_2nd] += coeff * D_xi[i,k]
        
                if idx_unknown_p == 1:
                    for s in range(0, M+1):
                        ind_2nd = map_idx(i, s, idx_unknown, e, element_list)
                        C[ind_1st, ind_2nd] += coeff * D_eta[j,s]
    
    # D Term phi u
    for idx_unknown in range(num_unknown):
        ind_2nd = map_idx(i, j, idx_unknown, e, element_list)          
        C[ind_1st, ind_2nd] += w_xi[i] * w_eta[j] * Dc[idx_eqn, idx_unknown, i, j]




    

def set_C(C, num_unknown, Eprop, element_list):
    # Set C[0], C[1], C[2]
    for npart in [0,1,2]:
        print("npart", npart)
        for e in element_list:
            if npart == 0:
                cal_Coeff_NG_3DAxisymmetricElasticity_Fourier_1term(e, Eprop)
            if npart == 1:
                cal_Coeff_NG_3DAxisymmetricElasticity_Fourier_nterm(e, Eprop)
            if npart == 2:
                cal_Coeff_NG_3DAxisymmetricElasticity_Fourier_n2term(e, Eprop)
        
        # Set equations at all nodes
        location = (0, 0, 0, 0)
        for idx_eqn in range(num_unknown):
            print("Equation", idx_eqn)
            for idx_e, e in enumerate(element_list):
                #print("Element", idx_e)
                imin, imax, jmin, jmax = set_LocationRange(e, 'inner', location)
        
                for i in range(imin, imax+1):
                    for j in range(jmin, jmax+1):
                        set_LHS_SingleEqn_SingleNode(C[npart], idx_eqn, e, i, j, num_unknown, element_list)
                        
    
    



def set_d(d, num_unknown, Eprop, n, element_list):
    # Set the RHS for solving a system for a single Fourier mode
    
    #print("RHS")
    
    # Reset d
    # d is repeatedly used for all n, thus must be reset  
    d *= 0
    
    
    # Set equations for all nodes
    location = (0, 0, 0, 0)
    for idx_eqn in range(num_unknown):
        #print("Equation", idx_eqn)
        for e in element_list:
            imin, imax, jmin, jmax = set_LocationRange(e, 'inner', location)
            for i in range(imin, imax+1):
                for j in range(jmin, jmax+1):
                    ind_1st = map_idx(i, j, idx_eqn, e, element_list)
                    idx_Fourier = Utility.map_idx_Fourier(n)
                    d[ind_1st] += e.RHS[idx_Fourier, idx_eqn, i, j]
                    
  

def combine_LHS(C, n):
    """
    Params:
      C: coefficient arrays
          The LHS of the linear system is C[0] + C[1]*n + C[2]*n^2
      n: Fourier mode (Fourier eigenvalues)
    Returns: 
      The LHS of the linear system: C[0] + C[1]*n + C[2]*n^2
    """
    return C[0] + C[1] * n + C[2] * n**2

    
##################################################
# Setting up BCs
##################################################

################################################
# Displacement BC (Dirichlet)
################################################

        
def set_BC_Dirichlet(C, d, n, idx_unknown, func, tag_nodes, element_list):
    # tag_nodes: labels of Dirichlet nodes
    
    # There are rare cases where an element has an edge and a corner on the Dirichlet boundary
    # and the corner does not belong to the edge.
    # Thus, we first consider all Dirichlet edges and they all Dirichlet corners
    # Some nodes may be treated multiple times, which are unnecessary, 
    # but this is the safest approach.
    
    # Dirichlet edges
    for e in element_list:
        # Determine whether an edge is on the Dirichlet boundary
        for idx_edge, idx_c1, idx_c2 in ((1,1,2),(2,2,3),(3,4,3),(4,1,4)):
            if (e.label_corners[idx_c1-1] in tag_nodes) and (e.label_corners[idx_c2-1] in tag_nodes): 
                #print("idx_edge", idx_edge)
                imin, imax, jmin, jmax = set_LocationBCEquation(e, 'edge', (idx_edge,0,0))
                for i in range(imin, imax+1):
                    for j in range(jmin, jmax+1):
                        idx_global = map_idx(i, j, idx_unknown, e, element_list)
                        x = e.nodes_phy_x[i,j]
                        y = e.nodes_phy_y[i,j]
                        
                        C[idx_global, :] = 0
                        C[idx_global, idx_global] = 1 
                        d[idx_global] = func(x, y, n)
                        
        # Determine whether a corner is on the Dirichlet boundary
        for idx_corner in (1,2,3,4):
            if e.label_corners[idx_corner-1] in tag_nodes:
                #print("idx_corner", idx_corner)
                i, j = set_LocationCorner(e, idx_corner)
                idx_global = map_idx(i, j, idx_unknown, e, element_list)
                x = e.nodes_phy_x[i,j]
                y = e.nodes_phy_y[i,j]
                
                C[idx_global, :] = 0
                C[idx_global, idx_global] = 1 
                d[idx_global] = func(x,y,n)
    



###############################################
# Patching
###############################################
def patch_Edge(C, d, edge, num_unknown, element_list):
    # Patching along a single edge, only inner edge nodes
    # edge: a common edge
    # Instead of creating new equations, this function modifies the existing matrix C
    
    
    for idx_unknown in range(num_unknown):
        # idx_unknown here represents both the equation and unknown indices.
    
        # Patching inner edge nodes
        for s in range(1, edge.N): # 1...N-1
    
            # idx_1 is both the location of the idx_unknown th equation 
            # and the location of the idx_unknown th variable
            edge_element_idx = 0
            e, curve_idx, mark_align = edge.element_info[edge_element_idx]
            i, j = map_idx_edgecommon_to_element(s, edge, edge_element_idx)
            idx_1 = map_idx(i, j, idx_unknown, e, element_list)
            
            edge_element_idx = 1
            e, curve_idx, mark_align = edge.element_info[edge_element_idx]
            i, j = map_idx_edgecommon_to_element(s, edge, edge_element_idx)
            idx_2 = map_idx(i, j, idx_unknown, e, element_list)
            
            C[idx_1, :] = C[idx_1, :] + C[idx_2, :]
            d[idx_1] = d[idx_1] + d[idx_2]
            
            C[idx_2, :] = 0
            d[idx_2] = 0
            C[idx_2, idx_2] = 1
            C[idx_2, idx_1] = -1
            


def patch_Corner(C, d, corner, num_unknown, element_list):
    
    
    # Finding the number of elements sharing the corner
    # We single out this treatment since multiple elements may share a single corner.
    num_element = len(corner.element_info)
    
    
    for idx_unknown in range(num_unknown):
        e, idx_corner = corner.element_info[0]
        i, j = set_LocationCorner(e, idx_corner)
        idx0 = map_idx(i, j, idx_unknown, e, element_list)
        
    
        for k in range(1, num_element):
            e, idx_corner = corner.element_info[k]
            i, j = set_LocationCorner(e, idx_corner)
            idx = map_idx(i, j, idx_unknown, e, element_list)
        
            C[idx0, :] = C[idx0, :] + C[idx, :]
            d[idx0] = d[idx0] + d[idx]
        
            C[idx, :] = 0
            C[idx, idx] = 1
            C[idx, idx0] = -1
            d[idx] = 0



    
################################################
# Solver
#################################################

def solve_System(C, d, n, element_list, num_unknown):
    # Solving the matrix Cu = d and reshape the solution
    # C: coefficient arrays
    # d: RHS array
    # n: Fourier eigenvalue
    
    # Index corresponding to n
    idx_Fourier = Utility.map_idx_Fourier(n)

    
    
    u = np.linalg.solve(C, d)
    
    print('residue', np.max(np.absolute(d - np.dot(C, u))))
    
    
    # Create a list of number of nodes of all elements
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
    # Total number of nodes   
    num_nodes_total = np.sum(num_nodes_list, dtype=np.int64)
    
    # Reshape the solution
    for idx_unknown in range(num_unknown):
        u_single_unknown = u[idx_unknown*num_nodes_total:(idx_unknown+1)*num_nodes_total]
    
        for k, element in enumerate(element_list):
            idx_start = np.sum(num_nodes_list[0:k], dtype=np.int64)
            idx_end = np.sum(num_nodes_list[0:k+1], dtype=np.int64)
            Nx = element.grid.Nx
            Ny = element.grid.Ny
            element.U[idx_Fourier, idx_unknown] = u_single_unknown[idx_start : idx_end].reshape(Nx+1, Ny+1)
        
          
































####################################################
# Old codes
####################################################
def set_d_SingleNode(d, n, idx_eqn, e, i, j, element_list):
    # Constructing the RHS of a single equation at a single node
    # d as in Cx = d
    # n: Fourier mode
    # ind_st: the current index
    # idx_eqn: 0 (x), 1 (y)
    # e: element
    # i,j: element grid point indices where the equation belongs to
        # e, i, j fully determine the location of the grid
    
    ind_1st = map_idx(i, j, idx_eqn, e, element_list)
    idx_Fourier = Utility.map_idx_Fourier(n)

    d[ind_1st] += e.RHS[idx_Fourier, idx_eqn, i, j]



def cal_Coeff_NG_3DAxisymmetricElasticity_Fourier(e, Eprop, n):
    # Finding the coefficients involved in NG 
    # 3D Elasticity problem in axisymmetric domain
    # The coefficients are in the Fourier transformed equations
    
    # This function generates all coefficients, including constants, n-coefficients, n^2-coefficients
    # Used only for test purpose
    
    X_xi = e.X_xi
    X_eta = e.X_eta
    Y_xi = e.Y_xi
    Y_eta = e.Y_eta
    J = e.J
    X = e.nodes_phy_x
    
    
    G = Eprop.G
    l = Eprop.lamb
    
    
    Nx = e.grid.Nx
    Ny = e.grid.Ny
    
    # Initialize coefficient matrices
    # A: coeff_p_phi_p_u
    # Coefficient in front of (\partial phi) * (\partial u)
    # Meaning of the dimensions (idx_eqn, idx_unknown, idx_phi_partial, idx_unknown_partial, n, m)
        # idx_eqn: 0->r, 1->phi, 2->z 
        # idx_unknown: 0->ur, 1->uphi, 2->uz
        # idx_phi_partial: 0->\partial r, 1->\partial z
        # idx_unknown_partial: 0->\partial r, 1->\partial z
        
    # B: coeff_p_phi_u
    # Coefficient in front of (\partial phi) * (u)
    # (idx_eqn, idx_unknown, idx_phi_partial, n, m)
    
    # C: coeff_phi_p_u
    # Coefficient in front of (phi) * (\partial u)
    # (idx_eqn, idx_unknown, idx_unknown_partial, n, m)
    
    # D: coeff_phi_u
    # Coefficient in front of (phi) * (u)
    # (idx_eqn, idx_unknown, n, m)
    
    A = np.zeros((3, 3, 2, 2, Nx+1, Ny+1), dtype=complex)
    B = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    C = np.zeros((3, 3, 2, Nx+1, Ny+1), dtype=complex)
    D = np.zeros((3, 3, Nx+1, Ny+1), dtype=complex)
    
    
    
    A[0,0,0,0] =  G*X*X_eta**2/J + 2*G*X*Y_eta**2/J + X*Y_eta**2*l/J
    A[0,0,0,1] =  -G*X*X_eta*X_xi/J - 2*G*X*Y_eta*Y_xi/J - X*Y_eta*Y_xi*l/J
    A[0,0,1,0] =  -G*X*X_eta*X_xi/J - 2*G*X*Y_eta*Y_xi/J - X*Y_eta*Y_xi*l/J
    A[0,0,1,1] =  G*X*X_xi**2/J + 2*G*X*Y_xi**2/J + X*Y_xi**2*l/J
    A[0,2,0,0] =  -G*X*X_eta*Y_eta/J - X*X_eta*Y_eta*l/J
    A[0,2,0,1] =  G*X*X_eta*Y_xi/J + X*X_xi*Y_eta*l/J
    A[0,2,1,0] =  G*X*X_xi*Y_eta/J + X*X_eta*Y_xi*l/J
    A[0,2,1,1] =  -G*X*X_xi*Y_xi/J - X*X_xi*Y_xi*l/J
    B[0,0,0] =  Y_eta*l
    B[0,0,1] =  -Y_xi*l
    B[0,1,0] =  1j*Y_eta*l*n
    B[0,1,1] =  -1j*Y_xi*l*n
    C[0,0,0] =  Y_eta*l
    C[0,0,1] =  -Y_xi*l
    C[0,1,0] =  -1j*G*Y_eta*n
    C[0,1,1] =  1j*G*Y_xi*n
    C[0,2,0] =  -X_eta*l
    C[0,2,1] =  X_xi*l
    D[0,0] =  G*J*n**2/X + 2*G*J/X + J*l/X
    D[0,1] =  3*1j*G*J*n/X + 1j*J*l*n/X

    

    A[1,1,0,0] =  G*X*X_eta**2/J + G*X*Y_eta**2/J
    A[1,1,0,1] =  -G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J
    A[1,1,1,0] =  -G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J
    A[1,1,1,1] =  G*X*X_xi**2/J + G*X*Y_xi**2/J
    B[1,0,0] =  1j*G*Y_eta*n
    B[1,0,1] =  -1j*G*Y_xi*n
    B[1,1,0] =  -G*Y_eta
    B[1,1,1] =  G*Y_xi
    B[1,2,0] =  -1j*G*X_eta*n
    B[1,2,1] =  1j*G*X_xi*n
    C[1,0,0] =  -1j*Y_eta*l*n
    C[1,0,1] =  1j*Y_xi*l*n
    C[1,1,0] =  -G*Y_eta
    C[1,1,1] =  G*Y_xi
    C[1,2,0] =  1j*X_eta*l*n
    C[1,2,1] =  -1j*X_xi*l*n
    D[1,0] =  -3*1j*G*J*n/X - 1j*J*l*n/X
    D[1,1] =  2*G*J*n**2/X + G*J/X + J*l*n**2/X
    
    
    

    A[2,0,0,0] =  -G*X*X_eta*Y_eta/J - X*X_eta*Y_eta*l/J
    A[2,0,0,1] =  G*X*X_xi*Y_eta/J + X*X_eta*Y_xi*l/J
    A[2,0,1,0] =  G*X*X_eta*Y_xi/J + X*X_xi*Y_eta*l/J
    A[2,0,1,1] =  -G*X*X_xi*Y_xi/J - X*X_xi*Y_xi*l/J
    A[2,2,0,0] =  2*G*X*X_eta**2/J + G*X*Y_eta**2/J + X*X_eta**2*l/J
    A[2,2,0,1] =  -2*G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J - X*X_eta*X_xi*l/J
    A[2,2,1,0] =  -2*G*X*X_eta*X_xi/J - G*X*Y_eta*Y_xi/J - X*X_eta*X_xi*l/J
    A[2,2,1,1] =  2*G*X*X_xi**2/J + G*X*Y_xi**2/J + X*X_xi**2*l/J
    B[2,0,0] =  -X_eta*l
    B[2,0,1] =  X_xi*l
    B[2,1,0] =  -1j*X_eta*l*n
    B[2,1,1] =  1j*X_xi*l*n
    C[2,1,0] =  1j*G*X_eta*n
    C[2,1,1] =  -1j*G*X_xi*n
    D[2,2] =  G*J*n**2/X
    
    
    

    e.coeff_p_phi_p_u = A
    e.coeff_p_phi_u = B
    e.coeff_phi_p_u = C
    e.coeff_phi_u = D  

def init_System_old(element_list, num_unknown):
    # Initializing the linear system Ax = d
    # element_list: list of all elements
    # num_unknown: number of unknown variables
    
    # Return: C, d, ind_1st
        # C, d as in Cx = d 
        # ind_1st: 0, the index of the first row
    
    
    # Create a list of number of nodes of all elements
    # The number of nodes of each element is its total number of grid points
    num_nodes_list = []
    for e in element_list:
        num_nodes_list.append(e.num_nodes)
        
    # Get the total number of unknown grid values
    # For each unknown variable, there are np.sum(num_nodes_list, dtype=np.int64) grid values
    
    # Note that np.sum returns float as default
    # We must convert it to int in order to use it as an index
    num_nodes_total = np.sum(num_nodes_list, dtype=np.int64) * num_unknown
    
    # The linear system is Cx = d
    C = np.zeros((num_nodes_total, num_nodes_total), dtype=complex)
    d = np.zeros(num_nodes_total, dtype=complex)
    
    
    
    return C, d

def set_LocationInnerNodeEquation(e, location):
    # Given the gaps stored in location, compute the grid point range 
    # for inner node equations
    # Used in the construction of the linear system
    # e: element
    # location = (gap_i_start, gap_i_end, gap_j_start, gap_j_end)
        # gap_i_start: the number of points skipped at the beginning of the x direction
        # gap_i_end: the number of points skipped at the end of the x direction
        # gap_j_start, gap_j_end are similar
    
    N = e.grid.Nx
    M = e.grid.Ny
    
    gap_i_start, gap_i_end, gap_j_start, gap_j_end = location
    # Starting i index
    imin = gap_i_start
    # End i index
    imax = N - gap_i_end
    # Starting j index
    jmin = gap_j_start
    # End j index
    jmax = M - gap_j_end
    
    return imin, imax, jmin, jmax

def set_LocationBCEquation(e, location_type, location):
    # Given the information in location, compute the grid point range
    # for BC equations
    # Used in the construction of the linear system
    
    # location_type: 'edge', 'corner'
        # Most BCs are applied to the whole or part of an edge
        # Corner BCs are used when two neighboring edges both have stress BCs
    
    # location: location of the BC
        # if location_type == 'edge'
        # location = (edge_idx, gap_start, gap_end), or list, or 1D array of the three quantities
        # edge_idx: the index of the edge to which the BC is applied
        # gap_start: the number of grid points skipped in the beginning
        # gap_end: the number of grid points skipped at the end
        # e.g. location = (3, 1, 1), the BC is applied to edge 3
        # with the first and the last grid points on that layer skipped. Then the grid
        # index range is i = 1,...,N-1, j=M-1
        
        # if location_type == 'corner'
        # location = 1,2,3,4, which are corner indices
        
        
    N = e.grid.Nx
    M = e.grid.Ny    
    
    if location_type == 'edge':
    
        edge_idx, gap_start, gap_end = location
    
        if edge_idx == 1:
            imin = gap_start
            imax = N - gap_end
            jmin = 0
            jmax = jmin
        elif edge_idx == 2:
            imin = N
            imax = imin
            jmin = gap_start
            jmax = M - gap_end
        elif edge_idx == 3:
            imin = gap_start
            imax = N - gap_end
            jmin = M
            jmax = jmin
        elif edge_idx == 4:
            imin = 0
            imax = imin
            jmin = gap_start
            jmax = M - gap_end
            
    elif location_type == 'corner':
        
        edge_idx = location
        
        if edge_idx == 1:
            imin = 0
            imax = 0
            jmin = 0
            jmax = 0
        elif edge_idx == 2:
            imin = N
            imax = N
            jmin = 0
            jmax = 0
        elif edge_idx == 3:
            imin = N
            imax = N
            jmin = M
            jmax = M
        elif edge_idx == 4:
            imin = 0
            imax = 0
            jmin = M
            jmax = M
        
    return imin, imax, jmin, jmax











def set_LHS_SingleEqn_SingleNode_old(C, idx_eqn, e, i, j, num_unknown, element_list):
    # Constructing the LHS of a single equation at a given node
    # C: 
        # If the LHS is constructed together, C as in Cx=d
        # If the LHS is splitted into 1, n, n^2 parts, C[0], C[1], C[2] correspond to the coefficients
    
    
    # idx_eqn: 0 (x), 1 (y)
    # e: element
    # i,j: element grid point indices where the equation belongs to
        # e, i, j fully determine the location of the grid
    # num_unknown: number of unknown
    # element_list: list of element, used for index mapping
    
    D_xi = e.grid.Dx
    D_eta = e.grid.Dy
    
    w_xi = e.grid.w_x
    w_eta = e.grid.w_y
    
    
    Ac = e.coeff_p_phi_p_u
    Bc = e.coeff_p_phi_u
    Cc = e.coeff_phi_p_u
    Dc = e.coeff_phi_u
    
    N = e.grid.Nx
    M = e.grid.Ny
    
    # The location of this equation 
    ind_1st = map_idx(i, j, idx_eqn, e, element_list)
    
                
    # A term, \partial phi \partial u 
    idx_phi_p = 0
    for idx_unknown in range(num_unknown):
        for idx_unknown_p in [0,1]:
            for n in range(0, N+1):
                coeff = w_xi[n] * w_eta[j] * D_xi[n,i] 
                coeff *= Ac[idx_eqn, idx_unknown, idx_phi_p, idx_unknown_p, n, j]
                
                if idx_unknown_p == 0:
                    for k in range(0, N+1):
                        ind_2nd = map_idx(k, j, idx_unknown, e, element_list)
                        C[ind_1st, ind_2nd] += coeff * D_xi[n,k]
                    
                elif idx_unknown_p == 1:
                    for s in range(0, M+1):
                        ind_2nd = map_idx(n, s, idx_unknown, e, element_list)
                        C[ind_1st, ind_2nd] += coeff * D_eta[j,s]
                    
                    
                
    idx_phi_p = 1
    for idx_unknown in range(num_unknown):
        for idx_unknown_p in [0,1]:
            for m in range(0, M+1):
                coeff = w_xi[i] * w_eta[m] * D_eta[m,j] 
                coeff *= Ac[idx_eqn, idx_unknown, idx_phi_p, idx_unknown_p, i, m]
                
                if idx_unknown_p == 0:
                    for k in range(0, N+1):
                        ind_2nd = map_idx(k, m, idx_unknown, e, element_list)
                        C[ind_1st, ind_2nd] += coeff * D_xi[i,k]
                    
                elif idx_unknown_p == 1:
                    for s in range(0, M+1):
                        ind_2nd = map_idx(i, s, idx_unknown, e, element_list)
                        C[ind_1st, ind_2nd] += coeff * D_eta[m,s]
                        
                        
    # B Term, \partial phi u
    idx_phi_p = 0
    for idx_unknown in range(num_unknown):
        for n in range(0, N+1):
            coeff = w_xi[n] * w_eta[j] * D_xi[n,i]
            coeff *= Bc[idx_eqn, idx_unknown, idx_phi_p, n, j]
            ind_2nd = map_idx(n, j, idx_unknown, e, element_list)
            C[ind_1st, ind_2nd] += coeff
            
    idx_phi_p = 1
    for idx_unknown in range(num_unknown):
        for m in range(0, M+1):
            coeff = w_xi[i] * w_eta[m] * D_eta[m,j] 
            coeff *= Bc[idx_eqn, idx_unknown, idx_phi_p, i, m]
            ind_2nd = map_idx(i, m, idx_unknown, e, element_list)
            C[ind_1st, ind_2nd] += coeff
    
    # C Term phi \partial u
    
    for idx_unknown in range(num_unknown):
        for idx_unknown_p in [0,1]:
            coeff = w_xi[i] * w_eta[j] * Cc[idx_eqn, idx_unknown, idx_unknown_p, i, j]
            if idx_unknown_p == 0:
                for k in range(0, N+1):
                    ind_2nd = map_idx(k, j, idx_unknown, e, element_list)
                    C[ind_1st, ind_2nd] += coeff * D_xi[i,k]
        
            if idx_unknown_p == 1:
                for s in range(0, M+1):
                    ind_2nd = map_idx(i, s, idx_unknown, e, element_list)
                    C[ind_1st, ind_2nd] += coeff * D_eta[j,s]
    
    # D Term phi u
    for idx_unknown in range(num_unknown):
        ind_2nd = map_idx(i, j, idx_unknown, e, element_list)          
        C[ind_1st, ind_2nd] += w_xi[i] * w_eta[j] * Dc[idx_eqn, idx_unknown, i, j]

def set_Eqn_old(C, d, num_unknown, Eprop, n, element_list):
    
    for e in element_list:
        # Changes here
        cal_Coeff_NG_3DAxisymmetricElasticity_Fourier(e, Eprop, n)
    
    # Set equations at all nodes
    location = (0, 0, 0, 0)
    for idx_eqn in range(num_unknown):
        print("Equation", idx_eqn)
        for idx_e, e in enumerate(element_list):
            print("Element", idx_e)
            
            # Changes here
            imin, imax, jmin, jmax = set_LocationInnerNodeEquation(e, location)
        
            for i in range(imin, imax+1):
                for j in range(jmin, jmax+1):
                    #print(f"node ({i}, {j})")
                    set_LHS_SingleEqn_SingleNode(C, idx_eqn, e, i, j, num_unknown, element_list)
                    set_RHS_SingleEqn_SingleNode(d, idx_eqn, e, i, j, element_list)
    
   




def set_BC_Dirichlet_Edge_old(C, d, e, location, idx_unknown, func, element_list):
    # Set up the RHS of displacement (Dirichlet) BC equations at the specified location
    # e: element
    # location = (idx_edge, gap_start, gap_end)
        # see Element class set_LocationBCEquation for details
    # idx_unknown: index of the unknown variable
    # func: function specifying the boundary values
    
    # Changes here
    imin, imax, jmin, jmax = set_LocationBCEquation(e, 'edge', location)
    for i in range(imin, imax+1):
        for j in range(jmin, jmax+1):
            idx = map_idx(i, j, idx_unknown, e, element_list)
            x = e.nodes_phy_x[i,j]
            y = e.nodes_phy_y[i,j]
            C[idx, :] = 0
            C[idx, idx] = 1 
            d[idx] = func(x,y)
            
            

def set_BC_Dirichlet_CommonCorner_old(C, d, corner, idx_unknown, func, element_list):
    # Set up Dirichlet BC at a common corner
    # conrer: corner object
    # idx_unknown: index of the unknown
    # func: function specifying the boundary values

    for e, idx_corner in corner.element_info: 
        # Changes here
        i, j = set_LocationCorner(e, idx_corner)
        idx = map_idx(i, j, idx_unknown, e, element_list)
        C[idx, :] = 0
        C[idx, idx] = 1
        
        x = e.nodes_phy_x[i,j]
        y = e.nodes_phy_y[i,j]
        d[idx] = func(x,y)




#pydoc.writedoc('LinearSystem_3DAxisymmetric')