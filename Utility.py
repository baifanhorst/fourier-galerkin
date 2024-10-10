import numpy as np
from scipy.interpolate import griddata





import shapely.geometry as sg





# Reloading the module
import importlib

import LinearSystem_3DAxisymmetric
importlib.reload(LinearSystem_3DAxisymmetric)
from LinearSystem_3DAxisymmetric import *

import SEM2D_Line
importlib.reload(SEM2D_Line)
from SEM2D_Line import *

import SEM2D_Grid
importlib.reload(SEM2D_Grid)
from SEM2D_Grid import *

import SEM2D_Edge
importlib.reload(SEM2D_Edge)
from SEM2D_Edge import *

import SEM2D_EdgeCommon
importlib.reload(SEM2D_EdgeCommon)
from SEM2D_EdgeCommon import *

import SEM2D_Element_Quad
importlib.reload(SEM2D_Element_Quad)
from SEM2D_Element_Quad import *

import SEM2D_Corner
importlib.reload(SEM2D_Corner)
from SEM2D_Corner import *

###################################################
# Element construction
###################################################


def set_Element_Quad(elements, nodes, grid_sizes, grid_types):
    # Create elements according to mesh data
    # elements: mesh data for element
    # nodes: node coordinates
    # grid_sizes: (N, N), N is the maximum node label in each direction
    # grid_types: grid types for the two directions. 
        # For nodal Galerkin methods, grid_types = ("Legendre", "Legendre")
        
    
    # Initialize element list
    element_list = []
    
    # Grid
    Nx, Ny = grid_sizes
    node_type_x, node_type_y = grid_types
    grid = Grid(Nx, Ny, node_type_x, node_type_y)
    
    # Corner coordinates
    for idx_e in range(elements.shape[0]):
        corners = np.zeros((4,2))
        for i in range(4):
            corners[i] = nodes[elements[idx_e, i]]
            
    
        element_list.append( Element(idx_e, corners, elements[idx_e], grid) )
    

    #element.visualizing_curves()
    #element.visualizing_Grid_NormalVector()
    
    return element_list


def set_Edge(element_list):
    # Create a list of all elemental edges
    
    edge_list = []
    # Create a list of all edges
    for e in element_list:
        for idx_edge in (1,2,3,4):
            edge_list.append(Edge(e, idx_edge))
            
    return edge_list




def set_EdgeCommon(edge_list):
    # Create a list of common edges
    # edge_list: a list of all element edges
    edgecommon_list = []
    idx = 0
    for i in range(len(edge_list)):
        edge1 = edge_list[i]
        for j in range(i+1, len(edge_list)):
            edge2 = edge_list[j]
                
            if np.array_equal(edge1.c1, edge2.c1) and np.array_equal(edge1.c2, edge2.c2):
                element_info = [(edge1.e, edge1.idx_edge, 1), (edge2.e, edge2.idx_edge, 1)]
                edgecommon = EdgeCommon(idx, element_info)
                edgecommon_list.append(edgecommon)
                idx += 1
            
            if np.array_equal(edge1.c1, edge2.c2) and np.array_equal(edge1.c2, edge2.c1):
                element_info = [(edge1.e, edge1.idx_edge, 1), (edge2.e, edge2.idx_edge, -1)]
                edgecommon = EdgeCommon(idx, element_info)
                edgecommon_list.append(edgecommon)
                idx += 1
                
    return edgecommon_list


def set_CornerCommon(corners, element_list):
    # Create a list of common corners
    # corners: corner coordinate arrays, each row stores the coordinates of a corner
    cornercommon_list = []
    for idx_corner, c in enumerate(corners):
        element_info = []
        for e in element_list:
            for k, c_ in enumerate(e.corners):
                if np.array_equal(c, c_):
                    element_info.append((e, k+1))
        if len(element_info)>1:
            cornercommon = CornerCommon(idx_corner, c, element_info)
            cornercommon_list.append(cornercommon)   
            
    return cornercommon_list



   
    




#########################################
# Linear system
###########################################

def map_idx_Fourier(n):
    # Given the Fourier mode n, find the index in the solution array
    # The Fourier modes are ordered as: 0, -1, 1, -2, 2, ...
    
    if n >= 0:
        return 2*n
    else:
        return -2*n - 1
    


def init_Solution_RHS(num_unknown, e, nmax, exact=True):
    # Initialize solution and RHS arrays for a single element
    # nmax: max Fourier mode
        # the Fourier mode = -nmax, ..., 0, ... nmax
    
    # We order solutions for all Fourier modes as
    # U0, U-1, U1, U-2, U2 
    # Given index i, the corresponding Fourier mode is 
        # i even: Fourier mode = i//2
        # i odd: Fourier mode = -(i//2+1) 
    
    N = e.grid.Nx
    M = e.grid.Ny
    
    # Numerical solution
    e.U = np.zeros((2*nmax+1, num_unknown, N+1, M+1), dtype=complex)
    
    # RHS
    e.RHS = np.zeros((2*nmax+1, num_unknown, N+1, M+1), dtype=complex) 
    
    # Exact solution
    if exact:
        e.Uth = np.zeros((2*nmax+1, num_unknown, N+1, M+1), dtype=complex)
        
    
    
    

def set_Solution_Exact(n, num_unknown, element_list, u_Fourier):
    # Set the exact solution
    # u_Fourier: (3,), functions for the 3 transformed displacement components
    
    idx_Fourier = map_idx_Fourier(n)
    
    for e in element_list:
        X = e.nodes_phy_x
        Y = e.nodes_phy_y
        for i in range(num_unknown):
            e.Uth[idx_Fourier, i] = u_Fourier[i](X, Y, n)
            
    
    
    


    
def set_RHS_BodyForce(n, num_unknown, element_list, f_Fourier, Eprop):
    # Set the contribution of the bodyforce to the RHS
    
    idx_Fourier = map_idx_Fourier(n)
    G = Eprop.G
    lamb = Eprop.lamb
    
    for e in element_list:
        X = e.nodes_phy_x
        Y = e.nodes_phy_y
        N = e.grid.Nx
        M = e.grid.Ny
    
        for idx_eqn, f in zip(range(num_unknown), f_Fourier):
            for i in range(0, N+1):
                for j in range(0, M+1):
                    e.RHS[idx_Fourier, idx_eqn, i, j] = (f(X[i,j], Y[i,j], n, G, lamb) 
                                            * e.grid.w_x[i] * e.grid.w_y[j] 
                                            * X[i,j] * e.J[i,j])
    
    
    
    
    
                
                
def set_RHS_ExternalSress(n, num_unknown, element_list, p_Fourier, Eprop, 
                                        tag_nodes):
    
    # Set the contribution of the external load to the RHS
    # tag_nodes: nodes on the Neumann boundary
    G = Eprop.G
    lamb = Eprop.lamb
    
    idx_Fourier = map_idx_Fourier(n)
    
    for e in element_list:
        
        # Factors in the contribution
        # See notes 'SEM2D, Axisymmetric Elasticity, Nonsquare Nodal Galerkin'
        factor_xi = np.sqrt(e.X_xi**2 + e.Y_xi**2)
        factor_eta = np.sqrt(e.X_eta**2 + e.Y_eta**2)
        X = e.nodes_phy_x
        Y = e.nodes_phy_y
        
        
        # Determine whether an edge is on the Neumann boundary
        for idx_edge, idx_c1, idx_c2 in ((1,1,2),(2,2,3),(3,4,3),(4,1,4)):
            if (e.label_corners[idx_c1-1] in tag_nodes) and (e.label_corners[idx_c2-1] in tag_nodes): 
                imin, imax, jmin, jmax = LinearSystem_3DAxisymmetric.set_LocationRange(e, 'edge', (idx_edge,0,0))
                for idx_eqn in range(num_unknown):
                    for i in range(imin, imax+1):
                        for j in range(jmin, jmax+1):
                            r = X[i,j]
                            z = Y[i,j]

                            if idx_edge == 1:
                                nr, nz = e.norm_vect_lower[i]
                                factor = factor_xi[i,j]
                                w = e.grid.w_x[i]
                            elif idx_edge == 3:
                                nr, nz = e.norm_vect_upper[i] 
                                factor = factor_xi[i,j]
                                w = e.grid.w_x[i]
                            elif idx_edge == 4:
                                nr, nz = e.norm_vect_left[j]
                                factor = factor_eta[i,j]
                                w = e.grid.w_y[j]
                            elif idx_edge == 2:
                                nr, nz = e.norm_vect_right[j]
                                factor = factor_eta[i,j]
                                w = e.grid.w_y[j]

                            e.RHS[idx_Fourier, idx_eqn, i, j] += p_Fourier[idx_eqn](r, z, n, nr, nz, G, lamb) * factor * w * r
                
                
    



def cal_U_Physical(solid):
    # Add all Fourier terms
    nmax = solid.nmax
    Nphi = solid.Nphi
    # Broadcast angular grid
    phi = solid.phi[:, np.newaxis, np.newaxis]
    
    # Initialize element solution in polar coordinates
    for e in solid.element_list:
        N = e.grid.Nx
        M = e.grid.Ny
        e.U_polar = np.zeros((solid.num_unknown, Nphi+1, N+1, M+1), dtype=complex)
    
    # Adding up all terms in the Fourier truncated series
    for n in range(-nmax, nmax+1):
        idx_Fourier = map_idx_Fourier(n)
        factor = np.exp(1j*n*phi)
        
        for e in solid.element_list:
            for idx_unknown in range(solid.num_unknown):
                e.U_polar[idx_unknown] += factor * e.U[idx_Fourier, idx_unknown] 
                
    
                
                
def cal_U_Physical_SingleAngle(solid, phi):
    # Compute the displacement at a single angle
    # phi: the specified scalar angle
    
    # Add all Fourier terms
    nmax = solid.nmax
            
    # Initialize element solution in polar coordinates
    for e in solid.element_list:
        N = e.grid.Nx
        M = e.grid.Ny
        e.U_polar_SingleAngle = np.zeros((solid.num_unknown, N+1, M+1), dtype=complex)
    
    # Adding up all terms in the Fourier truncated series
    for n in range(-nmax, nmax+1):
        idx_Fourier = map_idx_Fourier(n)
        factor = np.exp(1j*n*phi)
        
        for e in solid.element_list:
            for idx_unknown in range(solid.num_unknown):
                e.U_polar_SingleAngle[idx_unknown] += factor * e.U[idx_Fourier, idx_unknown] 
                
    
    for e in solid.element_list:
        e.U_polar_SingleAngle = np.real(e.U_polar_SingleAngle)
 

###########################################
# Stress 
###########################################                   


def cal_Stress_Fourier_Element(e, n, Eprop):
    # Compute Fourier transformed stress for a single element and a single Fourier mode
    G = Eprop.G
    l = Eprop.lamb
    
    idx_Fourier = map_idx_Fourier(n)
  
    # Metrics
    X = e.nodes_phy_x
    J = e.J
    X_xi = e.X_xi
    X_eta = e.X_eta
    Y_xi = e.Y_xi
    Y_eta = e.Y_eta
    
    D_xi = e.grid.Dx
    D_eta = e.grid.Dy
    
    # Fourier components of displacement
    Ur = e.U[idx_Fourier, 0]
    Uphi = e.U[idx_Fourier, 1]
    Uz = e.U[idx_Fourier, 2]
    
    # Derivatives in the computational domain
    Ur_xi = np.dot(D_xi, Ur)
    Ur_eta = np.dot(Ur, D_eta.T)
    
    Uphi_xi = np.dot(D_xi, Uphi)
    Uphi_eta = np.dot(Uphi, D_eta.T)
    
    Uz_xi = np.dot(D_xi, Uz)
    Uz_eta = np.dot(Uz, D_eta.T)
    
    
    
    
    # Computation of stress
    e.stress[idx_Fourier, 0, 0] =  1j*Uphi*l*n/X + 2*G*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + Ur*l/X + l*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + l*(Uz_eta*X_xi/J - Uz_xi*X_eta/J)
    e.stress[idx_Fourier, 0, 1] =  1j*G*Ur*n/X - G*Uphi/X + G*(-Uphi_eta*Y_xi/J + Uphi_xi*Y_eta/J)
    e.stress[idx_Fourier, 0, 2] =  G*(Ur_eta*X_xi/J - Ur_xi*X_eta/J) + G*(-Uz_eta*Y_xi/J + Uz_xi*Y_eta/J)
    e.stress[idx_Fourier, 1, 0] =  1j*G*Ur*n/X - G*Uphi/X + G*(-Uphi_eta*Y_xi/J + Uphi_xi*Y_eta/J)
    e.stress[idx_Fourier, 1, 1] =  2*1j*G*Uphi*n/X + 1j*Uphi*l*n/X + 2*G*Ur/X + Ur*l/X + l*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + l*(Uz_eta*X_xi/J - Uz_xi*X_eta/J)
    e.stress[idx_Fourier, 1, 2] =  1j*G*Uz*n/X + G*(Uphi_eta*X_xi/J - Uphi_xi*X_eta/J)
    e.stress[idx_Fourier, 2, 0] =  G*(Ur_eta*X_xi/J - Ur_xi*X_eta/J) + G*(-Uz_eta*Y_xi/J + Uz_xi*Y_eta/J)
    e.stress[idx_Fourier, 2, 1] =  1j*G*Uz*n/X + G*(Uphi_eta*X_xi/J - Uphi_xi*X_eta/J)
    e.stress[idx_Fourier, 2, 2] =  1j*Uphi*l*n/X + 2*G*(Uz_eta*X_xi/J - Uz_xi*X_eta/J) + Ur*l/X + l*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + l*(Uz_eta*X_xi/J - Uz_xi*X_eta/J)
    
    
    
    
    
    
def cal_Stress_Fourier(solid):
    # Compute Stress within the solid (all elements)
    nmax = solid.nmax
    for e in solid.element_list:
        N = e.grid.Nx
        M = e.grid.Ny
        e.stress = np.zeros((2*nmax+1, 3, 3, N+1, M+1), dtype=complex)
        for n in range(-nmax, nmax+1):
            cal_Stress_Fourier_Element(e, n, solid.Eprop)
            

def cal_Stress_Physical(solid):
    # Add all Fourier terms
    nmax = solid.nmax
    Nphi = solid.Nphi
    # Broadcast angular grid
    phi = solid.phi[:, np.newaxis, np.newaxis]
    
    # Initialize element solution in polar coordinates
    for e in solid.element_list:
        N = e.grid.Nx
        M = e.grid.Ny
        e.stress_polar = np.zeros((3, 3, Nphi+1, N+1, M+1), dtype=complex)
        e.stress_v_polar = np.zeros((Nphi+1, N+1, M+1))

    # Adding up all terms in the Fourier truncated series
    for n in range(-nmax, nmax+1):
        idx_Fourier = map_idx_Fourier(n)
        factor = np.exp(1j*n*phi)
        
        for e in solid.element_list:
            for i in range(3):
                for j in range(3):
                    e.stress_polar[i,j] += factor * e.stress[idx_Fourier, i, j]
    
    
    for e in solid.element_list:
        e.stress_polar = np.real(e.stress_polar)
        
    
    # Von mises
    for e in solid.element_list:
        e.stress_v_polar = (e.stress_polar[0,0] - e.stress_polar[1,1])**2
        e.stress_v_polar += (e.stress_polar[1,1] - e.stress_polar[2,2])**2
        e.stress_v_polar += (e.stress_polar[2,2] - e.stress_polar[0,0])**2
        e.stress_v_polar += 6 * e.stress_polar[0,1]**2
        e.stress_v_polar += 6 * e.stress_polar[1,2]**2
        e.stress_v_polar += 6 * e.stress_polar[0,2]**2
        e.stress_v_polar = np.sqrt(e.stress_v_polar / 2)
        
        



def cal_Stress_Physical_SingleAngle(solid, phi):
    # Add all Fourier terms
    nmax = solid.nmax
    
    
    # Initialize element solution in polar coordinates
    for e in solid.element_list:
        N = e.grid.Nx
        M = e.grid.Ny
        e.stress_polar_SingleAngle = np.zeros((3, 3, N+1, M+1), dtype=complex)
        e.stress_v_polar_SingleAngle = np.zeros((N+1, M+1))

    # Adding up all terms in the Fourier truncated series
    for n in range(-nmax, nmax+1):
        idx_Fourier = map_idx_Fourier(n)
        factor = np.exp(1j*n*phi)
        
        for e in solid.element_list:
            for i in range(3):
                for j in range(3):
                    e.stress_polar_SingleAngle[i,j] += factor * e.stress[idx_Fourier, i, j]
    
    
    for e in solid.element_list:
        e.stress_polar_SingleAngle = np.real(e.stress_polar_SingleAngle)
        
    
    # Von mises
    for e in solid.element_list:
        stress = e.stress_polar_SingleAngle
        stress_v = (stress[0,0] - stress[1,1])**2
        stress_v += (stress[1,1] - stress[2,2])**2
        stress_v += (stress[2,2] - stress[0,0])**2
        stress_v += 6 * stress[0,1]**2
        stress_v += 6 * stress[1,2]**2
        stress_v += 6 * stress[0,2]**2
        e.stress_v_polar_SingleAngle = np.sqrt(stress_v / 2) 
        
        
    # patch stress
    for edge in solid.edgecommon_list:
        # Averaging stress for inner nodes on common edges
        for s in range(1, edge.N): # 1...N-1
            stress = np.zeros((3, 3)) 
            stress_v = 0
            Jsum = 0
            for edge_element_idx in [0,1]:
                e, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = LinearSystem_3DAxisymmetric.map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                Jsum += 1
                stress += e.stress_polar_SingleAngle[:, :, i, j]
                stress_v += e.stress_v_polar_SingleAngle[i, j]
            
            stress /= Jsum
            stress_v /= Jsum
            
            for edge_element_idx in [0,1]:
                e, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = LinearSystem_3DAxisymmetric.map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                e.stress_polar_SingleAngle[:, :, i, j] = stress
                e.stress_v_polar_SingleAngle[i, j] = stress_v
    
    for corner in solid.cornercommon_list:
        
        stress = np.zeros((3, 3)) 
        stress_v = 0
        Jsum = 0
        
        for e, idx_corner in corner.element_info:
            i, j = LinearSystem_3DAxisymmetric.set_LocationCorner(e, idx_corner)
            Jsum += 1
            stress += e.stress_polar_SingleAngle[:, :, i, j]
            stress_v += e.stress_v_polar_SingleAngle[i, j]
            
        stress /= Jsum
        stress_v /= Jsum

        for e, idx_corner in corner.element_info:
            i, j = LinearSystem_3DAxisymmetric.set_LocationCorner(e, idx_corner)
            e.stress_polar_SingleAngle[:, :, i, j] = stress
            e.stress_v_polar_SingleAngle[i, j] = stress_v
            
 





def patch_Stress(solid):
    
    # edgecommon_list, cornercommon_list
    
    # Patching the values
    # Simple average based on how many elements share a node
    # Better averaging should be based on corner angles
    # However, for stress_{z, *} and stress_{r, *}, it is not clear
    # how to use the angles
    # For stress_{phi, *}, weighted averaging wrt corner angles is straightforward
    
    Nphi = solid.Nphi
    
    
    for edge in solid.edgecommon_list:
        # Averaging stress for inner nodes on common edges
        for s in range(1, edge.N): # 1...N-1
            stress = np.zeros((3, 3, Nphi+1)) 
            stress_v = np.zeros(Nphi+1)
            Jsum = 0
            for edge_element_idx in [0,1]:
                e, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = LinearSystem_3DAxisymmetric.map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                Jsum += 1
                stress += e.stress_polar[:, :, :, i, j]
                stress_v += e.stress_v_polar[:, i, j]
            
            stress /= Jsum
            stress_v /= Jsum
            
            for edge_element_idx in [0,1]:
                e, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = LinearSystem_3DAxisymmetric.map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                e.stress_polar[:, :, :, i, j] = stress
                e.stress_v_polar[:, i, j] = stress_v
    
    for corner in solid.cornercommon_list:
        
        stress = np.zeros((3, 3, Nphi+1)) 
        stress_v = np.zeros(Nphi+1)
        Jsum = 0
        
        for e, idx_corner in corner.element_info:
            i, j = LinearSystem_3DAxisymmetric.set_LocationCorner(e, idx_corner)
            Jsum += 1
            stress += e.stress_polar[:, :, :, i, j]
            stress_v += e.stress_v_polar[:, i, j]
            
        stress /= Jsum
        stress_v /= Jsum

        for e, idx_corner in corner.element_info:
            i, j = LinearSystem_3DAxisymmetric.set_LocationCorner(e, idx_corner)
            e.stress_polar[:, :, :, i, j] = stress
            e.stress_v_polar[:, i, j] = stress_v
            
            
            
##############################################
# Locate the element containing a point
###############################################
def get_Element(point, solid):
    # Given the coordinates of a point, find the element that contains it.
    # point: numpy 1D array, or list, or tuple
    p = sg.Point(point[0], point[1])
    
    for e in solid.element_list:
        poly = sg.Polygon(e.corners)
        if p.within(poly):
            return e
        




    










####################################
# Currently not working, deprecated
#####################################

def cal_Element_Angle(e):
    # Compute the corner angles of an element
    d = np.linalg.norm
    
    c1 = e.corners[0]
    c2 = e.corners[1]
    c3 = e.corners[2]
    c4 = e.corners[3]
    
    l1 = d(c2 - c1)
    l2 = d(c3 - c2)
    l3 = d(c3 - c4)
    l4 = d(c4 - c1)
    
    l24 = d(c4 - c2)
    l13 = d(c3 - c1)
    
    e.angles = np.zeros(4)
    
    a, b, c = l1, l4, l24
    e.angles[0] = np.arccos( (a**2 + b**2 - c**2) / (2 * a * b) )
    
    a, b, c = l1, l2, l13
    e.angles[1] = np.arccos( (a**2 + b**2 - c**2) / (2 * a * b) )
    
    a, b, c = l2, l3, l24
    e.angles[2] = np.arccos( (a**2 + b**2 - c**2) / (2 * a * b) )
    
    a, b, c = l3, l4, l13
    e.angles[3] = np.arccos( (a**2 + b**2 - c**2) / (2 * a * b) )
    
    
def cal_Stress_Fourier_Corner(element_list, nodes):
    
    # Compute the stress at corner nodes
    num_node = nodes.shape[0]
    
    stress_node = np.zeros((num_node, 3, 3), dtype=complex)
    stress_v_node = np.zeros(num_node, dtype=complex)
    angle_node = np.zeros(num_node)
    
    for e in element_list:
        cal_Element_Angle(e)
        
        for idx_corner in [1,2,3,4]:
            idx_node = e.label_corners[idx_corner-1]
            angle_node[idx_node] += e.angles[idx_corner-1]
            i, j = set_LocationCorner(e, idx_corner)
            stress_node[idx_node] += e.stress[:,:,i,j] * e.angles[idx_corner-1]
            
            
    stress_node /= (angle_node[:, np.newaxis, np.newaxis] + 1e-8)
    
    stress_v_node = (stress_node[:,0,0] - stress_node[:,1,1])**2
    stress_v_node += (stress_node[:,1,1] - stress_node[:,2,2])**2
    stress_v_node += (stress_node[:,2,2] - stress_node[:,0,0])**2
    stress_v_node += 6 * stress_node[:,0,1]**2
    stress_v_node += 6 * stress_node[:,1,2]**2
    stress_v_node += 6 * stress_node[:,0,2]**2
    stress_v_node = np.sqrt(stress_v_node / 2)
    
    
    return stress_node, stress_v_node

def cal_Stress_Fourier_Interpolation(stress_node, stress_v_node, nodes, element_list):
    x = nodes[:,0]
    y = nodes[:,1]
    
    for e in element_list:
        N = e.grid.Nx
        M = e.grid.Ny
        X = e.nodes_phy_x
        Y = e.nodes_phy_y
        e.stress_interp = np.zeros((3, 3, N+1, M+1), dtype=complex)
        e.stress_interp[0,0] = griddata((x, y), stress_node[:,0,0], (X, Y), method='cubic')
        e.stress_interp[1,1] = griddata((x, y), stress_node[:,1,1], (X, Y), method='cubic')
        e.stress_interp[2,2] = griddata((x, y), stress_node[:,2,2], (X, Y), method='cubic')
        e.stress_interp[0,1] = griddata((x, y), stress_node[:,0,1], (X, Y), method='cubic')
        e.stress_interp[0,2] = griddata((x, y), stress_node[:,0,2], (X, Y), method='cubic')
        e.stress_interp[1,2] = griddata((x, y), stress_node[:,1,2], (X, Y), method='cubic')

        e.stress_interp[1,0] = e.stress_interp[0,1]
        e.stress_interp[2,0] = e.stress_interp[0,2]
        e.stress_interp[2,1] = e.stress_interp[1,2]
        
        e.stress_v_interp = griddata((x, y), stress_v_node, (X, Y), method='cubic')


        
    
    
    
    
    
    

            
           
            
           
            
           
            
           
            
           
            
           
            
           
            
           
            
           
            
           
            
########################################
# Old codes
########################################
def patch_Stress_old(edgecommon_list, cornercommon_list):
    
    # Patching the values
    # Simple average based on how many elements share a node
    # Better averaging should be based on corner angles
    # However, for stress_{z, *} and stress_{r, *}, it is not clear
    # how to use the angles
    # For stress_{phi, *}, weighted averaging wrt corner angles is straightforward
    
    for edge in edgecommon_list:
        # Averaging stress for inner nodes on common edges
        for s in range(1, edge.N): # 1...N-1
            stress = np.zeros((3, 3), dtype=complex) 
            stress_v = 0
            Jsum = 0
            for edge_element_idx in [0,1]:
                e, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                Jsum += 1
                stress += e.stress[:,:,i,j]
                stress_v += e.stress_v[i,j]
            
            stress /= Jsum
            stress_v /= Jsum
            
            for edge_element_idx in [0,1]:
                e, curve_idx, mark_align = edge.element_info[edge_element_idx]
                i, j = map_idx_edgecommon_to_element(s, edge, edge_element_idx)
                e.stress[:,:,i,j] = stress
                e.stress_v[i,j] = stress_v
    
    for corner in cornercommon_list:
        stress = np.zeros((3, 3), dtype=complex) 
        stress_v = 0
        Jsum = 0
        
        for e, idx_corner in corner.element_info:
            i, j = set_LocationCorner(e, idx_corner)
            Jsum += 1
            stress += e.stress[:,:,i,j]
            stress_v += e.stress_v[i,j]
            
        stress /= Jsum
        stress_v /= Jsum

        for e, idx_corner in corner.element_info:
            i, j = set_LocationCorner(e, idx_corner)
            e.stress[:,:,i,j] = stress
            e.stress_v[i,j] = stress_v   


def cal_Stress_Fourier_old(e, n, Eprop):
    # Compute Fourier transformed stress
    G = Eprop.G
    l = Eprop.lamb
    
    N = e.grid.Nx
    M = e.grid.Ny
    
    # Initialize stress, 9 components in total (essentially 6)
    e.stress = np.zeros((3, 3, N+1, M+1), dtype=complex)
    # Metrics
    X = e.nodes_phy_x
    J = e.J
    X_xi = e.X_xi
    X_eta = e.X_eta
    Y_xi = e.Y_xi
    Y_eta = e.Y_eta
    
    D_xi = e.grid.Dx
    D_eta = e.grid.Dy
    
    # Derivatives in the computational domain
    Ur = e.U[0]
    Uphi = e.U[1]
    Uz = e.U[2]
    
    Ur_xi = np.dot(D_xi, Ur)
    Ur_eta = np.dot(Ur, D_eta.T)
    
    Uphi_xi = np.dot(D_xi, Uphi)
    Uphi_eta = np.dot(Uphi, D_eta.T)
    
    Uz_xi = np.dot(D_xi, Uz)
    Uz_eta = np.dot(Uz, D_eta.T)
    
    # Computation of stress
    e.stress[0, 0] =  1j*Uphi*l*n/X + 2*G*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + Ur*l/X + l*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + l*(Uz_eta*X_xi/J - Uz_xi*X_eta/J)
    e.stress[0, 1] =  1j*G*Ur*n/X - G*Uphi/X + G*(-Uphi_eta*Y_xi/J + Uphi_xi*Y_eta/J)
    e.stress[0, 2] =  G*(Ur_eta*X_xi/J - Ur_xi*X_eta/J) + G*(-Uz_eta*Y_xi/J + Uz_xi*Y_eta/J)
    e.stress[1, 0] =  1j*G*Ur*n/X - G*Uphi/X + G*(-Uphi_eta*Y_xi/J + Uphi_xi*Y_eta/J)
    e.stress[1, 1] =  2*1j*G*Uphi*n/X + 1j*Uphi*l*n/X + 2*G*Ur/X + Ur*l/X + l*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + l*(Uz_eta*X_xi/J - Uz_xi*X_eta/J)
    e.stress[1, 2] =  1j*G*Uz*n/X + G*(Uphi_eta*X_xi/J - Uphi_xi*X_eta/J)
    e.stress[2, 0] =  G*(Ur_eta*X_xi/J - Ur_xi*X_eta/J) + G*(-Uz_eta*Y_xi/J + Uz_xi*Y_eta/J)
    e.stress[2, 1] =  1j*G*Uz*n/X + G*(Uphi_eta*X_xi/J - Uphi_xi*X_eta/J)
    e.stress[2, 2] =  1j*Uphi*l*n/X + 2*G*(Uz_eta*X_xi/J - Uz_xi*X_eta/J) + Ur*l/X + l*(-Ur_eta*Y_xi/J + Ur_xi*Y_eta/J) + l*(Uz_eta*X_xi/J - Uz_xi*X_eta/J)
    
    
    e.stress_v = (e.stress[0,0] - e.stress[1,1])**2
    e.stress_v += (e.stress[1,1] - e.stress[2,2])**2
    e.stress_v += (e.stress[2,2] - e.stress[0,0])**2
    e.stress_v += 6 * e.stress[0,1]**2
    e.stress_v += 6 * e.stress[1,2]**2
    e.stress_v += 6 * e.stress[0,2]**2
    e.stress_v = np.sqrt(e.stress_v / 2)
    
    
    
def set_RHS_ExternalSress_ExternalLoad_V1(num_unknown, p_normal, mesh_nodes, tag_nodes, element_list):
    
    # mesh_nodes: node coordinates
    # tag_nodes: Neumann boundary node labels
    
    # This function is only for axisymmetric load
    
    p_Fourier = np.empty(3, dtype=object)
    
    p_Fourier[0] = lambda p, nr, nz: p * nr
    p_Fourier[1] = lambda p, nr, nz: 0
    p_Fourier[2] = lambda p, nr, nz: p * nz
    
    
    for e in element_list:
        
        factor_xi = np.sqrt(e.X_xi**2 + e.Y_xi**2)
        factor_eta = np.sqrt(e.X_eta**2 + e.Y_eta**2)
        X = e.nodes_phy_x
        Y = e.nodes_phy_y
        
        
        # Determine whether an edge is on the Neumann boundary
        for idx_edge, idx_c1, idx_c2 in ((1,1,2),(2,2,3),(3,4,3),(4,1,4)):
            if (e.label_corners[idx_c1-1] in tag_nodes) and (e.label_corners[idx_c2-1] in tag_nodes): 
                imin, imax, jmin, jmax = LinearSystem_3DAxisymmetric.set_LocationBCEquation(e, 'edge', (idx_edge,0,0))
                for idx_eqn in range(num_unknown):
                    for i in range(imin, imax+1):
                        for j in range(jmin, jmax+1):
                            r = X[i,j]
                            z = Y[i,j]
                            

                            if idx_edge == 1:
                                nr, nz = e.norm_vect_lower[i]
                                factor = factor_xi[i,j]
                                w = e.grid.w_x[i]
                            elif idx_edge == 3:
                                nr, nz = e.norm_vect_upper[i] 
                                factor = factor_xi[i,j]
                                w = e.grid.w_x[i]
                            elif idx_edge == 4:
                                nr, nz = e.norm_vect_left[j]
                                factor = factor_eta[i,j]
                                w = e.grid.w_y[j]
                            elif idx_edge == 2:
                                nr, nz = e.norm_vect_right[j]
                                factor = factor_eta[i,j]
                                w = e.grid.w_y[j]
                                
                            p = p_normal(r, z, mesh_nodes)

                            e.RHS[idx_eqn, i, j] += p_Fourier[idx_eqn](p, nr, nz) * factor * w * r







def set_RHS_ExternalStress(num_unknown, e, p_Fourier, n, locations):
    factor_xi = np.sqrt(e.X_xi**2 + e.Y_xi**2)
    factor_eta = np.sqrt(e.X_eta**2 + e.Y_eta**2)
    X = e.nodes_phy_x
    Y = e.nodes_phy_y
    location_type = 'edge'
    for location in locations:
        idx_edge = location[0]
        imin, imax, jmin, jmax = e.set_LocationBCEquation(location_type, location)
        for idx_eqn in range(num_unknown):
            for i in range(imin, imax+1):
                for j in range(jmin, jmax+1):
                    r = X[i,j]
                    z = Y[i,j]

                    if idx_edge == 1:
                        nr, nz = e.norm_vect_lower[i]
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 3:
                        nr, nz = e.norm_vect_upper[i] 
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 4:
                        nr, nz = e.norm_vect_left[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]
                    elif idx_edge == 2:
                        nr, nz = e.norm_vect_right[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]

                    e.RHS[idx_eqn, i, j] += p_Fourier[idx_eqn](r, z, n, nr, nz, e, idx_edge) * factor * w * r


def set_RHS_ExternalStress_V2(num_unknown, e, p_normal, n, locations):
    
    p_Fourier = np.empty(3, dtype=object)
    p_Fourier[0] = lambda p, nr, nz: p * nr
    p_Fourier[1] = lambda p, nr, nz: 0
    p_Fourier[2] = lambda p, nr, nz: p * nz
    
    
    factor_xi = np.sqrt(e.X_xi**2 + e.Y_xi**2)
    factor_eta = np.sqrt(e.X_eta**2 + e.Y_eta**2)
    X = e.nodes_phy_x
    Y = e.nodes_phy_y
    location_type = 'edge'
    for location in locations:
        idx_edge = location[0]
        imin, imax, jmin, jmax = e.set_LocationBCEquation(location_type, location)
        for idx_eqn in range(num_unknown):
            for i in range(imin, imax+1):
                for j in range(jmin, jmax+1):
                    r = X[i,j]
                    z = Y[i,j]
                    
                    p = p_normal(r, z)

                    if idx_edge == 1:
                        nr, nz = e.norm_vect_lower[i]
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 3:
                        nr, nz = e.norm_vect_upper[i] 
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 4:
                        nr, nz = e.norm_vect_left[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]
                    elif idx_edge == 2:
                        nr, nz = e.norm_vect_right[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]

                    e.RHS[idx_eqn, i, j] += p_Fourier[idx_eqn](p, nr, nz) * factor * w * r


def set_RHS_ExternalStress_V3(num_unknown, e, p_Fourier, n, locations):
    factor_xi = np.sqrt(e.X_xi**2 + e.Y_xi**2)
    factor_eta = np.sqrt(e.X_eta**2 + e.Y_eta**2)
    X = e.nodes_phy_x
    Y = e.nodes_phy_y
    location_type = 'edge'
    for location in locations:
        idx_edge = location[0]
        imin, imax, jmin, jmax = e.set_LocationBCEquation(location_type, location)
        for idx_eqn in range(num_unknown):
            for i in range(imin, imax+1):
                for j in range(jmin, jmax+1):
                    r = X[i,j]
                    z = Y[i,j]

                    if idx_edge == 1:
                        nr, nz = e.norm_vect_lower[i]
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 3:
                        nr, nz = e.norm_vect_upper[i] 
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 4:
                        nr, nz = e.norm_vect_left[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]
                    elif idx_edge == 2:
                        nr, nz = e.norm_vect_right[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]

                    e.RHS[idx_eqn, i, j] += p_Fourier[idx_eqn](r, z, n, nr, nz) * factor * w * r
                    
                    
                    
def set_RHS_ExternalStress_V4(num_unknown, e, p_normal, n, locations):
    
    p_Fourier = np.empty(3, dtype=object)
    p_Fourier[0] = lambda p, nr, nz: p * nr
    p_Fourier[1] = lambda p, nr, nz: 0
    p_Fourier[2] = lambda p, nr, nz: p * nz
    
    
    factor_xi = np.sqrt(e.X_xi**2 + e.Y_xi**2)
    factor_eta = np.sqrt(e.X_eta**2 + e.Y_eta**2)
    X = e.nodes_phy_x
    Y = e.nodes_phy_y
    location_type = 'edge'
    for location in locations:
        idx_edge = location[0]
        
        # Changes made here.
        # Previously, set_LocationBCEquation is a element class function
        imin, imax, jmin, jmax = set_LocationBCEquation(e, location_type, location)
        for idx_eqn in range(num_unknown):
            for i in range(imin, imax+1):
                for j in range(jmin, jmax+1):
                    r = X[i,j]
                    z = Y[i,j]
                    
                    p = p_normal(r, z)

                    if idx_edge == 1:
                        nr, nz = e.norm_vect_lower[i]
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 3:
                        nr, nz = e.norm_vect_upper[i] 
                        factor = factor_xi[i,j]
                        w = e.grid.w_x[i]
                    elif idx_edge == 4:
                        nr, nz = e.norm_vect_left[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]
                    elif idx_edge == 2:
                        nr, nz = e.norm_vect_right[j]
                        factor = factor_eta[i,j]
                        w = e.grid.w_y[j]

                    e.RHS[idx_eqn, i, j] += p_Fourier[idx_eqn](p, nr, nz) * factor * w * r
            



        
        
        
        