import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize # For scatter plot color adjustment

# Reloading the module
import importlib

import Utility
importlib.reload(Utility)

###########################################
# Visualization
###########################################    

    
def visualizing_elements(elements, legend='on'):
    # Showing all elements' grids
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for k, element in enumerate(elements):
        ax.scatter(element.nodes_phy_x, element.nodes_phy_y, label=f"Element {k}")
    if legend == 'on':
        ax.legend()
    ax.set_aspect('equal')
    # Save figure
    filename = "./figs/elements_.jpg"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    
def visualizing_C(C):
    # Showing nonzero entries of C
    idx_nonzero_x, idx_nonzero_y = np.nonzero(C)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(idx_nonzero_x, idx_nonzero_y, s=0.25, color='black')
    
    
###############################################
# Full Fourier Solution Visualization
###############################################
def vis_Displacement_Scatter(solid):
    
    element_list = solid.element_list
    num_unknown = solid.num_unknown
    
    
    
    fig, ax = plt.subplots(nrows=1, ncols=num_unknown, subplot_kw={"projection": "3d"})
    
    if num_unknown == 1:
        ax = np.expand_dims(ax, axis=0)   
        
    cmap = plt.get_cmap('viridis')
    data_mins = []
    data_maxs = []
   
    for e in element_list:
        U = np.real(e.U_polar)
        data_mins.append(np.min(U))
        data_maxs.append(np.max(U))

    vmin = np.min(data_mins)
    vmax = np.max(data_maxs)
    norm = Normalize(vmin=vmin, vmax=vmax)
        
        
    for idx_unknown, name_component in zip(range(num_unknown), (r'$u_r$', r'$u_{\phi}$', r'$u_z$')):  
        
        for e in element_list:
            for idx_phi in range(solid.Nphi+1):
                phi = solid.phi[idx_phi]
                U = np.real(e.U_polar[idx_unknown, idx_phi])
                r = e.nodes_phy_x
                z = e.nodes_phy_y
                x = r * np.cos(phi)
                y = r * np.sin(phi)
                ax[idx_unknown].scatter(x, y, z, c=U, cmap=cmap, norm=norm)
        
        
        
        # x,y labels
        # Set x, y, z labels
        ax[idx_unknown].set_xlabel(r'$x$', labelpad=-10)
        ax[idx_unknown].set_ylabel(r'$y$', labelpad=-10)
        #ax[idx_unknown].set_zlabel(name_component, labelpad=-10)
        #ax[idx_unknown].zaxis.set_rotate_label(False)  # Disable automatic rotation
        #ax[idx_unknown].zaxis.label.set_rotation(0)
        size = 6 
        ax[idx_unknown].tick_params(axis='x', labelsize=size)
        ax[idx_unknown].tick_params(axis='y', labelsize=size)
        ax[idx_unknown].tick_params(axis='z', labelsize=size)
        
        
        # Adjust tick label distance (reduce pad to move them closer)
        ax[idx_unknown].xaxis.set_tick_params(pad=-5)  # Decrease pad value to bring tick labels closer
        ax[idx_unknown].yaxis.set_tick_params(pad=-5)  
        ax[idx_unknown].zaxis.set_tick_params(pad=-5) 
        
        # Titles
        title_size = 8
        ax[idx_unknown].set_title(name_component, fontsize=title_size)
       
    # Hozirontal spacing
    fig.subplots_adjust(hspace=1)  # Adjust the vertical spacing

    # Save figure
    filename = f"./figs/Displacement_Scatter_.jpg"
    fig.savefig(filename, dpi=600, bbox_inches="tight")
    
    
def vis_Stress_Scatter(solid, component):
    
    # component = (i,j) or 'v'
    
    if component == 'v':
        name_component = r'$\sigma_v$'
        filename_component = 'sigma_v'
        
    elif component == (0,0):
        name_component = r'$\sigma_r$'
        filename_component = 'sigma_r'
        
    elif component == (1,1):
        name_component = r'$\sigma_{\phi}$'
        filename_component = 'sigma_phi'
        
    elif component == (2,2):
        name_component = r'$\sigma_z$'
        filename_component = 'sigma_z'
        
    elif component == (0,1) or component == (1,0):
        name_component = r'$\tau_{r\phi}$'
        filename_component = 'tau_r_phi'
        
    elif component == (0,2) or component == (2,0):
        name_component = r'$\tau_{rz}$'
        filename_component = 'tau_r_z'
        
    elif component == (1,2) or component == (2,1):
        name_component = r'$\tau_{\phi z}$'
        filename_component = 'tau_phi_z'
    
    
    element_list = solid.element_list
    
    
    fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d"})
       
    cmap = plt.get_cmap('viridis')
    data_mins = []
    data_maxs = []
   
    for e in element_list:
        
        if component=='v':
            stress = np.real(e.stress_v_polar)
        
        else:
            i,j = component
            stress = np.real(e.stress_polar[i,j])
            
        
        data_mins.append(np.min(stress))
        data_maxs.append(np.max(stress))

    vmin = np.min(data_mins)
    vmax = np.max(data_maxs)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    
    
    for e in element_list:
        if component=='v':
            stress = np.real(e.stress_v_polar)
        
        else:
            i,j = component
            stress = np.real(e.stress_polar[i,j])
        
        for idx_phi in range(solid.Nphi+1):
            # Broadcast angular grid
            phi = solid.phi[:, np.newaxis, np.newaxis]
            r = e.nodes_phy_x
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            z = e.nodes_phy_y
            z = np.tile(z, (solid.Nphi+1, 1, 1))
            
            ax.scatter(x, y, z, c=stress, cmap=cmap, norm=norm)
        
        
        
    # x,y labels
    # Set x, y, z labels
    ax.set_xlabel(r'$x$', labelpad=-10)
    ax.set_ylabel(r'$y$', labelpad=-10)
    #ax[idx_unknown].set_zlabel(name_component, labelpad=-10)
    #ax[idx_unknown].zaxis.set_rotate_label(False)  # Disable automatic rotation
    #ax[idx_unknown].zaxis.label.set_rotation(0)
    size = 6 
    ax.tick_params(axis='x', labelsize=size)
    ax.tick_params(axis='y', labelsize=size)
    ax.tick_params(axis='z', labelsize=size)
        
        
    # Adjust tick label distance (reduce pad to move them closer)
    ax.xaxis.set_tick_params(pad=-5)  # Decrease pad value to bring tick labels closer
    ax.yaxis.set_tick_params(pad=-5)  
    ax.zaxis.set_tick_params(pad=-5) 
        
    # Titles
    title_size = 8
    ax.set_title(name_component, fontsize=title_size)
       
    # Hozirontal spacing
    fig.subplots_adjust(hspace=1)  # Adjust the vertical spacing

    # Save figure
    filename = f"./figs/Stress_Scatter_{filename_component}.jpg"
    fig.savefig(filename, dpi=600, bbox_inches="tight")




###########################################
# Single cross-section visualization (single angle, RZ plane)
###########################################
def vis_Displacement_RZ_SurfacePlot(solid, phi):
    # phi: angle of the rz plane
    
    element_list = solid.element_list
    num_unknown = solid.num_unknown
    
    Utility.cal_U_Physical_SingleAngle(solid, phi)
    
    for idx_unknown, name_component in zip(range(num_unknown), (r'$u_r$', r'$u_{\phi}$', r'$u_z$')):
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d"})
        
        for e in element_list:
            U = np.real(e.U_polar_SingleAngle[idx_unknown])
            ax.plot_surface(e.nodes_phy_x, e.nodes_phy_y, U, cmap=cm.coolwarm)
            
         
        # x,y labels
        # Set x, y, z labels
        ax.set_xlabel(r'$r$', labelpad=-10)
        ax.set_ylabel(r'$z$', labelpad=-10)
        #ax[idx_unknown].set_zlabel(name_component, labelpad=-10)
        #ax[idx_unknown].zaxis.set_rotate_label(False)  # Disable automatic rotation
        #ax[idx_unknown].zaxis.label.set_rotation(0)
        size = 6 
        ax.tick_params(axis='x', labelsize=size)
        ax.tick_params(axis='y', labelsize=size)
        ax.tick_params(axis='z', labelsize=size)
        
        
        # Adjust tick label distance (reduce pad to move them closer)
        ax.xaxis.set_tick_params(pad=-5)  # Decrease pad value to bring tick labels closer
        ax.yaxis.set_tick_params(pad=-5)  
        ax.zaxis.set_tick_params(pad=-5) 
        
        '''
        
        
        pad = -3
        ax[idx_unknown].tick_params(axis='x', pad=pad)
        ax[idx_unknown].tick_params(axis='y', pad=pad)
        ax[idx_unknown].tick_params(axis='z', pad=pad)
        '''
        

        # Titles
        title_size = 10
        
        ax.set_title(name_component + '(' + r'$\phi=$' + f'{phi/np.pi:.3f}' + r'$\pi$'+')', 
                fontsize=title_size)
        
        
        # Save figure
        filename = f"./figs/Displacement_RZ_SurfacePlot_U{idx_unknown}_phi={phi/np.pi:.3f}pi.jpg"
        fig.savefig(filename, dpi=600, bbox_inches="tight")
        
        
def vis_Displacement_RZ_ContourPlot(solid, phi, num_level):
    # phi: angle of the rz plane
    # num_level: number of contour level
    
    element_list = solid.element_list
    num_unknown = solid.num_unknown
    
    Utility.cal_U_Physical_SingleAngle(solid, phi)
    
    for idx_unknown, name_component in zip(range(num_unknown), (r'$u_r$', r'$u_{\phi}$', r'$u_z$')):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        # Creating levels
        data_mins = []
        data_maxs = []
        for e in element_list:
            U = np.real(e.U_polar_SingleAngle[idx_unknown])
            data_mins.append(np.min(U))
            data_maxs.append(np.max(U))
            
        data_min = np.min(data_mins)
        data_max = np.max(data_maxs)
        if data_max - data_min<1e-8:
            data_min -= 1e-8
            data_max += 1e-8
            
        levels = np.linspace(data_min, data_max, num_level+1)
        
        for e in element_list:
            U = np.real(e.U_polar_SingleAngle[idx_unknown])
            ax.contourf(e.nodes_phy_x, e.nodes_phy_y, U, 
                levels=levels, vmin=data_min, vmax=data_max)
            
         
        # x,y labels
        # Set x, y, z labels
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(r'$z$')
        
        size = 6 
        ax.tick_params(axis='x', labelsize=size)
        ax.tick_params(axis='y', labelsize=size)
              
        

        # Titles
        title_size = 10
        
        ax.set_title(name_component + '(' + r'$\phi=$' + f'{phi/np.pi:.3f}' + r'$\pi$'+')', 
                fontsize=title_size)
        
        
        # Save figure
        filename = f"./figs/Displacement_RZ_ContourPlot_U{idx_unknown}_phi={phi/np.pi:.3f}pi.jpg"
        fig.savefig(filename, dpi=600, bbox_inches="tight")
        
        
        
 
    
def vis_Stress_RZ_SurfacePlot(solid, phi):
    # phi: angle of the rz plane
    
    element_list = solid.element_list
    
    
    Utility.cal_Stress_Physical_SingleAngle(solid, phi)
    
    for component in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2), 'v']:
        
        if component == 'v':
            name_component = r'$\sigma_v$'
            filename_component = 'sigma_v'
            
        elif component == (0,0):
            name_component = r'$\sigma_r$'
            filename_component = 'sigma_r'
            
        elif component == (1,1):
            name_component = r'$\sigma_{\phi}$'
            filename_component = 'sigma_phi'
            
        elif component == (2,2):
            name_component = r'$\sigma_z$'
            filename_component = 'sigma_z'
            
        elif component == (0,1) or component == (1,0):
            name_component = r'$\tau_{r\phi}$'
            filename_component = 'tau_r_phi'
            
        elif component == (0,2) or component == (2,0):
            name_component = r'$\tau_{rz}$'
            filename_component = 'tau_r_z'
            
        elif component == (1,2) or component == (2,1):
            name_component = r'$\tau_{\phi z}$'
            filename_component = 'tau_phi_z'
    
    
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d"})
    
        for e in element_list:
            if component=='v':
                stress = e.stress_v_polar_SingleAngle
            else:
                i,j = component
                stress = e.stress_polar_SingleAngle[i,j]
            
            ax.plot_surface(e.nodes_phy_x, e.nodes_phy_y, stress, cmap=cm.coolwarm)
            
        
        # x,y labels
        # Set x, y, z labels
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(r'$z$')
    
        size = 6 
        ax.tick_params(axis='x', labelsize=size)
        ax.tick_params(axis='y', labelsize=size)
    
        # Titles
        title_size = 10
        ax.set_title(name_component + '(' + r'$\phi=$' + f'{phi/np.pi:.3f}' + r'$\pi$'+')', 
            fontsize=title_size)
    
        # Save figure
        filename = f"./figs/Stress_RZ_SurfacePlot_{filename_component}_phi={phi/np.pi:.3f}pi.jpg"
        fig.savefig(filename, dpi=600, bbox_inches="tight")
    
    
    
    
    
def vis_Stress_RZ_ContourPlot(solid, phi, num_level):
    # phi: angle of the rz plane
    
    element_list = solid.element_list
    
    
    Utility.cal_Stress_Physical_SingleAngle(solid, phi)
    
    for component in [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2), 'v']:
        
        if component == 'v':
            name_component = r'$\sigma_v$'
            filename_component = 'sigma_v'
            
        elif component == (0,0):
            name_component = r'$\sigma_r$'
            filename_component = 'sigma_r'
            
        elif component == (1,1):
            name_component = r'$\sigma_{\phi}$'
            filename_component = 'sigma_phi'
            
        elif component == (2,2):
            name_component = r'$\sigma_z$'
            filename_component = 'sigma_z'
            
        elif component == (0,1) or component == (1,0):
            name_component = r'$\tau_{r\phi}$'
            filename_component = 'tau_r_phi'
            
        elif component == (0,2) or component == (2,0):
            name_component = r'$\tau_{rz}$'
            filename_component = 'tau_r_z'
            
        elif component == (1,2) or component == (2,1):
            name_component = r'$\tau_{\phi z}$'
            filename_component = 'tau_phi_z'
            
            
            
        data_mins = []
        data_maxs = []    
        for e in element_list:
            if component=='v':
                stress = e.stress_v_polar_SingleAngle
            else:
                i,j = component
                stress = e.stress_polar_SingleAngle[i,j]    
                
            data_mins.append(np.min(stress))
            data_maxs.append(np.max(stress))
            
        data_min = np.min(data_mins)
        data_max = np.max(data_maxs)
        if data_max - data_min<1e-8:
            data_min -= 1e-8
            data_max += 1e-8
            
        levels = np.linspace(data_min, data_max, num_level+1)
    
    
        fig, ax = plt.subplots(nrows=1, ncols=1)
    
        for e in element_list:
            if component=='v':
                stress = e.stress_v_polar_SingleAngle
            else:
                i,j = component
                stress = e.stress_polar_SingleAngle[i,j]
            
            ax.contourf(e.nodes_phy_x, e.nodes_phy_y, stress, 
                levels=levels, vmin=data_min, vmax=data_max)
            
        
        # x,y labels
        # Set x, y, z labels
        ax.set_xlabel(r'$r$')
        ax.set_ylabel(r'$z$')
    
        size = 6 
        ax.tick_params(axis='x', labelsize=size)
        ax.tick_params(axis='y', labelsize=size)
    
        # Titles
        title_size = 10
        ax.set_title(name_component + '(' + r'$\phi=$' + f'{phi/np.pi:.3f}' + r'$\pi$'+')', 
            fontsize=title_size)
    
        # Save figure
        filename = f"./figs/Stress_RZ_ContourPlot_{filename_component}_phi={phi/np.pi:.3f}pi.jpg"
        fig.savefig(filename, dpi=600, bbox_inches="tight")
    
    
    

    

        
        


############################################
# Old codes
############################################
def vis_Displacement_RZ_SurfacePlot_old(phi, solid):
    # phi: angle of the rz plane
    # There are two ways of inputing phi
        # (1) phi in [-pi, pi]
        # (2) phi in [0, 2*pi]
    # In (1), we convert it to [0, 2*pi]
    
    element_list = solid.element_list
    num_unknown = solid.num_unknown
    
    if phi<0:
        phi = 2*np.pi + phi
        
    idx_phi = int( np.floor( phi * solid.Nphi / (2*np.pi) ) )
    phi = idx_phi * 2*np.pi/solid.Nphi
    
    fig, ax = plt.subplots(nrows=1, ncols=num_unknown, subplot_kw={"projection": "3d"})
    
    if num_unknown == 1:
        ax = np.expand_dims(ax, axis=0)   
    
    
    
    for idx_unknown, name_component in zip(range(num_unknown), (r'$u_r$', r'$u_{\phi}$', r'$u_z$')):  
    
        for e in element_list:
            U = np.real(e.U_polar[idx_unknown, idx_phi])
            ax[idx_unknown].plot_surface(e.nodes_phy_x, e.nodes_phy_y, U, cmap=cm.coolwarm)
            
         
        # x,y labels
        # Set x, y, z labels
        ax[idx_unknown].set_xlabel(r'$r$', labelpad=-10)
        ax[idx_unknown].set_ylabel(r'$z$', labelpad=-10)
        #ax[idx_unknown].set_zlabel(name_component, labelpad=-10)
        #ax[idx_unknown].zaxis.set_rotate_label(False)  # Disable automatic rotation
        #ax[idx_unknown].zaxis.label.set_rotation(0)
        size = 6 
        ax[idx_unknown].tick_params(axis='x', labelsize=size)
        ax[idx_unknown].tick_params(axis='y', labelsize=size)
        ax[idx_unknown].tick_params(axis='z', labelsize=size)
        
        
        # Adjust tick label distance (reduce pad to move them closer)
        ax[idx_unknown].xaxis.set_tick_params(pad=-5)  # Decrease pad value to bring tick labels closer
        ax[idx_unknown].yaxis.set_tick_params(pad=-5)  
        ax[idx_unknown].zaxis.set_tick_params(pad=-5) 
        
        '''
        
        
        pad = -3
        ax[idx_unknown].tick_params(axis='x', pad=pad)
        ax[idx_unknown].tick_params(axis='y', pad=pad)
        ax[idx_unknown].tick_params(axis='z', pad=pad)
        '''
        
    
        

        # Titles
        title_size = 10
        
        ax[idx_unknown].set_title(name_component + '(' + r'$\phi=$' + f'{phi/np.pi}' + r'$\pi$'+')', 
                fontsize=title_size)
        
        
        
        
    # Hozirontal spacing
    fig.subplots_adjust(hspace=1)  # Adjust the vertical spacing
    

    # Save figure
    filename = f"./figs/Displacement_RZ_SurfacePlot_phi={phi/np.pi}pi.jpg"
    fig.savefig(filename, dpi=600, bbox_inches="tight")