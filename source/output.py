import numpy as np
from dolfinx.fem import Expression
import sys, os

def output_setup(md):
    error_code = 0      # code for catching io errors
    
    md.comm.barrier()
    # create arrays for saving solution
    if md.rank == 0:
        try:
            os.makedirs(md.results_name,exist_ok=False)
        except FileExistsError:
            print(f"Error: Directory '{md.results_name}' already exists.\nChoose another name in setup file or delete this directory.")  
            error_code = 1
   
    md.comm.barrier()    
    error_code = md.comm.bcast(error_code, root=0)
    
    if error_code == 1:
        sys.exit(1)
    
    # save mesh nodes for plotting
    nodes_x = md.comm.gather(md.x[md.mask_dofs],root=0)
    nodes_z = md.comm.gather(md.z[md.mask_dofs],root=0) 
    
    # save dofmap to reconstruct mesh for plotting
    md.save_dofmap()
    
    # get surface elevations
    h,xh,s,xs = md.get_surfaces()
    h = md.comm.gather(h,root=0)
    s = md.comm.gather(s,root=0)  
    
    if md.rank == 0:
        nodes_x = np.concatenate(nodes_x)
        nodes_z = np.concatenate(nodes_z)
        
        nti = int(md.timesteps.size/md.nt_save)     # number of time slices saved
        t_i = np.linspace(0,md.timesteps.max(),nti) # time array
        nd = md.V0.dofmap.index_map.size_global     # number of dofs (CG1)
        
        # arrays for solution dof's at each timestep
        md.u_arr = np.zeros((nti,nd))
        md.w_arr = np.zeros((nti,nd))
        md.p_arr = np.zeros((nti,nd))
        md.z_arr = np.zeros((nti,nd))
        
        np.save(md.results_name+'/t.npy',t_i)
        np.save(md.results_name+'/nodes_x.npy',nodes_x)
        np.save(md.results_name+'/nodes_z.npy',nodes_z)
         
        nd_h = np.concatenate(h).size    # number of dofs for surface elevation
        nd_s = np.concatenate(s).size    # number of dofs for base elevation
        md.h_arr = np.zeros((nti,nd_h))
        md.xh_arr = np.zeros((nti,nd_h))
        md.s_arr = np.zeros((nti,nd_s))
        md.xs_arr = np.zeros((nti,nd_s))
        
        md.j = 0 # index for saving results at nt_save time intervals
        
    # create dolfinx expressions for interpolating solutions
    md.u_expr = Expression(md.sol.sub(0).sub(0), md.V0.element.interpolation_points())
    md.w_expr = Expression(md.sol.sub(0).sub(1), md.V0.element.interpolation_points())
    md.p_expr = Expression(md.sol.sub(1), md.V0.element.interpolation_points())

def output_process(md):
    # interpolate solution components for saving
    md.u.interpolate(md.u_expr)
    md.w.interpolate(md.w_expr)
    md.p.interpolate(md.p_expr)
    
    # get surface elevations (h, s) and respective 
    # horizontal coordinates (xh, xs, resp.)
    h,xh,s,xs = md.get_surfaces()
    
    # mask out the ghost dofs and gather to root
    u__ = md.comm.gather(md.u.x.array[md.mask_dofs],root=0)
    w__ = md.comm.gather(md.w.x.array[md.mask_dofs],root=0)
    p__ = md.comm.gather(md.p.x.array[md.mask_dofs],root=0)
    z__ = md.comm.gather(md.z[md.mask_dofs],root=0)
    x__ = md.comm.gather(md.x[md.mask_dofs],root=0)
                
    h__ = md.comm.gather(h,root=0)
    s__ = md.comm.gather(s,root=0)
    xh__ = md.comm.gather(xh,root=0)
    xs__ = md.comm.gather(xs,root=0)

    if md.rank == 0:
        z__ = np.concatenate(z__)
        x__ = np.concatenate(x__)
        
        h__ = np.concatenate(h__)
        xh__ = np.concatenate(xh__)
        s__ = np.concatenate(s__)
        xs__ = np.concatenate(xs__)
        
        # sort upper/lower surface dofs for easy plotting
        h__ = h__[np.argsort(xh__)]
        xh__.sort()
        
        s__ = s__[np.argsort(xs__)]
        xs__.sort()

        # save the dof's as numpy arrays
        md.u_arr[md.j,:] = np.concatenate(u__)
        md.w_arr[md.j,:] = np.concatenate(w__)
        md.p_arr[md.j,:] = np.concatenate(p__)
        md.z_arr[md.j,:] = z__
        
        md.h_arr[md.j,:] = h__
        md.s_arr[md.j,:] = s__
        md.xh_arr[md.j,:] = xh__
        md.xs_arr[md.j,:] = xs__
        
        # update time save index
        md.j += 1

def output_save(md):
    # save all results arrays in results_name directory
    if md.rank == 0:
        np.save(md.results_name+f'/u.npy',md.u_arr)
        np.save(md.results_name+f'/w.npy',md.w_arr)
        np.save(md.results_name+f'/p.npy',md.p_arr)
        np.save(md.results_name+f'/nodes_z.npy',md.z_arr)
        np.save(md.results_name+f'/h.npy',md.h_arr)
        np.save(md.results_name+f'/s.npy',md.s_arr)
        np.save(md.results_name+f'/xh.npy',md.xh_arr)
        np.save(md.results_name+f'/xs.npy',md.xs_arr)