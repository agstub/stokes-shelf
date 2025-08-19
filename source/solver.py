# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import Constant,Expression, Function
from dolfinx.log import set_log_level, LogLevel
from ufl import SpatialCoordinate, Dx
import sys
import os
import shutil
from pathlib import Path

def solve(md):
    # solve the hydrology problem given setup file

    # *see {repo root}/setup/setup_example.py for an example of how to set these

    # The solution is saved in a directory {repo root}/results/results_name:
    # u = horizontal velocity [units]
    # w = vertical velocity [units]
    # p = pressure [units]
    
    # set dolfinx log output to desired level
    set_log_level(LogLevel.WARNING)
    
    error_code = 0      # code for catching io errors

    nt = np.size(md.timesteps)
    dt_ = np.abs(md.timesteps[1]-md.timesteps[0])
    dt = Constant(md.domain, dt_)
    t = Constant(md.domain,md.timesteps[0])

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
        
    # save nodes so that in post-processing we can create a
    # parallel-to-serial mapping between dof's for plotting
    nodes_x = md.comm.gather(md.x[md.mask_dofs],root=0)
    nodes_z = md.comm.gather(md.z[md.mask_dofs],root=0) 
    md.save_dofmap()
    
    h,xh,s,xs = md.get_surfaces()
    h = md.comm.gather(h,root=0)
    s = md.comm.gather(s,root=0)  
    
    if md.rank == 0:
        # some io setup
        parent_dir = str((Path(__file__).resolve()).parent.parent)
        nodes_x = np.concatenate(nodes_x)
        nodes_z = np.concatenate(nodes_z)
        
        nti = int(nt/md.nt_save)
        t_i = np.linspace(0,md.timesteps.max(),nti)
        nd = md.V0.dofmap.index_map.size_global
        
        # arrays for solution dof's at each timestep
        u_arr = np.zeros((nti,nd))
        w_arr = np.zeros((nti,nd))
        p_arr = np.zeros((nti,nd))
        z_arr = np.zeros((nti,nd))
        lapw_arr = np.zeros((nti,nd))
        
        # some other things we want to save
        # residual_arr = np.zeros((nti,nd))
        
        np.save(md.results_name+'/t.npy',t_i)
        np.save(md.results_name+'/nodes_x.npy',nodes_x)
        np.save(md.results_name+'/nodes_z.npy',nodes_z)
         
        nd_h = np.concatenate(h).size
        nd_s = np.concatenate(s).size
        h_arr = np.zeros((nti,nd_h))
        xh_arr = np.zeros((nti,nd_h))
        s_arr = np.zeros((nti,nd_s))
        xs_arr = np.zeros((nti,nd_s))
        
        divuh_arr = np.zeros((nti,nd_h))
        divus_arr = np.zeros((nti,nd_s))
        
        # copy setup file into results directory to for plotting/post-processing
        # and to keep record of input 
        shutil.copy(parent_dir+'/setups/{}.py'.format(md.setup_name), md.results_name+'/{}.py'.format(md.setup_name))
        j = 0 # index for saving results at nt_save time intervals

    # create dolfinx expressions for interpolating water flux
    u_expr = Expression(md.sol.sub(0).sub(0), md.V0.element.interpolation_points())
    w_expr = Expression(md.sol.sub(0).sub(1), md.V0.element.interpolation_points())
    p_expr = Expression(md.sol.sub(1), md.V0.element.interpolation_points())
    lapw_expr = Expression(Dx(Dx(md.sol.sub(0).sub(1),0),0), md.V0.element.interpolation_points())
    
    lapw = Function(md.V0)
    
    # some other things we want to save
    divu = Function(md.V0)
    divu_expr = Expression(Dx(md.sol.sub(0).sub(0),0), md.V0.element.interpolation_points())
    
    # displacement at upper and lower boundaries
    x = SpatialCoordinate(md.domain)

    # expressions for updating the domain 
    dh_expr = Expression(dt*(md.sol.sub(0).sub(1) - md.sol.sub(0).sub(0)*(-md.slope) + md.smb_surf(x[0],t)),md.V0.element.interpolation_points())
    ds_expr = Expression(dt*(md.sol.sub(0).sub(1) - md.sol.sub(0).sub(0)*md.slope + md.smb_base(x[0],t)),md.V0.element.interpolation_points())

    # define solvers
    stokes_solver = md.stokes_solver(dt,t)
    slope_solver = md.slope_solver()
    mesh_solver = md.mesh_solver()
        
    # time-stepping loop
    for i in range(nt):

        if md.rank == 0 and (i+1)%10==0:
            print(f"Time step {i+1} of {nt} completed ({(i+1)/nt*100:.1f}%)", end='\r')
            sys.stdout.flush()

        if i>0:
            dt_ = np.abs(md.timesteps[i]-md.timesteps[i-1])
            dt.value = dt_
            t.value = md.timesteps[i]
    
        # solve for the solution sol = ((u,w),p)
        niter, converged = stokes_solver.solve(md.sol)
        assert (converged)

        if converged == False:
            break
        
        if i % md.nt_save == 0:
            # interpolate water flux components for saving
            md.u.interpolate(u_expr)
            md.w.interpolate(w_expr)
            md.p.interpolate(p_expr)
            
            lapw.interpolate(lapw_expr)
                 
            h,xh,s,xs = md.get_surfaces()
            
            # mask out the ghost points and gather
            u__ = md.comm.gather(md.u.x.array[md.mask_dofs],root=0)
            w__ = md.comm.gather(md.w.x.array[md.mask_dofs],root=0)
            p__ = md.comm.gather(md.p.x.array[md.mask_dofs],root=0)
            z__ = md.comm.gather(md.z[md.mask_dofs],root=0)
            x__ = md.comm.gather(md.x[md.mask_dofs],root=0)
            
            lapw__ = md.comm.gather(lapw.x.array[md.mask_dofs],root=0)
            
            h__ = md.comm.gather(h,root=0)
            s__ = md.comm.gather(s,root=0)
            xh__ = md.comm.gather(xh,root=0)
            xs__ = md.comm.gather(xs,root=0)
            
            divu.interpolate(divu_expr)
            divuh__ = divu.x.array[md.dofs_top]
            divuh__ = md.comm.gather(divuh__,root=0)
            
            divus__ = divu.x.array[md.dofs_base]
            divus__ = md.comm.gather(divus__,root=0)
  
            if md.rank == 0:
                z__ = np.concatenate(z__)
                x__ = np.concatenate(x__)
                
                h__ = np.concatenate(h__)
                xh__ = np.concatenate(xh__)
                s__ = np.concatenate(s__)
                xs__ = np.concatenate(xs__)
                
                divuh__ = np.concatenate(divuh__)
                divuh__ = divuh__[np.argsort(xh__)]
                
                divus__ = np.concatenate(divus__)
                divus__ = divus__[np.argsort(xs__)]
                
                h__ = h__[np.argsort(xh__)]
                xh__.sort()
                
                s__ = s__[np.argsort(xs__)]
                xs__.sort()

                # save the dof's as numpy arrays
                u_arr[j,:] = np.concatenate(u__)
                w_arr[j,:] = np.concatenate(w__)
                p_arr[j,:] = np.concatenate(p__)
                z_arr[j,:] = z__
                lapw_arr[j,:] = np.concatenate(lapw__)
                
                h_arr[j,:] = h__
                s_arr[j,:] = s__
                xh_arr[j,:] = xh__
                xs_arr[j,:] = xs__
                
                divuh_arr[j,:] = divuh__
                divus_arr[j,:] = divus__
                
                # some other things we want to save
                # residual_arr[j,:] = np.concatenate(residual__)
                
                if i % md.nt_check == 0:
                # checkpoint saves: e.g., to not wait until
                # the end of simulation for plotting
                    np.save(md.results_name+f'/u.npy',u_arr)
                    np.save(md.results_name+f'/w.npy',w_arr)
                    np.save(md.results_name+f'/p.npy',p_arr)
                    np.save(md.results_name+f'/nodes_z.npy',z_arr)
                    np.save(md.results_name+f'/h.npy',h_arr)
                    np.save(md.results_name+f'/s.npy',s_arr)
                    np.save(md.results_name+f'/xh.npy',xh_arr)
                    np.save(md.results_name+f'/xs.npy',xs_arr)
                    
                    # some other things we want to save
                    np.save(md.results_name+f'/divuh.npy',divuh_arr)
                    np.save(md.results_name+f'/divus.npy',divus_arr)
                    np.save(md.results_name+f'/lapw.npy',lapw_arr)
                    
                                    
                j += 1
 
        # update the domain
        n0 = slope_solver.solve()
        md.slope = n0/((1-n0**2)**0.5)
        md.dh.interpolate(dh_expr)
        md.ds.interpolate(ds_expr)
        displacement = mesh_solver.solve()
        md.domain.geometry.x[:,1] += displacement.x.array
        
        # set solution to zero for initial Newton guess at next time step
        md.sol.x.array[:] = 0
        md.sol.x.scatter_forward()
    
    # post-processing: put time-slices into big arrays
    if md.rank == 0:
        np.save(md.results_name+f'/u.npy',u_arr)
        np.save(md.results_name+f'/w.npy',w_arr)
        np.save(md.results_name+f'/p.npy',p_arr)
        np.save(md.results_name+f'/nodes_z.npy',z_arr)
        np.save(md.results_name+f'/h.npy',h_arr)
        np.save(md.results_name+f'/s.npy',s_arr)
        np.save(md.results_name+f'/xh.npy',xh_arr)
        np.save(md.results_name+f'/xs.npy',xs_arr)

        # some other things we want to save
        np.save(md.results_name+f'/divuh.npy',divuh_arr)
        np.save(md.results_name+f'/divus.npy',divus_arr)
        np.save(md.results_name+f'/lapw.npy',lapw_arr)


    return 