# This file contains the functions needed for solving the subglacial hydrology problem.
import numpy as np
from dolfinx.fem import Constant,Expression
from dolfinx.log import set_log_level, LogLevel
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
    nodes_x = md.comm.gather(md.x[md.mask],root=0)
    nodes_z = md.comm.gather(md.z[md.mask],root=0) #NOTE: save z's at each time?
    # NOTE: md.domain.geometry.dofmap

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
        
        np.save(md.results_name+'/t.npy',t_i)
        np.save(md.results_name+'/nodes_x.npy',nodes_x)
        np.save(md.results_name+'/nodes_z.npy',nodes_z)

        # copy setup file into results directory to for plotting/post-processing
        # and to keep record of input 
        shutil.copy(parent_dir+'/setups/{}.py'.format(md.setup_name), md.results_name+'/{}.py'.format(md.setup_name))
        j = 0 # index for saving results at nt_save time intervals


    # create dolfinx expressions for interpolating water flux
    u_expr = Expression(md.sol.sub(0).sub(0), md.V0.element.interpolation_points())
    w_expr = Expression(md.sol.sub(0).sub(1), md.V0.element.interpolation_points())
    p_expr = Expression(md.sol.sub(1), md.V0.element.interpolation_points())
    
    solver = md.stokes_solver(dt)
        
    # time-stepping loop
    for i in range(nt):

        if md.rank == 0 and (i+1)%1==0:
            print(f"Time step {i+1} of {nt} completed ({(i+1)/nt*100:.1f}%)", end='\r')
            sys.stdout.flush()

        if i>0:
            dt_ = np.abs(md.timesteps[i]-md.timesteps[i-1])
            dt.value = dt_
    
        # solve for the solution sol = ((u,w),p)
        niter, converged = solver.solve(md.sol)
        assert (converged)

        if converged == False:
            break
        
        if i % md.nt_save == 0:
            # interpolate water flux components for saving
            md.u.interpolate(u_expr)
            md.w.interpolate(w_expr)
            md.p.interpolate(p_expr)
            
            # mask out the ghost points and gather
            u__ = md.comm.gather(md.u.x.array[md.mask],root=0)
            w__ = md.comm.gather(md.w.x.array[md.mask],root=0)
            p__ = md.comm.gather(md.p.x.array[md.mask],root=0)

            if md.rank == 0:
                # save the dof's as numpy arrays
                u_arr[j,:] = np.concatenate(u__)
                w_arr[j,:] = np.concatenate(w__)
                p_arr[j,:] = np.concatenate(p__)
                
                if i % md.nt_check == 0:
                # checkpoint saves: e.g., to not wait until
                # the end of simulation for plotting
                    np.save(md.results_name+f'/u.npy',u_arr)
                    np.save(md.results_name+f'/w.npy',w_arr)
                    np.save(md.results_name+f'/p.npy',p_arr)

                j += 1
 
        # update the domain
        md.update_mesh(md.timesteps[i],dt_)     
    
    # post-processing: put time-slices into big arrays
    if md.rank == 0:
        np.save(md.results_name+f'/u.npy',u_arr)
        np.save(md.results_name+f'/w.npy',w_arr)
        np.save(md.results_name+f'/p.npy',p_arr)
    
    return 