# This file contains the functions needed for solving the stokes-flow ice-shelf problem
import numpy as np
from dolfinx.log import set_log_level, LogLevel
import sys

def solve(md):
    # solve the stokes-flow problem given the model class

    # *see {repo root}/setup/setup_example.py for an example of how to set these

    # The solution is saved in a directory {repo root}/results/results_name:
    # u = horizontal velocity [m/s]
    # w = vertical velocity [m/s]
    # p = pressure [Pa]
    # h = ice-surface elevation [m]
    # s = basal surface elevation [m]
    
    # set dolfinx log output to desired level
    set_log_level(LogLevel.WARNING)

    # initialize the solvers
    md.set_solvers()
        
    # time-stepping loop
    for i in range(md.timesteps.size):

        if md.rank == 0 and (i+1)%10==0:
            print(f"Time step {i+1} of {md.timesteps.size} completed ({(i+1)/md.timesteps.size*100:.1f}%)", end='\r')
            sys.stdout.flush()

        if i>0:
            md.dt.value = np.abs(md.timesteps[i]-md.timesteps[i-1])
            md.t.value = md.timesteps[i]
    
        # solve for the solution sol = ((u,w),p)
        niter, converged = md.stokes_solver.solve(md.sol)
        assert (converged)

        if converged == False:
            break
        
        if i % md.nt_save == 0:
            md.output_process()
                
            if i % md.nt_check == 0:
                # checkpoint saves: e.g., to not wait until
                # the end of simulation for plotting
                md.output_save()
                    
        # update the domain
        md.update_mesh()
        
        # set solution to zero for initial Newton guess at next time step
        md.sol.x.array[:] = 0
        md.sol.x.scatter_forward()

    return 