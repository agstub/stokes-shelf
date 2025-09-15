# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys
sys.path.insert(0, '../source')
import numpy as np
from pathlib import Path
from dolfinx.mesh import create_rectangle
from model import model

def initialize(comm):
    # generate mesh
    H = 400.0                         # Initial height of the domain
    L = 1000.0                        # Length of the domain
    surf = H                          # Initial surface elevation
    base = 0                          # Initial base elevation
    p0 = [-L/2.0,base]                # lower left corner of domain
    p1 = [L/2.0,surf]                 # upper right corner of domain
    res = [int(L/2.5), int(H/2.5)]    # uniform resolution
    domain = create_rectangle(comm,[p0,p1], res)
    
    # need functions of initial surfaces - note: can generalize via interpolation
    # used for marking mesh boundaries
    z_b = lambda x: 0*x + base
    z_s = lambda x: 0*x + surf
    
    # initialize model object
    md = model(comm,domain,z_b,z_s)
    
    # surface mass balance functions
    melt_const = -4/3.154e7      # background surface melt rate [m/s]
    melt_spike =  -8/3.154e7     # melt rate spikes [m/s]
    sigma_spike = 15.0/3         # standard deviation for Gaussian basal melt anomaly [m]
    T = 200                      # spacing between melt spikes [m]
    N = int(0.5*L/T)-1           # ~(1/2)*(number of melt spikes) 
    smb_surf = lambda x,t: melt_const + sum(melt_spike * np.exp(1)**(-((x - n*T)**2) / (2 * sigma_spike**2)) for n in range(-N, N+1))

    x_g = np.linspace(-L/2.0,L/2.0,1000)
    melt_mean = np.mean(smb_surf(x_g,0))

    #----------------------------------------------------------------------------------------
    # Trying some different setups here to see what results in steady-state:
    Idea = 0
    
    if Idea == 0:
        # Idea 0: do nothing--only use the original melting term (can't reach steady-state):
        md.smb_surf = lambda x,t: smb_surf(x,t)
    
    elif Idea == 1:
        # Idea 1: add bulk source term to try to balance surface melting:
        md.div_source = -melt_mean/H 
        md.smb_surf = lambda x,t: smb_surf(x,t)
    
    elif Idea == 2:
        # Idea 2: subtract off mean surface melt to try to balance:
        md.smb_surf = lambda x,t: smb_surf(x,t) - melt_mean 
    
    #----------------------------------------------------------------------------------------

    # ice viscosity (Newtonian)
    md.eta = 1e13 # [Pa s]: default value, can modify if desired

    # define time stepping 
    years =  2
    nt_per_year = 365.0
    t_final = years*3.154e7
    md.timesteps = np.linspace(0,t_final,int(years*nt_per_year))

    # frequency for saving files
    md.nt_save = 10
    md.nt_check = 50*md.nt_save # checkpoint save for real-time 
    
    # results directory name
    md.results_name = f'{(Path(__file__).resolve()).parent.parent}/results/stream_Idea{Idea}/'
    
    return md