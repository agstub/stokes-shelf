# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys, os
sys.path.insert(0, '../source')
import numpy as np
from params import rho_i, rho_w
from pathlib import Path
from dolfinx.mesh import create_rectangle
from model_setup import model_setup
from scipy.special import erf

def initialize(comm):

    # Define mesh (see create_mesh.ipynb notebook for example)
    # generate mesh
    L = 20*1000.0                      # Length of the domain
    H = 500.0                          # Height of the domain
    surf = (1-rho_i/rho_w)*H
    base = -(rho_i/rho_w)*H
    p0 = [-L/2.0,base]
    p1 = [L/2.0,surf]
    domain = create_rectangle(comm,[p0,p1], [2000, 50])
    
    # need functions for initial surfaces - can generalize via interpolation
    z_b = lambda x: 0*x + base
    z_s = lambda x: 0*x + surf
    
    # initialize model object
    md = model_setup(comm,domain,z_b,z_s)
    
    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    
    
    m0 =  5 / 3.154e7               # max basal melt(+) or freeze(-) rate (m/yr)
    stdev = 10*H/3                  # standard deviation for Gaussian basal melt anomaly
    md.smb_base = lambda x,t: m0*(np.exp(1)**(-x**2/(stdev**2)))
    md.smb_surf = lambda x,t: m0*np.sqrt(np.pi)*stdev*erf(L/(2*stdev)) / L 


    # define time stepping 
    years = 1
    nt_per_year = 100.0
    t_final = years*3.154e7
    md.timesteps = np.linspace(0,t_final,int(years*nt_per_year))

    # frequency for saving files
    md.nt_save = 10
    md.nt_check = 50*md.nt_save # checkpoint save for real-time 
    
    # results directory name
    md.results_name = f'{(Path(__file__).resolve()).parent.parent}/results/example/'
    
    return md