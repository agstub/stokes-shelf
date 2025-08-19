# this file sets the main model options like the spatial domain (horizontal map-plane), 
# surface elevation, bed elevation, and meltwater inputs (inflow and distributed source)
# see params.py where other model parameters are defined.
import sys, os
sys.path.insert(0, '../source')
import numpy as np
from params import rho_i, rho_w
from pathlib import Path
from dolfinx.mesh import create_rectangle,CellType
import dolfinx.cpp as _cpp
from model_setup import model_setup
from scipy.special import erf

def initialize(comm):
    # generate mesh
    L = 10*1000.0             # Length of the domain
    H = 500.0                 # Initial height of the domain
    surf = (1-rho_i/rho_w)*H  # Initial surface in flotation
    base = -(rho_i/rho_w)*H   # Initial base in flotation
    p0 = [-L/2.0,base]        # lower left corner of domain
    p1 = [L/2.0,surf]         # upper right corner of domain
    res = [400, 10]           # [nx, nz]
    domain = create_rectangle(comm,[p0,p1], res)#, cell_type=CellType.triangle, diagonal = _cpp.mesh.DiagonalType.crossed)    
    
    # need functions of initial surfaces - NOTE: can generalize via interpolation
    # used for marking mesh boundaries
    z_b = lambda x: 0*x + base
    z_s = lambda x: 0*x + surf
    
    # initialize model object
    md = model_setup(comm,domain,z_b,z_s)
    
    # setup name is module name
    md.setup_name = os.path.splitext(os.path.basename(__file__))[0]  
    
    # surface mass balance functions
    m0 =  10 / 3.154e7               # max basal melt(+) or freeze(-) rate (m/yr)
    stdev = 5*H/3                  # standard deviation for Gaussian basal melt anomaly
    md.smb_base = lambda x,t: m0*(np.exp(1)**(-x**2/(stdev**2)))
    md.smb_surf = lambda x,t: m0*np.sqrt(np.pi)*stdev*erf(L/(2*stdev)) / L 

    # define time stepping 
    years = 10
    nt_per_year = 200.0
    t_final = years*3.154e7
    md.timesteps = np.linspace(0,t_final,int(years*nt_per_year))

    # frequency for saving files
    md.nt_save = 10
    md.nt_check = 50*md.nt_save # checkpoint save for real-time 
    
    # results directory name
    md.results_name = f'{(Path(__file__).resolve()).parent.parent}/results/example/'
    
    return md