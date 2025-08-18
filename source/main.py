# This is a wrapper for solving the ice-shelf stokes-flow problem from command line with MPI
# See setup_example.py for an exanple of model setup options like bed and surface geometry,
# meltwater inputs, mesh creation, etc...

import sys
import importlib
from mpi4py import MPI
sys.path.insert(0, '../setups')

# Set up MPI 
comm = MPI.COMM_WORLD

# import model setup module from command line argument
setup = importlib.import_module(sys.argv[1])

# initialize md with MPI context
md = setup.initialize(comm)

# solve the problem, results are saved in a 'results' directory
# visualize the solution with solution-plots.ipynb notebook
md.solve()