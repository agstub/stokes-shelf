#-------------------------------------------------------------------------------------
# This module contains functions for updating the mesh at each timestep according 
# to the solution and forcings 
#-------------------------------------------------------------------------------------

from dolfinx.fem import Constant,dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from ufl import FacetNormal, TestFunction,TrialFunction, ds, dx, grad, inner
import numpy as np

# ------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def slope_solver(md):
    n0 = FacetNormal(md.domain)[0]
    u_ = TrialFunction(md.V0)
    v_ = TestFunction(md.V0)
    a = inner(u_,v_)*dx+inner(u_,v_)*ds
    l = inner(n0, v_)*ds
    slope_problem  = LinearProblem(a,l, bcs=[])
    return slope_problem


def mesh_solver(md):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices
    
    # solve for slope at surfaces (first component of normal vector)
    # and interpolate onto enitire domain
    dofs_base = locate_dofs_topological(md.V0, md.domain.topology.dim-1, md.facets_base)
    dofs_top = locate_dofs_topological(md.V0, md.domain.topology.dim-1, md.facets_top)

    # # define displacement boundary conditions on upper and lower surfaces
    bc_top = dirichletbc(md.dh, dofs_top)
    bc_base = dirichletbc(md.ds, dofs_base)

    bcs = [bc_top,bc_base]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement disp_bdry
    d = TrialFunction(md.V0)
    v = TestFunction(md.V0)
    a = inner(grad(d), grad(v))*dx 
    f = Constant(md.domain, ScalarType(0.0))
    L = f*v*dx

    mesh_problem = LinearProblem(a,L, bcs=bcs)

    return mesh_problem
    

def get_surfaces(X,Z):
    # retrieve numpy arrays of the upper and lower surface elevations
    x = np.sort(X)
    z = Z[np.argsort(X)]
    x_surfs = np.unique(X)
    h = np.zeros(x_surfs.size)      # upper surface elevation
    s = np.zeros(x_surfs.size)      # lower surface elevation

    for i in range(h.size):
        h[i] = np.max(z[np.where(np.isclose(x_surfs[i],x))])
        s[i] = np.min(z[np.where(np.isclose(x_surfs[i],x))])

    return h,s,x_surfs