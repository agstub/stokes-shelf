#-------------------------------------------------------------------------------------
# This module contains functions for updating the mesh at each timestep according 
# to the solution and mass-balance forcings 
#-------------------------------------------------------------------------------------

from dolfinx.fem import Constant,dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from ufl import FacetNormal, TestFunction,TrialFunction, ds, dx, grad, inner

# ------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def slope_solver(md):
    # solver for computing the slope of the upper and lower surfaces
    n0 = FacetNormal(md.domain)[0]
    u_ = TrialFunction(md.V0)
    v_ = TestFunction(md.V0)
    a = inner(u_,v_)*dx+inner(u_,v_)*ds
    l = inner(n0, v_)*ds
    slope_problem  = LinearProblem(a,l, bcs=[])
    return slope_problem


def mesh_solver(md):
    # this solves Laplace's equation for a smooth displacement function,
    # defined for all mesh vertices, that is used to update the mesh
    
    # define displacement boundary conditions on upper and lower surfaces
    bc_top = dirichletbc(md.dh, md.dofs_top)
    bc_base = dirichletbc(md.ds, md.dofs_base)
    bcs = [bc_top,bc_base]

    # solve Laplace's equation for a smooth displacement field on all vertices,
    # given the boundary displacement disp_bdry
    d = TrialFunction(md.V0)
    v = TestFunction(md.V0)
    a = inner(grad(d), grad(v))*dx 
    f = Constant(md.domain, ScalarType(0.0))
    L = f*v*dx

    mesh_problem = LinearProblem(a,L, bcs=bcs)

    return mesh_problem
    
def update_mesh(md):
    # update mesh according to velocity solution,
    # surface slope, and basal forcing
    n0 = md.slope_solver.solve()
    md.slope = n0/((1-n0**2)**0.5)
    md.dh.interpolate(md.dh_expr)
    md.ds.interpolate(md.ds_expr)
    displacement = md.mesh_solver.solve()
    md.domain.geometry.x[:,1] += displacement.x.array