#-------------------------------------------------------------------------------------
# This module moves the mesh at each timestep according the solution and forcings 
#-------------------------------------------------------------------------------------

from dolfinx.fem import Constant, Expression, Function,dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from ufl import FacetNormal, SpatialCoordinate, TestFunction,TrialFunction, ds, dx, grad, inner


# ------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------
def update_mesh(md,t,dt):
    # this function computes the surface displacements and moves the mesh
    # by solving Laplace's equation for a smooth displacement function
    # defined for all mesh vertices

    x = SpatialCoordinate(md.domain)

    # solve for slope at surfaces (first component of normal vector)
    # and interpolate onto enitire domain
    n0 = FacetNormal(md.domain)[0]
    u_ = TrialFunction(md.V0)
    v_ = TestFunction(md.V0)
    a = inner(u_,v_)*dx+inner(u_,v_)*ds
    l = inner(n0, v_)*ds
    prob0 = LinearProblem(a,l, bcs=[])
    n0 = prob0.solve()
    
    # displacement at upper and lower boundaries
    disp_h = dt*(md.sol.sub(0).sub(1) - md.sol.sub(0).sub(0)*(-n0) + md.smb_surf(x[0],t))
    disp_s = dt*(md.sol.sub(0).sub(1) - md.sol.sub(0).sub(0)*n0 + md.smb_base(x[0],t))

    disp_h_fcn = Function(md.V0)
    disp_s_fcn = Function(md.V0)
    
    disp_h_fcn.interpolate(Expression(disp_h, md.V0.element.interpolation_points()))
    disp_s_fcn.interpolate(Expression(disp_s, md.V0.element.interpolation_points()))

    dofs_base = locate_dofs_topological(md.V0, md.domain.topology.dim-1, md.facets_base)
    dofs_top = locate_dofs_topological(md.V0, md.domain.topology.dim-1, md.facets_top)

    # # define displacement boundary conditions on upper and lower surfaces
    bc_top = dirichletbc(disp_h_fcn, dofs_top)
    bc_base = dirichletbc(disp_s_fcn, dofs_base)

    bcs = [bc_top,bc_base]

    # # solve Laplace's equation for a smooth displacement field on all vertices,
    # # given the boundary displacement disp_bdry
    d = TrialFunction(md.V0)
    v = TestFunction(md.V0)
    a = inner(grad(d), grad(v))*dx 
    f = Constant(md.domain, ScalarType(0.0))
    L = f*v*dx

    problem = LinearProblem(a,L, bcs=bcs)
    displacement = problem.solve()

    md.domain.geometry.x[:,1] += displacement.x.array

    return 