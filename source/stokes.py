# This file contains the functions needed for solving the stokes flow problem
from dolfinx.fem import Constant,dirichletbc,locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from ufl import dx, TestFunctions,split,grad, FacetNormal, Measure, SpatialCoordinate, div, inner, sym

def get_bcs(md):
    # assign Dirichlet boundary conditions on lateral boundaries
    dofs_left = locate_dofs_topological(md.V.sub(0).sub(0), md.domain.topology.dim-1, md.facets_left)
    dofs_right = locate_dofs_topological(md.V.sub(0).sub(0), md.domain.topology.dim-1, md.facets_right)
    bc_left = dirichletbc(PETSc.ScalarType(0), dofs_left, md.V.sub(0).sub(0))
    bc_right = dirichletbc(PETSc.ScalarType(0), dofs_right, md.V.sub(0).sub(0))
    bcs = [bc_left,bc_right]
    return bcs

def stokes_solver(md,dt,t):
        # solve the stokes problem for (u,p) = (velocity,pressure)

        # define boundary conditions 
        bcs = get_bcs(md)
        
        # define weak form
        (u,p) = split(md.sol)
        (v,q) = TestFunctions(md.V)
        
        # outward-pointing unit normal to the boundary  
        nu = FacetNormal(md.domain) 
            
        # Neumann condition at ice-water boundary
        x = SpatialCoordinate(md.domain)
        g_base = -md.rho_w*md.g*(x[1]+dt*(md.smb_base(x[0],t)-inner(u,nu)))

        # Body force
        f = Constant(md.domain,PETSc.ScalarType((0,-md.rho_i*md.g)))      

        # Mark bounadries of mesh and define a measure for integration
        facet_tag = md.mark_boundary()
        ds = Measure('ds', domain=md.domain, subdomain_data=facet_tag)
        
        # define weak form residual (F)
        F = 2*md.eta*inner(sym(grad(u)),sym(grad(v)))*dx
        F += (- div(v)*p + q*div(u))*dx - inner(f, v)*dx
        F += g_base*inner(v,nu)*ds(3)
  
        # Solve (F==0) for (u,p) with Newton's method
        problem = NonlinearProblem(F, md.sol, bcs=bcs)
        solver = NewtonSolver(md.comm, problem)

        return solver