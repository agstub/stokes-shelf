# This file contains the functions needed for solving the stokes flow problem
from dolfinx.fem import Constant,dirichletbc,locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from ufl import dx, TestFunctions,split,grad, div, inner, sym

def get_bcs(md):
    # assign Dirichlet boundary conditions on lateral boundaries
    dofs_left = locate_dofs_topological(md.V.sub(0).sub(0), md.domain.topology.dim-1, md.facets_left)
    dofs_right = locate_dofs_topological(md.V.sub(0).sub(0), md.domain.topology.dim-1, md.facets_right)
   
    dofs_base_u = locate_dofs_topological(md.V.sub(0).sub(0), md.domain.topology.dim-1, md.facets_base)
    dofs_base_w = locate_dofs_topological(md.V.sub(0).sub(1), md.domain.topology.dim-1, md.facets_base)
   
    bc_left = dirichletbc(PETSc.ScalarType(0), dofs_left, md.V.sub(0).sub(0))
    bc_right = dirichletbc(PETSc.ScalarType(0), dofs_right, md.V.sub(0).sub(0))
    
    bc_base_u = dirichletbc(PETSc.ScalarType(0), dofs_base_u, md.V.sub(0).sub(0))
    bc_base_w = dirichletbc(PETSc.ScalarType(0), dofs_base_w, md.V.sub(0).sub(1))
    
    bcs = [bc_left,bc_right,bc_base_u,bc_base_w]
    return bcs

def stokes_solver(md):
        # solve the stokes problem for (u,p) = (velocity,pressure)

        # define boundary conditions 
        bcs = get_bcs(md)
        
        # define weak form
        (u,p) = split(md.sol)
        (v,q) = TestFunctions(md.V)
        
        # Body force
        f = Constant(md.domain,PETSc.ScalarType((0,-md.rho_i*md.g)))     
        
        # define weak form residual (F)
        F = 2*md.eta*inner(sym(grad(u)),sym(grad(v)))*dx
        F += (- div(v)*p + q*(div(u)-md.div_source))*dx - inner(f, v)*dx

        # Solve (F==0) for (u,p) with Newton's method
        problem = NonlinearProblem(F, md.sol, bcs=bcs)
        solver = NewtonSolver(md.comm, problem)
        solver.error_on_nonconvergence = False

        return solver