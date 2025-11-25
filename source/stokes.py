# This file contains the functions needed for solving the stokes flow problem
from dolfinx.fem import Constant,dirichletbc,locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc
from ufl import dx, TestFunctions,split,grad, div, inner, sym, SpatialCoordinate

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
        
        x = SpatialCoordinate(md.domain)
        
        # Body force
        f = Constant(md.domain,PETSc.ScalarType((0,-md.rho_i*md.g)))     
        
        B = (2**((md.n-1.0)/(2*md.n)))*(md.A**(-1/md.n)) # "2*Viscosity" constant in weak form (Pa s^{1/n})
        if md.n>1:
            eps_v = (2*md.eta/B)**(2.0/(1/md.n-1))           # Flow law regularization parameter 
        else:                                                # (bounds viscosity above by md.eta at zero strain rate)
            eps_v = 1e-17
            
        # Glen's law: 
        eta = 0.5*B*((inner(sym(grad(u)),sym(grad(u)))+eps_v)**((1/md.n-1)/2.0)) 
        
        if md.n == 1:
            eta = md.eta
        
        # define weak form residual (F)
        F = 2*eta*inner(sym(grad(u)),sym(grad(v)))*dx
        F += (- div(v)*p + q*(div(u)-md.div_source(x[0],x[1],md.t)))*dx - inner(f, v)*dx
    
        # do we need to set this
        petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_monitor": None,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_stol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        }  

        # Solve (F==0) for (u,p) with Newton's method 
        solver = NonlinearProblem(F, md.sol, bcs=bcs,petsc_options=petsc_options,petsc_options_prefix="stokes")
        solver.error_on_nonconvergence = False

        return solver