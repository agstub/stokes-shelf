from dolfinx.fem import Constant, Function, Expression, functionspace, locate_dofs_topological
from ufl import SpatialCoordinate
from basix.ufl import element, mixed_element
import numpy as np
from dolfinx.mesh import locate_entities, meshtags
from solver import solve
from stokes import stokes_solver
from dolfinx.mesh import locate_entities_boundary
from mesh_routine import mesh_solver, slope_solver, update_mesh
import params
from output import output_setup, output_process, output_save

# model input class file
class model:
    def __init__(self, comm, domain,z_b,z_s):
        # mpi 
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Domain, mesh, function spaces
        self.domain = domain
        self.x = self.domain.geometry.x[:,0]
        self.z = self.domain.geometry.x[:,1]        
        
        p_el = element('P',self.domain.basix_cell(),1)                                   # pressure element  
        u_el = element('P',self.domain.basix_cell(),2,shape=(self.domain.geometry.dim,)) # velocity element
        self.V = functionspace(self.domain,mixed_element([u_el,p_el]))                   # function space for (u,p)
        self.V0 = functionspace(self.domain, ("CG", 1))                                  # function space for saving results
        
        # masks to filter out ghost dofs and ghost cells
        self.mask_dofs = self.ghost_mask(self.V0.dofmap.index_map) 
        self.mask_cells = self.ghost_mask(self.domain.topology.index_map(self.domain.topology.dim))
        
        # define solution function sol = ((u,w),p) 
        self.sol = Function(self.V)
        
        # functions for saving the solution    
        self.u = Function(self.V0)
        self.w = Function(self.V0)
        self.p = Function(self.V0)
                
        # functions for updating mesh
        self.slope = Function(self.V0)
        self.dh = Function(self.V0)
        self.ds = Function(self.V0)

        # Physical input functions
        self.z_b = z_b                   # initial bed elevation [m]
        self.z_s = z_s                   # initial surface elevation [m]
        self.smb_surf = lambda x,t: 0*x  # surface mass balance
        self.smb_base = lambda x,t: 0*x  # basal mass balance

        # Output names
        self.results_name = None
        self.setup_name = None
        
        # time stepping & frequency for saving files
        self.timesteps = None  # timesteps array
        self.nt_save = None    # time-save frequency (size of output in time dimension)
        self.nt_check = None   # checkpoint save frequency
        self.t = None          # time clock for solver
        self.dt = None         # timestep size tracker for solver
        
        # boundary facets
        self.facets_left = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.LeftBoundary(x))        
        self.facets_right = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.RightBoundary(x))
        self.facets_top = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.TopBoundary(x))
        self.facets_base = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.BaseBoundary(x))
        
        # dofs for the upper and lower surfaces
        self.dofs_top = locate_dofs_topological(self.V0, self.domain.topology.dim-1, self.facets_top)
        self.dofs_base = locate_dofs_topological(self.V0, self.domain.topology.dim-1, self.facets_base)
        
        # physical constants; default values in params module
        self.A = params.A         # ice rigidity
        self.n = params.n         # flow-law exponent
        self.g = params.g         # gravitional acceleration
        self.rho_i = params.rho_i # ice density
        self.rho_w = params.rho_w # water density
        self.delta = params.delta # flotation factor
        self.eta = params.eta     # ice viscosity
        
        # expressions for interpolating onto functions 
        self.u_expr = None
        self.w_expr = None
        self.p_expr = None
        self.dh_expr = None
        self.ds_expr = None
        
        # arrays for saving solutions
        self.u_arr = None
        self.w_arr = None
        self.p_arr = None
        self.z_arr = None
        self.h_arr = None
        self.s_arr = None
        self.xh_arr = None
        self.xs_arr = None
    
    def save_dofmap(self):
        # Extract the local geometry dofmap for owned cells
        local_geom_dofmap = self.domain.geometry.dofmap

        # Access the index map for geometry dofs
        imap = self.domain.geometry.index_map()

        # Build local-to-global mapping for coordinate dofs
        local_to_global = np.empty(imap.size_local + imap.num_ghosts, dtype=np.int32)
        local_to_global[:imap.size_local] = np.arange(*imap.local_range)
        local_to_global[imap.size_local:] = imap.ghosts

        # Map local dofs in the geometry dofmap to global indices
        global_geom_dofmap = local_to_global[local_geom_dofmap]
        all_dofmaps = self.comm.gather(global_geom_dofmap[self.mask_cells], root=0)
        if self.rank == 0:
            full_global_dofmap = np.concatenate(all_dofmaps)            
            np.save(self.results_name+'/dofmap.npy',full_global_dofmap)
    
    def get_surfaces(self):
        h = self.domain.geometry.x[:,1][self.dofs_top]
        s = self.domain.geometry.x[:,1][self.dofs_base]
        xh = self.domain.geometry.x[:,0][self.dofs_top]         
        xs = self.domain.geometry.x[:,0][self.dofs_base] 
        return h,xh,s,xs

    def ghost_mask(self, index_map):
        # create mask for ghosts (e.g., dofs or cells) given index map
        ghosts = index_map.ghosts
        global_to_local = index_map.global_to_local
        ghosts_local = global_to_local(ghosts)
        size_local = index_map.size_local
        num_ghosts = index_map.num_ghosts
        mask = np.ones(size_local+num_ghosts,dtype=bool)
        mask[ghosts_local] = False
        return mask
    
    def LeftBoundary(self,x):
        # Left boundary (inflow/outflow)
        return np.isclose(x[0],self.x.min())

    def RightBoundary(self,x):
        # Right boundary (inflow/outflow)
        return np.isclose(x[0],self.x.max())
    
    def BaseBoundary(self,x):
        # Base boundary
        return np.isclose(x[1],self.z_b(x[0]))
    
    def TopBoundary(self,x):
        # Top boundary
        return np.isclose(x[1],self.z_s(x[0]))
    
    def mark_boundary(self):
        # Assign markers to each boundary segment (except the upper surface).
        # "This is used at each time step to update the markers"
        # NOTE: we shouldn't need to update the markers every timesetep unless
        #       grounding line is migrating...
        # Boundary marker numbering convention:
        # 1 - Left boundary
        # 2 - Right boundary
        # 3 - Base (ice-water) boundary
        # 4 - Top boundary
        boundaries = [(1, lambda x: self.LeftBoundary(x)),
                      (2, lambda x: self.RightBoundary(x)),
                      (3, lambda x: self.BaseBoundary(x)),
                      (4, lambda x: self.TopBoundary(x))]
        facet_indices, facet_markers = [], []
        fdim = self.domain.topology.dim - 1
        for (marker, locator) in boundaries:
            facets = locate_entities(self.domain, fdim, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        facet_tag = meshtags(self.domain, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        return facet_tag
    
    def solve(self):
        # call the main solve function
        solve(self)   
    
    def set_stokes_solver(self,dt,t):
        # define stokes solver
        self.stokes_solver = stokes_solver(self,dt,t)
    
    def set_slope_solver(self):
        # define solver that computes surface slopes
        self.slope_solver = slope_solver(self)

    def set_mesh_solver(self):
        # define solver that computes mesh displacement 
        # functions by solving Laplace's equation
        self.mesh_solver = mesh_solver(self)
    
    def set_solvers(self):
        # initialize the solvers
        
        # initialize time-stepping information
        self.dt = Constant(self.domain, np.abs(self.timesteps[1]-self.timesteps[0]))
        self.t = Constant(self.domain,self.timesteps[0])

        # initialize expressions for displacing mesh
        x = SpatialCoordinate(self.domain)
        self.dh_expr = Expression(dt*(self.sol.sub(0).sub(1) - self.sol.sub(0).sub(0)*(-self.slope) + self.smb_surf(x[0],t)),self.V0.element.interpolation_points())
        self.ds_expr = Expression(dt*(self.sol.sub(0).sub(1) - self.sol.sub(0).sub(0)*self.slope + self.smb_base(x[0],t)),self.V0.element.interpolation_points())

        # set solvers
        self.set_stokes_solver(dt,t)
        self.set_slope_solver()
        self.set_mesh_solver() 
    
    def update_mesh(self):
        # update mesh according to velocity solution,
        # surface slope, and basal forcing
        update_mesh(self)
    
    def output_setup(self):
        # set up output arrays and expressions for
        # interpolating solutions
        output_setup(self)
    
    def output_process(self):
        # interpolate solution and save in arrays
        output_process(self)
    
    def output_save(self):
        # save soluton arrays as .npy files in 
        # directory (results_name)
        output_save(self)