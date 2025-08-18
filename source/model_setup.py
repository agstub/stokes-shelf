from dolfinx.fem import Function, functionspace
from basix.ufl import element, mixed_element
import numpy as np
from dolfinx.mesh import locate_entities, meshtags
from solver import solve
from stokes import stokes_solver
from dolfinx.mesh import locate_entities_boundary
from mesh_routine import mesh_solver, slope_solver
import params

# model input class file
class model_setup:
    def __init__(self, comm, domain,z_b,z_s):
        # mpi 
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        
        # Domain, mesh, function spaces
        self.domain = domain
        self.x = self.domain.geometry.x[:,0]
        self.z = self.domain.geometry.x[:,1]        
        
        p_el = element('P',self.domain.basix_cell(),1)     # Pressure p  
        u_el = element('P',self.domain.basix_cell(),2,shape=(self.domain.geometry.dim,)) # Velocity u
        self.V = functionspace(self.domain,mixed_element([u_el,p_el]))  # Function space for (u,p)
        self.V0 = functionspace(self.domain, ("CG", 1))
        self.mask = self.ghost_mask(self.V0.dofmap.index_map) 
        self.mask_cells = self.ghost_mask(self.domain.topology.index_map(self.domain.topology.dim))
        
        # define solution function and set initial conditions
        # sol = ((u,w),p)
        self.sol = Function(self.V)
        
        # for saving the solution    
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
        self.timesteps = None
        self.nt_save = None
        self.nt_check = None
        
        # facets 
        self.facets_left = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.LeftBoundary(x))        
        self.facets_right = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.RightBoundary(x))
        self.facets_top = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.TopBoundary(x))
        self.facets_base = locate_entities_boundary(self.domain, self.domain.topology.dim-1, lambda x: self.BaseBoundary(x))
        
        # physical constants
        self.A = params.A         # ice rigidity
        self.n = params.n         # flow-law exponent
        self.g = params.g         # gravitional acceleration
        self.rho_i = params.rho_i # ice density
        self.rho_w = params.rho_w # water density
        self.delta = params.delta # flotation factor
    
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
                      
    def ghost_mask(self, index_map):
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
        # Left boundary (inflow/outflow)
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
        solve(self)
        
    def stokes_solver(self,dt):
        return stokes_solver(self,dt)
    
    def slope_solver(self):
        return slope_solver(self)

    def mesh_solver(self):
        return mesh_solver(self)