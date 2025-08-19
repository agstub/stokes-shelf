# default physical parameters are set here
# these can be modified in the setup file if desired

# physical parameters:
g = 9.81                  # gravitational acceleration [m/s^2]
rho_i = 917               # ice density [kg/m^3]
rho_w = 1000              # density of water [kg/m^3]
n = 3                     # Glen's flow law parameter [-]
A = 2.24e-24              # Glen's flow law coefficient [Pa^-n s^-1]
delta = rho_w/rho_i - 1   # flotation factor [-]
eta = 1e14                # default ice viscosity [Pa s]