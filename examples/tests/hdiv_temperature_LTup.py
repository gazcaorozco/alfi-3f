from firedrake import *

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

# Define function space W
mesh = RectangleMesh(80, 80, 1, 1)
grad_element = VectorElement("DG", mesh.ufl_cell(), 0, dim=3)
vel_element = FiniteElement("BDM", mesh.ufl_cell(), 1, variant="integral")
#vel_element = VectorElement("CG", mesh.ufl_cell(), 2)
pressure_element = FiniteElement("DG", mesh.ufl_cell(), 0)
#pressure_element = FiniteElement("CG", mesh.ufl_cell(), 1)
temp_element = FiniteElement("CG", mesh.ufl_cell(), 1)
Z = FunctionSpace(mesh, MixedElement(grad_element, temp_element, vel_element, pressure_element))
print(Z.dim())

# Define Function and TestFunction(s)
z = Function(Z)
(L_, theta, u, p) = split(z)
(LL_, theta_, v, q) = split(TestFunction(Z))
(L_1, L_2, L_3) = split(L_)
L = as_tensor(((L_1,L_2),(L_2,L_3)))
(LL_1, LL_2, LL_3) = split(LL_)
LL = as_tensor(((LL_1,LL_2),(LL_2,LL_3)))

# Exact solution
(x, y) = SpatialCoordinate(z.ufl_domain())
u1_e = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
u2_e = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
#u1_e = (1./4.)*(-2 + x)**2 * x**2 * y * (-2 + y**2)
#u2_e = -(1./4.)*x*(2 - 3*x + x**2)*y**2*(-4 + y**2)
u_exact = as_vector([u1_e, u2_e])
p_exact = y**2 - x**2
p_exact = p_exact - assemble(p*dx)
#p_exact = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5/0.5)
#p_exact = p_exact - 0.25*assemble(p*dx)
theta_exact = x**2 + y**4 + cos(x)
D_exact = sym(grad(u_exact))

#BCS
bcs = [DirichletBC(Z.sub(2), u_exact, 4),
       DirichletBC(Z.sub(2), u_exact, 1),
       DirichletBC(Z.sub(2), u_exact, 2),
       DirichletBC(Z.sub(2), u_exact, 3),
       DirichletBC(Z.sub(1), theta_exact, "on_boundary")]
nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), Z.sub(1), Z.sub(2), VectorSpaceBasis(constant=True)])

#Parameters
nu = Constant(0.5)#0.1
Theta = Constant(0.0)
Di = Constant(0.0)
Ra = Constant(1.0)
gamma = Constant(0.0)
advect = Constant(1.0)
n = FacetNormal(mesh)
h = CellDiameter(mesh)

#Const-Rel
D = sym(grad(u))
S = 2. * nu * (D + L)

#RHS
g = Constant((0, 1))
rhs_1 = -div(grad(theta_exact)) + div(u_exact*theta_exact) + Di*(theta_exact + Theta)*dot(g, u_exact) - (Di/Ra)*inner(2. * nu * D_exact, D_exact)
rhs_2 = -div(2.*nu*D_exact) + div(outer(u_exact, u_exact)) + grad(p_exact) - Ra*theta_exact*g
rhs_3 = -div(u_exact)

#Form
uflux_int_ = 0.5*(dot(u, n) + abs(dot(u, n)))*u
F = (
    inner(L, LL) * dx
    + inner(2*avg(outer(u,n)), avg(LL)) * dS
    + inner(S, sym(grad(v))) * dx
    + gamma * inner(div(u), div(v))*dx
    - advect * inner(outer(u,u), grad(v)) * dx
    + advect * dot(v('+')-v('-'), uflux_int_('+')-uflux_int_('-'))*dS
    - p * div(v) * dx
    - Ra * inner(theta * g, v) * dx
    - inner(rhs_2, v) * dx
    - div(u) * q * dx
    + inner(grad(theta), grad(theta_)) * dx
    - inner(theta*u, grad(theta_)) * dx
    + Di* (theta + Theta) * dot(g, u) * theta_ * dx
    - (Di/Ra) * inner(S,D+L) * theta_ * dx
    - rhs_1 * theta_ * dx
)

#For the jump terms
U_jmp = 2. * avg(outer(u,n))
sigma = Constant(5.0)
jmp_penalty = 2. * nu * (1./avg(h)) * U_jmp
F += (
#    - inner(avg(2.*nu*sym(grad(v))), 2*avg(outer(u, n))) * dS
    - inner(avg(S), 2*avg(outer(v, n))) * dS
    + sigma * inner(jmp_penalty, 2*avg(outer(v,n))) * dS
    )

#For BCs
def a_bc(u, v, bid, g_D):
    U_jmp_bdry = outer(u - g_D, n)
    jmp_penalty_bdry =  2. * nu * (1./h) * U_jmp_bdry
    abc = (-inner(outer(v,n), 2.*nu*sym(grad(u)))*ds(bid)
           - inner(outer(u - g_D, n), 2*nu*sym(grad(v)))*ds(bid)
           + 2.*nu*(sigma/h)*inner(v, u - g_D)*ds(bid)
           )
    return abc

def c_bc(u, v, bid, g_D, advect_):
    if g_D is None:
        uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
    else:
        uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g_D
    return advect * dot(v,uflux_ext)*ds(bid)

exterior_markers = list(mesh.exterior_facets.unique_markers)
for bc in bcs:
    if "DG" in str(bc._function_space) or "CG" in str(bc._function_space):
        continue
    g_D = bc.function_arg
    bid = bc.sub_domain
    exterior_markers.remove(bid)
    F += a_bc(u, v, bid, g_D)
    F += c_bc(u, v, bid, g_D, advect)
for bid in exterior_markers:
    F += c_bc(u, v, bid, None, advect)


#Solver params
params_lu = {
    "snes_type": "newtonls",
    "snes_max_it":100,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_monitor": None,
    "snes_linesearch_monitor": None,
    "snes_converged_reason": None,
    "snes_rtol": 1.0e-9,
    "snes_atol": 1.0e-8,
    "snes_stol": 1.0e-6,
    "mat_type": "aij",
    "ksp_type": "fgmres",
    "ksp_rtol": 1.0e-9,
    "ksp_atol": 1.0e-10,
    "ksp_max_it": 1,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_convergence_test": "skip",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 300,#,200,
     "mat_mumps_icntl_24": 1,
    # "mat_mumps_icntl_25": 1,
     "mat_mumps_cntl_1": 0.001,
     "mat_mumps_cntl_3": 0.0001,
}

problem_ = NonlinearVariationalProblem(F, z, bcs=bcs)
solver_ = NonlinearVariationalSolver(problem_, nullspace=nsp, solver_parameters=params_lu)

solver_.solve()

#Compute error
(L_, theta, u, p) = z.split()
(L_1, L_2, L_3) = split(L_)
L = as_tensor(((L_1,L_2),(L_2,L_3)))
#symvelgrad = Function(TensorFunctionSpace(mesh, "DG", 0)).interpolate(sym(grad(u))+L)
symvelgrad = Function(TensorFunctionSpace(mesh, "DG", 0)).interpolate(sym(grad(u)))
p = p - assemble(p*dx)
print("Temp error = ", norm(theta - theta_exact))
print("Vel error = ", norm(u - u_exact))
print("Pre error = ", norm(p - p_exact))
#print("SymGrad error = ", norm(symvelgrad - sym(grad(u_exact))))
print("SymGrad error = ", norm(symvelgrad - sym(grad(u_exact))))

## FOr tests...
theta_exact = Function(FunctionSpace(mesh, "CG", 1)).interpolate(theta_exact)
u_exact = Function(FunctionSpace(mesh, vel_element)).interpolate(u_exact)
p_exact = Function(FunctionSpace(mesh, "DG", 0)).interpolate(p_exact)
p = Function(FunctionSpace(mesh, "DG", 0)).interpolate(p)
File("exact_test.pvd").write(theta_exact, u_exact, p_exact)
File("computed_test.pvd").write(theta, u, p)
#??????????????????????
