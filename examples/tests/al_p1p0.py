from firedrake import *
import argparse

from alfi.solver import DGMassInv

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

class P1P0SchurPC(AuxiliaryOperatorPC):
    """ This one is for the problem without augmentation"""

    def form(self, pc, test, trial):

        appctx = self.get_appctx(pc)
        gamma = appctx["gamma"]
        nu = appctx["nu"]
        delta = appctx["delta"]
        discretisation = appctx["discretisation"]

        if discretisation == "th":
            a =   1/nu * inner(test, trial) * dx
        elif discretisation == "p1p0":
            a =  (
                  + 1/nu * inner(test, trial) * dx
                  + inner(avg(delta)*jump(test), jump(trial)) * dS
                  )
        bcs = None
        return (a, bcs)

class P1P0SchurALPC(DGMassInv):
    """ Option 3: The Schur complement inverse is approximated as:
        ( -1/nu*M - C)^{-1} * (I - gamma * C * M^{-1}) - gamma * M^{-1}"""

    def initialize(self, pc):
        from firedrake.assemble import allocate_matrix, create_assembly_callable
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "al_"
        _, P = pc.getOperators()

        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")
        V = dmhooks.get_function_space(pc.getDM())
        h = CellDiameter(V.ufl_domain())
        delta = appctx["delta"]
        self.gamma = appctx["gamma"]
        self.nu = appctx["nu"]
        self.discretisation = appctx["discretisation"]

        # get function spaces
        u = TrialFunction(V)
        v = TestFunction(V)

        schur_orig = -1./self.nu * inner(u,v) * dx - inner(avg(delta)*jump(u), jump(v)) * dS #The Schur complement for the system without augmentation
        c_stab = inner(avg(delta) * jump(u), jump(v))*dS
        self.massinv = assemble(Tensor(inner(u, v)*dx).inv)
        #Or With a minus?
#        schur_orig = 1./self.nu * inner(u,v) * dx + inner(delta*jump(u), jump(v)) * dS

        opts = PETSc.Options()
        # we're inverting S_orig and using matfree to apply C_stab
        default = parameters["default_matrix_type"]
        S_orig_mat_type = opts.getString(prefix+"So_mat_type", default)
        C_stab_mat_type = opts.getString(prefix+"Cs_mat_type", "matfree")

        S_orig = assemble(schur_orig, form_compiler_parameters=fcp,
                      mat_type=S_orig_mat_type,
                      options_prefix=prefix + "So_")

        Sksp = PETSc.KSP().create(comm=pc.comm)
        Sksp.incrementTabLevel(1, parent=pc)
        Sksp.setOptionsPrefix(prefix + "So_")
        Sksp.setOperators(S_orig.petscmat)
        Sksp.setUp()
        Sksp.setFromOptions()
        self.Sksp = Sksp

        self.C_stab = allocate_matrix(c_stab, form_compiler_parameters=fcp,
                                  mat_type=C_stab_mat_type,
                                  options_prefix=prefix + "Cs_")
        _assemble_C_stab = create_assembly_callable(c_stab, tensor=self.C_stab,
                                                     form_compiler_parameters=fcp,
                                                     mat_type=C_stab_mat_type)
        _assemble_C_stab()
        Cstabmat = self.C_stab.petscmat
        self.workspace = [Cstabmat.createVecLeft() for i in (0, 1, 2, 3)]

    def apply(self, pc, x, y):

        if self.discretisation == "p1p0":
            a, b, c, d = self.workspace
            self.massinv.petscmat.mult(x, a)  #a = -gamma * M^{-1}
            a.scale(-float(self.gamma))
            self.C_stab.petscmat.mult(a, b)   #b = C * a
            c.waxpy(1.0, x, b)                #c = x + b  #Should be negligible if gamma is large
            self.Sksp.solve(c, d)             #d = S^{-1} * c
            y.waxpy(1.0, d, a)                #y = d + a
            y.scale(-1.0)
            #With a minus
    #        a, b, c, d = self.workspace
    #        self.massinv.petscmat.mult(x, a)  #a = gamma * M^{-1}
    #        a.scale(float(self.gamma))
    #        self.C_stab.petscmat.mult(a, b)   #b = C * a
    #        c.waxpy(1.0, x, b)                #c = x + b  #Should be negligible if gamma is large
    #        self.Sksp.solve(c, d)             #d = S^{-1} * c
    #        y.waxpy(1.0, d, a)                #y = d + a
        elif self.discretisation == "p2p0": #We're doing the old one here
            self.massinv.petscmat.mult(x, y)
            y.scale(-(float(self.nu) + float(self.gamma)))


parser = argparse.ArgumentParser(add_help=False)
#parser.add_argument("--negative", dest="negative", default=True,
#                    action="store_true")
parser.add_argument("--gamma", type=float, default=0.0)
parser.add_argument("--disc", type=str, default="th",
                    choices=["th", "p2p0", "p1p0", "p1p1"])
parser.add_argument("--lsolver", type=str, default="lu",
                    choices=["lu", "schur", "schurAL"])
args, _ = parser.parse_known_args()
N = 2 * 64

mesh = UnitSquareMesh(N, N)

if args.disc == "th":
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "CG", 1)
elif args.disc == "p2p0":
    V = VectorFunctionSpace(mesh, "CG", 2)
    W = FunctionSpace(mesh, "DG", 0)
elif args.disc == "p1p0":
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)
elif args.disc == "p1p1":
    V = VectorFunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "CG", 1)
Z = V * W
print("Number of dofs: %i"%Z.dim())

z = Function(Z)
u, p = split(z)
v, q = TestFunctions(Z)

nu_s = [0.5, 0.1, 0.01, 0.005, 0.001]
nu = Constant(nu_s[0])
gamma = Constant(args.gamma); print("gamma = ",float(gamma))

F = (
    gamma * inner(div(u), div(v)) * dx
    + 2. * nu * inner(sym(grad(u)), sym(grad(v))) * dx
   + inner(dot(grad(u), u), v) * dx
    - p * div(v) * dx ######
    - div(u) * q * dx
)

delta = Constant(0.0)
if args.disc == "p1p0":
    h = CellDiameter(mesh)
    beta = Constant(0.1)
    delta = h * beta
    F += gamma * inner(avg(delta) * jump(p), jump(div(v)))*dS
    F -=  inner(avg(delta) * jump(p),jump(q))*dS

if args.disc == "p1p1":
    h = CellDiameter(mesh)
    beta = Constant(0.2)
    delta = h * h * beta
#    F += - delta * inner(grad(p), grad(q)) * dx
    F += - delta * inner(dot(grad(u), u) + grad(p), grad(q)) * dx


bcs = [DirichletBC(Z.sub(0), Constant((1, 0)), (4,)),
       DirichletBC(Z.sub(0), Constant((0, 0)), (1, 2, 3))]

nsp = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])

appctx = {"nu": nu, "gamma": gamma, "discretisation": args.disc, "delta": delta}

outer_base = {
    "snes_type": "newtonls",
    "snes_max_it": 100,
    "snes_linesearch_type": "basic",#"l2",
    "snes_linesearch_maxstep": 1.0,
    "snes_monitor": None,
    "snes_linesearch_monitor": None,
    "snes_converged_reason": None,
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
#            "snes_view": None,
    "ksp_rtol": 1.0e-9,
    "ksp_atol": 1.0e-10,
    "snes_rtol": 1.0e-10,
    "snes_atol": 1.0e-9,
#    "snes_stol": 1.0e-6,
}
outer_lu = {
    "mat_type": "aij",
    "ksp_max_it": 1,
    "ksp_convergence_test": "skip",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 300,
     "mat_mumps_icntl_24": 1,
    # "mat_mumps_icntl_25": 1,
     "mat_mumps_cntl_1": 0.001,
     "mat_mumps_cntl_3": 0.0001,
}
fieldsplit_0_lu = {
    "ksp_type": "preonly",
    "ksp_max_it": 1,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 150,
}
fieldsplit_mass = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "__main__.P1P0SchurPC",
    "aux_pc_type": "bjacobi",
    "aux_sub_pc_type": "icc",
}
fieldsplit_massAL = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "__main__.P1P0SchurALPC",
    "al_So_ksp_type": "preonly",
    "al_So_pc_type": "lu",
}
outer_fieldsplit = {
    "mat_type": "nest",
    "ksp_max_it": 200,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    # "pc_fieldsplit_schur_factorization_type": "upper",
    "pc_fieldsplit_schur_factorization_type": "full",
    "pc_fieldsplit_schur_precondition": "user",
    "fieldsplit_0": fieldsplit_0_lu,
    "fieldsplit_1": fieldsplit_mass if args.lsolver == "schur" else fieldsplit_massAL,
}

# LU
if args.lsolver == "lu":
    solver_params = {**outer_base, **outer_lu}
elif args.lsolver in ["schur", "schurAL"]:
    solver_params = {**outer_base, **outer_fieldsplit}

problem_ = NonlinearVariationalProblem(F, z, bcs)
solver_ = NonlinearVariationalSolver(problem_, solver_parameters=solver_params, nullspace=nsp, appctx=appctx)

for nu_ in nu_s:
    nu.assign(nu_)
    print("-----Solving with nu = %f"%float(nu))
    solver_.solve()

u, p = z.split()
pintegral = assemble(p*dx)
area = assemble(Constant(1., mesh)*dx)
p.assign(p - Constant(pintegral/area))
#File("alp1p10_text.pvd").write(u, p)

#import matplotlib.pyplot as plt
#fig, axes = plt.subplots()
#axes.set_aspect('equal')
#kwargs = {'resolution': 1/30, 'seed': 0, 'cmap': 'winter'}
#streamlines = firedrake.streamplot(u, axes=axes, **kwargs)
#fig.colorbar(streamlines);
#fig.show()
