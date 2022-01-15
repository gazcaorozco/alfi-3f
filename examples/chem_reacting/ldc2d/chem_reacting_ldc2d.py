#python chem_reacting_ldc2d.py --discretisation th --mh uniform --patch star --k 2 --nref 2 --gamma 0.0 --linearisation kacanov --solver-type lu --stabilisation-type-u none --stabilisation-type-t supg --scalar-conv forced2 --rheol synovial2
from firedrake import *
from alfi_3f import *

import os

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

class ChemicallyReactingLDC(NonNewtonianProblem_Tup):
    def __init__(self, rheol, **params):
        super().__init__(**params)
        self.rheol = rheol

    def mesh(self, distribution_parameters):
#        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/rectangle.msh",
#                    distribution_parameters=distribution_parameters)
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/rectangle_bdry.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.lid_vel(Z), 4),
               DirichletBC(Z.sub(1), Constant((0., 0.)), [1,2,3]),
               DirichletBC(Z.sub(0), Constant(0.3), (4,)),
               DirichletBC(Z.sub(0), Constant(0.1), (3,))
            ]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, D, c):
        viscosity = self.visc(D, c)
        S = 2. * viscosity * D
        return S

    def const_rel_picard(self,D, c, D0):
        viscosity = self.visc(D, c)
        S0 = 2. * viscosity * D0
        return S0

    def const_rel_temperature(self, c, gradc):
        kappa = Constant(1.0)
        q = kappa*gradc
        return q

    def visc(self, D, c):
        if self.rheol == "synovial":
            #Synovial shear-thinning
            nu_ = 2. * self.nu * pow(self.beta + (1./self.eps)*inner(D,D), 0.5*(exp(-self.alpha*c) - 1.0))
        elif self.rheol == "synovial2":
            #Synovial Carreau-Yasuda #nu = visc at zero, beta = nu_inf/nu_0
            nu_ = 2. * self.nu * self.beta + 2. * self.nu * (1. - self.beta)*pow(1.0 + (1./self.eps)*inner(D,D), 0.5*(exp(-self.alpha*c) - 1.0))
        elif self.rheol == "power-law":
            """ alpha is the power-law parameter (usually 'r' or 'p')"""
            n_exp = (self.alpha-2)/(2.)
            nu_ = self.nu * pow(self.beta + (1./self.eps)*inner(D,D), n_exp)
        elif self.rheol == "newtonian":
            nu_ = 2. * self.nu
        return nu_

    def interpolate_initial_guess(self, z):
        (x,y) = SpatialCoordinate(z.ufl_domain())
        w_expr = as_vector([(1./625)*x*x*(10-x)*(10-x), 0])
        z.sub(1).interpolate(w_expr)

    def lid_vel(self, Z):
        (x,y) = SpatialCoordinate(Z.ufl_domain())
        driver = as_vector([(1./625)*x*x*(10-x)*(10-x), 0])
        return driver

    def plaw_exponent(self, c):
        if self.rheol == "synovial":
            nn = 0.5*(exp(-self.alpha*c) - 1.0)
        elif self.rheol == "power-law":
            nn = (self.alpha-2)/(2.)
        elif self.rheol == "newtonian":
            nn = Constant(0.)
        return nn

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--rheol", type=str, default="newtonian",
                        choices=["synovial", "synovial2", "power-law", "newtonian"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    #Rheological parameters
    if args.rheol in ["synovial","synovial2"]:
        alphas = [0.0,2.0,3.0]; alpha = Constant(alphas[0]) #Newtonian alpha=0
        betas = [1e-4]; beta = Constant(betas[0])
        epss = [0.001]; eps = Constant(epss[0])
        nu_s = [0.5]; nu = Constant(nu_s[0])
        Pe_s = [100., 1000, 1e5]; Pe = Constant(Pe_s[0])
#        Pe_s = [1e4]; Pe = Constant(Pe_s[0])#For Kacanov
#        Pe_s = [500, 1e4]; Pe = Constant(Pe_s[0])#For Newton

    elif args.rheol == "power-law":
        alphas = [1.6]; alpha = Constant(alphas[0]) #Newtonian alpha=2
        betas = [1e-4]; beta = Constant(betas[0])
        epss = [0.001]; eps = Constant(epss[0])
        nu_s = [0.5]; nu = Constant(nu_s[0])
        Pe_s = [1000., 10000, 100000]; Pe = Constant(Pe_s[0])
        Pe_s = [100000]; Pe = Constant(Pe_s[0])

    else:
        nu_s = [0.5]; nu = Constant(nu_s[0])
        #Dummy parameters
        alphas = [2.0]; alpha = Constant(alphas[0])
        betas = [1e-4]; beta = Constant(betas[0])
        epss = [0.001]; eps = Constant(epss[0])
        Pe_s = [100.]; Pe = Constant(Pe_s[0])

    problem_ = ChemicallyReactingLDC(rheol=args.rheol, Pe=Pe, nu=nu, alpha=alpha, beta=beta, eps=eps)

    solver_ = get_solver(args, problem_)
    problem_.interpolate_initial_guess(solver_.z)
#    solver_.no_convection = True
#    results0 = run_solver(solver_,args, {"r": [2.0]})
#    solver_.no_convection = False

    z_cop = solver_.z.copy(deepcopy=True)
    energies = []
    def mymonitor(snes, it, norm):
        x = snes.getSolution()
        with z_cop.dat.vec_wo as y:
            x.copy(y)
        print("Energy of current solution: ")
        c_, u_, p_ = z_cop.split()
        if args.rheol == "power-law":
            energ = (solver_.eps/solver_.alpha) * solver_.nu * pow(solver_.beta +  (1./solver_.eps)*inner(sym(grad(u_)), sym(grad(u_))), 0.5*solver_.alpha) * dx
        elif args.rheol == "synovial2":
            D_ = sym(grad(u_))
            energ = solver_.nu * solver_.beta * inner(D_,D_) * dx
            energ += solver_.eps * solver_.nu * (1. - solver_.beta) * pow(1.0 + (1./solver_.eps)*inner(D_,D_), exp(-solver_.alpha*c_) + 1.0) * dx
        energ_ = assemble(energ)
        energies.append(energ_)
        print(energ_, float(solver_.alpha))

    if args.rheol in ["power-law", "synovial2"]:
        solver_.solver.snes.setMonitor(mymonitor)

    continuation_params = {"beta": betas, "eps": epss, "nu": nu_s, "alpha": alphas, "Pe": Pe_s}
    results = run_solver(solver_, args, continuation_params)

    if args.rheol in ["power-law", "synovial2"]:
        print(energies)

    if args.plots:
        k = solver_.z.sub(1).ufl_element().degree()
        c, u, p = solver_.z.split()
        S = problem_.const_rel(sym(grad(u)), c)
        D = sym(grad(u))
        SS = project(S,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        DD = project(D,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        nu_eff = problem_.visc(D, c)
        nu_eff = project(nu_eff, FunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        n_exp = problem_.plaw_exponent(c)
        n_exp = project(n_exp, FunctionSpace(solver_.z.ufl_domain(), "CG", k))
        u.rename("Velocity")
        p.rename("Pressure")
        c.rename("Concentration")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        nu_eff.rename("Viscosity")
        n_exp.rename("Shear-thinning index")
        string = args.rheol
        string += "_alpha%s"%(alphas[-1])
        string += "_Pe%s"%(Pe_s[-1])
        string += "_nu%s"%(nu_s[-1])
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        string += "_%s"%(args.stabilisation_type_t)
        string += "_%s"%(args.solver_type)

        File("plots/z_%s.pvd"%string).write(nu_eff,n_exp,u,p,c)
