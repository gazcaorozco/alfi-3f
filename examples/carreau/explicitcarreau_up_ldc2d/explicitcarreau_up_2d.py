#--k 2 --nref 1 --baseN 12 --solver-type almg --discretisation sv --mh bary --stabilisation-type-u burman --patch macro --smoothing 4 --cycles 2 --restriction
from firedrake import *
from alfi_3f import *

from firedrake.petsc import PETSc

PETSc.Sys.popErrorHandler()

class ImplicitCarreau_ldc(NonNewtonianProblem_up):
    def __init__(self,baseN,r,nu,eps,tau,diagonal=None,regularised=True):
        super().__init__(r=r,nu=nu,eps=eps,tau=tau)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.regularised = regularised
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 2, 2, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(0), Constant((0., 0.)), [1, 2, 3])]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, D):
        nn2 = (self.r-2)/(2.)
        visc_diff2 = (2.*self.nu)*(1. - self.tau)
        S = visc_diff2*pow(1 + (1./self.eps)*inner(D,D),nn2)*D + 2.*self.nu*self.tau*D
        return S

    def const_rel_picard(self, D, D0):
        nn2 = (self.r-2)/(2.)
        visc_diff2 = (2.*self.nu)*(1. - self.tau)
        S0 = visc_diff2*pow(1 + (1./self.eps)*inner(D,D),nn2)*D0 + 2.*self.nu*self.tau*D0
        return S0

    def driver(self, domain):
        (x, y) = SpatialCoordinate(domain)
        if self.regularised:
            driver = as_vector([x*x*(2-x)*(2-x)*(0.25*y*y), 0])
        else:
            driver = as_vector([(0.25*y*y), 0])
        return driver

    def relaxation_direction(self): return "0+:1-"

    def interpolate_initial_guess(self, z):
        w_expr = self.driver(z.ufl_domain())
        z.sub(0).interpolate(w_expr)

if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    #Initialize values
    r = Constant(2.5)
    nu = Constant(0.05)
    eps = Constant(1e-05)
    tau = Constant(0.)

    problem_up = ImplicitCarreau_ldc(args.baseN,r=r,nu=nu,eps=eps,tau=tau,diagonal=args.diagonal)
    solver_up = get_solver(args, problem_up)

    problem_up.interpolate_initial_guess(solver_up.z)

    #Test a simple problem
    r_s = [2.]
    epss = [1.]
    nus = [2.]
    taus = [1.]

    #Continuation for small nu (needs advective stabilisation)
    r_s =  [2.0]#[2.,2.5]
    taus = [1.0]#[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
    epss = [1.0]#[1.,0.5,0.1,0.05,0.01,0.008,0.005]
    res = [1, 10, 100] + list(range(200, 10000+200, 200))
    nus = [2./re for re in res]

#    #For continuation in r,r,eps,eps,tau,tau,nu,nu
#    continuation_params = {"r": [r_s[0]],"eps": [epss[0]],"tau": [taus[0]],"nu": [nus[0]]}
    continuation_params = {"r": r_s, "eps": epss, "tau": taus, "nu": nus}
    results = run_solver(solver_up, args, continuation_params)

#    #Test to see if solutions are too different for different gammas
#    z_gam = solver_up.z.copy(deepcopy=True)
#    solver_up.gamma.assign(0.01)
#    solver_up.z.assign(0)
#    results_1 = run_solver(solver_up, args, continuation_params)
#    z_1 = solver_up.z.copy(deepcopy=True)
#    print("Norm of the difference between gamma 0.01 and 10000: %.14e"%(errornorm(z_gam, z_1)))


    if args.plots:
        k = solver_Sup.z.sub(0).ufl_element().degree()
        u, p = solver_Sup.z.split()
        D = sym(grad(u))
        S = problem_up.const_rel(D)
        SS = interpolate(S,TensorFunctionSpace(solver_Sup.z.ufl_domain(),"DG",k-1))
        DD = interpolate(D,TensorFunctionSpace(solver_Sup.z.ufl_domain(),"DG",k-1))
        u.rename("Velocity")
        p.rename("Pressure")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        string = "_up"
        string += "_nu%s"%(nus[-1])
#        string += "_r%s"%(r_s[-1])
        File("output_plots/z_newt%s.pvd"%string).write(DD,SS,u,p)
