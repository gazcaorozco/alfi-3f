#--k 2 --nref 1 --baseN 10 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 4 --cycles 2 --restriction --stabilisation-weight 5e-3
from firedrake import *
from alfi_3f import *

from firedrake.petsc import PETSc

#PETSc.Sys.popErrorHandler()

class ImplicitCarreau_ldc(NonNewtonianProblem_Sup):
    def __init__(self,baseN,r,nu,eps,tau,r2,nu2,eps2,tau2,diagonal=None,regularised=True,explicit=False):
        super().__init__(r=r,nu=nu,eps=eps,tau=tau,r2=r2,nu2=nu2,eps2=eps2,tau2=tau2)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.regularised = regularised
        self.baseN = baseN
        self.explicit = explicit

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 2, 2, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(1), Constant((0., 0.)), 1),
               DirichletBC(Z.sub(1), Constant((0., 0.)), 2),
               DirichletBC(Z.sub(1), Constant((0., 0.)), 3)]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, S, D):
        nn = (2.- self.r)/(2.*(self.r-1.))
        nn2 = (self.r2-2)/(2.)
        visc_diff = (1./(2.*self.nu))*(1. - self.tau)
        visc_diff2 = (2.*self.nu2)*(1. - self.tau2)

        if self.explicit:
            G = D - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S
        else:
            G = visc_diff2*pow(1 + (1./self.eps2)*inner(D,D),nn2)*D + 2.*self.nu2*self.tau2*D -  (1./(2.*self.nu))*self.tau*S - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S
        return G

    def const_rel_picard(self,S, D, S0, D0):
        nn = (2.- self.r)/(2.*(self.r-1.))
        nn2 = (self.r2-2)/(2.)
        visc_diff = (1./(2.*self.nu))*(1. - self.tau)
        visc_diff2 = (2.*self.nu2)*(1. - self.tau2)

        if self.explicit:
            G0 = D0 - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S0
        else:
            G0 = visc_diff2*pow(1 + (1./self.eps2)*inner(D,D),nn2)*D0 + 2.*self.nu2*self.tau2*D0 -  (1./(2.*self.nu))*self.tau*S0 - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S0
        return G0

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
        z.sub(1).interpolate(w_expr)

class ImplicitCarreau_ldc_Hdiv(NonNewtonianProblem_LSup):
    def __init__(self,baseN,r,nu,eps,tau,r2,nu2,eps2,tau2,diagonal=None,regularised=True,explicit=False):
        super().__init__(r=r,nu=nu,eps=eps,tau=tau,r2=r2,nu2=nu2,eps2=eps2,tau2=tau2)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.regularised = regularised
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 2, 2, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(2), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(2), Constant((0., 0.)), 1),
               DirichletBC(Z.sub(2), Constant((0., 0.)), 2),
               DirichletBC(Z.sub(2), Constant((0., 0.)), 3)]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, S, D):
        nn = (2.- self.r)/(2.*(self.r-1.))
        nn2 = (self.r2-2)/(2.)
        visc_diff = (1./(2.*self.nu))*(1. - self.tau)
        visc_diff2 = (2.*self.nu2)*(1. - self.tau2)

        #If you're using this one it's because it's fully implicit...
        G = visc_diff2*pow(1 + (1./self.eps2)*inner(D,D),nn2)*D + 2.*self.nu2*self.tau2*D -  (1./(2.*self.nu))*self.tau*S - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S
        #For Tests
#        G = D - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S
        return G

    def const_rel_picard(self,S, D, S0, D0):
        nn = (2.- self.r)/(2.*(self.r-1.))
        nn2 = (self.r2-2)/(2.)
        visc_diff = (1./(2.*self.nu))*(1. - self.tau)
        visc_diff2 = (2.*self.nu2)*(1. - self.tau2)

        if self.explicit:
            G0 = D0 - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S0
        else:
            G0 = visc_diff2*pow(1 + (1./self.eps2)*inner(D,D),nn2)*D0 + 2.*self.nu2*self.tau2*D0 -  (1./(2.*self.nu))*self.tau*S0 - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S0
        return G0

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
        z.sub(2).interpolate(w_expr)

if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--explicit", dest="explicit", default=False,
                        action="store_true")
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    #Initialize values
    r = Constant(2.0)
    r2 = Constant(2.5)
    nu = Constant(2.0)
    nu2 = Constant(0.05)
    eps = Constant(1e-05)
    eps2 = Constant(1e-05)
    tau = Constant(0.)
    tau2 = Constant(0.)

    if args.explicit:
        pclass = ImplicitCarreau_ldc
    else:
        pclass = ImplicitCarreau_ldc_Hdiv if args.discretisation in ["bdm", "rt"] else ImplicitCarreau_ldc

    problem_Sup = pclass(args.baseN,r=r,nu=nu,eps=eps,tau=tau,r2=r2,nu2=nu2,eps2=eps2,tau2=tau2,diagonal=args.diagonal,explicit=args.explicit)
    solver_Sup = get_solver(args, problem_Sup)

    problem_Sup.interpolate_initial_guess(solver_Sup.z)

    #Test a simple problem
#    r2_s = [2.]
#    r_s = [2.]
#    epss = [1.]
#    epss2 = [1.]
#    nus = [2.]
#    nus2 = [2.]
#    taus = [1.]
#    taus2 = [1.]

    #Continuation for small nu (needs advective stabilisation)
#    r2_s = [2.,2.5]
#    taus2 = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
#    epss2 = [1.,0.5,0.1,0.05,0.01,0.008,0.005]
#    nus2 = [2.,1.,0.5]
#    r_s = [2.0,1.8]
#    taus = [0.95,0.9]
#    epss = [1.,0.5,0.1,0.05,0.01,0.008,0.005]
#    res = [1, 10, 100] + list(range(200, 10000+200, 200))
#    nus = [2./re for re in res]
    #Tests (this is better...)
    r2_s = [2.5]
    taus2 = [0.5]
    epss2 = [0.005]
    nus2 = [0.5]
    r_s = [1.8]
    taus = [0.9]
    epss = [0.005]
    res = [100,250] + list(range(500, 10000+500, 500))
    nus = [2./re for re in res]

##    taus = [1.]
#    #Continuation for small r
#    r2_s = [2,2.5]
#    taus2 = [1.]
#    epss2 = [1.]
#    nus2 = [2.,1.,0.5]
#    ys = list(np.arange(1.1,15.+0.1,0.1))
#    r_s = [(1./y_) +1 for y_ in ys]
#    taus = [0.95,0.9,0.85,0.8,0.75,0.7]
#    epss = [1.,0.5,0.1,0.05,0.01,0.008]
#    nus = [2.,1.,0.5,0.01]

#    #Continuation for large Gamma
#    r_s = [2.1,2.5,3.]
#    taus = [0.9]
#    gammas = [1, 10, 100] + list(range(200, 10000+200, 200))
#    epss = [1./gamma2_ for gamma2_ in gammas]
#    nus2 = [2.,1.,0.5]
#    r2_s = [1.9,1.7]
#    taus2 = [0.9,0.2]
#    epss2 = [1.,0.5,0.1]
#    nus = [2.,1.,0.5,0.01]

#    #For continuation in r2,r,eps2,eps,tau2,tau,nu2,nu
#    continuation_params = {"r2": [r2_s[0]],"r": [r_s[0]],"eps2": [epss2[0]],"eps": [epss[0]],"tau2": [taus2[0]],"tau": [taus[0]],"nu2": [nus2[0]],"nu": [nus[0]]}
    continuation_params = {"r2": r2_s,"r": r_s,"eps2": epss2,"eps": epss,"tau2": taus2,"tau": taus,"nu2": nus2,"nu": nus}
    #For continuation in  nu2,nu,r2,eps2,eps,tau2,tau,r
#    continuation_params = {"nu2": nus2,"nu": nus,"r2": r2_s,"eps2": epss2,"eps": epss,"tau2": taus2,"tau": taus,"r": r_s}
#    #For continuation in  nu2,nu,tau2,tau,eps2,r2,r1,eps
#    continuation_params = {"nu2": nus2,"nu": nus,"tau2": taus2,"tau": taus,"eps2": epss2,"r2": r2_s,"r": r_s,"eps": epss}
#    solver_Sup.z.assign(0)
    results = run_solver(solver_Sup, args, continuation_params)

#    #Test to see if solutions are too different for different gammas
#    z_gam = solver_Sup.z.copy(deepcopy=True)
#    solver_Sup.gamma.assign(0.01)
#    solver_Sup.z.assign(0)
#    results_1 = run_solver(solver_Sup, args, continuation_params)
#    z_1 = solver_Sup.z.copy(deepcopy=True)
#    print("Norm of the difference between gamma 0.01 and 10000: %.14e"%(errornorm(z_gam, z_1)))


    if args.plots:
        k = solver_Sup.z.sub(1).ufl_element().degree()
        S_, u, p = solver_Sup.z.split()
        (S_1,S_2) = split(S_)
        S = as_tensor(((S_1,S_2),(S_2,-S_1)))
        D = sym(grad(u))
        SS = interpolate(S,TensorFunctionSpace(solver_Sup.z.ufl_domain(),"DG",k-1))
        DD = interpolate(D,TensorFunctionSpace(solver_Sup.z.ufl_domain(),"DG",k-1))
        u.rename("Velocity")
        p.rename("Pressure")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        string = "_Sup"
        string += "_nu%s"%(nus[-1])
#        string += "_r%s"%(r_s[-1])
        File("output_plots/z_newt%s.pvd"%string).write(DD,SS,u,p)
