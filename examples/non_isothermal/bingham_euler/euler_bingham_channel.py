#Remember to use the l2 line-search (it converged with damping=0.7)
from firedrake import *
from alfi_3f import *

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

def PositivePart(a, eps): return (a + sqrt(a*a + eps*eps))/2.

def Threshold(x, a, b, d, eps):
    """ f(x) = max{0,min{d, d/(b-a) * (x-b) + d}}
    The function is equal to 'd' (a positive number) when x=b and zero when x=a, and varies linearly
    between a and b. If a<b the function is non-decreasing (and non-increasing if b<a). The corners are
    regularised using 'eps'"""
    z = 0.5 * (2.*d + (d/(b-a))*(x-b) - abs(d/(b-a))*sqrt((x-b)**2 + eps**2))
    return PositivePart(z, eps)

def PositivePartF(a, eps): return (a + sqrt(inner(a,a) + eps*eps))/2.

class BinghamEulerChannel2D(NonIsothermalProblem_Sup):
    def __init__(self, cr, **params):
        super().__init__(**params)
        self.cr = cr

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        (x, y) = SpatialCoordinate(Z.ufl_domain())
        bcs = [DirichletBC(Z.sub(2), Constant((0., 0.)), [12, 13, 14]),
               DirichletBC(Z.sub(2).sub(1), 0. , [11]),
               DirichletBC(Z.sub(2), self.bingham_poiseuille(Z.ufl_domain(), self.tau), [10]),
               DirichletBC(Z.sub(0), Constant(10.), [10,12]),
               DirichletBC(Z.sub(0), -x+20, [14]), #Linear decrease
               DirichletBC(Z.sub(0), Constant(0.0), [11,13])]
        return bcs

    def has_nullspace(self): return False

    def const_rel(self, S, D, theta):
        ystress = Threshold(theta, 7.0, 9.0, self.tau, self.eps2) #Vanishes for theta<7
        ystrain = Threshold(theta, 3.0, 1.0, self.sigm, self.eps2) #Vanishes for theta>3
        if self.cr == "newtonian":
            G = 2.*self.nu*D - S
        elif self.cr == "fully-implicit":
            G = 2.*self.nu*(PositivePartF(sqrt(inner(D,D)) - ystrain, self.eps) / sqrt(inner(D,D)))*D  - (PositivePartF(sqrt(inner(S,S)) - ystress, self.eps) / sqrt(inner(S,S)))*S
        return G

    def const_rel_picard(self,S, D, theta, S0, D0):
        ystress = Threshold(theta, 7.0, 9.0, self.tau, self.eps2) #Vanishes for theta<7
        ystrain = Threshold(theta, 3.0, 1.0, self.sigm, self.eps2) #Vanishes for theta>3
        if self.cr == "newtonian":
            G0 = 2.*self.nu*D0 - S0
        elif self.cr == "fully-implicit":
            G0 = 2.*self.nu*(PositivePartF(sqrt(inner(D,D)) - ystrain, self.eps) / sqrt(inner(D,D)))*D0  - (PositivePartF(sqrt(inner(S,S)) - ystress, self.eps) / sqrt(inner(S,S)))*S0

        return G0

    def const_rel_temperature(self, theta, gradtheta):
        kappa = Constant(1.0)
        q = kappa*gradtheta
        return q

    def bingham_poiseuille(self, domain, Bn_inlet):
        #Choose the pressure drop C such that the maximum of the (non-dimensional) velocity is 1.
        C = Bn_inlet + 1. + sqrt((Bn_inlet + 1.)**2 - Bn_inlet**2)
        (x, y) = SpatialCoordinate(domain)
        aux = conditional(le(y,-((Bn_inlet)/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1+y),0]),as_vector([(C/2.)*(1 - (Bn_inlet/C)**2) - Bn_inlet*(1 - (Bn_inlet/C)),0]))
        sols = conditional(ge(y,(Bn_inlet/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1-y),0]),aux)
        return sols

    def interpolate_initial_guess(self, z):
        (x,y) = SpatialCoordinate(z.ufl_domain())
        w_expr = as_vector([(y-1)*(y+1), 0.])
        z.sub(2).interpolate(w_expr)
        z.sub(2).interpolate(self.bingham_poiseuille(z.ufl_domain(), self.tau))
#        ts = as_vector([1., 1.])
        ts = as_vector([1., 1., 1.])
        z.sub(1).interpolate(ts)

    def effective_viscosity(self, S, D):
        ystress = Threshold(theta, 7.0, 9.0, self.tau, self.eps2) #Vanishes for theta<7
        ystrain = Threshold(theta, 3.0, 1.0, self.sigm, self.eps2) #Vanishes for theta>3
        nu_eff2 = 2.*self.nu*(PositivePartF(sqrt(inner(D,D)) - ystrain, self.eps) / sqrt(inner(D,D)))/(PositivePartF(sqrt(inner(S,S)) - ystress, self.eps) / sqrt(inner(S,S)))
        return nu_eff2

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    parser.add_argument("--cr", type=str, default="fully-implicit",
                        choices=["newtonian", "fully-implicit"])
    args, _ = parser.parse_known_args()

    nu = Constant(0.5)

    #Yield stress when theta=9 (or higher)
    taus = [0.,0.01,0.02, 0.025]
    taus2 = [0.025, 0.03,0.04,0.05]
    taus3 = [0.052, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059, 0.06]#, 0.07, 0.08, 0.09, 0.1]
    tau = Constant(taus[0])

    #Yield 'strain' when theta=1 (or lower)
    sigmas = [0., 0.1, 0.2]
    sigmas = [0., 0.05, 0.1]
    sigmas2 = [0.1, 0.2]
    sigm = Constant(sigmas[0])

    #Regularisation for the threshold function for the activation parameters
    epss_2 = [1e-4, 1e-5]
    epss_2 = [1e-4]
    epss_2 = [1e-3, 1e-4]
    eps2 = Constant(epss_2[0])

    #Regularisation for the constitutive relation
    zepss = [2., 10., 50., 100.]
    epss = [1./z_ for z_ in zepss]
    epss = [0.01,1e-3,1e-4,1e-5]
    epss = [1e-4]
    eps = Constant(epss[0])

    #Peclet number
    Pe_s = [1.]
    Pe = Constant(Pe_s[0])

    problem_ = BinghamEulerChannel2D(args.cr, nu=nu, Pe=Pe, eps2=eps2, tau=tau, sigm=sigm, eps=eps)
    solver_ = get_solver(args, problem_)

    problem_.interpolate_initial_guess(solver_.z)

    continuation_params = {"nu": [float(nu)],"Pe": Pe_s,"eps2": epss_2,"tau": taus,"sigm": sigmas, "eps": epss} #Works
#    continuation_params = {"nu": [float(nu)],"Pe": Pe_s,"eps2": epss_2,"sigm": sigmas,"tau": taus, "eps": epss}
    results = run_solver(solver_, args, continuation_params)

#    continuation_params = {"nu": [float(nu)],"Pe": [Pe_s[-1]],"eps2": [epss_2[-1]],"tau": taus2,"sigm": sigmas2, "eps": [epss[-1]]}
    continuation_params = {"nu": [float(nu)],"Pe": [Pe_s[-1]],"eps2": [epss_2[-1]],"sigm": sigmas2,"tau": taus2, "eps": [epss[-1]]} #Works
    results = run_solver(solver_, args, continuation_params)


    continuation_params = {"nu": [float(nu)],"Pe": [Pe_s[-1]],"eps2": [epss_2[-1]],"sigm": [sigmas2[-1]], "eps": [epss[-1]], "tau": taus3}
    results = run_solver(solver_, args, continuation_params, {"tau": "secant"})

    if args.plots:
        k = solver_.z.sub(2).ufl_element().degree()
        theta, S_, u, p = solver_.z.split()
#        (S_1,S_2) = split(S_)
#        S = as_tensor(((S_1,S_2),(S_2,-S_1)))
        (S_1,S_2,S_3) = split(S_)
        S = as_tensor(((S_1,S_2),(S_2,S_3)))
        D = sym(grad(u))
        nu_eff = 0.5 * sqrt(inner(S,S)) / (sqrt(inner(D,D)) + 1e-10)
        nu_eff2 = problem_.effective_viscosity(S, D)
        SS = project(S,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        DD = project(D,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        nu_eff_ = project(nu_eff, FunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        nu_eff2_ = project(nu_eff2, FunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        u.rename("Velocity")
        p.rename("Pressure")
        theta.rename("Temperature")
        nu_eff_.rename("Effective Viscosity")
        nu_eff2_.rename("Effective Viscosity B")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        string = "_tau%s"%(taus3[-1])
        string += "_sigm%s"%(sigmas2[-1])
        string += "_eps%s"%(epss[-1])
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        string += "_%s"%(args.solver_type)

        File("plots/z%s.pvd"%string).write(nu_eff_, nu_eff2_, u, theta, SS, DD)
