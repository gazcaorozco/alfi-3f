#mpirun -n 12 python3 --k 3 --baseN 4 --solver-type almg --discretisation sv --mh bary --stabilisation-type burman --patch macro --smoothing 6 --cycles 1 --restriction --stabilisation-weight 5e-3 2>&1
from firedrake import *
from alfi_3f import *

class ImplicitCarreau_ldc3D(NonNewtonianProblem_Sup):
    def __init__(self,baseN,r,nu,eps,tau,r2,nu2,eps2,tau2):
        super().__init__(r=r,nu=nu,eps=eps,tau=tau,r2=r2,nu2=nu2,eps2=eps2,tau2=tau2)
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = BoxMesh(self.baseN, self.baseN, self.baseN, 2, 2, 2,
                            distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.driver(Z.ufl_domain()), 4),
               DirichletBC(Z.sub(1), Constant((0., 0., 0.)), [1, 2, 3, 5, 6])]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self,S, D):
        nn = (2.- self.r)/(2.*(self.r-1.))
        nn2 = (self.r2-2)/(2.)
        visc_diff = (1./(2.*self.nu))*(1. - self.tau)
        visc_diff2 = (2.*self.nu2)*(1. - self.tau2)
        G = visc_diff2*pow(1 + (1./self.eps2)*inner(D,D),nn2)*D + 2.*self.nu2*self.tau2*D -  (1./(2.*self.nu))*self.tau*S - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S
        G = D - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S
        return G

    def const_rel_picard(self,S, D, S0, D0):
        nn = (2.- self.r)/(2.*(self.r-1.))
        nn2 = (self.r2-2)/(2.)
        visc_diff = (1./(2.*self.nu))*(1. - self.tau)
        visc_diff2 = (2.*self.nu2)*(1. - self.tau2)
        G0 = visc_diff2*pow(1 + (1./self.eps2)*inner(D,D),nn2)*D0 + 2.*self.nu2*self.tau2*D0 -  (1./(2.*self.nu))*self.tau*S0 - visc_diff*pow(1 + (1./self.eps)*inner(S,S),nn)*S0
        return G0

    def driver(self, domain):
        (x, y, z) = SpatialCoordinate(domain)
        driver = as_vector([x*x*(2-x)*(2-x)*z*z*(2-z)*(2-z)*(0.25*y*y), 0, 0])
        return driver

#    def char_length(self): return 2.0

    def relaxation_direction(self): return "0+:1-"


if __name__ == "__main__":

    parser = get_default_parser()
    args, _ = parser.parse_known_args()

#    print(args)

    #r2 greater than 2
    r2_s = [2,2.5]#,3.,3.5,4.]
    #r2 smaller than 2
#    r2_s = [2.0,1.8]#,1.7]
#    r2_s = [2.0]#[2.5]  #Mild
    r2 = Constant(r2_s[-1])

    #tau2 smaller than one
    taus2 = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5]
    #tau2 greater than one
#    taus2 = [10,100,300,500,700,900,1000]
#    taus2 = [0.95]#0
    taus2 = [0.5]
#    taus2 = [0.95] #Mild
    tau2 = Constant(taus2[-1])

    #eps2
    epss2 = [1.,0.5,0.1,0.05,0.01,0.008,0.005]
 #   epss2 = [1.]
    epss2 = [0.005] #Already mild
    eps2 = Constant(epss2[-1])

    #nu2
    nus2 = [2.,1.,0.5]
#    nus2 = [1.]   #Mild
    nu2 = Constant(nus2[-1])

    #r greater than 2
    r_s = [2,2.5]#,3.,3.5,4.]
    #r_smaller than 2
    r_s = [2.0,1.8]#,1.6]
#    r_s = [1.8] #Mild
    r = Constant(r_s[-1])

    #tau smaller than one
    taus = [0.95,0.9]
    #tau greater than one
   # taus = [1,10,100,300,500,700,900,1000]
#    taus = [0.95]#0
    taus = [0.9] #Already mild
    tau = Constant(taus[-1])

    #eps
    epss = [1.,0.5,0.1,0.05,0.01,0.008,0.005]
#    epss = [1.]
    epss = [0.005]# Already Mild
    eps = Constant(epss[-1])

    #nu
    res = [0, 1, 10, 100] + list(range(250, 5000+250, 250))
    res = [1, 10, 100]  # + list(range(start, end+step, step))
    res = [1, 10, 100] + list(range(250, 7500+250, 250))
    nus = [2./re for re in res]
#    nus = [2.] #Mild
    nu = Constant(nus[-1])

    problem_Sup = ImplicitCarreau_ldc3D(args.baseN,r=r,nu=nu,eps=eps,tau=tau,r2=r2,nu2=nu2,eps2=eps2,tau2=tau2)
    solver_Sup = get_solver(args, problem_Sup)

    continuation_params = {"r2": r2_s,"r": r_s,"eps2": epss2,"eps": epss,"tau2": taus2,"tau": taus,"nu2": nus2,"nu": nus}
#    continuation_params = {"r2": [r2_s[0]],"r": [r_s[0]],"eps2": [epss2[0]],"eps": [epss[0]],"tau2": [taus2[0]],"tau": [taus[0]],"nu2": [nus2[0]],"nu": [nus[0]]}
    results = run_solver(solver_Sup, args, continuation_params)
