#python expansion_contraction2d.py --patch star --mh uniform --k 1 --discretisation p1p1 --gamma 0 --solver-type lu-p1 --baseN 14 --stabilisation-type-u none --no-convection --cr semismoothMAX --nref 4  --plots
from firedrake import *
from alfi_3f import *

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

def PositivePart(a): return (a + abs(a))/2.

class SemismoothExpansionContraction(NonNewtonianProblem_Sup):
    def __init__(self,r,nu,eps,tau,cr_form):
        super().__init__(r=r,nu=nu,eps=eps,tau=tau)
        self.cr_form = cr_form

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel_L_4.0_delta_0.5_h_2.0.msh", distribution_parameters=distribution_parameters)
#        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel_L_3.0_delta_0.2_h_1.2.msh", distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.inflow_velocity(Z.ufl_domain(), self.tau), [20, 21]),
#               DirichletBC(Z.sub(1).sub(1), Constant(0.), [21]),
               DirichletBC(Z.sub(1), Constant((0.,0.)), [22])]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, S, D):
#        (x, y) = SpatialCoordinate(self)
        nn = (self.r - 2.)/(2.)
        nn_1 = (float(self.r) - 1.)/(2.)
        visc = self.nu
        ystress = self.tau
        #Newtonian
        if self.cr_form == "newtonian":
            G = S - 2.*self.nu*D
        elif self.cr_form == "BE":
            G = S - (2.*self.nu*pow(inner(D,D),nn) + self.tau/(sqrt(inner(D,D) + eps**2)))*D
        elif self.cr_form == "implicitBE":
            G = sqrt(inner(D,D) + eps**2)*S - (self.tau + 2.*self.nu*pow(inner(D,D),nn_1))*D
        elif self.cr_form == "semismoothBE":
            G =  sqrt(inner(D - self.eps*S,D - self.eps*S))*(S- self.eps*D) - (ystress + 2.*visc*pow(inner(D - self.eps*S,D - self.eps*S),nn_1))*(D - self.eps*S)
        elif self.cr_form == "semismoothActEuler":
            G =  sqrt(inner(S - self.eps*D,S - self.eps*D))*(D- self.eps*S) - (self.tau + 2.*self.nu*pow(inner(S - self.eps*D,S - self.eps*D),nn_1))*(S - self.eps*D)
        elif self.cr_form == "semismoothMAX":
            G = PositivePart(sqrt(inner(S - self.eps*D,S - self.eps*D)) - ystress)*(S - self.eps*D) - 2.*self.nu*(pow(inner(D - self.eps*S,D - self.eps*S),nn))*(ystress + PositivePart(sqrt(inner(S - self.eps*D,S - self.eps*D)) - ystress))*(D - self.eps*S)
#            G = Max(sqrt(inner(S - self.eps*D,S - self.eps*D)) - self.tau, 0)*(S - self.eps*D) - 2.*self.nu*sqrt(inner(S - self.eps*D, S - self.eps*D))*(D - self.eps*S)
        return G

    def const_rel_picard(self,S, D, S0, D0):
        nn = (float(self.r) - 2.)/(2.)
        nn_1 = (float(self.r) - 1.)/(2.)
        #Newtonian
        if cr_form == "newtonian":
            G0 = S0 - 2.*nu*D0
        elif cr_form == "BE":
            G0 = S0 - (2.*nu*pow(inner(D,D),nn) + tau/(sqrt(inner(D,D) + eps**2)))*D0
        elif cr_form == "implicitBE":
            G0 = sqrt(inner(D,D) + eps**2)*S0 - (tau + 2.*nu*pow(inner(D,D),nn_1))*D0
        elif cr_form == "semismoothBE":
            G0 =  sqrt(inner(D - eps*S,D - eps*S))*(S0- eps*D0) - (tau + 2.*nu*pow(inner(D - eps*S,D - eps*S),nn_1))*(D0 - eps*S0)
        elif cr_form == "semismoothActEuler":
            G0 =  sqrt(inner(S - eps*D,S - eps*D))*(D0- eps*S0) - (tau + 2.*nu*pow(inner(S - eps*D,S - eps*D),nn_1))*(S0 - eps*D0)
        return G0

    def inflow_velocity(self, domain, tau_obj):
        (x, y) = SpatialCoordinate(domain)
        if self.cr_form == "newtonian":
            #Newtonian
            sols = as_vector([2.*(1. - y**2), 0])
        else:
            #Bingham
#            sols = conditional(le(abs(y), 0.5), as_vector([sqrt(2.)*(tau_obj/4.), 0]), as_vector([sqrt(2.)*tau_obj*(abs(y) - y**2), 0]))
#======================from the older paper
#            C = 10.
#            aux = conditional(le(y,-((tau_obj)/C)),as_vector([(C/2.)*(1 - y**2) - tau_obj*(1+y),0]),as_vector([(C/2.)*(1 - (tau_obj/C)**2) - tau_obj*(1 - (tau_obj/C)),0]))
#            sols = conditional(ge(y,(tau_obj/C)),as_vector([(C/2.)*(1 - y**2) - tau_obj*(1-y),0]),aux)
#================New aproach
            #Transition point (from data fit)
            y_tr = 0.99846 + (-1.001252 - 0.99846)/(1 + pow(tau_obj/0.484535, 0.47067))
#            print("y_tr=  ",float(y_tr))
            C = tau_obj/y_tr
            aux = conditional(le(y,-((tau_obj)/C)),as_vector([(C/2.)*(1 - y**2) - tau_obj*(1+y),0]),as_vector([(C/2.)*(1 - (tau_obj/C)**2) - tau_obj*(1 - (tau_obj/C)),0]))
            sols = conditional(ge(y,(tau_obj/C)),as_vector([(C/2.)*(1 - y**2) - tau_obj*(1-y),0]),aux)
            norm_factor = (2.*C*C*C - 3.*C*C*tau_obj + tau_obj**3)/(6*C*C)
#            sols = sols/norm_factor
#            sols = sqrt(2.)*sols
#            print("=========Pressure drop:   ",float(C))
        return sols

    def relaxation_direction(self): return "0+:1-"

    def interpolate_initial_guess(self, z):
        (x, y) = SpatialCoordinate(z.ufl_domain())
        driver = self.inflow_velocity(z.ufl_domain(), self.tau)
#        driver = as_vector([x+y,y])
        z.sub(1).interpolate(driver)
        ts = as_vector([5., 0., 5.])
#        ts = as_vector([10., 0., 10.])
        z.sub(0).interpolate(ts)


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    parser.add_argument("--cr", type=str, default="semismoothBE",
                        choices=["newtonian", "semismoothBE", "semismoothActEuler", "implicitBE", "BE", "semismoothMAX"])
    args, _ = parser.parse_known_args()

    nu = Constant(1.0)

    r_s = [2.0]
    r = Constant(r_s[0])

    if args.cr == "semismoothMAX":
        taus_1 = [0.0, 0.5, 4., 5]#, 10]#, 20, 50]
        taus_1 = [0.1, 0.5, 2.0, 4., 5, 10, 20, 30, 50]
#        taus_1 = [0.5, 2.0]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.2, 0.5, 1./60]
        epss_0 = [0.1, 0.08, 1./60] #For old BCs
        zepss = [1000, 5000, 10000]
        epss_2 = [0.00008,0.00005,0.000035,0.00002]
        epss_2 = [0.00008,0.00005,0.000035, 0.00003, 0.000025, 0.00002] #For old BCs
    elif args.cr == "semismoothBE":
        taus_1 = [0.5,2.,4.,5.]
#        taus_2 = [5.,8.,15.,20.,30.,50.]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.5, 1./60]
        zepss = [1000, 10000]
        epss_2 = [0.00002]
#        epss_2 = [0.00008,0.00005,0.000035,0.00002,0.00001]
    elif args.cr == "implicitBE":
        taus_1 = [0.5, 2., 5.]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.5, 1./20, 1./50]
        zepss = list(range(100, 1000 + 100, 100)) + list(range(2000, 10000 + 1000, 1000))

        epss_2 = [1./zepss[-1]]
    elif args.cr == "BE":
        taus_1 = [0.5, 2., 5.]
        taus_2 = [taus_1[-1]]
        epss_0 = [1., 0.5, 1./10]
        zepss = [20.] + [30., 40.]+ list(range(50, 1000 + 50, 50))
        epss_2 = [1./zepss[-1]]
    else:
        taus_1 = [0.]
        taus_2 = [taus_1[-1]]
        epss_0 = [1.]
        zepss = [1.]
        epss_2 = [1./zepss[-1]]
    epss_1 = [1./z_ for z_ in zepss]
    eps = Constant(epss_0[0])
    tau = Constant(taus_1[0])

    problem_Sup = SemismoothExpansionContraction(r=r,nu=nu,eps=eps,tau=tau,cr_form=args.cr)

    solver_Sup = get_solver(args, problem_Sup)

    problem_Sup.interpolate_initial_guess(solver_Sup.z)


    continuation_params_0 = {"nu": [float(nu)],"tau": taus_1,"r": r_s,"eps": epss_0}
#    continuation_params_0 = {"nu": [float(nu)],"r": r_s,"tau": taus_1,"eps": epss_0}
    continuation_params_1 = {"nu": [float(nu)],"r": [r_s[-1]],"tau": [taus_1[-1]],"eps": epss_1}
    continuation_params_2 = {"nu": [float(nu)],"r": [r_s[-1]],"tau": taus_2,"eps": [epss_1[-1]]}
#    continuation_params_2 = {"nu": [float(nu)],"tau": taus_2,"r": r_s_2,"eps": [epss_1[-1]]}
    continuation_params_3 = {"nu": [float(nu)],"r": [r_s[-1]],"tau": [taus_2[-1]],"eps": epss_2}

    results_0 = run_solver(solver_Sup, args, continuation_params_0)
    if args.cr in ["semismoothBE"]:
        results_1 = run_solver(solver_Sup, args, continuation_params_1)
    else:
        results_1 = run_solver(solver_Sup, args, continuation_params_1, {"eps": "secant"})
    results_2 = run_solver(solver_Sup, args, continuation_params_2)
    if args.cr in ["semismoothMAX", "semismoothMAX2"]:
        results_3 = run_solver(solver_Sup, args, continuation_params_3, {"eps": "secant"})

    if args.plots:
        (S__p1, u_p1, p_p1) = solver_Sup.z.split()
        (S_p1_1,S_p1_2,S_p1_3) = split(S__p1)
        S_p1 = as_tensor(((S_p1_1,S_p1_2),(S_p1_2,S_p1_3)))
        SS_p1 = interpolate(S_p1,TensorFunctionSpace(solver_Sup.z.ufl_domain(), "DG", 0))
        Du_p1 = interpolate(sym(grad(u_p1)),TensorFunctionSpace(solver_Sup.z.ufl_domain(), "DG", 0))
        u_p1.rename("Velocity")
        p_p1.rename("Pressure")
        SS_p1.rename("Stress")
        Du_p1.rename("Sym vel gradient")
#        File("plots/sols_%s_P1_nref%i_r_%f_tau_%f_eps_%f.pvd"%(args.cr, args.nref, float(r), float(tau), float(eps))).write(SS_p1, Du_p1, u_p1, p_p1)
        File("plots/long/sols_%s_P1_nref%i_r_%f_tau_%f_eps_%f.pvd"%(args.cr, args.nref, float(r), float(tau), float(eps))).write(SS_p1, Du_p1, u_p1, p_p1)
