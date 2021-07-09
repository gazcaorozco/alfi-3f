#python semismooth_cavity2d.py --patch star --mh uniform --k 1 --discretisation p1p1 --gamma 0 --solver-type lu-p1 --baseN 14 --stabilisation-type-u none --no-convection --cr semismoothMAX --nref 3 --plots
from firedrake import *
from alfi_3f import *

def Max(a, b): return (a+b+abs(a-b))/Constant(2)

def PositivePart(a): return (a + abs(a))/2.

class SemismoothCavity2d(NonNewtonianProblem_Sup):
    def __init__(self,baseN,r,nu,eps,tau,cr_form,diagonal=None):
        super().__init__(r=r,nu=nu,eps=eps,tau=tau)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.cr_form = cr_form

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.driver(Z.ufl_domain()), [4]),
               DirichletBC(Z.sub(1), Constant((0.,0.)), [1,2,3])]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, S, D):
#        (x, y) = SpatialCoordinate(self)
        nn = (self.r - 2.)/(2.)
        nn_1 = (float(self.r) - 1.)/(2.)
#        nn_1 = (self.r - 1.)/(2.) + 1e-15
#        temperature = (x-1.)**2 + (y-1.)**2
        #Variable viscosity
#        visc = self.nu*exp(-0.1*temperature)
#        visc = self.nu*exp(-0.4*temperature)
        visc = self.nu
        #Variable yield stress
#        ystress = ((0.5 - self.tau)/2.)*temperature + self.tau
#        ystress = ((0.0 - self.tau)/2.)*temperature + self.tau
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

    def driver(self, domain):
        (x, y) = SpatialCoordinate(domain)
        #if self.regularised:
        driver = as_vector([x*x*(2-x)*(2-x)*(0.25*y*y), 0])
        #else:
        driver = as_vector([1.0, 0.0])
        return driver

    def relaxation_direction(self): return "0+:1-"

    def interpolate_initial_guess(self, z):
        (x, y) = SpatialCoordinate(z.ufl_domain())
        driver = as_vector([x*x*(2-x)*(2-x)*(0.25*y*y), x + 20.*y*y]) #Works with (tau) = [0.5]
        driver = as_vector([x*x*(1-x)*(1-x)*(y*y), x*y*(1-x)*(1-y)]) #Works with (tau) = [0.5]
#        driver = self.driver(z.ufl_domain())
#        z.sub(1).interpolate(driver)
        ts = as_vector([5., 0., 5.])
        z.sub(0).interpolate(ts)


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    parser.add_argument("--cr", type=str, default="semismoothBE",
                        choices=["newtonian", "semismoothBE", "semismoothActEuler", "implicitBE", "BE", "semismoothMAX"])
    args, _ = parser.parse_known_args()

    nu = Constant(1.0)

    r_s = [2.0]
    #Tests
#    r_s = [2.0, 1.95, 1.9, 1.85]#,1.5,1.3,1.15]
#    r_s = [2.0,2.5]#,2.6,3.0]#,3.5,4.0,4.5,5.0,6.0,7.0]
    r = Constant(r_s[0])
 #   r_s_2 = [2.75,3.0]

    if args.cr == "semismoothBE":
        taus_1 = [0.5,2.]#,4.]
        taus_1 = [0.1, 0.5, 4., 5, 7.5, 10]#, 20, 50]
#        taus_1 = [0.]
        taus_2 = [5.,8.,15.,20.,30.,50.]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.5, 1./60]
        zepss = [1000, 10000]
        epss_2 = [0.00002]
#        epss_2 = [0.00008,0.00005,0.000035,0.00002,0.00001]
    elif args.cr == "BE":
        taus_1 = [0.5, 4.]
        taus_2 = [taus_1[-1]]
        epss_0 = [1., 0.5, 1./10]
        zepss = [20.] + [30., 40.]#+ list(range(25, 40 + 5, 5))
        epss_2 = [1./zepss[-1]]
    elif args.cr == "implicitBE":
        taus_1 = [0.5, 4.]
        taus_1 = [0.1, 0.5, 4., 5, 7.5, 10, 20, 40]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.5, 1./20, 1./50]
        zepss = list(range(100, 1000 + 100, 100)) + list(range(2000, 5000 + 1000, 1000))
        epss_2 = [1./zepss[-1]]
    elif args.cr == "semismoothActEuler":
        taus_1 = [0.5]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.5, 1./40, 1./60]
#        zepss = [200, 300, 320, 340] + list(range(360, 600+20, 20)) + list(range(610, 770+10, 10))  + [775]#,780,785,790,795,800]#, 500, 600]#750, 875, 1000]#, 1500]#, 10000]
        zepss =  [100,110,120,150,200,300,340,400] + list(range(500, 5000+100, 100))
        zepss =  list(range(100, 1000+10, 10))
        epss_2 = [1./zepss[-1]]
    elif args.cr == "semismoothMAX":
        taus_1 = [0.05,0.1, 0.5, 3., 4., 5, 8, 10, 20, 30, 31, 32, 33, 34, 35, 36, 36.5, 37, 37.5, 38, 39, 40]#, 50]
#        taus_1 = [0.]
        taus_2 = [4.5,5.]#,8.,15.,20.,30.,50.]
#        taus_2 = [10.,20.]
        taus_2 = [taus_1[-1]]
        epss_0 = [0.15, 1./60]
#        epss_0 = [0.5, 1./30, 1./45, 1./60]
        zepss = [65, 75, 85, 100, 500, 600, 750, 850, 1000, 5000, 10000]
        epss_2 = [0.00009, 0.00008, 0.000065, 0.00005, 0.00004]#, 0.000035,0.000021]#,0.00001] #Works with constant parameters
#        epss_2 = []
#For tests
        epss_0 = [0.1, 1./40, 1./60]
        zepss = list(range(100, 1000 + 100, 100)) + list(range(2000, 50000 + 1000, 1000))
        epss_2 = [0.00002]
    else:
        taus_1 = [0.]
        taus_2 = [taus_1[-1]]
        epss_0 = [1.]
        zepss = [1.]
        epss_2 = [1./zepss[-1]]
    epss_1 = [1./z_ for z_ in zepss]
    eps = Constant(epss_0[0])
    tau = Constant(taus_1[0])

    #Newtonian first
    problem_0 = SemismoothCavity2d(args.baseN,r=r,nu=nu,eps=eps,tau=tau,cr_form="newtonian",diagonal=args.diagonal)
    solver_0 = get_solver(args, problem_0)
    rslts_0 = run_solver(solver_0, args, {"nu": [float(nu)]})
    S_, u_, p_ = solver_0.z.split()

    problem_Sup = SemismoothCavity2d(args.baseN,r=r,nu=nu,eps=eps,tau=tau,cr_form=args.cr,diagonal=args.diagonal)
    solver_Sup = get_solver(args, problem_Sup)

    problem_Sup.interpolate_initial_guess(solver_Sup.z)
###    solver_Sup.z.sub(0).assign(S_)
    solver_Sup.z.sub(1).assign(u_)
#    solver_Sup.z.sub(2).assign(p_)
#    solver_Sup.z.sub(0).interpolate(as_vector([5., 0., 5.]))


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
#    else:
#        results_3 = run_solver(solver_Sup, args, continuation_params_3)


    #==== Compute vorticity and stream function =====#
    _, u, _ = solver_Sup.z.split()
    # Compute vorticity by L2 projection
    print("Solving for the vorticity...")
    QQ = FunctionSpace(solver_Sup.z.ufl_domain(), "DG", 0)
    vv = TrialFunction(QQ)
    vv_ = TestFunction(QQ)
    PP = FunctionSpace(solver_Sup.z.ufl_domain(), "CG", 1)
    ww = TrialFunction(PP)
    ww_ = TestFunction(PP)
    a = vv*vv_*dx
#    L = (u[0].dx(1) - u[1].dx(0))*s*dx
    L = (Dx(u[0],1) - Dx(u[1],0))*vv_*dx
    vort = Function(QQ)
    solve(a == L, vort, solver_parameters={"ksp_type": "preonly", "ksp_max_it": 1, "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 150})
    # Compute stream function
    # Laplace(psi) = -vort
    print("Solving for the stream function...")
    h = CellDiameter(solver_Sup.z.ufl_domain())
    n = FacetNormal(solver_Sup.z.ufl_domain())
    a = inner(grad(ww), grad(ww_))*dx - inner(avg(grad(ww)), 2*avg(outer(ww_,n)))*dS - inner(avg(grad(ww_)), 2*avg(outer(ww,n)))*dS
    a += Constant(10.)/avg(h) * inner(2*avg(outer(ww,n)), 2*avg(outer(ww_,n))) * dS
    L = vort*ww_*dx
    psi = Function(PP)
    wall = DirichletBC(PP, Constant(0.), "on_boundary")
    solve(a == L, psi, bcs=wall, solver_parameters={"ksp_type": "preonly", "ksp_max_it": 1, "pc_type": "lu", "pc_factor_mat_solver_type": "mumps", "mat_mumps_icntl_14": 150, "ksp_monitor_true_residual": None})

    #Min...
    with vort.dat.vec_ro as w:
        print("Maximum of vorticity: ", w.max())
        print("Minimum of vorticity: ", w.min())

    with psi.dat.vec_ro as w:
        print("Maximum of stream function: ", w.max())
        print("Minimum of stream function: ", w.min())

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
        vort.rename("Vorticity")
        psi.rename("Stream function")
        File("plots2/sols_%s_P1_nref%i_r_%f_tau_%f_eps_%f.pvd"%(args.cr, args.nref, float(r), float(tau), float(eps))).write(SS_p1, Du_p1, u_p1, p_p1, vort, psi)
#        File("plots_cavity2d/ystress0_sols_%s_P1_nref%i_r_%f_tau_%f_eps_%f.pvd"%(args.cr, args.nref, float(r), float(tau), float(eps))).write(SS_p1, Du_p1, u_p1, p_p1)
#        File("plots_test/sols_%s_P1_nref%i_r_%f_tau_%f_eps_%f.pvd"%(args.cr, args.nref, float(r), float(tau), float(eps))).write(SS_p1, Du_p1, u_p1, p_p1)
