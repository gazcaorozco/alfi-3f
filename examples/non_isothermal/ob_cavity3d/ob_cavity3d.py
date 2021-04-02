#python temp_viscosity_ob_cavity3d.py --baseN 4 --discretisation sv --mh bary --patch macro --restriction --gamma 10000 --temp-bcs left-right --stabilisation-weight 5e-3 --stabilisation-type none --solver-type almg --non-dimensional rayleigh1 --fields Tup  --temp-dependent none --k 3 --nref 1 --cycles 2 --smoothing 4
from firedrake import *
from alfi_3f import *

class TempViscosityOBCavity_up(NonIsothermalProblem_up):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", non_dimensional="rayleigh1", **params):
        super().__init__(non_dimensional=non_dimensional, **params)
        self.baseN = baseN
        self.temp_bcs = temp_bcs
        self.temp_dependent = temp_dependent

    def mesh(self, distribution_parameters):
        base = BoxMesh(self.baseN, self.baseN, self.baseN, 1, 1, 1,
                            distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        if self.temp_bcs == "down-up":
            bcs = [DirichletBC(Z.sub(1), Constant((0., 0., 0.)), [1, 2, 3, 4, 5, 6]),
                   DirichletBC(Z.sub(0), Constant(1.0), (5,)),               #Hot (bottom) - Cold (top)
                   DirichletBC(Z.sub(0), Constant(0.0), (6,)),
                ]
        else:
            bcs = [DirichletBC(Z.sub(1), Constant((0., 0., 0.)), [1, 2, 3, 4, 5, 6]),
                   DirichletBC(Z.sub(0), Constant(1.0), (3,)),               #Hot (left) - Cold (right)
                   DirichletBC(Z.sub(0), Constant(0.0), (4,)),
                ]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, D, theta):
        nr = (self.r - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D
        return S

    def const_rel_picard(self,D,D0,theta):
        nr = (self.r - 2.)/2.
        K = self.viscosity(theta)
        S0 = K*pow(inner(D,D),nr)*D0
        return S0

    def const_rel_temperature(self, theta, gradtheta):
        kappa = self.heat_conductivity(theta)
        q = kappa*gradtheta
        return q

    def viscosity(self, theta):
        if self.temp_dependent in ["viscosity","viscosity-conductivity"]:
            nu = 0.5 + 0.5*theta + theta**2
            nu = exp(-0.1*theta)
        else:
            nu = Constant(1.)
        return nu

    def heat_conductivity(self, theta):
        if self.temp_dependent == "viscosity-conductivity":
            kappa = 0.5 + 0.5*theta + theta**2
        else:
            kappa = Constant(1.)
        return kappa

    def interpolate_initial_guess(self, w):
        (x,y,z) = SpatialCoordinate(w.ufl_domain())
        w_expr = as_vector([z,x + 5, 3.*y])
        w.sub(1).interpolate(w_expr)

class TempViscosityOBCavity_Sup(NonIsothermalProblem_Sup):
    def __init__(self, baseN, temp_bcs="left-right",temp_dependent="viscosity", non_dimensional="rayleigh1", **params):
        super().__init__(non_dimensional=non_dimensional,**params)
        self.baseN = baseN
        self.temp_bcs = temp_bcs
        self.temp_dependent = temp_dependent

    def mesh(self, distribution_parameters):
        base = BoxMesh(self.baseN, self.baseN, self.baseN, 1, 1, 1,
                            distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        if self.temp_bcs == "down-up":
            bcs = [DirichletBC(Z.sub(2), Constant((0., 0., 0.)), [1, 2, 3, 4, 5, 6]),
                   DirichletBC(Z.sub(0), Constant(1.0), (5,)),               #Hot (bottom) - Cold (top)
                   DirichletBC(Z.sub(0), Constant(0.0), (6,)),
                ]
        else:
            bcs = [DirichletBC(Z.sub(2), Constant((0., 0., 0.)), [1, 2, 3, 4, 5, 6]),
                   DirichletBC(Z.sub(0), Constant(1.0), (3,)),               #Hot (left) - Cold (right)
                   DirichletBC(Z.sub(0), Constant(0.0), (4,)),
                ]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, S, D, theta):
        nr = (self.r - 2.)/2.
#        nr2 = (2. - self.r)/(2.*(self.r - 1.))
        K = self.viscosity(theta)
        G = S - K*pow(inner(D,D),nr)*D
#        G = D - (1./(2.*self.nu))*pow(inner(S/(2.*self.nu),S/(2.*self.nu)),nr2)*S
        return G

    def const_rel_picard(self,S,D,S0,D0,theta):
        nr = (self.r - 2.)/2.
#        nr2 = (2. - self.r)/(2.*(self.r - 1.))
        K = self.viscosity(theta)
        G0 = S0 - K*pow(inner(D,D),nr)*D0
#        G0 = D0 - (1./(2.*self.nu))*pow(inner(S/(2.*self.nu),S/(2.*self.nu)),nr2)*S0
        return G0

    def const_rel_temperature(self, theta, gradtheta):
        kappa = self.heat_conductivity(theta)
        q = kappa*gradtheta
        return q

    def viscosity(self, theta):
        if self.temp_dependent in ["viscosity","viscosity-conductivity"]:
            nu = (0.5 + 0.5*theta + theta**2)
            nu = exp(-0.1*theta)
        else:
            nu = Constant(1.)
        return nu

    def heat_conductivity(self, theta):
        if self.temp_dependent == "viscosity-conductivity":
            kappa = 0.5 + 0.5*theta + theta**2
        else:
            kappa = Constant(1.)
        return kappa

    def interpolate_initial_guess(self, w):
        (x,y,z) = SpatialCoordinate(w.ufl_domain())
        w_expr = as_vector([z,x + 5, 3.*y])
        w.sub(2).interpolate(w_expr)

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup"])
    parser.add_argument("--temp-bcs", type=str, default="left-right",
                        choices=["left-right","down-up"])
    parser.add_argument("--temp-dependent", type=str, default="none",
                        choices=["none","viscosity","viscosity-conductivity"])
    parser.add_argument("--non-dimensional", type=str, default="rayleigh1",
                        choices=["rayleigh1","rayleigh2","grashof"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    #Prandtl number
    Pr_s = [2.,6.8]
    Pr_s = [1.,0.5,0.1,0.01]
    Pr_s = [1.]
    Pr = Constant(Pr_s[0])

    if args.non_dimensional in ["rayleigh1", "rayleigh2"]:
        #Rayleigh number
        Ra_s = [2500] + list(range(5000,500000 + 5000,5000))    #Meant for left-right, constant parameters
        Ra_s = [1,2500,5000] + list(range(10000, 20000 + 5000,5000)) #For Power-law (shear thinning)
#        Ra_s = [1.]
        Ra = Constant(Ra_s[0])
    else:
        #Grashof number
        Gr_s = [2500] + list(range(20000, 10000000 + 20000, 20000)) #Used to test
        Gr_s = [2500] + list(range(63000, 1260000 + 63000, 63000)) #For Navier-Stokes
#        Gr_s = [2500] + list(range(63000, 1000000 + 63000, 63000)) #For Navier-Stokes with temp-dependent viscosity and conductivity
        Gr = Constant(Gr_s[0])

    #Power-law
    r_s = [2.0,2.5,3.,3.5,4.,4.5,5.]
    r_s = [2.0,1.9,1.8,1.7,1.6]
#    r_s = [2.0]
    r = Constant(r_s[-1])

    #Dissipation number
    Di_s = [0.]
#    Di_s = [0.,0.2,0.4,0.6,0.8,1.,1.3]#,1.5,1.7,1.9,2.]
#    Di_s = [0.,0.6,1.3]#,2.] #For Navier-Stokes
    Di = Constant(Di_s[0])

    if args.non_dimensional in ["rayleigh1", "rayleigh2"]:
        if args.fields == "Tup":
            problem_ = TempViscosityOBCavity_up(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Ra=Ra, r=r, Di=Di)
        else:
            problem_ = TempViscosityOBCavity_Sup(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Ra=Ra, r=r, Di=Di)
    else:
        if args.fields == "Tup":
            problem_ = TempViscosityOBCavity_up(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Gr=Gr, r=r, Di=Di)
        else:
            problem_ = TempViscosityOBCavity_Sup(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Gr=Gr, r=r, Di=Di)
    solver_ = get_solver(args, problem_)
    problem_.interpolate_initial_guess(solver_.z)
    solver_.no_convection = True
#    solver_.stabilisation_type = "none"
    results0 = run_solver(solver_,args, {"r": [2.0]})
    solver_.no_convection = False
#    solver_.stabilisation_type = args.stabilisation_type

    if args.non_dimensional in ["rayleigh1", "rayleigh2"]:
        continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Ra": Ra_s}
    else:
        continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Gr": Gr_s}
    results = run_solver(solver_, args, continuation_params)

    #Quick visualisation
    if args.plots:
        if args.fields == "Tup":
            k = solver_.z.sub(1).ufl_element().degree()
            theta, u, p = solver_.z.split()
            nu_temp = problem_.viscosity(theta)
            S = problem_.const_rel(sym(grad(u)), theta)
        else:
            k = solver_.z.sub(2).ufl_element().degree()
            theta, S_, u, p = solver_.z.split()
            nu_temp = problem_.viscosity(theta)
            (S_1,S_2,S_3,S_4,S_5) = split(S_)
            S = as_tensor(((S_1,S_2,S_3),(S_2,S_5,S_4),(S_3,S_4,-S_1-S_5)))
        D = sym(grad(u))
        SS = interpolate(S,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        DD = interpolate(D,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        nu_temp_ = interpolate(nu_temp, FunctionSpace(solver_.z.ufl_domain(),"CG",k-1))
        u.rename("Velocity")
        p.rename("Pressure")
        theta.rename("Temperature")
        nu_temp_.rename("Viscosity_Temp")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        string = "_up" if args.fields == "Tup" else "_Sup"
        if problem_.non_dimensional in ["rayleigh1","rayleigh2"]:
            string += "%s_Ra%s"%(problem_.non_dimensional,Ra_s[-1])
        elif problem_.non_dimensional == "grashof":
            string += "_Gr%s"%(Gr_s[-1])
        string += "_Di%s"%(Di_s[-1])
        string += "_Pr%s"%(Pr_s[-1])
        string += "_r%s"%(r_s[-1])
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        if args.temp_dependent == "viscosity":
            string += "_visc"
        elif args.temp_dependent == "viscosity-conductivity":
            string += "_visccond"
        File("plots_new/z%s.pvd"%string).write(DD,SS,u,theta)
