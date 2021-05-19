#Temperature dependent viscosity and conductivity
#mpiexec -n 4 python ob_cavity2d.py --discretisation sv --mh bary --patch macro --restriction --baseN 10 --gamma 10000 --temp-bcs left-right --stabilisation-weight-u 5e-3 --solver-type almg --thermal-conv natural_Gr --fields Tup --temp-dependent viscosity-conductivity --stabilisation-type-u burman --k 2 --nref 2 --cycles 2 --smoothing 4
from firedrake import *
from alfi_3f import *

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

class TempViscosityOBCavity_up(NonIsothermalProblem_up):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.temp_bcs = temp_bcs
        self.temp_dependent = temp_dependent
        self.unstructured = unstructured

    def mesh(self, distribution_parameters):
        if self.unstructured:
            base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",distribution_parameters=distribution_parameters)
        else:
            base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        if self.temp_bcs == "down-up":
            bcs = [DirichletBC(Z.sub(1), Constant((0., 0.)), [1, 2, 3, 4]),
                   DirichletBC(Z.sub(0), Constant(1.0), (3,)),               #Hot (bottom) - Cold (top)
                   DirichletBC(Z.sub(0), Constant(0.0), (4,)),
                ]
        else:
            bcs = [DirichletBC(Z.sub(1), Constant((0., 0.)), [1, 2, 3, 4]),
                   DirichletBC(Z.sub(0), Constant(1.0), (1,)),               #Hot (left) - Cold (right)
                   DirichletBC(Z.sub(0), Constant(0.0), (2,)),
                ]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, D, theta):
        nr = (self.r - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D
        return S

    def const_rel_picard(self,D, theta, D0):
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

    def interpolate_initial_guess(self, z):
        X = SpatialCoordinate(z.ufl_domain())
        (x, y) = X
        w_expr = as_vector([x*x + 5, 3.*y])
        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        u = (1./4.)*(-2 + x)**2 * x**2 * y * (-2 + y**2)
        v = -(1./4.)*x*(2 - 3*x + x**2)*y**2*(-4 + y**2)
        u = replace(u, {X: 2.0 * X})
        v = replace(v, {X: 2.0 * X})
        w_expr = as_vector([u,v])
        z.sub(1).interpolate(w_expr)

class TempViscosityOBCavity_Sup(NonIsothermalProblem_Sup):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.temp_bcs = temp_bcs
        self.temp_dependent = temp_dependent
        self.unstructured = unstructured

    def mesh(self, distribution_parameters):
        if self.unstructured:
            base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",distribution_parameters=distribution_parameters)
        else:
            base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        if self.temp_bcs == "down-up":
            bcs = [DirichletBC(Z.sub(2), Constant((0., 0.)), [1, 2, 3, 4]),
                   DirichletBC(Z.sub(0), Constant(1.0), (3,)),               #Hot (bottom) - Cold (top)
                   DirichletBC(Z.sub(0), Constant(0.0), (4,)),
                ]
        else:
            bcs = [DirichletBC(Z.sub(2), Constant((0., 0.)), [1, 2, 3, 4]),
                   DirichletBC(Z.sub(0), Constant(1.0), (1,)),               #Hot (left) - Cold (right)
                   DirichletBC(Z.sub(0), Constant(0.0), (2,)),
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

    def const_rel_picard(self,S, D, theta, S0, D0):
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

    def interpolate_initial_guess(self, z):
        (x,y) = SpatialCoordinate(z.ufl_domain())
        w_expr = as_vector([x*x + 5, 3.*y])
        z.sub(2).interpolate(w_expr)

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup"])
    parser.add_argument("--temp-bcs", type=str, default="left-right",
                        choices=["left-right","down-up"])
    parser.add_argument("--temp-dependent", type=str, default="none",
                        choices=["none","viscosity","viscosity-conductivity"])
    parser.add_argument("--unstructured", dest="unstructured", default=False,
                        action="store_true")
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    assert args.thermal_conv in ["natural_Ra", "natural_Ra2", "natural_Gr"], "You need to select natural convection"

    #Prandtl number
    Pr_s = [1.,10.]
    Pr_s = [1.]
#    Pr_s = [1.,0.5,0.1,0.01]
#    Pr_s = [1.,1000,2500,5000,10000]
    Pr = Constant(Pr_s[0])

    if args.thermal_conv in ["natural_Ra", "natural_Ra2"]:
        #Rayleigh number (old code)
        Ra_s = [1,2500] + list(range(5000,50000000 + 2500,2500)) #To test how high can it get (very inefficient)
        Ra_s = [1,2500, 5000, 7500] + list(range(10000, 30000 + 2500, 2500)) #For Power-law (shear thinning)
        Ra_s = [1, 1000, 2500, 3500, 5000, 7500] + list(range(10000, 30000 + 2500, 2500)) #For Power-law (shear thinning) nref=3
#        Ra_s = [1,700,1250,2500,5000] + list(range(10000, 100000 + 5000,5000)) #For Power-law (shear thickenning)
        Ra = Constant(Ra_s[0])
    else:
        #Grashof number
        Gr_s = [2500] + list(range(20000, 100000000 + 20000, 20000)) #To test how high can it get (very inefficient)
        Gr_s = [2500] + list(range(50000, 500000 + 50000, 50000)) + list(range(500000, 12500000 + 500000, 500000)) + list(range(12500000 + 500000, 15000000 + 500000, 500000)) + list(range(15000000 + 500000, 100000000, 500000)) #To test how high can it get
        Gr_s = [2500] + list(range(50000, 500000 + 50000, 50000)) + list(range(500000, 12500000 + 500000, 500000)) + list(range(12500000 + 500000, 15000000 + 500000, 500000)) + list(range(15000000 + 1000000, 100000000 + 1000000, 1000000)) #Test: Works (k,nref)=(2,1),(2,2), T:supg,none,burman
        Gr_s = [2500, 50000] + list(range(100000, 1000000 + 100000, 100000)) + list(range(2000000, 20000000 + 1000000, 1000000)) + list(range(25000000, 100000000 + 5000000, 5000000)) #Test
#        Gr_s = [2500] + list(range(500000, 12500000 + 500000, 500000)) + list(range(12500000 + 500000, 15000000 + 500000, 500000)) #For Navier-Stokes
        Gr = Constant(Gr_s[0])

    #Power-law
    r_s = [2.0]
#    r_s = [2.0,2.3,2.6,2.7]#,3.5]
#    r_s = [2.0,1.9,1.8,1.7,1.6]
    r_s = [2.0,1.8,1.6] #For power-law
    r_s = [2.0,1.8,1.7,1.6] #For power-law
    r = Constant(r_s[0])

    #Dissipation number
    Di_s = [0.]
#    Di_s = [0.,0.2,0.4,0.6,0.8,1.,1.3,1.5,1.7,1.9,2.]
#    Di_s = [0.,0.6,1.3]#,2.]  #For Navier-Stokes
    Di = Constant(Di_s[0])

    if args.thermal_conv in ["natural_Ra", "natural_Ra2"]:
        if args.fields == "Tup":
            problem_ = TempViscosityOBCavity_up(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, diagonal=args.diagonal, unstructured=args.unstructured, Pr=Pr, Ra=Ra, r=r, Di=Di)
        else:
            problem_ = TempViscosityOBCavity_Sup(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, diagonal=args.diagonal, unstructured=args.unstructured, Pr=Pr, Ra=Ra, r=r, Di=Di)
    else:
        if args.fields == "Tup":
            problem_ = TempViscosityOBCavity_up(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, diagonal=args.diagonal, unstructured=args.unstructured, Pr=Pr, Gr=Gr, r=r, Di=Di)
        else:
            problem_ = TempViscosityOBCavity_Sup(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, diagonal=args.diagonal, unstructured=args.unstructured, Pr=Pr, Gr=Gr, r=r, Di=Di)

    solver_ = get_solver(args, problem_)
    problem_.interpolate_initial_guess(solver_.z)
    solver_.no_convection = True
    results0 = run_solver(solver_,args, {"r": [2.0]})
    solver_.no_convection = False

    if args.thermal_conv in ["natural_Ra", "natural_Ra2"]:
        continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Ra": Ra_s}
    else:
        continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Gr": Gr_s}
    results = run_solver(solver_, args, continuation_params)

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
            (S_1,S_2) = split(S_)
            S = as_tensor(((S_1,S_2),(S_2,-S_1)))
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
        string = "_up_" if args.fields == "Tup" else "_Sup_"
        string += args.temp_bcs
        if args.temp_dependent == "viscosity":
            string += "_visc"
        elif args.temp_dependent == "viscosity-conductivity":
            string += "_visccond"
        if args.thermal_conv in ["natural_Ra","natural_Ra2"]:
            string += "%s_Ra%s"%(args.thermal_conv,Ra_s[-1])
        elif args.thermal_conv == "grashof":
            string += "_Gr%s"%(Gr_s[-1])
        string += "_Pr%s"%(Pr_s[-1])
        string += "_Di%s"%(Di_s[-1])
        string += "_r%s"%(r_s[-1])
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        string += "_%s"%(args.stabilisation_type_t)
        string += "_%s"%(args.solver_type)

        File("plots/z%s.pvd"%string).write(DD,SS,u,theta)

#For power-law:
# python ob_cavity2d.py --discretisation sv --mh bary --patch macro --restriction --baseN 10 --gamma 1e4 --linearisation newton --temp-bcs left-right --stabilisation-weight-u 5e-3 --solver-type almg --thermal-conv natural_Ra --fields Tup --temp-dependent viscosity --stabilisation-type-u burman --stabilisation-type-t supg --k 2 --nref 4 --cycles 1 --smoothing 6 --unstructured
