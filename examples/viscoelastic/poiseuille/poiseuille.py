# python poiseuille.py --baseN 10 --nref 1 --k 2 --mh bary --patch macro --gamma 0.0 --discretisation oldroydSV --plots --solver-type lu
from firedrake import *
from alfi_3f import *

from firedrake.petsc import PETSc

PETSc.Sys.popErrorHandler()

class OldroydB_Poiseuille(NonNewtonianProblem_Sup):
    def __init__(self, baseN, nu, tau, G, delta=0.0, diagonal=None):
        super().__init__(nu=nu, tau=tau, G=G, delta=delta)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = RectangleMesh(4*self.baseN, self.baseN, 4, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):#TODO: Try with natural BCs?
        bcs = [DirichletBC(Z.sub(1), self.exact_vel(Z.ufl_domain()), "on_boundary"),
               DirichletBC(Z.sub(0), self.exact_stress(Z.ufl_domain()), 1)] #inflow
#        for bc in bcs: print(bc._function_space)
        return bcs

    def has_nullspace(self): return True

    def exact_vel(self, domain):
        (x, y) = SpatialCoordinate(domain)
        u_ex = as_vector([(y - y**2)/(2.*(self.nu + self.G*self.tau)), 0])
        return u_ex

    def exact_stress(self, domain):
        (x, y) = SpatialCoordinate(domain)
        B_11 = 1. + 2.*((self.tau * (1 - 2.*y)) / (2.* (self.nu + self.G*self.tau)))**2
        B_12 = (self.tau * (1 - 2.*y)) / (2. * (self.nu + self.G*self.tau))
        B_22 = 1.
        B = as_vector([B_11, B_12, B_22])
        return B

    def exact_pressure(self, domain):
        (x, y) = SpatialCoordinate(domain)
        p_ex = 4 - x
        p_ex = p_ex - 0.25*assemble(p_ex*dx) #Assumes we are using only Dirichlet BCs
        return p_ex

#    def rhs(self, Z):
#        u_ex = self.exact_velocity(Z)
#        p_ex = self.exact_pressure(Z)
#
#        S_ex = self.exact_stress(Z)
#
#        f1 = -div(S_ex) + div(outer(u_ex, u_ex)) + grad(p_ex)
#        f2 = -div(u_ex)
#
#        return f1, f2

if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    #Initialize values
    nu = Constant(1.0)
    tau = Constant(1.0)
    G = Constant(1.0)
    delta = Constant(0.1)

    problem_Sup = OldroydB_Poiseuille(args.baseN, nu=nu, tau=tau, G=G, delta=delta, diagonal=args.diagonal)
    solver_Sup = get_solver(args, problem_Sup)

    #Test a simple problem
    nu_s = [1.0]
    taus = [0.5, 1.0]
    G_s = [0.0, 1.0]
    deltas = [0.1]

    continuation_params = {"delta": deltas ,"G": G_s, "tau": taus, "nu": nu_s}
    results = run_solver(solver_Sup, args, continuation_params)

    ## Errors
    B_exact = problem_Sup.exact_stress(solver_Sup.z.ufl_domain())
    u_exact = problem_Sup.exact_vel(solver_Sup.z.ufl_domain())
    p_exact = problem_Sup.exact_pressure(solver_Sup.z.ufl_domain())
    B_, u, p = solver_Sup.z.split()
    print("L2 stress error: ", norm(B_exact - B_))
    print("L2 velocity error: ", norm(u_exact - u))
    print("L2 pressure error: ", norm(p_exact - p))

    if args.plots:
        k = solver_Sup.z.sub(1).ufl_element().degree()
        B_, u, p = solver_Sup.z.split()
        (B_1,B_2,B_3) = split(B_)
        B = as_tensor(((B_1,B_2),(B_2,-B_1)))
        D = sym(grad(u))
        BB = interpolate(B, TensorFunctionSpace(solver_Sup.z.ufl_domain(),"DG",k-1))
        DD = interpolate(D, TensorFunctionSpace(solver_Sup.z.ufl_domain(),"DG",k-1))
        u.rename("Velocity")
        p.rename("Pressure")
        BB.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        string = "tau_%s_G_%s_nu%s"%(taus[-1],G_s[-1],nu_s[-1])
        File("output_plots/z_%s.pvd"%string).write(DD,BB,u,p)
