#python ob_cavity.py --k 2 --nref 2 --discretisation sv --mh bary --solver-type allu
#python ob_cavity.py --k 2 --nref 1 --discretisation sv --mh bary --patch macro --solver-type almg --restriction |& tee output_tests/almg_k_2_nref_1.log
from firedrake import *
from implcfpc import *

class OberbeckBoussinesqCavity_up(BoussinesqProblem_up):
    def __init__(self,baseN,nu,Pr,Ra,diagonal=None):
        super().__init__(nu=nu,Pr=Pr,Ra=Ra)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), Constant((0., 0.)), [1, 2, 3, 4]),
               DirichletBC(Z.sub(0), Constant(1.0), (1,)), 
               DirichletBC(Z.sub(0), Constant(0.0), (2,)), 
            ]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self,D):    
        S = 2.*self.nu*D 
        return S

    def const_rel_picard(self,D,D0):
        S0 = 2.*self.nu*D0
        return S0

    def const_rel_temperature(self, theta, gradtheta):
        q = gradtheta
        return q

class OberbeckBoussinesqCavity_Sup(BoussinesqProblem_Sup):
    def __init__(self,baseN,nu,Pr,Ra,diagonal=None):
        super().__init__(nu=nu,Pr=Pr,Ra=Ra)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(2), Constant((0., 0.)), [1, 2, 3, 4]),
               DirichletBC(Z.sub(0), Constant(1.0), (1,)), 
               DirichletBC(Z.sub(0), Constant(0.0), (2,)), 
            ]
        return bcs

    def has_nullspace(self): return True

    def const_rel(self, S, D):    
        G = S - 2.*self.nu*D 
        return G

    def const_rel_picard(self,S,D,S0,D0):
        G0 = S0 - 2.*self.nu*D0
        return G0

    def const_rel_temperature(self, theta, gradtheta):
        q = gradtheta
        return q

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup"])
    args, _ = parser.parse_known_args()

    #Viscosities
    nus = [1.]
    nu = Constant(nus[0])

    #Prandtl number 
    Pr_s = [2.,6.8]
    Pr_s = [1.]
    Pr = Constant(Pr_s[0])

    #Rayleigh number
    Ra_s = [200]
    Ra_s = list(range(250,20000 + 250,250))
    Ra_s = list(range(250,200000 + 250,250))
    Ra = Constant(Ra_s[0])

    if args.fields == "Tup":
        problem_ = OberbeckBoussinesqCavity_up(args.baseN,nu=nu,Pr=Pr,Ra=Ra,diagonal=args.diagonal)
    else:
        problem_ = OberbeckBoussinesqCavity_Sup(args.baseN,nu=nu,Pr=Pr,Ra=Ra,diagonal=args.diagonal)
    solver_ = get_solver(args, problem_)

    continuation_params = {"nu": nus,"Pr": Pr_s,"Ra": Ra_s}
    results = run_solver(solver_, args, continuation_params)

    #Quick visualisation
    if args.fields == "Tup":
        theta, u, p = solver_.z.split()
    else:
        theta, S_, u, p = solver_.z.split()
    u.rename("Velocity")
    p.rename("Pressure")
    theta.rename("Temperature")
#    File("output_tests/OB_up_Ra_20e5.pvd").write(u, p, theta)
try:
  import matplotlib.pyplot as plt
except:
  warning("Matplotlib not imported")

try:
  plot(theta)
except Exception as e:
  warning("Cannot plot figure. Error msg '%s'" % e)
try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg '%s'" % e)
