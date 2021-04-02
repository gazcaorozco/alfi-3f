#python bingham_Sup_poiseuille.py --solver-type almg --discretisation sv --mh bary --patch macro  --smoothing 5 --cycles 2 --nref 1 --k 2 --high-accuracy
from firedrake import *
from alfi_3f import *
import numpy as np

class Bingham_Poiseuille_Sup(NonNewtonianProblem_Sup):
    def __init__(self,nu,eps,tau,C,tau_obj):
        super().__init__(nu=nu,eps=eps,tau=tau)
        self.C = C
        self.tau_obj = tau_obj

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/rectangle.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.exact_velocity(Z.ufl_domain(), self.tau_obj, self.C), "on_boundary")]

#        bcs_walls = DirichletBC(Z.sub(1), as_vector([0.,0.]), [3,4])
#        bcs_tangential = DirichletBC(Z.sub(1).sub(1),Constant(0),[1,2])
#        bcs = [bcs_walls,bcs_tangential]#,bcs_inflow
        return bcs

    def has_nullspace(self): return True #return True

    def const_rel(self,S, D):
        #Implicit
        G = 2.*nu*(self.tau + 2.*self.nu*sqrt(self.eps**2 + inner(D,D)))*D - sqrt(self.eps**2 + 2.*self.nu*inner(D,D))*S#Does it really need eps in the first term? It seems like it...
#        G = 2.*nu*(self.tau + 2.*self.nu*sqrt(inner(D,D)))*D - sqrt(self.eps**2 + 2.*self.nu*inner(D,D))*S
        #Papanastasiou
#        G = (((self.tau/sqrt(inner(D,D)))*(1-exp(-(1./self.eps)*sqrt(inner(D,D))))) + 2.*self.nu)*D - S
        #Bercovier-Engelman
#        G = (2.*self.nu + self.tau/sqrt(self.eps**2 + inner(D,D)))*D - S
        return G

    def exact_velocity(self, domain, tau_obj, C):
        (x, y) = SpatialCoordinate(domain)
        aux = conditional(le(y,-((tau_obj)/C)),as_vector([(C/2.)*(1 - y**2) - tau_obj*(1+y),0]),as_vector([(C/2.)*(1 - (tau_obj/C)**2) - tau_obj*(1 - (tau_obj/C)),0]))
        sols = conditional(ge(y,(tau_obj/C)),as_vector([(C/2.)*(1 - y**2) - tau_obj*(1-y),0]),aux)
        return sols

    def exact_pressure(self, domain, C):
        (x, y) = SpatialCoordinate(domain)
        return -C*(x - 2.)

    def interpolate_initial_guess(self, z):
        (x, y) = SpatialCoordinate(z.ufl_domain())
        driver = as_vector([x*x*(2-x)*(2-x)*(0.25*y*y), 0])
        z.sub(1).interpolate(driver)

if __name__ == "__main__":
    parser = get_default_parser()
    args, _ = parser.parse_known_args()

    #Choose Pressure Drop
    C = 2.

    #Set up lists of errors and values of eps
    k_nrefs = [(2,1),(2,2),(2,3),(3,1),(3,3)]
    epss = {}
    vel_errors = {}
    vel_profiles = {}
    y_slice = np.arange(-0.99, 1.00, 0.01)


    #The error will be computed for the last value in eps_list[j] (for each j)
    zeps_list = {}
    zeps_list[0] = [1.]
    zeps_list[1] = [5.]
    zeps_list[2] = [10]
    zeps_list[3] = [50,100]
    zeps_list[4] = [250,500,600] + list(range(650, 1000+50, 50))
    zeps_list[5] = list(range(1050, 10000+50, 50))
    eps_list = {}
    for j in [0,1,2,3,4,5]:
        eps_list[j] = [1./z_ for z_ in zeps_list[j]]
    eps = Constant(eps_list[0][0])

    #Use the same list of  eps for all the different parameters (in principle could be different)
    for pars in k_nrefs:
        epss[pars] = np.asarray([eps_list[i][-1] for i in range(6)])
#
#    zeps = [10,50,100,250,500,600]  + list(range(650, 10000+50,50))
#    epss_0 = [1.,1/5.]
#    epss = [1./z_ for z_ in zeps]
#    eps = Constant(epss_0[0])

    #Viscosities
    nus = [2.,1.5,1.]
    nu = Constant(nus[0])

    #Yield stresses
    taus = [0.,0.2,0.4,0.6,0.8,1.]
#    taus = [0.]
    tau = Constant(taus[0])

    problem_Sup = Bingham_Poiseuille_Sup(nu=nu,eps=eps,tau=tau,C=C,tau_obj=taus[-1])

    #Iterate over different number of refinements and polynomial degrees
    for params in k_nrefs:
        args.k = params[0]
        args.nref = params[1]
        vel_errors[params] = []
        solver_Sup = get_solver(args, problem_Sup)
#        problem_Sup.interpolate_initial_guess(solver_Sup.z)   #For Papanastasiou

        continuation_params = {}
        results = {}

        continuation_params[0] = {"nu": nus,"tau": taus,"eps": eps_list[0]}
        results[0] = run_solver(solver_Sup, args, continuation_params[0])
        (S_,u,p) = solver_Sup.z.split()
        vel_errors[params].append(norm(u - problem_Sup.exact_velocity(solver_Sup.Z.ufl_domain(), taus[-1], C)))
        if params == (2,1):
            vel_profiles[epss[params][0]] = [u((2.,y_))[0] for y_ in y_slice]
            np.savetxt('output/vel_profile_k_2_nref_1_eps_%f_rest.out'%(epss[params][0]),(y_slice,np.asarray(vel_profiles[epss[params][0]])))

        for i in range(1,6):
            continuation_params[i] = {"eps": eps_list[i]}#"nu": [nus[-1]],"tau": [taus[-1]],
            if i == 1:
                results[i] = run_solver(solver_Sup, args, continuation_params[i])
            else:
                results[i] = run_solver(solver_Sup, args, continuation_params[i], {"eps": "secant"})
            (S_,u,p) = solver_Sup.z.split()
            vel_errors[params].append(norm(u - problem_Sup.exact_velocity(solver_Sup.Z.ufl_domain(), taus[-1], C)))
            if params == (2,1):
                vel_profiles[epss[params][i]] = [u((2.,y_))[0] for y_ in y_slice]
                np.savetxt('output/vel_profile_k_2_nref_1_eps_%f_rest.out'%(epss[params][i]),(y_slice,np.asarray(vel_profiles[epss[params][i]])))
        vel_errors[params] = np.asarray(vel_errors[params])
        np.savetxt('output/vel_error_k_%i_nref_%i_rest.out'%(params[0],params[1]),(epss[params],vel_errors[params]))
#    solver_Sup = get_solver(args, problem_Sup)
#    continuation_params = {"nu" : nus,"tau": taus,"eps": epss_0}
#    results = run_solver(solver_Sup, args, continuation_params)
#
#    continuation_params2 = {"nu": [nus[-1]],"tau": [taus[-1]],"eps": epss}
#    results2 = run_solver(solver_Sup, args, continuation_params2, {"eps": "secant"})
#
#    (S_,u,p) = solver_Sup.z.split()
#    print("Error velocity ===   %f"%norm(u - problem_Sup.exact_velocity(solver_Sup.Z.ufl_domain(), taus[-1], C)))
#    print("Error wpressure  ===   %f"%norm(p - problem_Sup.exact_pressure(solver_Sup.Z.ufl_domain(), C)))

#    u_ex = project(problem_Sup.exact_velocity(solver_Sup.Z.ufl_domain(),taus[-1], C), solver_Sup.Z.sub(1))
#    File("exact_velocity.pvd").write(u_ex)
