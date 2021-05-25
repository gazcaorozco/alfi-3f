#Temperature-dependent viscosity
#python viscoplastic_cavity2d.py --discretisation sv --mh bary --patch macro --restriction --solver-type almg --cycles 4 --smoothing 4 --k 2 --nref 1 --fields TSup --non-isothermal viscosity --high-accuracy
#python viscoplastic_cavity2d.py --discretisation sv --mh bary --stabilisation-type-u none --solver-type allu --gamma 10000 --k 2 --nref 1 --fields TSup --non-isothermal viscosity --thermal-conv forced --high-accuracy
#mpiexec -n 4 python viscoplastic_cavity2d.py --discretisation sv --mh bary --patch macro --restriction --solver-type almg --cycles 4 --smoothing 4 --k 2 --nref 1 --high-accuracy --non-isothermal viscosity --fields TSup --gamma 100000  #last we tried
from firedrake import *
from alfi_3f import *

class TempViscosityOBCavity_up(NonIsothermalProblem_up):
    def __init__(self,Re,Pe,Bn,Br,Pa,temp_drop,non_isothermal):
        super().__init__(Re=Re,Pe=Pe,Bn=Bn,Br=Br,Pa=Pa,temp_drop=temp_drop)
        self.temp_drop = temp_drop
        self.non_isothermal = non_isothermal

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel_coarse_old_refined2.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), Constant((0., 0.)), [12, 13]),
               DirichletBC(Z.sub(1).sub(1), 0. , [11]),
               DirichletBC(Z.sub(1), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), [10]),
               DirichletBC(Z.sub(0), Constant(self.temp_drop), [10,12]),#Hot walls
               DirichletBC(Z.sub(0), Constant(0.0), [13])]
#        bcs = [DirichletBC(Z.sub(1), self.bingham_poiseuille(Z.ufl_domain(), 5., 10.), [10,11,12,13]),
#               DirichletBC(Z.sub(0), Constant(self.temp_drop), [10,12]),#Hot walls
#               DirichletBC(Z.sub(0), Constant(0.0), [13])]
        return bcs

    def has_nullspace(self): return False

    def const_rel(self, D, theta):
        #Choose reference values for temperature dependent yield stress
        Bn_outlet_reference = 5.#50
        temp_drop_reference = 10.#20
        #Choose an increase of a factor of 10 in the viscosity
        alpha = 10.
        nu = self.viscosity(alpha, theta)
        tau = self.yield_stress(temp_drop_reference, Bn_outlet_reference, theta)
        #Papanastasiou
        S = (((tau/sqrt(inner(D,D)))*(1-exp(-(self.Pa)*sqrt(inner(D,D))))) + 2.*nu)*D
        #Bercovier-Engelman
#        S = (2.*nu + tau/sqrt(self.eps**2 + inner(D,D)))*D
        return S

    def const_rel_picard(self,D,D0,theta):
        #Choose reference values for temperature dependent yield stress
        Bn_outlet_reference = 5.#50
        temp_drop_reference = 10.#20
        #Choose an increase of a factor of 10 in the viscosity
        alpha = 10.
        nu = self.viscosity(alpha, theta)
        tau = self.yield_stress(temp_drop_reference, Bn_outlet_reference, theta)
        #Papanastasiou
        S0 = (((tau/sqrt(inner(D,D)))*(1-exp(-(self.Pa)*sqrt(inner(D,D))))) + 2.*nu)*D0
        #Bercovier-Engelman
#        S = (2.*nu + tau/sqrt(self.eps**2 + inner(D,D)))*D
        return S0

    def const_rel_temperature(self, theta, gradtheta):
        q = gradtheta
        return q

    def bingham_poiseuille(self, domain, Bn_inlet):
        #Choose the pressure drop C such that the maximum of the (non-dimensional) velocity is 1.
        C = Bn_inlet + 1. + sqrt((Bn_inlet + 1.)**2 - Bn_inlet**2)
        (x, y) = SpatialCoordinate(domain)
        aux = conditional(le(y,-((Bn_inlet)/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1+y),0]),as_vector([(C/2.)*(1 - (Bn_inlet/C)**2) - Bn_inlet*(1 - (Bn_inlet/C)),0]))
        sols = conditional(ge(y,(Bn_inlet/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1-y),0]),aux)
        return sols

    def interpolate_initial_guess(self, z):
        w_expr = self.bingham_poiseuille(z.ufl_domain(), self.Bn)
        z.sub(1).interpolate(w_expr)

    def viscosity(self, alpha, theta):
        """
        Choosing a viscosity of the form a + b*theta such that it is 1 at the inlet and it increases by a factor "alpha" at the outlet.
        This means the Bingham number at the outlet will decrease by the same factor (assuming alpha is greater than 1).
        """
        if self.non_isothermal == "viscosity":
            visc = alpha
            visc += ((1. - alpha)/float(self.temp_drop))*theta
        else:
            visc = 1.
        return visc

    def yield_stress(self, temp_drop_reference, Bn_outlet_reference, theta):
        """
        Choosing a yield stress of the form tau = a + b*theta such that we get Bn_inlet at theta=temp_drop (hot wall) and Bn_outlet_reference
        when theta=0, for the value temp_drop_reference. The idea would be to do continuation on both self.temp_drop and self.Bn_inlet.
        """
        if self.non_isothermal == "yield-stress":
            if (abs(float(self.temp_drop)) <= 1e-9):
                tau = self.Bn
            else:
                tau = self.Bn - (self.temp_drop/temp_drop_reference)*(self.Bn - Bn_outlet_reference)
                tau += (1./temp_drop_reference)*(self.Bn - Bn_outlet_reference)*theta
        else:
            tau = self.Bn
        return tau

class TempViscosityOBCavity_Sup(NonIsothermalProblem_Sup):
    def __init__(self,Re,Pe,Bn,Br,eps,temp_drop,non_isothermal):
        super().__init__(Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop)
        self.temp_drop = temp_drop
        self.non_isothermal = non_isothermal

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel_coarse_old_refined2.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(2), Constant((0., 0.)), [12, 13]),
               DirichletBC(Z.sub(2).sub(1), 0. , [11]),
               DirichletBC(Z.sub(2), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), [10]),
               DirichletBC(Z.sub(0), Constant(self.temp_drop), [10,12]),#Hot walls
               DirichletBC(Z.sub(0), Constant(0.0), [13])]
#        bcs = [DirichletBC(Z.sub(1), self.bingham_poiseuille(Z.ufl_domain(), 5., 10.), [10,11,12,13]),
#               DirichletBC(Z.sub(0), Constant(self.temp_drop), [10,12]),#Hot walls
#               DirichletBC(Z.sub(0), Constant(0.0), [13])]
        return bcs

    def has_nullspace(self): return False

    def const_rel(self, S, D, theta):
        #Choose reference values for temperature dependent yield stress
        Bn_outlet_reference = 5.#50
        Bn_outlet_reference = 9.#10 converged
        temp_drop_reference = 10.#20
        temp_drop_reference = 15.#20
        #Choose an increase of a factor of 10 in the viscosity
        alpha = 10.
        alpha = 20.
        nu = self.viscosity(alpha, theta)
        tau = self.yield_stress(temp_drop_reference, Bn_outlet_reference, theta)
        #Implicit Bercovier-Engelman
        G = (tau + 2.*nu*sqrt(inner(D,D) + self.eps**2))*D - sqrt(self.eps**2 + inner(D,D))*S
        G = (tau + 2.*nu*sqrt(inner(D,D)))*D - sqrt(self.eps**2 + inner(D,D))*S
        #Alternative one
#        G = 2.*nu*(tau + 2.*nu*sqrt(inner(D,D)))*D - 0.5*(sqrt(inner(S,S)) - tau + sqrt((sqrt(inner(S,S)) - tau)**2 + self.eps**2))*S
        return G

    def const_rel_picard(self,S,D,theta,S0,D0,theta0):
        #Choose reference values for temperature dependent yield stress
        Bn_outlet_reference = 5.#50
        Bn_outlet_reference = 9.#10 converged
        temp_drop_reference = 10.#20
        temp_drop_reference = 15.#20
        #Choose an increase of a factor of 10 in the viscosity
        alpha = 10.
        alpha = 20.
        nu = self.viscosity(alpha, theta)
        tau = self.yield_stress(temp_drop_reference, Bn_outlet_reference, theta)
        #Implicit Bercovier-Engelman
        G0 = (tau + 2.*nu*sqrt(inner(D,D) + self.eps**2))*D0 - sqrt(self.eps**2 + inner(D,D))*S0
        G0 = (tau + 2.*nu*sqrt(inner(D,D)))*D0 - sqrt(self.eps**2 + inner(D,D))*S0
        return G0

    def const_rel_temperature(self, theta, gradtheta):
        q = gradtheta
        return q

    def bingham_poiseuille(self, domain, Bn_inlet):
        #Choose the pressure drop C such that the maximum of the (non-dimensional) velocity is 1.
        C = Bn_inlet + 1. + sqrt((Bn_inlet + 1.)**2 - Bn_inlet**2)
        (x, y) = SpatialCoordinate(domain)
        aux = conditional(le(y,-((Bn_inlet)/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1+y),0]),as_vector([(C/2.)*(1 - (Bn_inlet/C)**2) - Bn_inlet*(1 - (Bn_inlet/C)),0]))
        sols = conditional(ge(y,(Bn_inlet/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1-y),0]),aux)
        return sols

    def interpolate_initial_guess(self, z):
        w_expr = self.bingham_poiseuille(z.ufl_domain(), self.Bn)
        z.sub(2).interpolate(w_expr)
#        #Needed with the alternative Constitutive Relation:
#        (x, y) = SpatialCoordinate(z.ufl_domain())
#        z.sub(1).interpolate(as_vector([x*x,x+y]))

    def viscosity(self, alpha, theta):
        """
        Choosing a viscosity of the form a + b*theta such that it is 1 at the inlet and it increases by a factor "alpha" at the outlet.
        This means the Bingham number at the outlet will decrease by the same factor (assuming alpha is greater than 1).
        """
        if self.non_isothermal == "viscosity":
            visc = alpha
            visc += ((1. - alpha)/float(self.temp_drop))*theta
        else:
            visc = 1.
        return visc

    def yield_stress(self, temp_drop_reference, Bn_outlet_reference, theta):
        """
        Choosing a yield stress of the form tau = a + b*theta such that we get Bn_inlet at theta=temp_drop (hot wall) and Bn_outlet_reference
        when theta=0, for the value temp_drop_reference. The idea would be to do continuation on both self.temp_drop and self.Bn_inlet.
        """
        if self.non_isothermal == "yield-stress":
            if (abs(float(self.temp_drop)) <= 1e-9):
                tau = self.Bn
            else:
                tau = self.Bn - (self.temp_drop/temp_drop_reference)*(self.Bn - Bn_outlet_reference)
                tau += (1./temp_drop_reference)*(self.Bn - Bn_outlet_reference)*theta
        else:
            tau = self.Bn
        return tau

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup"])
    parser.add_argument("--non-isothermal", type=str, default="none",
                        choices=["none","viscosity","yield-stress"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    #Reynolds number
    Res = [1.]
    Re = Constant(Res[0])

    #Peclet number
    Pe_s = [1.,10.]
#    Pe_s = [1.]
    Pe = Constant(Pe_s[0])

    #Bingham number (at the inlet)
#    Bn_s = [0.]
    if args.non_isothermal == "yield-stress":
        Bn_s = [0.,0.5,1.,1.5]
    else:
        Bn_s = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6,6.5,7.] #Works with Br=0
        Bn_s = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.] #For use with Br=0.15
#        Bn_s = [0.0,0.1,0.5,1.]#Tests
    Bn = Constant(Bn_s[0])

    #Brinkman number
    if args.non_isothermal == "yield-stress":
        Br_s = [0.]
    else:
        Br_s = [0.]
        Br_s = [0.,0.1]
    Br = Constant(Br_s[0])

    #Regularisation (do something different for TSup and Tup?)
    if args.non_isothermal == "yield-stress":
        zepss_0 = [1., 5.]
        if args.fields == "Tup":
            zeps = [6.,7.,8.,9.,10,11.,12.,25,40,80] + list(range(120, 10000+30, 30))#allu works with 40 but takes a while...
        else:
            zeps = [6.,7.,200,500] + list(range(1000,50000 + 500, 500))
            zeps = [6.,7.,10.,100,125,150,175,200,220,240,260,280,300,340,380,420,460,500,600,700,800,900] + list(range(1000, 50000 + 250, 250))
            zeps = [6.,7.,10.,100,125,150,175,200,220,240,260,280,300,340,380,420,460,500,600,700,800,900] + [1000]
    else:
        zepss_0 = [1., 5.]
#        zepss_0 = [1., 1.5, 2., 2.5, 3., 5.]  #For use with Br=0.15
        if args.fields == "Tup":
            zeps = [6.,7.,8.,9.,10,11.,12.,25,40,80] + list(range(120, 10000+40, 40))#Works with allu
            zeps = [6.,7.,8.,9.,10,11.,12.,17,25,30,35,40,46,53,60,80] + list(range(120, 10000+40, 40))#Works with allu
        else:
            zeps = [6., 7., 20, 30, 40, 50, 70, 100, 120, 150, 170, 200, 350, 500] + list(range(1000, 100000 + 1000, 1000))
#            zeps = [5.25,5.5, 6.,7.,10,12,17,25,100,200,1000] + list(range(2000,100000 + 2000, 2000)) #For use with Br=0.15
            zeps = [6., 7., 20, 30, 40, 50, 70, 100, 120, 150, 170, 200, 350, 500]#, 1000]# + list(range(1000, 100000 + 1000, 1000))

    if args.fields == "Tup":
        Pa_s_0 = zepss_0
        Pa_s = zeps
        Pa = Constant(Pa_s[0])
    else:
        epss_0 = [1./z_ for z_ in zepss_0]
        epss = [1./z_ for z_ in zeps]
        eps = Constant(epss_0[0])

    #Temperature drop
    if args.fields == "yield-stress":
    #    temp_drop_s = [1.]
        temp_drop_s = [10.]
    else:
    #    temp_drop_s = [1.]
        temp_drop_s = [10.]
    temp_drop = Constant(temp_drop_s[0])

    if args.fields == "Tup":
        problem_ = TempViscosityOBCavity_up(Re=Re,Pe=Pe,Bn=Bn,Br=Br,Pa=Pa,temp_drop=temp_drop,non_isothermal=args.non_isothermal)
        solver_ = get_solver(args, problem_)
        solver_ = get_solver(args, problem_)
        problem_.interpolate_initial_guess(solver_.z)
    else:
        problem_ = TempViscosityOBCavity_Sup(Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop,non_isothermal=args.non_isothermal)
        solver_ = get_solver(args, problem_)
        problem_.interpolate_initial_guess(solver_.z)

    if args.fields == "Tup":
        continuation_params = {"Re": Res,"Br": Br_s,"Pe": Pe_s,"Bn": Bn_s,"temp_drop": temp_drop_s,"Pa": Pa_s_0}
        continuation_params2 = {"Re": [Res[-1]],"Br": [Br_s[-1]],"Pe": [Pe_s[-1]],"Bn": [Bn_s[-1]], "temp_drop": [temp_drop_s[-1]],"Pa": Pa_s}
        results = run_solver(solver_, args, continuation_params)
        results2 = run_solver(solver_, args, continuation_params2, {"Pa": "secant"})
    else:
        continuation_params = {"Re": Res,"Br": Br_s,"Pe": Pe_s,"Bn": Bn_s,"temp_drop": temp_drop_s,"eps": epss_0}
        continuation_params2 = {"Re": [Res[-1]],"Br": [Br_s[-1]],"Pe": [Pe_s[-1]],"Bn": [Bn_s[-1]], "temp_drop": [temp_drop_s[-1]],"eps": epss}
        results = run_solver(solver_, args, continuation_params)
        results2 = run_solver(solver_, args, continuation_params2, {"eps": "secant"})

    #Quick visualisation
    if args.plots:
        if args.fields == "Tup":
            k = solver_.z.sub(1).ufl_element().degree()
            theta, u, p = solver_.z.split()
            S = problem_.const_rel(sym(grad(u)), theta)
        else:
            k = solver_.z.sub(2).ufl_element().degree()
            theta, S_, u, p = solver_.z.split()
            (S_1,S_2) = split(S_)
            S = as_tensor(((S_1,S_2),(S_2,-S_1)))
        D = sym(grad(u))
        SS = interpolate(S,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        DD = interpolate(D,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        u_in = problem_.bingham_poiseuille(solver_.z.ufl_domain(), solver_.Bn)
        u_inflow = project(u_in,VectorFunctionSpace(solver_.z.ufl_domain(),"CG",k))
        u.rename("Velocity")
        p.rename("Pressure")
        theta.rename("Temperature")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        u_inflow.rename("Inlet velocity")
        string = "_Tup" if args.fields == "Tup" else "_TSup"
        if args.non_isothermal == "viscosity":
            string += "_visc"
        elif args.non_isothermal == "yield-stress":
            string += "_ystress"
        string += "_tdrop%s"%(temp_drop_s[-1])
        string += "_Bn%s"%(Bn_s[-1])
        string += "_Br%s"%(Br_s[-1])
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        File("output_plots/z%s.pvd"%string).write(DD,SS,u,theta,p,u_inflow)
