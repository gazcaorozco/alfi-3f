from firedrake import *
from alfi_3f import *

def PositivePart(a): return (a + abs(a))/2.

class BinghamCoolingChannel(NonNewtonianProblem):
    def __init__(self,baseN,non_isothermal,Re,Pe,Bn,Br,eps,temp_drop):
        super().__init__(Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop)
        self.temp_drop = temp_drop
        self.non_isothermal = non_isothermal
        self.vel_id = None
        self.temp_id = None
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        if self.baseN == 0: #Finer mesh for Hdiv
            base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel_hdiv.msh",
                        distribution_parameters=distribution_parameters)
        else:
            base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/channel.msh",
                        distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 13),
               DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 12),
               #DirichletBC(Z.sub(self.vel_id).sub(1), 0. , 11),
               DirichletBC(Z.sub(self.vel_id), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), 10),
               DirichletBC(Z.sub(self.temp_id), Constant(self.temp_drop), 12),#Hot wall
               DirichletBC(Z.sub(self.temp_id), Constant(self.temp_drop), 10),#Hot wall 
               DirichletBC(Z.sub(self.temp_id), Constant(0.0), 13)]
        bcs = [DirichletBC(Z.sub(self.vel_id), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), 10),
               DirichletBC(Z.sub(self.vel_id), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), 11),
               DirichletBC(Z.sub(self.vel_id), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), 12),
               DirichletBC(Z.sub(self.vel_id), self.bingham_poiseuille(Z.ufl_domain(), self.Bn), 13),
               DirichletBC(Z.sub(self.temp_id), Constant(self.temp_drop), 12),#Hot wall
               DirichletBC(Z.sub(self.temp_id), Constant(self.temp_drop), 10),#Hot wall 
               DirichletBC(Z.sub(self.temp_id), Constant(0.0), 13)]
        return bcs

    def has_nullspace(self): return True

    def const_rel_temperature(self, theta, gradtheta):
        q = gradtheta
        return q

    def interpolate_initial_guess(self, z):
        w_expr = self.bingham_poiseuille(z.ufl_domain(), self.Bn)
        z.sub(self.vel_id).interpolate(w_expr)

    def bingham_poiseuille(self, domain, Bn_inlet):
        #Choose the pressure drop C such that the maximum of the (non-dimensional) velocity is 1.
        C = Bn_inlet + 1. + sqrt((Bn_inlet + 1.)**2 - Bn_inlet**2)
        (x, y) = SpatialCoordinate(domain)
        aux = conditional(le(y,-((Bn_inlet)/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1+y),0]),as_vector([(C/2.)*(1 - (Bn_inlet/C)**2) - Bn_inlet*(1 - (Bn_inlet/C)),0]))
        sols = conditional(ge(y,(Bn_inlet/C)),as_vector([(C/2.)*(1 - y**2) - Bn_inlet*(1-y),0]),aux)
        return sols

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

    def explicit_cr(self, D, theta):
        raise(NotImplementedError("Use a quadratic jump penalisation!"))

class BinghamCoolingChannel_Tup(BinghamCoolingChannel):
    def __init__(self, baseN, non_isothermal, Re, Pe, Bn, Br, eps, temp_drop):
        super().__init__(baseN,non_isothermal,Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop)
        self.formulation = "T-u-p"
        self.vel_id = 1
        self.temp_id = 0

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), VectorSpaceBasis(constant=True)])

    def const_rel(self, D, theta):
        #Choose reference values for temperature dependent yield stress
        Bn_outlet_reference = 5.#50
        temp_drop_reference = 10.#20
        #Choose an increase of a factor of 10 in the viscosity
        alpha = 10.
        nu = self.viscosity(alpha, theta)
        tau = self.yield_stress(temp_drop_reference, Bn_outlet_reference, theta)
        #Papanastasiou
        S = (((tau/sqrt(inner(D,D)))*(1-exp(-(1./self.eps)*sqrt(inner(D,D))))) + 2.*nu)*D
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
        S0 = (((tau/sqrt(inner(D,D)))*(1-exp(-(1./self.eps)*sqrt(inner(D,D))))) + 2.*nu)*D0
        #Bercovier-Engelman
#        S0 = (2.*nu + tau/sqrt(self.eps**2 + inner(D,D)))*D0
        return S0

class BinghamCoolingChannel_LTup(BinghamCoolingChannel):
    def __init__(self, baseN, non_isothermal, Re, Pe, Bn, Br, eps, temp_drop):
        super().__init__(baseN,non_isothermal,Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop)
        self.formulation = "L-T-u-p"
        self.vel_id = 2
        self.temp_id = 1

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), VectorSpaceBasis(constant=True)])

    def const_rel(self, D, theta):
        #Choose reference values for temperature dependent yield stress
        Bn_outlet_reference = 5.#50
        temp_drop_reference = 10.#20
        #Choose an increase of a factor of 10 in the viscosity
        alpha = 10.
        nu = self.viscosity(alpha, theta)
        tau = self.yield_stress(temp_drop_reference, Bn_outlet_reference, theta)
        #Papanastasiou
        S = (((tau/sqrt(inner(D,D)))*(1-exp(-(1./self.eps)*sqrt(inner(D,D))))) + 2.*nu)*D
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
        S0 = (((tau/sqrt(inner(D,D)))*(1-exp(-(1./self.eps)*sqrt(inner(D,D))))) + 2.*nu)*D0
        #Bercovier-Engelman
#        S0 = (2.*nu + tau/sqrt(self.eps**2 + inner(D,D)))*D0
        return S0

class BinghamCoolingChannel_TSup(BinghamCoolingChannel):
    def __init__(self, baseN, non_isothermal, Re, Pe, Bn, Br, eps, temp_drop):
        super().__init__(baseN,non_isothermal,Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop)
        self.formulation = "T-S-u-p"
        self.vel_id = 2
        self.temp_id = 0

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), VectorSpaceBasis(constant=True)])

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
        #Bercovier-Engelman
        G = S - (2.*nu + tau/(sqrt(inner(D,D) + self.eps**2)))*D
        #Implicit Bercovier-Engelman
        G = (tau + 2.*nu*sqrt(inner(D,D) + self.eps**2))*D - sqrt(self.eps**2 + inner(D,D))*S
        G = (tau + 2.*nu*sqrt(inner(D,D)))*D - sqrt(self.eps**2 + inner(D,D))*S
        #Semismooth BE
#        G =  sqrt(inner(D - self.eps*S,D - self.eps*S))*(S- self.eps*D) - (tau + 2.*nu*sqrt(inner(D - self.eps*S,D - self.eps*S)))*(D - self.eps*S)
        #Semismooth MAX
#        G = PositivePart(sqrt(inner(S - self.eps*D,S - self.eps*D)) - tau)*(S - self.eps*D) - 2.*nu*(tau + PositivePart(sqrt(inner(S - self.eps*D,S - self.eps*D)) - tau))*(D - self.eps*S)
        #Alternative one
#        G = 2.*nu*(tau + 2.*nu*sqrt(inner(D,D)))*D - 0.5*(sqrt(inner(S,S)) - tau + sqrt((sqrt(inner(S,S)) - tau)**2 + self.eps**2))*S
        return G

    def interpolate_initial_guess(self, z):
        w_expr = self.bingham_poiseuille(z.ufl_domain(), self.Bn)
        z.sub(self.vel_id).interpolate(w_expr)
        ts = as_vector([10., 0.])
        z.sub(1).interpolate(ts)

class BinghamCoolingChannel_LTSup(BinghamCoolingChannel):
    def __init__(self, baseN, non_isothermal, Re, Pe, Bn, Br, eps, temp_drop):
        super().__init__(baseN,non_isothermal,Re=Re,Pe=Pe,Bn=Bn,Br=Br,eps=eps,temp_drop=temp_drop)
        self.formulation = "L-T-S-u-p"
        self.vel_id = 3
        self.temp_id = 1

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), Z.sub(3), VectorSpaceBasis(constant=True)])

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
        #Bercovier-Engelman
        G = S - (2.*nu + tau/(sqrt(inner(D,D) + self.eps**2)))*D
        #Implicit Bercovier-Engelman
        G = (tau + 2.*nu*sqrt(inner(D,D) + self.eps**2))*D - sqrt(self.eps**2 + inner(D,D))*S
        G = (tau + 2.*nu*sqrt(inner(D,D)))*D - sqrt(self.eps**2 + inner(D,D))*S
        #Semismooth BE
 #       G =  sqrt(inner(D - self.eps*S,D - self.eps*S))*(S- self.eps*D) - (tau + 2.*nu*sqrt(inner(D - self.eps*S,D - self.eps*S)))*(D - self.eps*S)
        #Semismooth MAX
#        G = PositivePart(sqrt(inner(S - self.eps*D,S - self.eps*D)) - tau)*(S - self.eps*D) - 2.*nu*(tau + PositivePart(sqrt(inner(S - self.eps*D,S - self.eps*D)) - tau))*(D - self.eps*S)
        #Alternative one
#        G = 2.*nu*(tau + 2.*nu*sqrt(inner(D,D)))*D - 0.5*(sqrt(inner(S,S)) - tau + sqrt((sqrt(inner(S,S)) - tau)**2 + self.eps**2))*S
        return G

    def interpolate_initial_guess(self, z):
        w_expr = self.bingham_poiseuille(z.ufl_domain(), self.Bn)
        z.sub(self.vel_id).interpolate(w_expr)
        ts = as_vector([10., 0.])
        z.sub(2).interpolate(ts)
