from firedrake import *
from alfi_3f import *
import os

class OBCavityMMS(NonNewtonianProblem):
    def __init__(self, temp_dependent="viscosity-conductivity", baseN=40, **const_rel_params):
        super().__init__(**const_rel_params)
        self.temp_dependent = temp_dependent
        self.baseN = baseN
        self.vel_id = None
        self.temp_id = None

    def mesh(self, distribution_parameters):
        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters)
#        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",
#                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(self.vel_id), self.exact_solution(Z)[1], 1),
               DirichletBC(Z.sub(self.vel_id), self.exact_solution(Z)[1], 2),
               DirichletBC(Z.sub(self.vel_id), self.exact_solution(Z)[1], 3),
               DirichletBC(Z.sub(self.vel_id), self.exact_solution(Z)[1], 4),
               DirichletBC(Z.sub(self.temp_id), self.exact_solution(Z)[0], [1, 2, 3, 4])
            ]
        return bcs

    def interpolate_initial_guess(self, z):
        w_expr = self.exact_solution(z.function_space())[1]
        z.sub(self.vel_id).interpolate(w_expr)
        w_expr0 = self.exact_solution(z.function_space())[0]
        z.sub(self.temp_id).interpolate(w_expr0)

    def const_rel_temperature(self, theta, gradtheta):
        kappa = self.heat_conductivity(theta)
        q = kappa*gradtheta
        return q

    def viscosity(self, theta):
        #Taken from Almonacid-Gatica-2020
        if self.temp_dependent in ["viscosity","viscosity-conductivity"]:
            nu = exp(-0.25*theta)
        else:
            nu = Constant(1.)
        return nu

    def heat_conductivity(self, theta):
        if self.temp_dependent == "viscosity-conductivity":
            kappa = exp(4*theta)
        else:
            kappa = Constant(1.)
        return kappa

    def explicit_cr(self, D, theta):
        nr = (float(self.r) - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D
        return S

    def quasi_norm(self, D):
        S = pow(sqrt(inner(D,D)), (float(self.r)-2.)/2)*D
        return S

    def exact_solution(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y) = X
        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        #u = (1./4.)*(-2 + x)**2 * x**2 * y * (-2 + y**2) #For a 2x2 RectangleMesh
        #v = -(1./4.)*x*(2 - 3*x + x**2)*y**2*(-4 + y**2)
        exact_vel = as_vector([u, v])
        p = y**2 - x**2
        #p = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5/0.5)
        #p = p - 0.25*assemble(p*dx)
        theta = x**2 + y**4 + cos(x)
        return (theta, exact_vel, p)

    def exact_stress(self, Z):
        (theta, u, _) = self.exact_solution(Z)
        return self.explicit_cr(sym(grad(u)), theta)

    def rhs(self, Z):
        if not("Theta" in list(self.const_rel_params.keys())): self.Theta = Constant(0.)
        if not("Di" in list(self.const_rel_params.keys())): self.Di = Constant(0.)

        (theta, u, p) = self.exact_solution(Z)
        g = Constant((0, 1))
        D = sym(grad(u))
        S = self.exact_stress(Z)
        q = self.const_rel_temperature(theta, grad(theta))
        #f1 = -div(q) + div(u*theta) + self.Di*(theta + self.Theta)*dot(g, u) - (self.Di/self.Ra)*inner(S,D)
        f1 = -div(grad(theta)) #+ div(u*theta)
#        f1 = -div(grad(theta)) + div(u*theta) + self.Di*(theta + self.Theta)*dot(g, u) - (self.Di/self.Ra)*inner(D,D)
        f2 = -div(D) + grad(p) #- theta*g
        #f2 = -self.Pr*div(D) + div(outer(u, u)) + grad(p) - (self.Ra*self.Pr)*theta*g
        f3 = -div(u)
        return (f1, f2, f3)

class OBCavityMMS_Tup(OBCavityMMS):
    def __init__(self, temp_dependent="viscosity-conductivity", baseN=40, **params):
        super().__init__(temp_dependent=temp_dependent, **params)
        self.formulation = "T-u-p"
        self.vel_id = 1
        self.temp_id = 0

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), VectorSpaceBasis(constant=True)])

    def const_rel(self, D, theta): #Make sure r=2 if using HDiv formulation
        nr = (float(self.r) - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D
        return S

class OBCavityMMS_LTup(OBCavityMMS):
    def __init__(self, temp_dependent="viscosity-conductivity", baseN=40, **params):
        super().__init__(temp_dependent=temp_dependent, **params)
        self.formulation = "L-T-u-p"
        self.vel_id = 2
        self.temp_id = 1

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), VectorSpaceBasis(constant=True)])

    def const_rel(self, D, theta):
        nr = (float(self.r) - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D
        return S

class OBCavityMMS_TSup(OBCavityMMS):
    def __init__(self, temp_dependent="viscosity-conductivity", baseN=40, **params):
        super().__init__(temp_dependent=temp_dependent, **params)
        self.formulation = "T-S-u-p"
        self.vel_id = 2
        self.temp_id = 0

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), VectorSpaceBasis(constant=True)])

    def const_rel(self, S, D, theta):
        nr = (float(self.r) - 2.)/2.
        nr_x = (2. - float(self.r)) / (2.*(float(self.r) - 1.))
        K = self.viscosity(theta)
        G = S - K*pow(inner(D,D),nr)*D
        #G = D - (1./K) * pow(inner(D,D) / K,nr)*S
        return G

class OBCavityMMS_LTSup(OBCavityMMS):
    def __init__(self, temp_dependent="viscosity-conductivity", baseN=40, **params):
        super().__init__(temp_dependent=temp_dependent, **params)
        self.formulation = "L-T-S-u-p"
        self.vel_id = 3
        self.temp_id = 1

    def nullspace(self, Z):
        MVSB = MixedVectorSpaceBasis
        return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), Z.sub(3), VectorSpaceBasis(constant=True)])

    def const_rel(self, S, D, theta):
        nr = (float(self.r) - 2.)/2.
        nr_x = (2. - float(self.r)) / (2.*(float(self.r) - 1.))
        K = self.viscosity(theta)
        G = S - K*pow(inner(D,D),nr)*D
#        G = D - (1./K) * pow(inner(D,D) / K,nr)*S
        return G
