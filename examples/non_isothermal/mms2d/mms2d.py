from firedrake import *
from implcfpc import *
import os

class OBCavityMMS_up(GeneralisedBoussinesqProblem_up):
    def __init__(self, baseN, temp_dependent="viscosity-conductivity", diagonal=None, non_dimensional="rayleigh1", **params):
        super().__init__(non_dimensional=non_dimensional, **params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.temp_dependent = temp_dependent

    def mesh(self, distribution_parameters):
#        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.exact_solution(Z)[1], [1, 2, 3, 4]),
               DirichletBC(Z.sub(0), self.exact_solution(Z)[0], [1, 2, 3, 4]) 
            ]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        w_expr = self.exact_solution(z.function_space())[1]
        z.sub(1).interpolate(w_expr)
        w_expr0 = self.exact_solution(z.function_space())[0]
        z.sub(0).interpolate(w_expr0)

    def const_rel(self, D, theta):    
        nr = (float(self.r) - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D 
        return S

    def const_rel_picard(self,D,D0,theta):
        nr = (float(self.r) - 2.)/2.
        K = self.viscosity(theta)
        S0 = K*pow(inner(D,D),nr)*D0 
        return S0

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
        
    def exact_solution(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y) = X
        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        exact_vel = as_vector([u, v])
        p = y**2 - x**2
        theta = x**2 - y**4
        return (theta, exact_vel, p)

    def rhs(self, Z):
        if not("Theta" in list(self.const_rel_params.keys())): self.Theta = Constant(0.)
        if not("Di" in list(self.const_rel_params.keys())): self.Di = Constant(0.)


        (theta, u, p) = self.exact_solution(Z)
        g = Constant((0, 1))
        D = sym(grad(u))
        S = self.const_rel(D, theta)
        q = self.const_rel_temperature(theta, grad(theta))
        f1 = -div(q) + div(u*theta) + self.Di*(theta + self.Theta)*dot(g, u) - (self.Di/self.Ra)*inner(S,D)
        f2 = -self.Pr*div(S) + div(outer(u, u)) + grad(p) - (self.Ra*self.Pr)*theta*g
        f3 = -div(u)
        return (f1, f2, f3)

class OBCavityMMS_Sup(GeneralisedBoussinesqProblem_Sup):
    def __init__(self, baseN, temp_dependent="viscosity-conductivity", diagonal=None, non_dimensional="rayleigh1", **params):
        super().__init__(non_dimensional=non_dimensional, **params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.temp_dependent = temp_dependent

    def mesh(self, distribution_parameters):
#        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(2), self.exact_solution(Z)[2], [1, 2, 3, 4]),
               DirichletBC(Z.sub(0), self.exact_solution(Z)[0], [1, 2, 3, 4]) 
            ]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        w_expr = self.exact_solution(z.function_space())[2]
        z.sub(2).interpolate(w_expr)
        w_expr0 = self.exact_solution(z.function_space())[0]
        z.sub(0).interpolate(w_expr0)

    def const_rel(self, S, D, theta):    
        nr = (float(self.r) - 2.)/2.
        nr2 = (2. - float(self.r))/(2.*(float(self.r) - 1.))
        K = self.viscosity(theta)
        G = S - K*pow(inner(D,D),nr)*D 
#        G = D - (1./(2.*K))*pow(inner(S/(2.*K),S/(2.*K)),nr2)*S
        return G

    def const_rel_picard(self,S,D,S0,D0,theta):
        nr = (float(self.r) - 2.)/2.
        nr2 = (2. - float(self.r))/(2.*(float(self.r) - 1.))
        K = self.viscosity(theta)
        G0 = S0 - K*pow(inner(D,D),nr)*D0 
        G0 = D0 - (1./(2.*self.nu))*pow(inner(S/(2.*self.nu),S/(2.*self.nu)),nr2)*S0
        return G0

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

    def exact_solution(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y) = X
        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        exact_vel = as_vector([u, v])
        p = y**2 - x**2
        theta = x**2 - y**4
        K = self.viscosity(theta)
        D = sym(grad(exact_vel))
        nr = (float(self.r) - 2.)/2.
        S = K*pow(inner(D,D),nr)*D 
        return (theta, S, exact_vel, p)

    def rhs(self, Z):
        if not("Theta" in list(self.const_rel_params.keys())): self.Theta = Constant(0.)
        if not("Di" in list(self.const_rel_params.keys())): self.Di = Constant(0.)

        (theta, S, u, p) = self.exact_solution(Z)
        g = Constant((0, 1))
        D = sym(grad(u))
        q = self.const_rel_temperature(theta, grad(theta))
        f1 = -div(q) + div(u*theta) + self.Di*(theta + self.Theta)*dot(g, u) - (self.Di/self.Ra)*inner(S,D)
        f2 = -self.Pr*div(S) + div(outer(u, u)) + grad(p) - (self.Ra*self.Pr)*theta*g
        f3 = -div(u)
        return (f1, f2, f3)
