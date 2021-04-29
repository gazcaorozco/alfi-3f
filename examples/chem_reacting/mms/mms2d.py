from firedrake import *
from alfi_3f import *
import os

class SynovialMMS2D(NonNewtonianProblem_Tup):
    def __init__(self, advect, rheol, **params):
        super().__init__(**params)
        self.rheol = rheol
        self.advect = advect

    def mesh(self, distribution_parameters):
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

    def const_rel(self, D, c):
        nu = self.visc(D, c)
        S = 2.*nu*D
        return S

    def const_rel_picard(self,D, c, D0):
        nu = self.visc(D, c)
        S0 = 2.*nu*D0
        return S0

    def const_rel_temperature(self, c, gradc):
        kappa = Constant(1.0)
        q = kappa*gradc
        return q

    def visc(self, D, c):
        if self.rheol == "synovial":
            #Synovial shear-thinning
            nu = pow(self.beta + (1./self.eps)*inner(D,D), 0.5*(exp(-self.alpha*c) - 1.0))
        elif self.rheol == "power-law":
            """ alpha is the power-law parameter (usually 'r' or 'p')"""
            n_exp = (self.alpha-2)/(2.)
            nu = pow(self.beta + (1./self.eps)*inner(D,D), n_exp)
        elif self.rheol == "newtonian":
            nu = Constant(1.)
        return nu

    def plaw_exponent(self, c):
        if self.rheol == "synovial":
            nn = 0.5*(exp(-self.alpha*c) - 1.0)
        elif self.rheol == "power-law":
            nn = (self.alpha-2)/(2.)
        elif self.rheol == "newtonian":
            nn = Constant(0.)
        return nn

    def exact_solution(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y) = X
        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        exact_vel = as_vector([u, v])
        p = y**2 - x**2
        p = p - assemble(p*dx)
        conc = x**2 - y**4
        return (conc, exact_vel, p)

    def rhs(self, Z):
        (conc, u, p) = self.exact_solution(Z)
        D = sym(grad(u))
        S = self.const_rel(D, conc)
        q = self.const_rel_temperature(conc, grad(conc))
        f1 = -(1./self.Pe) * div(q) + div(u*conc)
        f2 = -(1./self.Re) * div(S) +  self.advect*div(outer(u, u)) + grad(p)
        f3 = -div(u)
        return (f1, f2, f3)
