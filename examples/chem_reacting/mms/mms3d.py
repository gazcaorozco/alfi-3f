from firedrake import *
from alfi_3f import *
import os

class SynovialMMS3D(NonNewtonianProblem_Tup):
    def __init__(self, advect, rheol, **params):
        super().__init__(**params)
        self.rheol = rheol
        self.advect = advect

    def mesh(self, distribution_parameters):
        base = BoxMesh(self.baseN, self.baseN, self.baseN, 1., 1., 1.,
                        distribution_parameters=distribution_parameters)
#        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/cube.msh",
#                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.exact_solution(Z)[1], "on_boundary"),
               DirichletBC(Z.sub(0), self.exact_solution(Z)[0], "on_boundary")
            ]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        w_expr = self.exact_solution(z.function_space())[1]
        z.sub(1).interpolate(w_expr)

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
        (x, y, z) = X

        u = 8.*x*x*y*z*(x-1.)*(x-1.)*(y-1.)*(z-1.)*(y-z)
        v = -8.*x*y*y*z*(x-1.)*(y-1.)*(y-1.)*(z-1.)*(x-z)
        w = 8.*x*y*z*z*(x-1.)*(y-1.)*(z-1.)*(z-1.)*(x-y)
        exact_vel = as_vector([u, v, w])
        p = (x - 0.5)**3 * sin(y + z)
        p = p - assemble(p*dx)
        conc = sin(pi*x)*sin(pi*x)*sin(pi*y)*sin(pi*y)*(z-1)*(z-1)
        return (conc, exact_vel, p)

    def rhs(self, Z):
        (conc, u, p) = self.exact_solution(Z)
        D = sym(grad(u))
        S = self.const_rel(D, conc)
        q = self.const_rel_temperature(conc, grad(conc))
        f1 = -(1./self.Pe) * div(q) + div(u*theta)
        f2 = -(1./self.Re) * div(S) +  self.advect*div(outer(u, u)) + grad(p)
        f3 = -div(u)
        return (f1, f2, f3)
