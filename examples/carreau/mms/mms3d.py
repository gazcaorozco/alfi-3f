from firedrake import *
from alfi_3f import *
import os

class CarreauMMS3D_up(NonNewtonianProblem_up):
    def __init__(self, baseN, diagonal=None, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
#        base = BoxMesh(self.baseN, self.baseN, self.baseN, 1., 1., 1.,
#                        distribution_parameters=distribution_parameters)
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/cube.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(0), self.exact_velocity(Z), "on_boundary")]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        w_expr = self.exact_velocity(z.function_space())
        z.sub(0).interpolate(w_expr)

    def const_rel(self, D):
        nr = (float(self.r) - 2.)/2.
        S = 2. * self.nu * pow(self.eps**2 + inner(D,D), nr)*D
        return S

    def quasi_norm(self, D):
        S = pow(self.eps + sqrt(inner(D,D)), (float(self.r)-2.)/2)*D
        return S

    def exact_velocity(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y, z) = X

        u = 8.*x*x*y*z*(x-1.)*(x-1.)*(y-1.)*(z-1.)*(y-z)
        v = -8.*x*y*y*z*(x-1.)*(y-1.)*(y-1.)*(z-1.)*(x-z)
        w = 8.*x*y*z*z*(x-1.)*(y-1.)*(z-1.)*(z-1.)*(x-y)
        u = replace(u, {X: 2.0 * X})
        v = replace(v, {X: 2.0 * X})
        w = replace(w, {X: 2.0 * X})
        exact_vel = as_vector([u, v, w])
        return exact_vel

    def exact_pressure(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y, z) = X
        p = (x - 0.5)**3 * sin(y + z)
        ## Alternative
        # p = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5*self.Re)
        #p = replace(p, {X: 2.0 * X})

        p = p - assemble(p*dx)
        return p

    def exact_stress(self, Z):
        D = sym(grad(self.exact_velocity(Z)))
        S = self.const_rel(D)
        return S

    def rhs(self, Z):
        u = self.exact_velocity(Z)
        p = self.exact_pressure(Z)

        D = sym(grad(u))
        S = self.exact_stress(Z)

        f1 = -div(S) + div(outer(u, u)) + grad(p)
        f2 = -div(u)
        return (f1, f2)

class CarreauMMS3D_Sup(NonNewtonianProblem_Sup):
    def __init__(self, baseN, diagonal=None, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/cube.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.exact_velocity(Z), "on_boundary")]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        w_expr = self.exact_velocity(z.function_space())
        z.sub(1).interpolate(w_expr)

    def quasi_norm(self, D):
        S = pow(self.eps + sqrt(inner(D,D)), (float(self.r)-2.)/2)*D
        return S

    def const_rel(self, S, D):
        nr = (float(self.r) - 2.)/2.
#        nr2 = (2. - float(self.r))/(2.*(float(self.r) - 1.))
        G = 2. * self.nu * pow(self.eps**2 + inner(D,D), nr)*D - S
        return G

    def exact_velocity(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y, z) = X

        u = 8.*x*x*y*z*(x-1.)*(x-1.)*(y-1.)*(z-1.)*(y-z)
        v = -8.*x*y*y*z*(x-1.)*(y-1.)*(y-1.)*(z-1.)*(x-z)
        w = 8.*x*y*z*z*(x-1.)*(y-1.)*(z-1.)*(z-1.)*(x-y)
        u = replace(u, {X: 2.0 * X})
        v = replace(v, {X: 2.0 * X})
        w = replace(w, {X: 2.0 * X})
        exact_vel = as_vector([u, v, w])
        return exact_vel

    def exact_pressure(self, Z):
        X = SpatialCoordinate(Z.mesh())
        (x, y, z) = X
        p = (x - 0.5)**3 * sin(y + z)
        ## Alternative
        # p = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5*self.Re)
        #p = replace(p, {X: 2.0 * X})

        p = p - assemble(p*dx)
        return p

    def exact_stress(self, Z):
        D = sym(grad(self.exact_velocity(Z)))
        nr = (float(self.r) - 2.)/2.
        S = 2. * self.nu * pow(self.eps**2 + inner(D,D), nr)*D
        return S

    def rhs(self, Z):
        u = self.exact_velocity(Z)
        p = self.exact_pressure(Z)

        D = sym(grad(u))
        S = self.exact_stress(Z)

        f1 = -div(S) + div(outer(u, u)) + grad(p)
        f2 = -div(u)
        return (f1, f2)
