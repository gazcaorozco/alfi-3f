from firedrake import *
from alfi_3f import *
import os

class CarreauMMS_up(NonNewtonianProblem_up):
    def __init__(self, baseN, diagonal=None, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
#        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs_ = [DirichletBC(Z.sub(0), self.exact_velocity(Z), [1, 2, 3, 4])]
        return bcs_

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
        X = SpatialCoordinate(Z.ufl_domain())
        (x, y) = X
        u = (1./4.)*(-2 + x)**2 * x**2 * y * (-2 + y**2)
        v = -(1./4.)*x*(2 - 3*x + x**2)*y**2*(-4 + y**2)
        u = replace(u, {X: 2.0 * X})
        v = replace(v, {X: 2.0 * X})
        #Alternative
#        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
#        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        exact_vel = as_vector([u, v])
        return exact_vel

    def exact_pressure(self, Z):
        X = SpatialCoordinate(Z.ufl_domain())
        (x, y) = X
        ##This one has the wrong scaling
        p = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5/self.nu)
        p = replace(p, {X: 2.0 * X})
#        p = p - (-(1408./33075.) + 8./(5/self.nu))
        p = p - assemble(p*dx)

#        ## Alternative
#        # Taken from 'EFFECTS OF GRID STAGGERING ON NUMERICAL SCHEMES - Shih, Tan, Hwang'
#        f = x**4 - 2 * x**3 + x**2
#        g = y**4 - y**2
#
#        from ufl.algorithms.apply_derivatives import apply_derivatives
#        df = apply_derivatives(grad(f)[0])
#        dg = apply_derivatives(grad(g)[1])
#        ddg = apply_derivatives(grad(dg)[1])
#        dddg = apply_derivatives(grad(ddg)[1])
#
#        F = 0.2 * x**5 - 0.5 * x**4 + (1./3.) * x**3
#        F2 = 0.5 * f**2
#        p = (8.*self.nu) * (F * dddg + df*dg) + 64 * F2 * (g*ddg-dg**2)
#        p = p - assemble(p*dx)
        return p

    def exact_stress(self, Z):
        D = sym(grad(self.exact_velocity(Z)))
        S = self.const_rel(D)
        return S

    def rhs(self, Z):
        u_ex = self.exact_velocity(Z)
        p_ex = self.exact_pressure(Z)

        S_ex = self.exact_stress(Z)

        f1 = -div(S_ex) + div(outer(u_ex, u_ex)) + grad(p_ex)
        f2 = -div(u_ex)

        return f1, f2

class CarreauMMS_Sup(NonNewtonianProblem_Sup):
    def __init__(self, baseN, diagonal=None, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN

    def mesh(self, distribution_parameters):
#        base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",
                    distribution_parameters=distribution_parameters)
        return base

    def bcs(self, Z):
        bcs = [DirichletBC(Z.sub(1), self.exact_velocity(Z), [1, 2, 3, 4])]
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
        G = S - 2. * self.nu * pow(self.eps**2 + inner(D,D), nr)*D
        return G

    def exact_velocity(self, Z):
        X = SpatialCoordinate(Z.ufl_domain())
        (x, y) = X
        u = (1./4.)*(-2 + x)**2 * x**2 * y * (-2 + y**2)
        v = -(1./4.)*x*(2 - 3*x + x**2)*y**2*(-4 + y**2)
        u = replace(u, {X: 2.0 * X})
        v = replace(v, {X: 2.0 * X})
        #Alternative
#        u = 2.*y*sin(pi*x)*sin(pi*y)*(x**2 - 1.) + pi*sin(pi*x)*cos(pi*y)*(x**2 -1.)*(y**2-1.)
#        v = -2.*x*sin(pi*x)*sin(pi*y)*(y**2 - 1.) - pi*cos(pi*x)*sin(pi*y)*(x**2 - 1.)*(y**2 - 1.)
        exact_vel = as_vector([u, v])
        return exact_vel

    def exact_pressure(self, Z):
        X = SpatialCoordinate(Z.ufl_domain())
        (x, y) = X
        p = -(1./128.)*(-2+x)**4*x**4*y**2*(8-2*y**2+y**4)+(x*y*(-15*x**3+3*x**4+10*x**2*y**2+20*(-2+y**2)-30*x*(-2+y**2)))/(5/self.nu)
        p = replace(p, {X: 2.0 * X})
        p = p - assemble(p*dx)

#        ## Alternative
#        # Taken from 'EFFECTS OF GRID STAGGERING ON NUMERICAL SCHEMES - Shih, Tan, Hwang'
#        f = x**4 - 2 * x**3 + x**2
#        g = y**4 - y**2
#
#        from ufl.algorithms.apply_derivatives import apply_derivatives
#        df = apply_derivatives(grad(f)[0])
#        dg = apply_derivatives(grad(g)[1])
#        ddg = apply_derivatives(grad(dg)[1])
#        dddg = apply_derivatives(grad(ddg)[1])
#
#        F = 0.2 * x**5 - 0.5 * x**4 + (1./3.) * x**3
#        F2 = 0.5 * f**2
#        p = (8.*self.nu) * (F * dddg + df*dg) + 64 * F2 * (g*ddg-dg**2)
#        p = p - assemble(p*dx)
        return p

    def exact_stress(self, Z):
        D = sym(grad(self.exact_velocity(Z)))
        nr = (float(self.r) - 2.)/2.
        S = 2. * self.nu * pow(self.eps**2 + inner(D,D), nr)*D
        return S

    def rhs(self, Z):
        u = self.exact_velocity(Z)
        p = self.exact_pressure(Z)

        S = self.exact_stress(Z)

        f1 = -div(S) + div(outer(u, u)) + grad(p)
        f2 = -div(u)

        return f1, f2
