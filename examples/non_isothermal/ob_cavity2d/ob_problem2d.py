from firedrake import *
from alfi_3f import *

class OBCavity2D(NonNewtonianProblem):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(**params)
        if diagonal is None:
            diagonal = "left"
        self.diagonal = diagonal
        self.baseN = baseN
        self.temp_bcs = temp_bcs
        self.temp_dependent = temp_dependent
        self.unstructured = unstructured
        self.vel_id = None
        self.temp_id = None

    def mesh(self, distribution_parameters):
        if self.unstructured:
            if self.baseN == 0:
                base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square_hdiv.msh",distribution_parameters=distribution_parameters)
            else:
                base = Mesh(os.path.dirname(os.path.abspath(__file__)) + "/square.msh",distribution_parameters=distribution_parameters)
        else:
            base = RectangleMesh(self.baseN, self.baseN, 1, 1, distribution_parameters=distribution_parameters, diagonal = self.diagonal)
        return base

    def bcs(self, Z):
        if self.temp_bcs == "down-up":
            bcs = [DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 1),
                   DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 2),
                   DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 3),
                   DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 4),
                   DirichletBC(Z.sub(self.temp_id), Constant(1.0), 3),               #Hot (bottom) - Cold (top)
                   DirichletBC(Z.sub(self.temp_id), Constant(0.0), 4),
                ]
        else:
            bcs = [DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 1),
                   DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 2),
                   DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 3),
                   DirichletBC(Z.sub(self.vel_id), Constant((0., 0.)), 4),
                   DirichletBC(Z.sub(self.temp_id), Constant(1.0), 1),               #Hot (left) - Cold (right)
                   DirichletBC(Z.sub(self.temp_id), Constant(0.0), 2),
                ]
        return bcs

    def has_nullspace(self): return True

    def interpolate_initial_guess(self, z):
        (x,y) = SpatialCoordinate(z.ufl_domain())
        w_expr = as_vector([x*x + 5, 3.*y])
        z.sub(self.vel_id).interpolate(w_expr)

    def const_rel_temperature(self, theta, gradtheta):
        kappa = self.heat_conductivity(theta)
        q = kappa*gradtheta
        return q

    def viscosity(self, theta):
        if self.temp_dependent in ["viscosity","viscosity-conductivity"]:
            nu = 0.5 + 0.5*theta + theta**2
            nu = exp(-0.1*theta)
        else:
            nu = Constant(1.)
        return nu

    def heat_conductivity(self, theta):
        if self.temp_dependent == "viscosity-conductivity":
            kappa = 0.5 + 0.5*theta + theta**2
        else:
            kappa = Constant(1.)
        return kappa

    def explicit_cr(self, D, theta):
        nr = (float(self.r) - 2.)/2.
        K = self.viscosity(theta)
        S = K*pow(inner(D,D),nr)*D
        return S

class OBCavity2D_Tup(OBCavity2D):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(baseN, temp_bcs, temp_dependent, diagonal, unstructured, **params)
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

class OBCavity2D_LTup(OBCavity2D):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(baseN, temp_bcs, temp_dependent, diagonal, unstructured, **params)
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

class OBCavity2D_TSup(OBCavity2D):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(baseN, temp_bcs, temp_dependent, diagonal, unstructured, **params)
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
        #G = S - K*pow(inner(D,D),nr)*D
        G = D - (1./K) * pow(inner(D,D) / K, nr_x)*S
        return G

class OBCavity2D_LTSup(OBCavity2D):
    def __init__(self, baseN, temp_bcs="left-right", temp_dependent="viscosity", diagonal=None, unstructured=False, **params):
        super().__init__(baseN, temp_bcs, temp_dependent, diagonal, unstructured, **params)
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
#        G = D - (1./K) * pow(inner(D,D) / K, nr_x)*S
        return G
