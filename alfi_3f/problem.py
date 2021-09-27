from firedrake import *
from alfi.bary import BaryMeshHierarchy, bary
from alfi.problem import NavierStokesProblem

class NonNewtonianProblem(NavierStokesProblem):
    def __init__(self, **const_rel_params):
        self.const_rel_params = const_rel_params
        for param in const_rel_params.keys():
            setattr(self, param, const_rel_params[param])

    def nullspace(self, Z):
        raise NotImplementedError

    def const_rel(self, *args):
        raise NotImplementedError

    def const_rel_picard(self, *args):
        raise NotImplementedError

class NonNewtonianProblem_Sup(NonNewtonianProblem):
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "S-u-p"

    def nullspace(self, Z):
        if self.has_nullspace():
            MVSB = MixedVectorSpaceBasis
            return MVSB(Z, [Z.sub(0), Z.sub(1), VectorSpaceBasis(constant=True)])
        else:
            return None

class NonNewtonianProblem_up(NonNewtonianProblem):
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "u-p"

    def nullspace(self, Z):
        if self.has_nullspace():
            MVSB = MixedVectorSpaceBasis
            return MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        else:
            return None

class NonNewtonianProblem_TSup(NonNewtonianProblem):
    """For problems coupled with something else (like a temperature field)"""
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "T-S-u-p"

    def nullspace(self, Z):
        if self.has_nullspace():
            MVSB = MixedVectorSpaceBasis
            return MVSB(Z, [Z.sub(0), Z.sub(1), Z.sub(2), VectorSpaceBasis(constant=True)])
        else:
            return None

class NonNewtonianProblem_Tup(NonNewtonianProblem_Sup):
    """For problems coupled with something else (like a temperature field)"""
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "T-u-p"

class NonNewtonianProblem_LSup(NonNewtonianProblem_TSup):
    """For problems coupled with another stress variable (like some lifting or the symmetric gradient)"""
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "L-S-u-p"

class NonNewtonianProblem_Lup(NonNewtonianProblem_Sup):
    """For problems coupled with another stress variable (like some lifting or the symmetric gradient)"""
    def __init__(self, **const_rel_params):
        super().__init__(**const_rel_params)
        self.formulation = "L-u-p"

class NonIsothermalProblem_Sup(NonNewtonianProblem_TSup):
    def const_rel_temperature(self, *args):
        raise NotImplementedError

class NonIsothermalProblem_up(NonNewtonianProblem_Tup):
    def const_rel_temperature(self, *args):
        raise NotImplementedError

