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

    def explicit_cr(self, U):
        if "r" not in self.const_rel_params.keys(): r = Constant(2.0)
        if "eps" not in self.const_rel_params.keys(): eps = Constant(0.0)
        q = 2.* self.nu * pow(self.eps**2 + inner(U,U), (float(self.r)-2.)/2.0)*U
        return q

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

#### I think this is not needed...
#class BoussinesqProblem_Sup(NonIsothermalProblem_Sup):
#    def __init__(self, non_dimensional="rayleigh1", **const_rel_params):
#        super().__init__(**const_rel_params)
#        self.non_dimensional = non_dimensional
#        assert non_dimensional in {"rayleigh1", "rayleigh2", "grashof"}, "Invalid non-dimensional form of the system %s"%non_dimensional
#        assert all(elem in list(const_rel_params.keys()) for elem in ["Pr"]), "You must define the Prandtl number when working with an Oberbeck-Boussinesq approximation"
#        if self.non_dimensional in ["rayleigh1","rayleigh2"]:
#            assert "Ra" in self.const_rel_params.keys(), "You must define the Rayleigh number when working with the non-dimensional form %s"%self.non_dimensional
#        else:
#            assert "Gr" in self.const_rel_params.keys(), "You must define the Grashof number when working with the non-dimensional form %s"%self.non_dimensional
#
#class BoussinesqProblem_up(NonIsothermalProblem_up):
#    def __init__(self, non_dimensional="rayleigh1", **const_rel_params):
#        super().__init__(**const_rel_params)
#        self.formulation = "OB_Temp-u-p"
#        self.non_dimensional = non_dimensional
#        assert non_dimensional in {"rayleigh1", "rayleigh2", "grashof"}, "Invalid non-dimensional form of the system %s"%non_dimensional
#        assert all(elem in list(const_rel_params.keys()) for elem in ["Pr"]), "You must define the Prandtl number when working with an Oberbeck-Boussinesq approximation"
#        if self.non_dimensional in ["rayleigh1","rayleigh2"]:
#            assert "Ra" in self.const_rel_params.keys(), "You must define the Rayleigh number when working with the non-dimensional form %s"%self.non_dimensional
#        else:
#            assert "Gr" in self.const_rel_params.keys(), "You must define the Grashof number when working with the non-dimensional form %s"%self.non_dimensional
#
#
#class GeneralisedNonIsothermalProblem_Sup(NonIsothermalProblem_Sup):
#    def __init__(self, **const_rel_params):
#        super().__init__(**const_rel_params)
#        self.formulation = "General_Temp-S-u-p"
#
#    def const_rel(self, S, D, theta):
#        raise NotImplementedError
#
#    def const_rel_picard(self, S, D, S0, D0, theta):
#        raise NotImplementedError
#
#    def const_rel_picard_regularised(self, S, D, S0, D0, theta):
#        raise NotImplementedError
#
#class GeneralisedNonIsothermalProblem_up(NonIsothermalProblem_up):
#    def __init__(self, **const_rel_params):
#        super().__init__(**const_rel_params)
#        self.formulation = "General_Temp-u-p"
#
#    def const_rel(self, D, theta):
#        raise NotImplementedError
#
#    def const_rel_picard(self, D, D0, theta):
#        raise NotImplementedError
#
#class GeneralisedBoussinesqProblem_Sup(BoussinesqProblem_Sup):
#    def __init__(self, non_dimensional="rayleigh1", **const_rel_params, ):
#        super().__init__(non_dimensional, **const_rel_params)
#        self.formulation = "GeneralOB_Temp-S-u-p"
#
#    def const_rel(self, S, D, theta):
#        raise NotImplementedError
#
#    def const_rel_picard(self, S, D, S0, D0, theta):
#        raise NotImplementedError
#
#class GeneralisedBoussinesqProblem_up(BoussinesqProblem_up):
#    def __init__(self, non_dimensional="rayleigh1", **const_rel_params):
#        super().__init__(non_dimensional, **const_rel_params)
#        self.formulation = "GeneralOB_Temp-u-p"
#
#    def const_rel(self, D, theta):
#        raise NotImplementedError
#
#    def const_rel_picard(self, D, D0, theta):
#        raise NotImplementedError
