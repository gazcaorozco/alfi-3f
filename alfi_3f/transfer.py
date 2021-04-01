from firedrake import *
from firedrake.dmhooks import get_appctx

class DGInjection(object):

    def __init__(self):
        self._DG_inv_mass = {}
        self._mixed_mass = {}
        self._tmp_function = {}


    def DG_inv_mass(self, DG):
        """
        Inverse DG mass matrix
        :arg DG: the DG space
        :returns: A PETSc Mat.
        """
        key = DG.dim()
        try:
            return self._DG_inv_mass[key]
        except KeyError:
            assert DG.ufl_element().family() == "Discontinuous Lagrange"
            M = assemble(Tensor(inner(TestFunction(DG), TrialFunction(DG))*dx).inv)
            return self._DG_inv_mass.setdefault(key, M.petscmat)

    def mixed_mass(self, V_A, V_B):
        """
        Compute the mixed mass matrix of two function spaces.
        :arg V_A: the donor space
        :arg V_B: the target space
        :returns: A PETSc Mat.
        """
        from firedrake.supermeshing import assemble_mixed_mass_matrix
        key = (V_A.dim(), V_B.dim())
        try:
            return self._mixed_mass[key]
        except KeyError:
            M = assemble_mixed_mass_matrix(V_A, V_B)
            return self._mixed_mass.setdefault(key, M)

    def tmp_function(self, V):
        """
        Construct a temporary work function on a function space.
        """
        key = V.dim()
        try:
            return self._tmp_function[key]
        except KeyError:
            u = Function(V)
            return self._tmp_function.setdefault(key, u)

    def inject(self, fine, coarse):
        V_fine = fine.function_space()
        V_coarse = coarse.function_space()

        mixed_mass = self.mixed_mass(V_fine, V_coarse)
        mass_inv = self.DG_inv_mass(V_coarse)
        tmp = self.tmp_function(V_coarse)

        # Can these be bundled into one with statement?
        with fine.dat.vec_ro as src, tmp.dat.vec_wo as rhs:
            mixed_mass.mult(src, rhs)
        with tmp.dat.vec_ro as rhs, coarse.dat.vec_wo as dest:
            mass_inv.mult(rhs, dest)

    def form(self, V):
        a = get_appctx(V.dm).J
        return a

    def prolong(self, fine, coarse):
        V_fine = fine.function_space()
        V_coarse = coarse.function_space()

        mixed_mass = self.mixed_mass(V_fine, V_coarse)
        mass_inv = self.DG_inv_mass(V_coarse)
        tmp = self.tmp_function(V_coarse)

        # Can these be bundled into one with statement?
        with fine.dat.vec_ro as src, tmp.dat.vec_wo as rhs:
            mixed_mass.mult(src, rhs)
        with tmp.dat.vec_ro as rhs, coarse.dat.vec_wo as dest:
            mass_inv.mult(rhs, dest)

#        def energy_norm(S): #This is not working for some reason...
#            return assemble(action(action(self.form(S.function_space()), S), S))

#        warning("Stress - From mesh %i to %i" % (coarse.function_space().dim(), fine.function_space().dim()))
#        warning("Stress - Ratio |fine|/|coarse|:   %f"%(norm(fine)/norm(coarse)))#Print norms

#        warning("Stress -  norm ratio: %.2f" % (energy_norm(fine)/energy_norm(coarse)))


#    prolong = inject

class NullTransfer(object):
    def transfer(self, src, dest):
        with dest.dat.vec_wo as x:
            x.set(numpy.nan)

    inject = transfer
    prolong = transfer
    restrict = transfer
