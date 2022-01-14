from firedrake import *

from alfi.transfer import *

from alfi_3f.solver import NonNewtonianSolver

class OldroydBSolver(NonNewtonianSolver):

    def strong_residual(self, z, wind, rhs=None, type="momentum"):
        raise NotImplementedError

    def residual(self):

        assert self.formulation_Sup, "Viscoelastic OldroydB Model has only been implemted with a 3-field formulation B-u-p..."

        #Define functions and test functions
        fields = self.split_variables(self.z)
        u = fields["u"]
        v = fields["v"]
        p = fields["p"]
        q = fields["q"]
        B = fields.get("S")
        BT = fields.get("ST")

        D = sym(grad(u))
        L = grad(u)


        F = (
            2. * self.nu * inner(D, sym(grad(v))) * dx
            + self.G * inner(B - Identity(self.tdim), sym(grad(v))) * dx
            + self.gamma * inner(div(u), div(v)) * dx
            + self.advect * inner(dot(grad(u), u), v) * dx
            #- self.advect * inner(outer(u,u), sym(grad(v))) * dx
            - p * div(v) * dx
            - div(u) * q * dx
            + inner(dot(grad(B), u), BT) * dx
            - inner(dot(L, B), BT) * dx
            - inner(dot(B, transpose(L)), BT) * dx
            + (1./self.tau) * inner(B - Identity(self.tdim), BT) * dx
            + self.delta * inner(grad(B), grad(BT)) * dx
        )

        return F


class OldroydBSVSolver(OldroydBSolver):

    def function_space(self, mesh, k): #TODO: Should think about whether stresses can be taken traceless here...
        if self.tdim == 2:
            eles = VectorElement("CG", mesh.ufl_cell(), k-1, dim=3)
        else:
            eles = VectorElement("CG", mesh.ufl_cell(), k-1, dim=6)
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k-1)
        Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))

        return Z

    def get_transfers(self):

        V = self.Z.sub(self.velocity_id)
        Q = self.Z.sub(self.pressure_id)

        if self.stabilisation_type_u in ["burman", None]:
            qtransfer = NullTransfer()
        elif self.stabilisation_type_u in ["gls", "supg"]:
            qtransfer = EmbeddedDGTransfer(Q.ufl_element())
        else:
            raise ValueError("Unknown stabilisation")
        self.qtransfer = qtransfer

        if self.hierarchy == "bary":
            vtransfer = SVSchoeberlTransfer((self.nu, self.gamma), self.tdim, self.hierarchy)
            self.vtransfer = vtransfer

        transfers = {V.ufl_element(): (vtransfer.prolong, vtransfer.restrict if self.restriction else restrict, inject),
                    Q.ufl_element(): (prolong, restrict, qtransfer.inject)}

        return transfers

    def configure_patch_solver(self, opts):
        patchlu3d = "mkl_pardiso" if self.use_mkl else "umfpack"
        patchlu2d = "petsc"
        opts["patch_pc_patch_sub_mat_type"] = "seqaij"
        opts["patch_sub_pc_factor_mat_solver_type"] = patchlu3d if self.tdim > 2 else patchlu2d

    def distribution_parameters(self):
        return {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}
