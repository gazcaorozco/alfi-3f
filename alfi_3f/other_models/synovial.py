from firedrake import *

from alfi.transfer import *

from alfi_3f.solver import ConformingSolver

class SynovialSolver(ConformingSolver):

    def get_jacobian(self):
        if self.linearisation == "newton":
            J0 = derivative(self.F, self.z)
        elif self.linearisation in ["picard", "kacanov", "zarantonello"]:
            """ 'picard': full Picard
                'kacanov': 'Picard' only in the constitutive relation (not on the advective terms)
                'zarantonello': Iteration based on solving the system induced by the Riesz map
                """

            assert self.formulation_Tup, "This synovial model has only been implemented with a 3-field formulation c-u-p..."

            #Split variables for current solution and test function
            fields = self.split_variables(self.z)
            u = fields["u"]
            v = fields["v"]
            p = fields["p"]
            q = fields["q"]
            conc = fields.get("theta")
            conc_ = fields.get("theta_")

            D = sym(grad(u))

            #Split trial function
            w = TrialFunction(self.Z)
            fields0 = self.split_variables(w)
            u0 = fields0["u"]
            p0 = fields0["p"]
            conc0 = fields0.get("theta")

            D0 = sym(grad(u0))

            #Constitutive Relation
            S0 = self.problem.const_rel_picard(D, conc, D0)

            c_flux0 = self.problem.const_rel_temperature(conc, grad(conc0))  #TODO: Assumes the constitutive relation is linear in the second entry

            J0 = (
                self.gamma * inner(div(u0), div(v))*dx
                - p0 * div(v) * dx
                - div(u0) * q * dx
                + inner(dot(grad(conc0), u), conc_) * dx
                + self.advect * inner(dot(grad(u0), u), v) * dx
                + (1./self.Pe) * inner(c_flux0, grad(conc_)) * dx
                + (1./self.Re) * inner(S0, sym(grad(v))) * dx
            )


            if (self.linearisation == "kacanov"):
                #Add missing terms in the Newton linearisation of the convective terms
                J0 += self.advect * inner(dot(grad(u), u0), v)*dx
                J0 += inner(dot(grad(conc), u0), conc_) * dx

            #Terms arising from stabilisation
            if self.stabilisation_form_u is not None:
               J0 += derivative(self.stabilisation_form_u, self.z, w)

            if self.stabilisation_form_t is not None:
               J0 += derivative(self.stabilisation_form_t, self.z, w)

            if (self.linearisation == "zarantonello"):
            #TODO: How should one choose the Riesz map? Instead of this we could e.g. use the main part of the Kacanov operator
            #TODO: Should the SUPG, etc. terms be added here?
                J0 = (
    #                self.gamma * inner(div(u0), div(v))*dx
                    - p0 * div(v) * dx
                    - div(u0) * q * dx
                    + (1./self.Pe) * inner(grad(conc0), grad(conc_)) * dx
                    + (1./self.Re) * inner(sym(grad(u0)), sym(grad(v))) * dx
                )


        return J0

class SynovialSVSolver(SynovialSolver):

    def function_space(self, mesh, k):
        elec = FiniteElement("Lagrange", mesh.ufl_cell(), k)
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k-1)
        Z = FunctionSpace(mesh, MixedElement([elec,eleu,elep]))

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

class SynovialTHSolver(SynovialSVSolver):

    def function_space(self, mesh, k):
        elec = FiniteElement("Lagrange", mesh.ufl_cell(), k)
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Lagrange", mesh.ufl_cell(), k-1)
        Z = FunctionSpace(mesh, MixedElement([elec,eleu,elep]))

        return Z

    def get_transfers(self):

        V = self.Z.sub(self.velocity_id)
        Q = self.Z.sub(self.pressure_id)
        C = self.Z.sub(self.temperature_id)

        transfers = {V.ufl_element(): (prolong, restrict, inject),
                    Q.ufl_element(): (prolong, restrict, inject),
                     C.ufl_element(): (prolong, restrict, inject)}

        return transfers
