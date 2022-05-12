from firedrake import *

from alfi.transfer import *

from alfi_3f.solver import ConformingSolver

import copy

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
                J0 = (
    #                self.gamma * inner(div(u0), div(v))*dx
                    - self.zdamping * p0 * div(v) * dx
                    - div(u0) * q * dx
#                    + (1./self.Pe) * inner(grad(conc0), grad(conc_)) * dx
                    + inner(grad(conc0), grad(conc_)) * dx
#                    + (1./self.Re) * inner(sym(grad(u0)), sym(grad(v))) * dx
                    + inner(sym(grad(u0)), sym(grad(v))) * dx
                )


        return J0

    def get_parameters(self):
#================================================================================================
#       MG + Vanka (linear solver)
#================================================================================================
        mg_vanka = {"ksp_type": "cg",
                    "ksp_monitor_true_residual": None,
                    "ksp_converged_reason": None,
                    "ksp_max_it": 100,
                    "pc_type": "fieldsplit",
                    "pc_fieldsplit_0_fields": "0",
                    "pc_fieldsplit_1_fields": "1,2",
                    "pc_fieldsplit_type": "additive",
                    "fieldsplit_0_": {
                        "ksp_type": "preonly",
                        "pc_type": "mg",
                        "pc_mg_cycle_type": "v",
                        "pc_mg_type": "multiplicative",
                        "mg_levels": {
                            "ksp_type": "chebyshev",
                            "ksp_max_it": 3,
                            "ksp_convergence_test": "skip",
                            "pc_type": "jacobi",
                        },
                    },
                    "fieldsplit_1_": {
                        "ksp_type": "preonly",
                        "pc_type": "mg",
                        "pc_mg_cycle_type": "v",
                        "pc_mg_type": "multiplicative",
                        "mg_levels": {
                            "ksp_type": "chebyshev",
                            "ksp_chebyshev_esteig": "0,0,0,0",
                            "ksp_chebyshev_eigenvalues": "1.5,8",
                            "ksp_max_it": 3,
                            "ksp_convergence_test": "skip",
                            "pc_type": "python",
                            "pc_python_type": "firedrake.PatchPC",
                            "patch_pc_patch_save_operators": True,
                            "patch_pc_patch_partition_of_unity": False,
                            "patch_pc_patch_sub_mat_type":
                            "seqdense", "patch_pc_patch_construct_dim": 0,
                            "patch_pc_patch_vanka_dim": 0,
                            "patch_pc_patch_construct_type": "vanka",
                            "patch_pc_patch_exclude_subspaces": 1,
                            "patch_pc_patch_local_type": "additive",
                            "patch_pc_patch_precompute_element_tensors": True,
                            "patch_pc_patch_symmetrise_sweep": False,
                            "patch_sub_ksp_type": "preonly",
                            "patch_sub_pc_type": "lu",
                            "patch_sub_pc_factor_shift_type": "nonzero",
                        },
                        "mg_coarse" : {
                            "mg_coarse_ksp_type": "richardson",
                            "mg_coarse_pc_type": "python",
                            "mg_coarse_pc_python_type": "firedrake.AssembledPC",
                            "mg_coarse_assembled_pc_type": "lu",
                            "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
                        }
                    },
        }
#================================================================================================
#       Newton/Zarantonello \ LU
#================================================================================================
        newton_lu = {
                "snes_type": "newtonls",
                "snes_monitor": None,
                "snes_max_it": 1000,
                "snes_linesearch_type": "basic" if (self.linearisation == "zarantonello") else "basic",
                "snes_linesearch_maxstep": 1.0,
                "snes_linesearch_damping": float(self.zdamping) if (self.linearisation == "zarantonello") else 1.0,
                "snes_linesearch_monitor": None,
                "snes_converged_reason": None,
                "ksp_type": "fgmres",
#                "ksp_monitor_true_residual": None,
                "snes_divtol": "1e14",
    #            "snes_view": None,
                "snes_rtol": 1.0e-8,
                "snes_atol": 1.0e-8,
                "snes_stol": 0,
                "mat_type": "aij",
                "ksp_max_it": 1,
                "ksp_convergence_test": "skip",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_14": 300,#,200,
                 "mat_mumps_icntl_24": 1,
                # "mat_mumps_icntl_25": 1,
                 "mat_mumps_cntl_1": 0.001,
                 "mat_mumps_cntl_3": 0.0001,
        }
#================================================================================================
#       Zarantonello + EW \ Vanka
#================================================================================================
        zar_ew_vanka = {
                "mat_type": "aij",
                "snes_type": "newtonls",
                "snes_monitor": None,
                "snes_max_it": 1000,
                "snes_linesearch_type": "basic",
                "snes_linesearch_maxstep": 1.0,
                "snes_linesearch_damping": float(self.zdamping) if (self.linearisation == "zarantonello") else 1.0,
                "snes_linesearch_monitor": None,
                "snes_converged_reason": None,
                "snes_max_linear_solve_fail": 10,
                "snes_ksp_ew": None,
                "snes_force_iteration": 1,
                "snes_divtol": "1e14",
    #            "snes_view": None,
                "snes_rtol": 1.0e-8, #1.0e-7
                "snes_atol": 1.0e-8, #1.0e-7
                "snes_stol": 0,
        }
        zar_ew_vanka = {**zar_ew_vanka, **mg_vanka}
#================================================================================================
#       Zarantonello \ Vanka
#================================================================================================
        zar_vanka = {
                "mat_type": "aij",
                "snes_type": "newtonls",
                "snes_monitor": None,
                "snes_max_it": 1000,
                "snes_linesearch_type": "basic",
                "snes_linesearch_maxstep": 1.0,
                "snes_linesearch_damping": float(self.zdamping) if (self.linearisation == "zarantonello") else 1.0,
                "snes_linesearch_monitor": None,
                "snes_converged_reason": None,
    #            "snes_view": None,
                "snes_rtol": 1.0e-8, #1.0e-7
                "snes_atol": 1.0e-8, #1.0e-7
                "snes_stol": 0,
        }
        zar_vanka = {**zar_vanka, **mg_vanka}
#================================================================================================
#       NRICH_R - Newton/Zarantonello \ LU
#================================================================================================
        nrich_R_newtonlu = {
            "snes_type": "nrichardson",
            "snes_max_it": 200,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-8,
            "snes_stol": 0,
        }
        nrich_R_newtonlu = {**nrich_R_newtonlu, "npc_": copy.deepcopy(newton_lu)}
        nrich_R_newtonlu["npc_snes_max_it"] = 20 if self.linearisation == "zarantonello" else 3
        nrich_R_newtonlu["npc_snes_convergence_test"] = "skip"
        del nrich_R_newtonlu["npc_"]["snes_monitor"]
        del nrich_R_newtonlu["npc_"]["snes_converged_reason"]

#================================================================================================
#       Newton/Zarantonello \ LU_R - NRICH
#================================================================================================
        newton_lu_R_nrich = copy.deepcopy(newton_lu)
        newton_lu_R_nrich["npc_snes_type"] = "nrichardson"
        newton_lu_R_nrich["npc_snes_max_it"] = 5

#================================================================================================
#       NRICH_R - FAS(GS)
#================================================================================================
        nrich_fas_gs = {
            "snes_type": "nrichardson",
            "snes_max_it": 200,
#            "snes_view": None,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-8,
            "snes_stol": 0,
            "npc_snes_type": "fas",
            "npc_snes_max_it": 1,
            "npc_snes_fas_monitor": None,
            "npc_snes_fas_smoothup": 6,
            "npc_snes_fas_smoothdown": 6,
            "npc_fas_levels_snes_type": "ngs",
            "npc_fas_levels_snes_ngs_atol": 1e-4,
            "npc_fas_levels_snes_ngs_rtol": 1e-3,
            "npc_fas_levels_snes_monitor": None,
            "npc_fas_levels_snes_ngs_sweeps": 3,
            "npc_fas_levels_convergence_test": "skip",
#            "npc_fas_levels_snes_max_it": 6,
#            "npc_fas_coarse_snes_converged_reason": None,
#            "npc_fas_coarse_snes_linesearch_type": "basic",
            "npc_fas_coarse_": copy.deepcopy(newton_lu),
        }
        nrich_fas_gs["npc_fas_coarse_"]["snes_max_it"] = 10
        nrich_fas_gs["npc_fas_coarse_"]["snes_atol"] = 1e-4
        nrich_fas_gs["npc_fas_coarse_"]["snes_rtol"] = 1e-3

#================================================================================================
#       NGMRES_R - FAS(Newton \ LU)
#================================================================================================
        ngmres_fas_newton_lu = {
            "snes_type": "ngmres",
            "snes_max_it": 200,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-8,
            "snes_stol": 0,
            "npc_snes_type": "fas",
            "npc_snes_max_it": 1,
            "npc_snes_fas_type": "multiplicative",
            "npc_snes_fas_monitor": None,
            "npc_snes_fas_smoothup": 6,
            "npc_snes_fas_smoothdown": 6,
            "npc_fas_levels": copy.deepcopy(newton_lu),
#            "npc_fas_coarse_": copy.deepcopy(newton_lu),
            "npc_fas_coarse_snes_linesearch_type": "basic",
        }
        ngmres_fas_newton_lu["npc_fas_levels"]["snes_max_it"] = 4 if self.linearisation == "zarantonello" else 6
#        ngmres_fas_newton_lu["npc_fas_levels"]["snes_convergence_test"] = "skip"
#        ngmres_fas_newton_lu["npc_fas_coarse_"]["snes_max_it"] = 10 if self.linearisation == "zarantonello" else 5
#        ngmres_fas_newton_lu["npc_fas_coarse_"]["snes_convergence_test"] = "skip"
#        del ngmres_fas_newton_lu["npc_fas_coarse_"]["snes_monitor"]
##        del ngmres_fas_newton_lu["npc_fas_levels"]["snes_monitor"]

#================================================================================================
#       NGMRES_R - FAS(Newton \ Vanka) 
#================================================================================================
        ngmres_fas_newton_mg = {
            "snes_type": "ngmres",
            "snes_max_it": 200,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-8,
            "snes_stol": 0,
            "npc_snes_type": "fas",
            "npc_snes_max_it": 1,
            "npc_snes_fas_type": "multiplicative",
            "npc_snes_fas_monitor": None,
            "npc_snes_fas_smoothup": 6,
            "npc_snes_fas_smoothdown": 6,
            "npc_fas_levels": copy.deepcopy(zar_ew_vanka),
#            "npc_fas_coarse_": copy.deepcopy(newton_lu),
            "npc_fas_coarse_snes_linesearch_type": "basic",
        }
        ngmres_fas_newton_lu["npc_fas_levels"]["snes_max_it"] = 4
#        ngmres_fas_newton_lu["npc_fas_levels"]["snes_convergence_test"] = "skip"

#================================================================================================
#       NGMRES_R - FAS(NGS)
#================================================================================================
        ngmres_fas_ngs = {
            "snes_type": "ngmres",
            "snes_max_it": 200,
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-8,
            "snes_stol": 0,
            "npc_snes_type": "fas",
            "npc_snes_max_it": 1,
            "npc_snes_fas_type": "multiplicative",
            "npc_snes_fas_monitor": None,
            "npc_snes_fas_smoothup": 6,
            "npc_snes_fas_smoothdown": 6,
            "npc_fas_levels_snes_type": "ngs",
#            "npc_fas_levels_snes_ngs_sweeps": 3,
            "npc_fas_levels_snes_ngs_max_it": 6,
            "npc_fas_levels_snes_ngs_atol": 1e-4,
            "npc_fas_levels_snes_ngs_rtol": 1e-3,
            "npc_fas_levels_snes_monitor": None,
            "npc_fas_levels_convergence_test": "skip",
            "npc_fas_coarse_snes_linesearch_type": "basic",
        }

#================================================================================================
#       FAS(NGS) * Newton
#================================================================================================
        fas_c_newton_lu = {
            "snes_type": "composite",
            "snes_composite_type": "multiplicative",
            "snes_composite_sneses": "fas,newtonls",
            "snes_converged_reason": None,
            "snes_monitor": None,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-8,
            "snes_stol": 0,
            "sub_0": {"snes_fas_type": "multiplicative",
                      "snes_fas_monitor": None,
                      "snes_fas_smoothup": 6,
                      "snes_fas_smoothdown": 6,
                      "fas_levels_snes_type": "ngs",
                      "fas_levels_snes_ngs_atol": 1e-4,
                      "fas_levels_snes_ngs_rtol": 1e-3,
#                      "fas_levels_snes_monitor": None,
                      "fas_levels_snes_ngs_sweeps": 3,
                      "fas_levels_snes_convergence_test": "skip",
                      "fas_levels_snes_max_it": 6,
                      "fas_coarse_": copy.deepcopy(newton_lu),
                      },
            "sub_1": copy.deepcopy(newton_lu),
                           }
        fas_c_newton_lu["sub_0"]["snes_max_it"] = 10
        fas_c_newton_lu["sub_0"]["fas_coarse_"]["snes_max_it"] = 10
        fas_c_newton_lu["sub_0"]["fas_coarse_"]["snes_atol"] = 1e-4
        fas_c_newton_lu["sub_0"]["fas_coarse_"]["snes_rtol"] = 1e-3

#================================================================================================
#       Newton * FAS(NGS)
#================================================================================================
        newton_lu_c_fas = {
            "snes_type": "composite",
            "snes_composite_type": "multiplicative",
            "snes_composite_sneses": "newtonls,fas",
            "snes_converged_reason": None,
            "snes_monitor": None,
            "sub_1": {"snes_fas_type": "multiplicative",
                      "snes_fas_monitor": None,
                      "snes_fas_smoothup": 6,
                      "snes_fas_smoothdown": 6,
                      "fas_levels_snes_type": "ngs",
                      "fas_levels_snes_ngs_atol": 1e-4,
                      "fas_levels_snes_ngs_rtol": 1e-3,
#                      "fas_levels_snes_monitor": None,
                      "fas_levels_snes_ngs_sweeps": 3,
                      "fas_levels_snes_convergence_test": "skip",
                      "fas_levels_snes_max_it": 6,
                      "fas_coarse_": copy.deepcopy(newton_lu),
                      },
            "sub_0": copy.deepcopy(newton_lu),
                           }
        newton_lu_c_fas["sub_0"]["snes_max_it"] = 10
        newton_lu_c_fas["sub_1"]["fas_coarse_"]["snes_max_it"] = 10
        newton_lu_c_fas["sub_1"]["fas_coarse_"]["snes_atol"] = 1e-4
        newton_lu_c_fas["sub_1"]["fas_coarse_"]["snes_rtol"] = 1e-3
#========== Choose one======================================================
#===========================================================================
        solvers = {
            "newton_lu": newton_lu,
            "zar_ew_vanka": zar_ew_vanka,
            "nrich_R-newton_lu": nrich_R_newtonlu,
            "newton_lu_R-nrich": newton_lu_R_nrich,
            "nrich_fas_gs": nrich_fas_gs,
            "ngmres_fas_ngs": ngmres_fas_ngs,
            "ngmres_fas_newton_lu": ngmres_fas_newton_lu,
            "ngmres_fas_newton_mg": ngmres_fas_newton_mg,
            "fas_c_newton_lu": fas_c_newton_lu,
            "newton_lu_c_fas": newton_lu_c_fas,
        }
        return solvers[self.solver_type]

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
