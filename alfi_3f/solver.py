from firedrake import *
from firedrake.petsc import *
import numpy as np
from mpi4py import MPI

#from alfi.stabilisation import *
from alfi.transfer import *

from alfi_3f.stabilisation import *
from alfi_3f.transfer import DGInjection

import pprint
import sys
from datetime import datetime


class NonNewtonianSolver(object):

    def function_space(self, mesh, k):
        raise NotImplementedError

    def residual(self):
        raise NotImplementedError

    def update_wind(self, z):
        raise NotImplementedError

    def set_transfers(self):
        raise NotImplementedError

    def __init__(self, problem, nref=1, solver_type="almg",
                 stabilisation_type_u=None, stabilisation_type_t=None,
                 supg_method_u="shakib", supg_method_t="shakib", supg_magic=9.0, gamma=10000, k=3,
                 patch="macro", hierarchy="bary", use_mkl=False, stabilisation_weight_u=None, stabilisation_weight_t=None,
                 patch_composition="additive", restriction=False, smoothing=None, cycles=None,
                 rebalance_vertices=False, hierarchy_callback=None, high_accuracy=False, thermal_conv="none",
                 linearisation = "newton", low_accuracy = False, no_convection = False,
                 exactly_div_free = True, fluxes=None, ip_magic=5.0):

        assert solver_type in {"almg", "allu", "lu", "aljacobi", "alamg", "simple", "lu-hdiv", "allu-hdiv", "almg-hdiv"}, "Invalid solver type %s" % solver_type
        if solver_type in {"lu-hdiv", "allu-hdiv", "almg-hdiv"}: assert problem.formulation in {"L-u-p", "L-S-u-p"}, "That solver_type only makes sense with L-u-p or L-S-u-p formulations"
        if stabilisation_type_u == "none":
            stabilisation_type_u = None
        if stabilisation_type_t == "none":
            stabilisation_type_t = None
        assert stabilisation_type_u in {None, "gls", "supg", "burman"}, "Invalid stabilisation type %s" % stabilisation_type_u
        assert stabilisation_type_t in {None, "supg", "burman"}, "Invalid stabilisation type %s" % stabilisation_type_t
        assert hierarchy in {"uniform", "bary", "uniformbary"}, "Invalid hierarchy type %s" % hierarchy
        assert patch in {"macro", "star"}, "Invalid patch type %s" % patch
        assert linearisation in {"newton", "picard", "kacanov"}, "Invalid linearisation type %s" % linearisation
        if fluxes == "none":
            fluxes = None
        assert fluxes in {None, "ldg", "mixed", "ip"}, "Invalid choice of fluxes %s" % fluxes
        if thermal_conv == "none":
            thermal_conv = None
        assert thermal_conv in {None, "natural_Ra", "natural_Ra2", "natural_Gr", "forced"}, "Invalid thermal convection regime %s" % thermal_conv
        if hierarchy != "bary" and patch == "macro":
            raise ValueError("macro patch only makes sense with a BaryHierarchy")
        self.hierarchy = hierarchy
        self.problem = problem
        self.nref = nref
        self.solver_type = solver_type
        self.stabilisation_type_u = stabilisation_type_u
        self.stabilisation_type_t = stabilisation_type_t
        self.patch = patch
        self.use_mkl = use_mkl
        self.patch_composition = patch_composition
        self.restriction = restriction
        self.smoothing = smoothing
        self.cycles = cycles
        self.high_accuracy = high_accuracy
        self.low_accuracy = low_accuracy
        self.no_convection = no_convection
        self.exactly_div_free = exactly_div_free
        self.thermal_conv = thermal_conv
        self.formulation = self.problem.formulation
        self.linearisation = linearisation
        self.fluxes = fluxes
        self.ip_magic = ip_magic
        assert self.formulation in {
                "T-S-u-p",
                "T-u-p",
                "L-S-u-p",
                "L-u-p",
                "S-u-p",
                "u-p",
                "L-T-u-p",
                "L-T-S-u-p"
                }, "Invalid formulation %s" % self.formulation

        self.formulation_LSup = (self.formulation == "L-S-u-p")
        self.formulation_Lup = (self.formulation == "L-u-p")
        self.formulation_Sup = (self.formulation == "S-u-p")
        self.formulation_up = (self.formulation == "u-p")
        self.formulation_TSup = (self.formulation == "T-S-u-p")
        self.formulation_Tup = (self.formulation == "T-u-p")
        self.formulation_LTup = (self.formulation == "L-T-u-p")
        self.formulation_LTSup = (self.formulation == "L-T-S-u-p")
        self.formulation_has_stress = self.formulation_Sup or self.formulation_LSup or self.formulation_TSup or self.formulation_LTSup
        if self.stabilisation_type_u == "gls": assert formulation_up or formulation_Tup, "GLS stabilisation has not implemented for formulations including the stress"
        if (self.formulation_Tup or self.formulation_TSup): assert not(self.thermal_conv is None), "You have to choose the convection regime (natural or forced)"

        def rebalance(dm, i):
            if rebalance_vertices:
                # if not dm.rebalanceSharedPoints(useInitialGuess=False, parallel=False):
                #     warning("Vertex rebalancing from scratch failed on level %i" % i)
                # if not dm.rebalanceSharedPoints(useInitialGuess=True, parallel=True):
                #     warning("Vertex rebalancing from initial guess failed on level %i" % i)
                try:
                    dm.rebalanceSharedPoints(useInitialGuess=False, parallel=False)
                except:
                    warning("Vertex rebalancing in serial from scratch failed on level %i" % i)
                try:
                    dm.rebalanceSharedPoints(useInitialGuess=True, parallel=True)
                except:
                    warning("Vertex rebalancing from initial guess failed on level %i" % i)

        def before(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+1)

        def after(dm, i):
            for p in range(*dm.getHeightStratum(1)):
                dm.setLabelValue("prolongation", p, i+2)
            rebalance(dm, i)

        mh = problem.mesh_hierarchy(hierarchy, nref, (before, after), self.distribution_parameters())

        if hierarchy_callback is not None:
            mh = hierarchy_callback(mh)
        self.parallel = mh[0].comm.size > 1
        self.tdim = mh[0].topological_dimension()
        self.mh = mh
        self.area = assemble(Constant(1, domain=mh[0])*dx)

        #Obtain parameters from the constitutive relation and make sure they are Constants
        self.const_rel_params = {}
        for param_str, param in self.problem.const_rel_params.items():
            setattr(self, param_str, param)
            if not isinstance(getattr(self,param_str), Constant):
                setattr(self, param_str, Constant(getattr(self, param_str)))
            self.const_rel_params[param_str] = getattr(self, param_str)
        #Make sure either nu or Re are defined
        if self.thermal_conv in ["natural_Ra", "natural_Ra2", "natural_Gr"]:
            self.Re = sqrt(self.Gr) if self.thermal_conv == "natural_Gr" else self.Ra
            self.nu = 1./self.Re
        else:
            assert any(elem in list(self.problem.const_rel_params.keys()) for elem in ["nu","Re"]), "The constitutive relation must include either the Reynolds Re number or visosity nu"
        if not("nu" in list(self.problem.const_rel_params.keys())):
            self.nu = 1./self.Re
        if not("Re" in list(self.problem.const_rel_params.keys())):
            self.Re = 1/self.nu

        if not isinstance(gamma, Constant):
            gamma = Constant(gamma)
        self.gamma = gamma
        if self.solver_type == "simple":
            self.gamma.assign(0)
            warning("Setting gamma to 0")
        self.advect = Constant(0)

        mesh = mh[-1]

        self.mesh = mesh
        self.load_balance(mesh)
        Z = self.function_space(mesh, k)
        self.Z = Z
        comm = mesh.mpi_comm()

        if self.formulation_up:
            self.velocity_id = 0
            self.pressure_id = 1
        elif self.formulation_Tup:
            self.temperature_id = 0
            self.velocity_id = 1
            self.pressure_id = 2
        elif self.formulation_Lup:
            self.gradient_id = 0
            self.velocity_id = 1
            self.pressure_id = 2
        elif self.formulation_Sup:
            self.stress_id = 0
            self.velocity_id = 1
            self.pressure_id = 2
        elif self.formulation_LSup:
            self.gradient_id = 0
            self.stress_id = 1
            self.velocity_id = 2
            self.pressure_id = 3
        elif self.formulation_TSup:
            self.temperature_id = 0
            self.stress_id = 1
            self.velocity_id = 2
            self.pressure_id = 3
        elif self.formulation_LTup:
            self.gradient_id = 0
            self.temperature_id = 1
            self.velocity_id = 2
            self.pressure_id = 3
        elif self.formulation_LTSup:
            self.gradient_id = 0
            self.temperature_id = 1
            self.stress_id = 2
            self.velocity_id = 3
            self.pressure_id = 4

        #Define constants to use in the secant predictor (We probably don't need to store these always...)
        for param_str in self.problem.const_rel_params.keys():
            setattr(self, param_str+"_last", Constant(float(getattr(self, param_str))))
            setattr(self, param_str+"_last2", Constant(float(getattr(self, param_str))))

        Zdim = self.Z.dim()
        size = comm.size
        if comm.rank == 0:
            print("Number of degrees of freedom: %s (avg %.2f per core)" % (Zdim, Zdim / size))
        z = Function(Z, name="Solution")
        if comm.rank == 0:
            if self.formulation_has_stress:
                Vdim = self.Z.sub(self.stress_id).dim() + self.Z.sub(self.velocity_id).dim()
                print("Number of stress and velocity degrees of freedom: %s (avg %.2f per core)" % (Vdim, Vdim / size))
            else:
                Vdim = self.Z.sub(self.velocity_id).dim()
                print("Number of velocity degrees of freedom: %s (avg %.2f per core)" % (Vdim, Vdim / size))
        self.z = z

        #Split and define velocity u and test function v
        fields = self.split_variables(self.z)
        u = fields["u"]
        v = fields["v"]
        q = fields["q"]
        theta = fields.get("theta")
        theta_ = fields.get("theta_")

        bcs = problem.bcs(Z)
        self.bcs = bcs
        nsp = problem.nullspace(Z)

        if nsp is not None and solver_type == "lu":
            """ Pin the pressure because LU sometimes fails for the saddle
            point problem with a nullspace """
            bcs.append(DirichletBC(Z.sub(self.pressure_id), Constant(0), None))
            if Z.mesh().comm.rank == 0:
                bcs[-1].nodes = np.asarray([0])
            else:
                bcs[-1].nodes = np.asarray([], dtype=np.int64)
            self.nsp = None
        else:
            self.nsp = nsp

        params = self.get_parameters()
        if mesh.mpi_comm().rank == 0:
            pprint.pprint(params)
            sys.stdout.flush()

        self.z_last = z.copy(deepcopy=True)

        F = self.residual()

        """ Velocity Stabilisation (for Scott-Vogelius 'burman' seems to be best)"""
        wind = split(self.z_last)[self.velocity_id]
        rhs = problem.rhs(Z)
        if self.stabilisation_type_u in ["gls", "supg"]:
            #Define the parameter that causes dominant convection as it grows (should we use an effective viscosity for this?)
            if self.formulation_Tup or self.formulation_TSup:
                if self.thermal_conv in ["natural_Ra", "natural_Ra2"]:
                    supg_diffusion_parameter_u = self.Ra/self.Pr
                elif self.thermal_conv == "natural_Gr":
                    supg_diffusion_parameter_u = sqrt(self.Gr)
                elif self.thermal_conv == "forced":
                    supg_diffusion_parameter_u = self.Re
            else:
                supg_diffusion_parameter_u = 1./self.nu

            if supg_method_u == "turek":
                self.stabilisation_vel = TurekSUPG(supg_diffusion_parameter_u, self.Z, wind_id=self.velocity_id, magic=supg_magic, h=problem.mesh_size(u), state=z, weight=stabilisation_weight_u)
            elif supg_method_u == "shakib":
                self.stabilisation_vel = ShakibHughesZohanSUPG(supg_diffusion_parameter_u, self.Z, wind_id=self.velocity_id, magic=supg_magic, h=problem.mesh_size(u, "cell"), state=z, weight=stabilisation_weight_u)
            else:
                raise NotImplementedError

            Lu, Lv = self.strong_residual(self.z, wind, rhs)
            k = Z.sub(self.velocity_id).ufl_element().degree()
            if self.stabilisation_type_u == "gls":
                self.message( RED % "CAUTION:  The GLS stabilisation has been implemented for Newtonian flow only!!!")
                self.stabilisation_form_u = self.stabilisation_vel.form_gls(Lu, Lv, dx(degree=2*k))
            elif self.stabilisation_type_u == "supg":
                self.stabilisation_form_u = self.stabilisation_vel.form(Lu, v, dx(degree=2*k))
            else:
                raise NotImplementedError

        elif self.stabilisation_type_u in ["burman"]:
            self.stabilisation_vel = BurmanStabilisation(self.Z, wind_id=self.velocity_id, state=z, h=problem.mesh_size(u, "facet"), weight=stabilisation_weight_u)
            self.stabilisation_form_u = self.stabilisation_vel.form(u, v)

        else:
            self.stabilisation_vel = None
            self.stabilisation_form_u = None

        #Temperature stabilisation
        if self.stabilisation_type_t in ["supg"]:
            assert self.formulation_Tup or self.formulation_TSup, "Setting stabilisation-type-t only makes sense for non-isothermal problems..."
            if self.thermal_conv in ["natural_Ra", "natural_Ra2"]:
                supg_diffusion_parameter_t = Constant(1.)
            elif self.thermal_conv == "natural_Gr":
                supg_diffusion_parameter_t = self.Pr*sqrt(self.Gr)
            elif self.thermal_conv == "forced":
                supg_diffusion_parameter_t = self.Pe

            if supg_method_t == "turek":
                self.stabilisation_temp = TurekSUPG(supg_diffusion_parameter_t, self.Z, wind_id=self.velocity_id, magic=supg_magic, h=problem.mesh_size(u), state=z, field_id=self.temperature_id, weight=stabilisation_weight_t)
            elif supg_method_t == "shakib":
                self.stabilisation_temp = ShakibHughesZohanSUPG(supg_diffusion_parameter_t, self.Z, wind_id=self.velocity_id, magic=supg_magic, h=problem.mesh_size(u, "cell"), state=z, field_id=self.temperature_id, weight=stabilisation_weight_u)
            else:
                raise NotImplementedError

            Lth, _ = self.strong_residual(self.z, wind, rhs, type="temperature")
            k_t = Z.sub(self.temperature_id).ufl_element().degree()
            if self.stabilisation_type_t == "supg":
                self.stabilisation_form_t = self.stabilisation_temp.form(Lth, theta_, dx(degree=2*k_t))
            else:
                raise NotImplementedError

        elif self.stabilisation_type_t == "burman":
                self.stabilisation_temp = BurmanStabilisation(self.Z, wind_id=self.velocity_id, state=z, h=problem.mesh_size(theta, "facet"), weight=stabilisation_weight_t)
                self.stabilisation_form_t = self.stabilisation_temp.form(theta, theta_)
        else:
            self.stabilisation_temp = None
            self.stabilisation_form_t = None

        if self.stabilisation_form_u is not None:
            F += (self.advect * self.stabilisation_form_u)
        if self.stabilisation_form_t is not None:
            F += self.stabilisation_form_t

        if rhs is not None:
            if self.formulation_up or self.formulation_Sup or self.formulation_LSup or self.formulation_Lup: #Assumes the equations for S and L do NOT have right-hand-sides
                F -= inner(rhs[0], v) * dx + inner(rhs[1], q) * dx
            elif self.formulation_Tup or self.formulation_TSup:
                F -= inner(rhs[0], theta_) *dx + inner(rhs[1], v) * dx + inner(rhs[2], q) * dx

        ### Define constitutive relation
        D = sym(grad(u))

        #Compute Jacobian
        self.F = F
        self.J = self.get_jacobian()

        appctx = self.const_rel_params
        appctx["gamma"] = self.gamma
        appctx["problem"] = self.problem
        appctx["advect"] = self.advect
        if not("nu" in list(self.problem.const_rel_params.keys())):
            appctx["nu"] = self.nu
        problem = NonlinearVariationalProblem(F, z, bcs=bcs, J=self.J)
        self.bcs = bcs
        self.params = params
        self.nsp = nsp
        self.appctx = appctx
        self.solver = NonlinearVariationalSolver(problem, solver_parameters=params,
                                                 nullspace=nsp, options_prefix="ns_",
                                                 appctx=appctx)
        self.transfers = self.get_transfers()
        self.transfermanager = TransferManager(native_transfers=self.get_transfers())
        self.solver.set_transfer_manager(self.transfermanager)
        self.check_nograddiv_residual = True
        if self.check_nograddiv_residual:
            self.message(GREEN % "Checking residual without grad-div term")
            self.F_nograddiv = replace(F, {gamma: 0})

    def get_jacobian(self):
        if self.linearisation == "newton":
            J0 = derivative(self.F, self.z)
        elif self.linearisation in ["picard", "kacanov"]:
            """ 'picard': full Picard
                'kacanov': 'Picard' only in the constitutive relation (not on the advective terms) """
            #Split variables for current solution and test function
            fields = self.split_variables(self.z)
            u = fields["u"]
            v = fields["v"]
            p = fields["p"]
            q = fields["q"]
            S = fields.get("S")
            ST = fields.get("ST")
            L = fields.get("L")
            LT = fields.get("LT")
            theta = fields.get("theta")
            theta_ = fields.get("theta_")
            D = sym(grad(u))

            #Split trial function
            w = TrialFunction(self.Z)
            fields0 = self.split_variables(w)
            u0 = fields0["u"]
            p0 = fields0["p"]
            S0 = fields0.get("S")
            L0 = fields0.get("L")
            theta0 = fields0.get("theta")
            D0 = sym(grad(u0))

            #Constitutive Relation
            if self.formulation_Sup:
                G0 = self.problem.const_rel_picard(S,D,S0,D0)
            elif self.formulation_TSup:
                G0 = self.problem.const_rel_picard(S,D,theta,S0,D0)
            elif self.formulation_up:
                G0 = self.problem.const_rel_picard(D, D0)
            elif self.formulation_Tup:
                G0 = self.problem.const_rel_picard(D, theta, D0)
                G = self.problem.const_rel(D, theta)
            elif self.formulation_LSup:
                G0 = self.problem.const_rel_picard(S, L, S0, L0)
            elif self.formulation_Lup:
                G0 = self.problem.const_rel_picard(L0)

            if self.formulation_Tup or self.formulation_TSup:
                th_flux0 = self.problem.const_rel_temperature(theta, grad(theta0))  #TODO: Assumes the constitutive relation is linear in the second entry

            #Common to all formulations
            J0 = (
                self.gamma * inner(div(u0), div(v))*dx
                + self.advect * inner(dot(grad(u0), u), v)*dx
                - p0 * div(v) * dx
                - div(u0) * q * dx
            )

            if self.formulation_Sup:
                J0 += (
                    inner(S0,sym(grad(v)))*dx
                    - inner(G0,ST) * dx
                    )

            elif self.formulation_LSup:
                J0 += (
                    inner(S0,sym(grad(v)))*dx
                    - inner(D0 - L0, SL) * dx
                    - inner(G0, ST) * dx
                    )
            elif self.formulation_up:
                J0 += (
                    inner(G0, sym(grad(v)))*dx
                    )

            elif self.formulation_Tup or self.formulation_TSup:
                """
                The non-dimensional forms "rayleigh1" and "rayleigh2" use a time-scale based on heat diffusivity and only differ in the
                choice of how to balance the pressure term. With the non-dimensional form "grashof" one assumes that all the gravitational
                potential energy gets transformed into kinetic energy and so the characteristic velocity scale is chosen accordingly.
                For a non-Newtonian fluid (we have tried the Ostwald-de Waele power law relation), more non-dimensional numbers will
                arising from the constitutive relation will be necessary.
    #                """
                #If the Dissipation number is not defined, set it to zero.
                if not("Di" in list(self.problem.const_rel_params.keys())): self.Di = Constant(0.)
                if not("Theta" in list(self.problem.const_rel_params.keys())): self.Theta = Constant(0.)
                g = Constant((0, 1)) if self.tdim == 2 else Constant((0, 0, 1))

                #Common to all formulations with temperature
                J0 = (
                    self.gamma * inner(div(u0), div(v))*dx
                    - p0 * div(v) * dx
                    - div(u0) * q * dx
                    + inner(dot(grad(theta0), u), theta_) * dx
                )

                if self.thermal_conv == "natural_Ra":
                    J0 += (
                        self.advect * inner(dot(grad(u0), u), v)*dx
                        - (self.Ra*self.Pr) * inner(theta0*g, v) * dx
                        + inner(th_flux0, grad(theta_)) * dx
                        + self.Di * inner((theta0)*dot(g, u), theta_) * dx
                        + self.Di * inner((theta + self.Theta)*dot(g, u0), theta_) * dx
                        )
                    if self.formulation_Tup:
                        J0 += (
                            self.Pr * inner(G0,sym(grad(v)))*dx
                            )
                        #Newton linearisation of the dissipation term (not sure if this is the best way)
                        J0 -= (self.Di/self.Ra) * derivative(inner(inner(G,sym(grad(u))), theta_)*dx, self.z, w)
                    elif self.formulation_TSup:
                        J0 += (
                            self.Pr * inner(S0,sym(grad(v)))*dx
                            - inner(G0,ST) * dx
                            - (self.Di/self.Ra) * inner(inner(S0,sym(grad(u))), theta_) * dx
                            - (self.Di/self.Ra) * inner(inner(S,sym(grad(u0))), theta_) * dx
                            )

                elif self.thermal_conv == "natural_Ra2":
                    J0 += (
                        + (1./self.Pr) * self.advect * inner(dot(grad(u0), u), v)*dx
                        - (self.Ra) * inner(theta0*g, v) * dx
                        + inner(th_flux0, grad(theta_)) * dx
                        + self.Di * inner((theta0)*dot(g, u), theta_) * dx
                        + self.Di * inner((theta + self.Theta)*dot(g, u0), theta_) * dx
                        )
                    if self.formulation_Tup:
                        J0 += inner(G0,sym(grad(v)))*dx
                        #Newton linearisation of the dissipation term (not sure if this is the best way)
                        J0 -= (self.Di/self.Ra) * derivative(inner(inner(G,sym(grad(u))), theta_)*dx, self.z, w)
                    elif self.formulation_TSup:
                        J0 += (
                            inner(S0,sym(grad(v)))*dx
                            - inner(G0,ST) * dx
                            - (self.Di/self.Ra) * inner(inner(S0,sym(grad(u))), theta_) * dx
                            - (self.Di/self.Ra) * inner(inner(S,sym(grad(u0))), theta_) * dx
                            )
                elif self.thermal_conv == "natural_Gr":
                    J0 += (
                        self.advect * inner(dot(grad(u0), u), v)*dx
                        - inner(theta0*g, v) * dx
                        + (1./(self.Pr * sqrt(self.Gr))) * inner(th_flux0, grad(theta_)) * dx
                        + self.Di * inner((theta0)*dot(g, u), theta_) * dx
                        + self.Di * inner((theta + self.Theta)*dot(g, u0), theta_) * dx
                        )
                    if self.formulation_Tup:
                        J0 += (1./sqrt(self.Gr)) * inner(G0,sym(grad(v)))*dx
                        #Newton linearisation of the dissipation term (not sure if this is the best way)
                        J0 -= (self.Di/sqrt(self.Gr)) * derivative(inner(inner(G,sym(grad(u))), theta_)*dx, self.z, w)
                    elif self.formulation_TSup:
                        J0 += (
                            (1./sqrt(self.Gr)) * inner(S0,sym(grad(v)))*dx
                            - inner(G0,ST) * dx
                            - (self.Di/sqrt(self.Gr)) * inner(inner(S0,sym(grad(u))), theta_) * dx
                            - (self.Di/sqrt(self.Gr)) * inner(inner(S,sym(grad(u0))), theta_) * dx
                            )

                elif self.thermal_conv == "forced":
                    """
                    For the forced convection regime I choose a characteristic velocity, and so the Peclet and
                    Reynolds numbers will appear in the formulation

                    """
                    if not("Br" in list(self.problem.const_rel_params.keys())): self.Br = Constant(0.)
                    J0 += (
                        + self.Re * self.advect * inner(dot(grad(u0), u), v)*dx
                        + (1./self.Pe) * inner(th_flux0, grad(theta_)) * dx
                        )
                    if self.formulation_Tup:
                        J0 += inner(G0,sym(grad(v)))*dx
                        #Newton linearisation of the dissipation term (not sure if this is the best way)
                        J0 -= (self.Br/self.Pe) * derivative(inner(inner(G,sym(grad(u))), theta_)*dx, self.z, w)
                    elif self.formulation_TSup:
                        J0 += (
                            inner(S0,sym(grad(v)))*dx
                            - inner(G0,ST) * dx
                            - (self.Br/self.Pe) * inner(inner(S0,sym(grad(u))), theta_) * dx
                            - (self.Br/self.Pe) * inner(inner(S,sym(grad(u0))), theta_) * dx
                            )
            if (self.linearisation == "kacanov"):
                #Add extra term for Newton linearisation of the convective term
                if (self.formulation_Tup or self.formulation_TSup) and (self.thermal_conv == "natural_Ra2"):
                    J0 += (1./self.Pr) * self.advect * inner(dot(grad(u), u0), v)*dx
                elif (self.formulation_Tup or self.formulation_TSup) and (self.thermal_conv == "forced"):
                    J0 += self.Re * self.advect * inner(dot(grad(u), u0), v)*dx
                else:
                    J0 += self.advect * inner(dot(grad(u), u0), v)*dx
                #Same for the convective term in the temperature equation
                if self.formulation_Tup or self.formulation_TSup:
                    J0 += inner(dot(grad(theta), u0), theta_) * dx

            if self.stabilisation_form_u is not None:
               J0 += derivative(self.stabilisation_form_u, self.z, w)

            if self.stabilisation_form_t is not None:
               J0 += derivative(self.stabilisation_form_t, self.z, w)

        return J0
#        return inner(self.nu*grad(u0),grad(v))*dx - p0*div(v)*dx - q*div(u0)*dx

    def split_variables(self, z):
        Z = self.Z
        fields = {}
        if self.formulation_Sup:
            (S_,u, p) = split(z)
            (ST_,v, q) = split(TestFunction(Z))
        elif self.formulation_LSup:
            (L_,S_,u, p) = split(z)
            (LT_,ST_,v, q) = split(TestFunction(Z))
        elif self.formulation_Lup:
            (L_,u, p) = split(z)
            (LT_,v, q) = split(TestFunction(Z))
        elif self.formulation_up:
            (u, p) = split(z)
            (v, q) = split(TestFunction(Z))
        elif self.formulation_TSup:
            (theta,S_,u, p) = split(z)
            (theta_,ST_,v, q) = split(TestFunction(Z))
        elif self.formulation_Tup:
            (theta,u, p) = split(z)
            (theta_,v, q) = split(TestFunction(Z))
        elif self.formulation_LTSup:
            (L_,theta,S_,u, p) = split(z)
            (LT_,theta_,ST_,v, q) = split(TestFunction(Z))
        elif self.formulation_LTup:
            (L_,theta,u, p) = split(z)
            (LT_,theta_,v, q) = split(TestFunction(Z))
        if self.formulation_Sup or self.formulation_Lup or self.formulation_LSup or self.formulation_LTup or self.formulation_LTSup:
            L = self.stress_to_matrix(L_, False)
            LT = self.stress_to_matrix(LT_, False)
            fields["L"] = L
            fields["LT"] = LT
        if self.formulation_Tup or self.formulation_TSup or self.formulation_LTSup or self.formulation_LTup:
            fields["theta"] = theta
            fields["theta_"] = theta_
        fields["u"] = u
        fields["v"] = v
        fields["p"] = p
        fields["q"] = q

        #Split stress variables appropriately
        if self.formulation_has_stress:
            S = self.stress_to_matrix(S_, self.exactly_div_free)
            ST = self.stress_to_matrix(ST_, self.exactly_div_free)
            fields["S"] = S
            fields["ST"] = ST
        return fields

    def stress_to_matrix(self, S_, div_free=True):
        if self.tdim == 2:
            if div_free:
                (S_1,S_2) = split(S_)
                S = as_tensor(((S_1,S_2),(S_2,-S_1)))
            else:
                (S_1,S_2,S_3) = split(S_)
                S = as_tensor(((S_1,S_2),(S_2,S_3)))
        else:
            if div_free:
                (S_1,S_2,S_3,S_4,S_5) = split(S_)
                S = as_tensor(((S_1,S_2,S_3),(S_2,S_5,S_4),(S_3,S_4,-S_1-S_5)))
            else:
                (S_1,S_2,S_3,S_4,S_5,S_6) = split(S_)
                S = as_tensor(((S_1,S_2,S_3),(S_2,S_5,S_4),(S_3,S_4,S_6)))
        return S

    def ip_penalty_jump(self, h_factor, vec, form="cr"):
        assert form in ["cr", "plaw", "quadratic"], "That is not a valid form for the penalisation term"
        U_jmp = h_factor * vec
        if form == "cr":
            if self.formulation_Lup:
                jmp_penalty = self.problem.const_rel(U_jmp)
            elif self.formulation_LTup or self.formulation_Tup:
                theta = self.z.split()[self.temperature_id]
                jmp_penalty = self.problem.const_rel(U_jmp, theta)
            elif self.formulation_LTSup or self.formulation_TSup:
                theta = self.z.split()[self.temperature_id]
                jmp_penalty = self.problem.explicit_cr(U_jmp, theta)
            else:
                jmp_penalty = self.problem.explicit_cr(U_jmp)
        elif form == "plaw":
            jmp_penalty = 2. * self.nu * pow(inner(U_jmp,U_jmp), (float(self.r)-2.)/2.)*U_jmp
        elif form == "quadratic":
            jmp_penalty = 2. * self.nu * U_jmp
        return jmp_penalty

    def strong_residual(self, z, wind, rhs=None, type="momentum"):
        fields = self.split_variables(z)
        if rhs is not None:
            if self.formulation_up or self.formulation_Sup or self.formulation_LSup:
                rhs_u = rhs[0]
            elif self.formulation_Tup or self.formulation_TSup:
                rhs_theta = rhs[0]
                rhs_u = rhs[1]

        g = Constant((0, 1)) if self.tdim == 2 else Constant((0, 0, 1))
        u = fields["u"]
        v = fields["v"]
        p = fields["p"]
        q = fields["q"]
        theta = fields.get("theta")
        theta_ = fields.get("theta_")
        D = sym(grad(u))
        if self.formulation_Lup or self.formulation_up:
            S = self.problem.const_rel(sym(grad(u)))
#                ST = self.problem.const_rel(sym(grad(v))) #TODO: gls only works in the Newtonian case
            ST = 2. * self.nu * sym(grad(v))
        elif self.formulation_LSup or self.formulation_Sup or self.formulation_TSup:
            th_flux = self.problem.const_rel_temperature(theta, grad(theta))
            S = fields["S"]
            ST = fields["ST"]
        elif self.formulation_Tup:
            th_flux = self.problem.const_rel_temperature(theta, grad(theta))
            S = self.problem.const_rel(sym(grad(u)), theta)
            ST = 2. * self.nu * sym(grad(v))

        if type == "momentum":
            if self.formulation_up or self.formulation_Lup or self.formulation_LSup or self.formulation_Tup:
                LL = - div(S) + self.advect*dot(grad(u),u) + grad(p)
                LL_ = -div(ST) + self.advect*dot(grad(v), wind) + grad(q)  #TODO: What if linearisation="newton"?
            elif self.formulation_Tup or self.formulation_TSup:

                if self.thermal_conv == "natural_Ra":
                    LL = - self.Pr*div(S) + self.advect*dot(grad(u),u) + grad(p) - self.Ra*self.Pr*theta*g
                    LL_ = - self.Pr*div(ST) + self.advect*dot(grad(v), wind) + grad(q) - self.Ra*self.Pr*theta_*g
                elif self.thermal_conv == "natural_Ra2":
                    LL = -div(S) + (self.advect/self.Pr)*dot(grad(u),u) + grad(p) - self.Ra*theta*g
                    LL_ = -div(ST) + (self.advect/self.Pr)*dot(grad(v), wind) + grad(q) - self.Ra*theta_*g
                elif self.thermal_conv == "natural_Gr":
                    LL = -(1./sqrt(self.Gr))*div(S) + self.advect*dot(grad(u),u) + grad(p) - theta*g
                    LL_ = -(1./sqrt(self.Gr))*div(ST) + self.advect*dot(grad(v), wind) + grad(q) - theta_*g
                elif self.thermal_conv == "forced":
                    LL = -div(S) + self.Re*self.advect*dot(grad(u),u) + grad(p)
                    LL_ = -div(ST) + self.Re*self.advect*dot(grad(v), wind) + grad(q)

            if rhs is not None:
                LL -= rhs_u

        elif type == "temperature":
           LL_ = None

           if self.thermal_conv in ["natural_Ra", "natural_Ra2"]:
               LL = -div(th_flux) + div(theta*u) + self.Di*(theta + self.Theta)*dot(g,u) - (self.Di/self.Ra)*inner(S,D)
           elif self.thermal_conv == "natural_Gr":
               LL = -(1./(self.Pr*sqrt(self.Gr)))*div(th_flux) + div(theta*u) + self.Di*(theta + self.Theta)*dot(g,u) - (self.Di/sqrt(self.Gr))*inner(S,D)
           elif self.thermal_conv == "forced":
               LL = -(1./self.Pe)*div(th_flux) + div(theta*u) - (self.Br/self.Pe)*inner(S,D)

           if rhs is not None:
               LL -= rhs_theta

        return LL, LL_

    def solve(self, param, info_param, predictor="trivial"):
        """
        param is a string with the name of the parameter we are solving for
        info_param is an iterable with all the parameters of relevance in the problem
        predictor can be "trivial", "secant" or "tangent"
        """
        if (predictor == "trivial"):
            prev = self.z.copy(deepcopy=True)

        elif (predictor == "secant"):    #TODO: Here we have to make sure that the problem has been solved at least twice...
            prev = self.z.copy(deepcopy=True)
            param_current = getattr(self, param)
            param_last = getattr(self, param+"_last")
            param_last2 = getattr(self, param+"_last2")
            self.z.assign(((param_current-param_last)/(param_last - param_last2))*(self.z-self.z_last) + self.z)
            sec_prediction = self.z.copy(deepcopy=True)#For comparing with trivial continuation
        else:
            raise NotImplementedError("Don't know that predictor")

        output_info = "Solving for "
        for param_ in info_param:
            output_info += param_
            output_info += " = %.8f, "%float(getattr(self, param_))
        self.message(GREEN % (output_info))
        if self.no_convection:
            self.message(GREEN % ("Solving Stokes-like problem (no convective term)"))
            self.advect.assign(0)
        else:
            self.advect.assign(1)
        self.advect.assign(1)

        if self.stabilisation_vel is not None:
            self.stabilisation_vel.update(self.z.split()[self.velocity_id])
        if self.stabilisation_temp is not None:
            self.stabilisation_temp.update(self.z.split()[self.velocity_id])

        start = datetime.now()
        self.solver.solve()
        end = datetime.now()

        #========== Tests =====================
        if (predictor == "secant"):
            warning("Error squared between computed z and the secant prediction: ")
            warning(assemble(inner(self.z - sec_prediction,self.z - sec_prediction)*dx))
            warning("Error squared between computed z and the previous z: ")
            warning(assemble(inner(self.z - self.z_last,self.z - self.z_last)*dx))
        elif (predictor == "tangent"):
            print("Error squared between computed z and the tangent prediction: ")
            print(assemble(inner(self.z - tan_prediction,self.z - tan_prediction)*dx))
            print("Error squared between computed z and the previous z: ")
            print(assemble(inner(self.z - current,self.z - current)*dx))
        #=========================================

        #Prepare param_last and param_last2 in case we use secant prediction next
        getattr(self, param+"_last2").assign(getattr(self, param+"_last"))
        getattr(self, param+"_last").assign(getattr(self, param))
        self.z_last.assign(prev)

        if self.nsp is not None:
            # Hardcode that pressure is constant
            p = self.z.split()[self.pressure_id]

            pintegral = assemble(p*dx)
            p.assign(p - Constant(pintegral/self.area))

        if self.check_nograddiv_residual:
            F_ngd = assemble(self.F_nograddiv)
            for bc in self.bcs:
                bc.zero(F_ngd)
            F = assemble(self.F)
            for bc in self.bcs:
                bc.zero(F)
            with F_ngd.dat.vec_ro as v_ngd, F.dat.vec_ro as v:
                self.message(BLUE % ("Residual without grad-div term: %.14e" % v_ngd.norm()))
                self.message(BLUE % ("Residual with grad-div term:    %.14e" % v.norm()))
        linear_its = self.solver.snes.getLinearSolveIterations()
        nonlinear_its = self.solver.snes.getIterationNumber()
        time = (end-start).total_seconds() / 60
        self.message(GREEN % ("Time taken: %.2f min in %d iterations (%.2f Krylov iters per Newton step)" % (time, linear_its, linear_its/max(1.,float(nonlinear_its)))))

        info_dict = {}
        for param_ in info_param:
            info_dict[param_] = float(getattr(self.problem, param_))
        info_dict.update({"linear_iter": linear_its, "nonlinear_iter": nonlinear_its, "time": time})
        return (self.z, info_dict)

    def get_parameters(self):
        multiplicative = self.patch_composition == "multiplicative"
        if multiplicative and self.problem.relaxation_direction() is None:
            raise NotImplementedError("Need to specify a relaxation_direction in the problem.")
        if self.smoothing is None:
            self.smoothing = 10 if self.tdim > 2 else 6
        if self.cycles is None:
            self.cycles = 3

        if self.high_accuracy:
            tolerances = {
                "ksp_rtol": 1.0e-12,
                "ksp_atol": 1.0e-12,
                "snes_rtol": 1.0e-10,
                "snes_atol": 1.0e-10,
                "snes_stol": 1.0e-10,
            }
        elif self.low_accuracy:
            tolerances = {
                "ksp_rtol": 1.0e-6,#1.0e-7,#
                "ksp_atol": 1.0e-6,#1.0e-6,#
                "snes_rtol": 1.0e-6,#1.0e-5#
                "snes_atol": 1.0e-5,#1.0e-4$
                "snes_stol": 1.0e-6,#1.0e-4$
            }
        else:
            if self.tdim == 2:
                tolerances = {
                    "ksp_rtol": 1.0e-9,
                    "ksp_atol": 1.0e-10,
                    "snes_rtol": 1.0e-9,#10
                    "snes_atol": 1.0e-8,
                    "snes_stol": 1.0e-6,
                }
            else:
                tolerances = {
                    "ksp_rtol": 1.0e-8,
                    "ksp_atol": 1.0e-8,
                    "snes_rtol": 1.0e-8,
                    "snes_atol": 1.0e-8,
                    "snes_stol": 1.0e-6,
                }

################# For tests  #####################
        mg_levels_solver_jacobi = {
#            "ksp_type": "fgmres",
            "ksp_type": "preonly",
            "ksp_norm_type": "unpreconditioned",
            "ksp_max_it": self.smoothing,
            "ksp_convergence_test": "skip",
            #"ksp_monitor_true_residual": None,
            "ksp_divtol": 1.0e10,
            "pc_type": "jacobi",
        }
###################################################

        mg_levels_solver = {
            "ksp_type": "fgmres",
            "ksp_norm_type": "unpreconditioned",
            "ksp_max_it": self.smoothing,
            "ksp_convergence_test": "skip",
#            "ksp_monitor_true_residual": None,##
            "ksp_divtol": 1.0e10,
            "pc_type": "python",
            "pc_python_type": "firedrake.PatchPC",
            "patch_pc_patch_save_operators": True,
            "patch_pc_patch_partition_of_unity": False,
            "patch_pc_patch_local_type": "multiplicative" if multiplicative else "additive",
            "patch_pc_patch_statistics": False,
            "patch_pc_patch_symmetrise_sweep": multiplicative,
            "patch_pc_patch_precompute_element_tensors": True,
            "patch_sub_ksp_type": "preonly",
            "patch_sub_pc_type": "lu",
        }
        self.configure_patch_solver(mg_levels_solver)

        if self.patch == "star":
            if multiplicative:
                mg_levels_solver["patch_pc_patch_construct_type"] = "python"
                mg_levels_solver["patch_pc_patch_construct_python_type"] = "alfi.Star"
                mg_levels_solver["patch_pc_patch_construction_Star_sort_order"] = self.problem.relaxation_direction()
            else:
                mg_levels_solver["patch_pc_patch_construct_type"] = "star"
                mg_levels_solver["patch_pc_patch_construct_dim"] = 0
        elif self.patch == "macro":
            mg_levels_solver["patch_pc_patch_construct_type"] = "python"
            mg_levels_solver["patch_pc_patch_construct_python_type"] = "alfi.MacroStar"
            mg_levels_solver["patch_pc_patch_construction_MacroStar_sort_order"] = self.problem.relaxation_direction()
        else:
            raise NotImplementedError("Unknown patch type %s" % self.patch)

        fieldsplit_0_lu = {
            "ksp_type": "preonly",
            "ksp_max_it": 1,
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mkl_pardiso" if self.use_mkl else "mumps",
            "mat_mumps_icntl_14": 150,
        }

        size = self.mesh.mpi_comm().size
        if size > 24:
            telescope_factor = round(size/24.0)
        else:
            telescope_factor = 1
        fieldsplit_0_mg = {
            "ksp_type": "richardson",
            "ksp_richardson_self_scale": False,
            "ksp_max_it": self.cycles,
            "ksp_norm_type": "unpreconditioned",
            "ksp_convergence_test": "skip",
#            "ksp_monitor_true_residual": None,##
            "pc_type": "mg",
            "pc_mg_type": "full",
            "pc_mg_log": None,
            "mg_levels": mg_levels_solver_jacobi if self.solver_type == "aljacobi" else mg_levels_solver,
            "mg_coarse_pc_type": "python",
            "mg_coarse_pc_python_type": "firedrake.AssembledPC",
            "mg_coarse_assembled": {
                "mat_type": "aij",
                "pc_type": "telescope",
                "pc_telescope_reduction_factor": telescope_factor,
                "pc_telescope_subcomm_type": "contiguous",
                "telescope_pc_type": "lu",
                "telescope_pc_factor_mat_solver_type": "superlu_dist",
            }
        }
        ####### For tests #######
        fieldsplit_0_amg = {
            "ksp_type": "richardson",
            "ksp_max_it": self.cycles,
#            "pc_type": "hypre",
#            "pc_type": "ml",
            "pc_type": "gamg",
        }

        #######################

        fieldsplit_1 = {
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "alfi.solver.DGMassInv"
        }

        use_mg = self.solver_type == "almg"

        outer_lu = {
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

        outer_fieldsplit = {
            "mat_type": "nest" if self.formulation_up else "aij",
            "ksp_max_it": 200,#4000,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            # "pc_fieldsplit_schur_factorization_type": "upper",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_schur_precondition": "user",
            "fieldsplit_0": {
                "allu": fieldsplit_0_lu,
                "almg": fieldsplit_0_mg,
                "aljacobi": fieldsplit_0_mg,
                "alamg": fieldsplit_0_amg,
                "lu": None,
                "lu-hdiv": None,
                "allu-hdiv": None,
                "almg-hdiv": None,
                "lu-p1": None,
                "simple": None}[self.solver_type],
            "fieldsplit_1": fieldsplit_1,
        }

        if self.formulation_Sup or self.formulation_Tup or self.formulation_Lup:
            outer_fieldsplit["pc_fieldsplit_0_fields"] = "0,1"
            outer_fieldsplit["pc_fieldsplit_1_fields"] = "2"
        elif self.formulation_TSup or self.formulation_LSup or self.formulation_LTup:
            outer_fieldsplit["pc_fieldsplit_0_fields"] = "0,1,2"
            outer_fieldsplit["pc_fieldsplit_1_fields"] = "3"
        elif self.formulation_LTSup:
            outer_fieldsplit["pc_fieldsplit_0_fields"] = "0,1,2,3"
            outer_fieldsplit["pc_fieldsplit_1_fields"] = "4"

        outer_simple = {
            "mat_type": "nest" if self.formulation_up else "aij",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_factorization_type": "full",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0": {
                "ksp_type": "richardson",
                "ksp_richardson_self_scale": False,
                "ksp_max_it": 1,
                "ksp_norm_type": "none",
                "ksp_convergence_test": "skip",
                "pc_type": "ml",
                "pc_mg_cycle_type": "v",
                "pc_mg_type": "full",
            },
            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "ml",
            },
            "fieldsplit_1_upper_ksp_type": "preonly",
            "fieldsplit_1_upper_pc_type": "jacobi",
        }

        outer_base = {
            "snes_type": "newtonls",
            "snes_max_it": 100,
            "snes_linesearch_type": "basic",#"l2",
            "snes_linesearch_maxstep": 1.0,
            "snes_monitor": None,
            "snes_linesearch_monitor": None,
            "snes_converged_reason": None,
            "ksp_type": "fgmres",
            "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
#            "snes_view": None,
        }

        #For the Hdiv formulations; we want to solve locally for the lifting of the jumps L (not working ATM)
        outer_sc = {
            "snes_type": "newtonls",
            "snes_max_it": 100,
            "snes_linesearch_type": "basic",#"l2",
            "snes_linesearch_maxstep": 1.0,
            "snes_monitor": None,
            "snes_linesearch_monitor": None,
            "snes_converged_reason": None,
            "ksp_type": "preonly",
            "ksp_monitor_true_residual": None,
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.SCPC",
            "pc_sc_eliminate_fields": "0",
            "condensed_field_ksp_type": "fgmres",
            "condensed_field_ksp_rtol": tolerances["ksp_rtol"],
            "condensed_field_ksp_atol": tolerances["ksp_atol"],
            "condensed_field": {
                "lu": None,
                "allu": None,
                "almg": None,
                "lu-hdiv": outer_lu,
                "allu-hdiv": outer_fieldsplit,
                "almg-hdiv": outer_fieldsplit}[self.solver_type]
        }

        outer_base = {**outer_base, **tolerances}
        outer_sc = {**outer_sc, **tolerances}

        if self.solver_type == "lu":
            outer = {**outer_base, **outer_lu}
        elif self.solver_type == "simple":
            outer = {**outer_base, **outer_simple}
        elif self.solver_type in ["lu-hdiv", "allu-hdiv", "almg-hdiv"]:
            outer = outer_sc
        else:
            outer = {**outer_base, **outer_fieldsplit}

        parameters["default_sub_matrix_type"] = "aij" if self.use_mkl or self.solver_type == "simple" else "baij"

        return outer

    def message(self, msg):
        if self.mesh.comm.rank == 0:
            warning(msg)

    def setup_adjoint(self, J):
        F = self.F
        self.z_adj = self.z.copy(deepcopy=True)
        F = replace(F, {F.arguments()[0]: self.z_adj})
        L = F + J
        F_adj = derivative(L, self.z)
        problem = NonlinearVariationalProblem(
            F_adj, self.z_adj, bcs=homogenize(self.bcs))
        solver = NonlinearVariationalSolver(problem, nullspace=self.nsp,
                                            transpose_nullspace=self.nsp,
                                            solver_parameters=self.params,
                                            options_prefix="ns_adj",
                                            appctx=self.appctx)

        solver.set_transfer_manager(self.transfermanager)
        self.solver_adjoint = solver

    def load_balance(self, mesh):
        Z = FunctionSpace(mesh, "CG", 1)
        owned_dofs = Z.dof_dset.sizes[1]
        # if macro patch (i.e. on bary mesh) then subtract the number of
        # vertices that are new due to the bary refinement.  This number can be
        # calculated by divding the number of cell on the bary refined mesh by
        # (tdim+1)
        if self.patch == "macro":
            ZZ = FunctionSpace(mesh, "DG", 0)
            owned_dofs -= ZZ.dof_dset.sizes[1]/(self.tdim+1)
        comm = Z.mesh().mpi_comm()
        min_owned_dofs = comm.allreduce(owned_dofs, op=MPI.MIN)
        mean_owned_dofs = np.mean(comm.allgather(owned_dofs))
        max_owned_dofs = comm.allreduce(owned_dofs, op=MPI.MAX)
        self.message(BLUE % ("Load balance: %i vs %i vs %i (%.3f, %.3f)" % (
            min_owned_dofs, mean_owned_dofs, max_owned_dofs,
            max_owned_dofs/mean_owned_dofs, max_owned_dofs/min_owned_dofs
        )))

class ConformingSolver(NonNewtonianSolver):

    def residual(self):

        #Define functions and test functions
        fields = self.split_variables(self.z)
        u = fields["u"]
        v = fields["v"]
        p = fields["p"]
        q = fields["q"]
        S = fields.get("S")
        ST = fields.get("ST")
        L = fields.get("L")
        LT = fields.get("LT")
        theta = fields.get("theta")
        theta_ = fields.get("theta_")
        D = sym(grad(u))

        #For the constitutive relation
        if self.formulation_Sup:
            G = self.problem.const_rel(S,D)
        elif self.formulation_TSup:
            G = self.problem.const_rel(S,D,theta)
        elif self.formulation_up:
            G = self.problem.const_rel(D)
        elif self.formulation_Tup:
            G = self.problem.const_rel(D,theta)
        elif self.formulation_LSup:
            G = self.problem.const_rel(S,L)
        elif self.formulation_Lup:
            G = self.problem.const_rel(L)

        if self.formulation_Tup or self.formulation_TSup:
            th_flux = self.problem.const_rel_temperature(theta, grad(theta))

        F = (
            self.gamma * inner(div(u), div(v))*dx
            #+ self.advect * inner(dot(grad(u), u), v)*dx
            - self.advect * inner(outer(u,u), sym(grad(v)))*dx
            - p * div(v) * dx
            - div(u) * q * dx
        )

        if self.formulation_Sup:
            F += (
                inner(S,sym(grad(v)))*dx
                - inner(G,ST) * dx
                )

        elif self.formulation_LSup:
            F += (
                inner(S,sym(grad(v)))*dx
                - inner(D - L, SL) * dx
                - inner(G, ST) * dx
                )
        elif self.formulation_up:
            F += (
                inner(G,sym(grad(v)))*dx
                )
        elif self.formulation_Tup or self.formulation_TSup:
            """
            The non-dimensional forms "natural_Ra" and "natural_Ra2" use a time-scale based on heat diffusivity and only differ in the
            choice of how to balance the pressure term. With the non-dimensional form "natural_Gr" one assumes that all the gravitational
            potential energy gets transformed into kinetic energy and so the characteristic velocity scale is chosen accordingly.
            For a non-Newtonian fluid (we have tried the Ostwald-de Waele power law relation), more non-dimensional numbers will
            arising from the constitutive relation will be necessary.
            """
            #If the Dissipation number is not defined, set it to zero.
            if not("Di" in list(self.problem.const_rel_params.keys())): self.Di = Constant(0.)
            if not("Theta" in list(self.problem.const_rel_params.keys())): self.Theta = Constant(0.)
            g = Constant((0, 1)) if self.tdim == 2 else Constant((0, 0, 1))

            F = (
                self.gamma * inner(div(u), div(v))*dx
                - p * div(v) * dx
                - div(u) * q * dx
                + inner(dot(grad(theta), u), theta_) * dx
            )

            if self.thermal_conv == "natural_Ra":
                F += (
                    self.advect * inner(dot(grad(u), u), v)*dx
                    - (self.Ra*self.Pr) * inner(theta*g, v) * dx
                    + inner(th_flux, grad(theta_)) * dx
                    + self.Di * inner((theta + self.Theta)*dot(g, u), theta_) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        self.Pr * inner(G,sym(grad(v)))*dx
                        - (self.Di/self.Ra) * inner(inner(G,sym(grad(u))), theta_) * dx
                        )
                elif self.formulation_TSup:
                    F += (
                        self.Pr * inner(S,sym(grad(v)))*dx
                        - inner(G,ST) * dx
                        - (self.Di/self.Ra) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )

            elif self.thermal_conv == "natural_Ra2":
                F += (
                    + (1./self.Pr) * self.advect * inner(dot(grad(u), u), v)*dx
                    - (self.Ra) * inner(theta*g, v) * dx
                    + inner(th_flux, grad(theta_)) * dx
                    + self.Di * inner((theta + self.Theta)*dot(g, u), theta_) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        inner(G,sym(grad(v)))*dx
                        - (self.Di/self.Ra) * inner(inner(G,sym(grad(u))),theta_) * dx
                        )
                elif self.formulation_TSup:
                    F += (
                        inner(S,sym(grad(v)))*dx
                        - inner(G,ST) * dx
                        - (self.Di/self.Ra) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )

            elif self.thermal_conv == "natural_Gr":
                F += (
                    self.advect * inner(dot(grad(u), u), v)*dx
                    - inner(theta*g, v) * dx
                    + (1./(self.Pr * sqrt(self.Gr))) * inner(th_flux, grad(theta_)) * dx
                    + self.Di * inner((theta + self.Theta)*dot(g, u), theta_) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        (1./sqrt(self.Gr)) * inner(G,sym(grad(v)))*dx
                        - (self.Di/sqrt(self.Gr)) * inner(inner(G,sym(grad(u))),theta_) * dx
                        )
                elif self.formulation_TSup:
                    F += (
                        (1./sqrt(self.Gr)) * inner(S,sym(grad(v)))*dx
                        - inner(G,ST) * dx
                        - (self.Di/sqrt(self.Gr)) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
            elif self.thermal_conv == "forced":
                """
                For the forced convection regime I choose a characteristic velocity, and so the Peclet and
                Reynolds numbers will appear in the formulation

                """
                if not("Br" in list(self.problem.const_rel_params.keys())): self.Br = Constant(0.)
                F += (
                    + self.Re * self.advect * inner(dot(grad(u), u), v)*dx
                    + (1./self.Pe) * inner(th_flux, grad(theta_)) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        inner(G,sym(grad(v)))*dx
                        - (self.Br/self.Pe) * inner(inner(G,sym(grad(u))), theta_) * dx
                        )
                elif self.formulation_TSup:
                    F += (
                        inner(S,sym(grad(v)))*dx
                        - inner(G,ST) * dx
                        - (self.Br/self.Pe) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )

        return F

class ScottVogeliusSolver(ConformingSolver):

    def function_space(self, mesh, k):
        eleth = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.tdim == 2:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1)
        else:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=5)
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k-1)
        if self.formulation_Sup or self.formulation_Lup:
            Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))
        elif self.formulation_LSup:
            Z = FunctionSpace(mesh, MixedElement([eles,eles,eleu,elep]))
        elif self.formulation_up:
            Z = FunctionSpace(mesh, MixedElement([eleu,elep]))
        elif self.formulation_TSup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eles,eleu,elep]))
        elif self.formulation_Tup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eleu,elep]))
        return Z

    def get_transfers(self):

        V = self.Z.sub(self.velocity_id)
        Q = self.Z.sub(self.pressure_id)
        if self.formulation_Sup or self.formulation_LSup or self.formulation_TSup:
            SigmaS = self.Z.sub(self.stress_id)
            stransfer = DGInjection()
            self.stransfer = stransfer
        if self.formulation_LSup:
            SigmaD = self.Z.sub(self.strain_id)
        if self.formulation_Tup or self.formulation_TSup:
            V_temp = self.Z.sub(self.temperature_id)

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
        if self.formulation_Sup or self.formulation_LSup or self.formulation_TSup:
            transfers[SigmaS.ufl_element()] = (stransfer.prolong, restrict, stransfer.inject)
        if self.formulation_LSup:
            transfers[SigmaD.ufl_element()] = (stransfer.prolong, restrict, stransfer.inject)

        return transfers

    def configure_patch_solver(self, opts):
        patchlu3d = "mkl_pardiso" if self.use_mkl else "umfpack"
        patchlu2d = "petsc"
        opts["patch_pc_patch_sub_mat_type"] = "seqaij"
        opts["patch_sub_pc_factor_mat_solver_type"] = patchlu3d if self.tdim > 2 else patchlu2d

    def distribution_parameters(self):
        return {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 2)}

class TaylorHoodSolver(ScottVogeliusSolver): #TODO: try a star patch anyway?
    """Only for MMS, not meant for multigrid..."""

    def function_space(self, mesh, k):
        eleth = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.tdim == 2:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=3)
        else:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=6)
        eleu = VectorElement("Lagrange", mesh.ufl_cell(), k)
        elep = FiniteElement("Lagrange", mesh.ufl_cell(), k-1)
        if self.formulation_Sup:
            Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))
        elif self.formulation_LSup:
            Z = FunctionSpace(mesh, MixedElement([eles,eles,eleu,elep]))
        elif self.formulation_up:
            Z = FunctionSpace(mesh, MixedElement([eleu,elep]))
        elif self.formulation_TSup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eles,eleu,elep]))
        elif self.formulation_Tup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eleu,elep]))
        return Z

    def get_transfers(self):

        V = self.Z.sub(self.velocity_id)
        Q = self.Z.sub(self.pressure_id)
        if self.formulation_Sup or self.formulation_LSup or self.formulation_TSup:
            SigmaS = self.Z.sub(self.stress_id)
            stransfer = DGInjection()
            self.stransfer = stransfer
        if self.formulation_LSup:
            SigmaD = self.Z.sub(self.strain_id)
        if self.formulation_Tup or self.formulation_TSup:
            V_temp = self.Z.sub(self.temperature_id)

        transfers = {V.ufl_element(): (prolong, restrict, inject),
                    Q.ufl_element(): (prolong, restrict, inject)}
        if self.formulation_Sup or self.formulation_LSup or self.formulation_TSup:
            transfers[SigmaS.ufl_element()] = (stransfer.prolong, restrict, stransfer.inject)
        if self.formulation_LSup:
            transfers[SigmaD.ufl_element()] = (stransfer.prolong, restrict, stransfer.inject)

        return transfers

class P1P1Solver(TaylorHoodSolver):
    """not meant for multigrid..."""

    def function_space(self, mesh, k):
        assert k in [1], "P1P1 is only meant for k=1"
#        assert self.solver_type == "lu-p1", "P1P1 only makes sense with LU (for now)"
        eleth = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.tdim == 2:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=3)
        else:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=6)
        eleu = VectorElement("CG", mesh.ufl_cell(), k)
        elep = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.formulation_Sup:
            Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))
        elif self.formulation_LSup:
            Z = FunctionSpace(mesh, MixedElement([eles,eles,eleu,elep]))
        elif self.formulation_up:
            Z = FunctionSpace(mesh, MixedElement([eleu,elep]))
        elif self.formulation_TSup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eles,eleu,elep]))
        elif self.formulation_Tup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eleu,elep]))
        return Z

    def residual(self):

        F = super().residual()

        if self.stabilisation_type_u == "gls":
            return F
        else:
            #Define functions and test functions
            fields = self.split_variables(self.z)
            u = fields["u"]
            p = fields["p"]
            q = fields["q"]

            #Stabilisation (maybe it could be cleaner by using self.strong_residual)
            h = CellDiameter(self.mesh)
            beta = Constant(0.2)
            delta = beta*h*h
            if self.formulation_Tup or self.formulation_TSup:
                g = Constant((0, 1)) if self.tdim == 2 else Constant((0, 0, 1))
                theta = fields["theta"]
                if self.thermal_conv == "natural_Ra":
                    F += - delta * inner(self.advect*dot(grad(u), u) + grad(p) - self.Ra*self.Pr*theta*g, grad(q)) * dx
                elif self.thermal_conv == "natural_Ra2":
                    F += - delta * inner((self.advect/self.Pr)*dot(grad(u), u) + grad(p) - self.Ra*theta*g, grad(q)) * dx
                elif self.thermal_conv == "natural_Gr":
                    F += - delta * inner(self.advect*dot(grad(u), u) + grad(p) - theta*g, grad(q)) * dx
                elif self.thermal_conv == "forced":
                    F += - delta * inner(self.Re*self.advect*dot(grad(u), u) + grad(p), grad(q)) * dx

            else:
                F += - delta * inner(self.advect*dot(grad(u), u) + grad(p), grad(q)) * dx

            rhs = self.problem.rhs(self.Z)
            if rhs is not None:
                if self.formulation_up or self.formulation_Sup or self.formulation_LSup or self.formulation_Lup: #Assumes the equations for S and L do NOT have right-hand-sides
                    F += delta * inner(rhs[0], grad(q)) * dx
                elif self.formulation_Tup or self.formulation_TSup:
                    F += delta * inner(rhs[1], grad(q)) * dx

            return F

    def get_jacobian(self):

        J0 = super().get_jacobian()
        if self.linearisation == "newton" or self.stabilisation_type_u == "gls":
            return J0  #Already contains the pressure stabilisation
        else:

            #Split variables for current solution and test function
            fields = self.split_variables(self.z)
            u = fields["u"]
            q = fields["q"]
            #Split trial function
            w = TrialFunction(self.Z)
            fields0 = self.split_variables(w)
            u0 = fields0["u"]
            p0 = fields0["p"]

            #Stabilisation (maybe it could be cleaner by using self.strong_residual)
            h = CellDiameter(self.mesh)
            beta = Constant(0.2)
            delta = beta*h*h
            if self.formulation_Tup or self.formulation_TSup:
                theta = fields["theta"]
                theta0 = fields0["theta"]
                g = Constant((0, 1)) if self.tdim == 2 else Constant((0, 0, 1))
                if self.thermal_conv == "natural_Ra":
                    J0 += - delta * inner(self.advect*dot(grad(u0), u) + grad(p0) - self.Ra*self.Pr*theta0*g, grad(q)) * dx
                elif self.thermal_conv == "natural_Ra2":
                    J0 += - delta * inner((self.advect/self.Pr)*dot(grad(u0), u) + grad(p0) - self.Ra*theta0*g, grad(q)) * dx
                elif self.thermal_conv == "natural_Gr":
                    J0 += - delta * inner(self.advect*dot(grad(u0), u) + grad(p0) - theta0*g, grad(q)) * dx
                elif self.thermal_conv == "forced":
                    J0 += - delta * inner(self.Re*self.advect*dot(grad(u0), u) + grad(p0), grad(q)) * dx

            else:
                J0 += - delta * inner(self.advect*dot(grad(u0), u) + grad(p0), grad(q)) * dx

            if self.linearisation == "kacanov": #Newton linearisation of the convective term
                if self.formulation_Tup or self.formulation_TSup:
                    if self.thermal_conv == "natural_Ra":
                        J0 += - delta * inner(self.advect*dot(grad(u), u0), grad(q)) * dx
                    elif self.thermal_conv == "natural_Ra2":
                        J0 += - delta * inner((self.advect/self.Pr)*dot(grad(u), u0), grad(q)) * dx
                    elif self.thermal_conv == "natural_Gr":
                        J0 += - delta * inner(self.advect*dot(grad(u), u0), grad(q)) * dx
                    elif self.thermal_conv == "forced":
                        J0 += - delta * inner(self.Re*self.advect*dot(grad(u), u0), grad(q)) * dx

                else:
                    J0 += - delta * inner(self.advect*dot(grad(u), u0), grad(q)) * dx

            return J0

class P1P0Solver(ScottVogeliusSolver):

    def function_space(self, mesh, k):
        assert k in [1], "P1P0 is only meant for k=1"
        eleth = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.tdim == 2:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=3)
        else:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=6)
        eleu = VectorElement("CG", mesh.ufl_cell(), k)
        elep = FiniteElement("DG", mesh.ufl_cell(), k-1)
        if self.formulation_Sup:
            Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))
        elif self.formulation_LSup:
            Z = FunctionSpace(mesh, MixedElement([eles,eles,eleu,elep]))
        elif self.formulation_up:
            Z = FunctionSpace(mesh, MixedElement([eleu,elep]))
        elif self.formulation_TSup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eles,eleu,elep]))
        elif self.formulation_Tup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eleu,elep]))
        return Z

    def get_transfers(self):

        Q = self.Z.sub(self.pressure_id)
        if self.formulation_Sup or self.formulation_LSup or self.formulation_TSup:
            SigmaS = self.Z.sub(self.stress_id)
            stransfer = DGInjection()
            self.stransfer = stransfer

        qtransfer = NullTransfer()
        self.qtransfer = qtransfer


        transfers = {Q.ufl_element(): (prolong, restrict, qtransfer.inject)}
        if self.formulation_Sup or self.formulation_LSup or self.formulation_TSup:
            transfers[SigmaS.ufl_element()] = (stransfer.prolong, restrict, stransfer.inject)

        return transfers

    def residual(self):

        F = super().residual()
        #Define functions and test functions
        fields = self.split_variables(self.z)
        p = fields["p"]
        q = fields["q"]

        #Stabilisation
        h = CellDiameter(self.mesh)
        beta = Constant(0.1)
        delta = h * beta
        F -=  inner(avg(delta) * jump(p),jump(q))*dS

        return F

    def get_jacobian(self):

        J0 = super().get_jacobian()
        if self.linearisation == "newton":
            return J0  #Already contains the pressure stabilisation
        else:

            #Split variables
            fields = self.split_variables(self.z)
            q = fields["q"]
            #Split trial function
            w = TrialFunction(self.Z)
            fields0 = self.split_variables(w)
            p0 = fields0["p"]

            #Stabilisation
            h = CellDiameter(self.mesh)
            beta = Constant(0.1)
            delta = h * beta
            J0 -=  inner(avg(delta) * jump(p0),jump(q))*dS

            return J0

class HDivSolver(NonNewtonianSolver):
    """
    formulation u-p: Implemented only for Newtonian problems
    formulation S-u-p: Works only for an explicit CR of the form D=D(S); implements LDG and Mixed fluxes
    formulation L-u-p: L is a lifting of the velocity jumps; this is meant for explicit non-Newtonian relations S=S(D); implements LDG and IP fluxes;
    formulation L-S-u-p: meant to work in the general implicit case; implements LDG and Mixed Fluxes.
    """

    def residual(self):

        #Define functions and test functions
        fields = self.split_variables(self.z)
        u = fields["u"]
        v = fields["v"]
        p = fields["p"]
        q = fields["q"]
        S = fields.get("S")
        ST = fields.get("ST")
        L = fields.get("L")
        LT = fields.get("LT")
        theta = fields.get("theta")
        theta_ = fields.get("theta_")
        D = sym(grad(u))


        #For the constitutive relation
        if self.formulation_Sup or self.formulation_TSup:
            assert self.fluxes in ["ldg", "mixed"], "The Hdiv S-u-p formulation has only been implemented with LDG or Mixed fluxes"
            self.message(RED % "This discretisation/formulation only makes sense for constitutive relations of the form G = D - D*(S)  !!!!")
            G = self.problem.const_rel(S, D) if self.formulation_Sup else self.problem.const_rel(S,D,theta)
        elif self.formulation_up or self.formulation_Tup:
            assert self.fluxes in ["ip"], "The Hdiv u-p formulation has only been implemented with the Interior Penalty method"
            self.message(RED % "This Hdiv interior penalty u-p formulation only makes sense for a Newtonian constitutive relation S = 2. * nu * D  !!!!")
            G = 2. * self.nu * D if self.formulation_up else self.problem.const_rel(D, theta)
        elif self.formulation_LSup or self.formulation_LTSup:
            assert self.fluxes in ["ldg", "mixed"], "The Hdiv L-S-u-p formulation has only been implemented with LDG or Mixed fluxes"
            G = self.problem.const_rel(S, D + L) if self.formulation_LSup else self.problem.const_rel(S,D+L,theta)
        elif self.formulation_Lup or self.formulation_LTup:
            assert self.fluxes in ["ip", "ldg"], "The Hdiv L-u-p formulation has only been implemented with LDG or IP fluxes"
            G = self.problem.const_rel(D + L) if self.formulation_Lup else self.problem.const_rel(D+L, theta)
        else:
            raise NotImplementedError("This formulation hasn't been implemented with Hdiv-type spaces")

        n = FacetNormal(self.Z.ufl_domain())
        h = CellDiameter(self.Z.ufl_domain())

        #Common for all formulations
        uflux_int_ = 0.5*(dot(u, n) + abs(dot(u, n)))*u
        F = (
            self.gamma * inner(div(u), div(v))*dx
            - p * div(v) * dx
            - div(u) * q * dx
            #----------  Old form --------------------#
#            - self.advect * dot(u ,div(outer(v,u)))*dx
            + self.advect * dot(v('+')-v('-'), uflux_int_('+')-uflux_int_('-'))*dS  #Upwinding
            #----------  New form --------------------#
            - self.advect * inner(outer(u,u), grad(v)) * dx
            #+ 0.5 * self.advect * div(u) * dot(u, v) * dx #This one should vanish
            #+ 0.5 * self.advect * jump(u, n) * avg(dot(u, v)) * dS #This one should vanish
#            + 0.5 * self.advect * dot(avg(u), jump(dot(u, v), n)) * dS
#            + 0.5 * self.advect * dot(u, n) * dot(u, v) * ds
#            + 0.5 * self.advect * abs(avg(dot(u, n))) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS #Upwinding
        )

        #For the jump penalisation
        U_jmp = 2. * avg(outer(u,n))
        penalty_form = "cr" #"plaw", "quadratic", "cr"
        sigma = Constant(self.ip_magic) * self.Z.sub(self.velocity_id).ufl_element().degree()**2
        sigma_ = Constant(0.5) * self.Z.sub(self.velocity_id).ufl_element().degree()**2  #Or take them equal??

        if self.formulation_up:
            F += (
                inner(G, sym(grad(v))) * dx
                - self.nu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
                - self.nu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
                + 2. * self.nu * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
            )
        elif self.formulation_Sup:
            F += (
                inner(G, ST) * dx
                - inner(avg(ST), 2*avg(outer(u, n))) * dS
                + inner(S, sym(grad(v))) * dx
                - inner(avg(S), 2*avg(outer(v, n))) * dS
            )
            if self.fluxes == "ldg":
                jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                F += (
                    sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                )
            elif self.fluxes == "mixed":
                jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                F += (
                    sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                    - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                )
        elif self.formulation_LSup:
            F += (
                inner(L, LT) * dx
                + inner(2*avg(outer(u,n)), avg(LT)) * dS
                + inner(G, ST) * dx
                + inner(S, sym(grad(v))) * dx
                - inner(avg(S), 2*avg(outer(v, n))) * dS
                )
            if self.fluxes == "ldg":
                jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                F += (
                    sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                )
            elif self.fluxes == "mixed":
                jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form) #try 1/avg(h)
                F += (
                    sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                    - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                )
        elif self.formulation_Lup:
            jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
            F += (
                inner(L, LT) * dx
                + inner(2*avg(outer(u,n)), avg(LT)) * dS
                + inner(G, sym(grad(v))) * dx
                + sigma * inner(jmp_penalty, 2*avg(outer(v,n))) * dS
                - inner(avg(G), 2*avg(outer(v, n))) * dS
                )
            if self.fluxes == "ip":
                S_rh = self.problem.const_rel(-L)
                F -= inner(avg(S_rh), 2*avg(outer(v,n))) * dS

        elif self.formulation_Tup or self.formulation_TSup or self.formulation_LTup or self.formulation_LTSup:
            """
            The non-dimensional forms "natural_Ra" and "natural_Ra2" use a time-scale based on heat diffusivity and only differ in the
            choice of how to balance the pressure term. With the non-dimensional form "natural_Gr" one assumes that all the gravitational
            potential energy gets transformed into kinetic energy and so the characteristic velocity scale is chosen accordingly.
            For a non-Newtonian fluid (we have tried the Ostwald-de Waele power law relation), more non-dimensional numbers will
            arising from the constitutive relation will be necessary.
            """
            #If the Dissipation number is not defined, set it to zero.
            if not("Di" in list(self.problem.const_rel_params.keys())): self.Di = Constant(0.)
            if not("Theta" in list(self.problem.const_rel_params.keys())): self.Theta = Constant(0.)
            g = Constant((0, 1)) if self.tdim == 2 else Constant((0, 0, 1))
            th_flux = self.problem.const_rel_temperature(theta, grad(theta))

            uflux_int_ = 0.5*(dot(u, n) + abs(dot(u, n)))*u
            uflux_int_th_ = 0.5*(dot(u, n) + abs(dot(u, n)))*theta
            F = (
                self.gamma * inner(div(u), div(v))*dx
                - p * div(v) * dx
                - div(u) * q * dx
                + inner(dot(grad(theta), u), theta_) * dx
                #+ dot(theta_('+')-theta_('-'), uflux_int_th_('+')-uflux_int_th_('-'))*dS  #Upwinding
                #Anti-symmetrization
                #+ 0.5 * dot(u,n) * theta * theta_ * ds
            )

            if self.thermal_conv == "natural_Ra":
                F += (
                    #----------  Old form --------------------#
                    - self.advect * inner(outer(u,u), grad(v)) * dx
                    + self.advect * dot(v('+')-v('-'), uflux_int_('+')-uflux_int_('-'))*dS  #Upwinding
                    #-----------New form ---------------#
                    - (self.Ra*self.Pr) * inner(theta*g, v) * dx
                    + inner(th_flux, grad(theta_)) * dx
                    + self.Di * inner((theta + self.Theta)*dot(g, u), theta_) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        self.Pr * inner(G,sym(grad(v)))*dx
                        - self.Pr * self.nu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
                        - self.Pr * self.nu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
                        + 2. * self.Pr * self.nu * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
                        - (self.Di/self.Ra) * inner(inner(G,sym(grad(u))), theta_) * dx
                        )
                elif self.formulation_LTup:
                    jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                    F += (
                        self.Pr * inner(G,sym(grad(v)))*dx
                        + inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + self.Pr * inner(G, sym(grad(v))) * dx
                        + self.Pr * sigma * inner(jmp_penalty, 2*avg(outer(v,n))) * dS
                        - self.Pr * inner(avg(G), 2*avg(outer(v, n))) * dS
                        - (self.Di/self.Ra) * inner(inner(G,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ip":
                        S_rh = self.problem.const_rel(-L)
                        F -= self.Pr * inner(avg(S_rh), 2*avg(outer(v,n))) * dS
                elif self.formulation_TSup:
                    F += (
                        self.Pr * inner(S,sym(grad(v)))*dx
                        - self.Pr * inner(avg(ST), 2*avg(outer(u, n))) * dS
                        - self.Pr * inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Di/self.Ra) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            self.Pr * sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            self.Pr * sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )
                elif self.formulation_LTSup:
                    F += (
                        inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + self.Pr * inner(S,sym(grad(v)))*dx
                        - self.Pr * inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Di/self.Ra) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form) #try 1/avg(h)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )

            elif self.thermal_conv == "natural_Ra2":
                F += (
                    + (1./self.Pr) * self.advect * inner(dot(grad(u), u), v)*dx
                    #----------  Old form --------------------#
                    - (1./self.Pr) * self.advect * inner(outer(u,u), grad(v)) * dx
                    + (1./self.Pr) * self.advect * dot(v('+')-v('-'), uflux_int_('+')-uflux_int_('-'))*dS  #Upwinding
                    #-----------New form ---------------#
                    - (self.Ra) * inner(theta*g, v) * dx
                    + inner(th_flux, grad(theta_)) * dx
                    + self.Di * inner((theta + self.Theta)*dot(g, u), theta_) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        inner(G,sym(grad(v)))*dx
                        - self.nu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
                        - self.nu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
                        + 2. * self.nu * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
                        - (self.Di/self.Ra) * inner(inner(G,sym(grad(u))),theta_) * dx
                        )
                elif self.formulation_LTup:
                    F += (
                        inner(G,sym(grad(v)))*dx
                        - inner(avg(S), 2*avg(outer(v, n))) * dS
                        + inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        - (self.Di/self.Ra) * inner(inner(G,sym(grad(u))),theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form) #try 1/avg(h)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )
                elif self.formulation_TSup:
                    F += (
                        inner(S,sym(grad(v)))*dx
                        - inner(avg(ST), 2*avg(outer(u, n))) * dS
                        - inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Di/self.Ra) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )
                elif self.formulation_LTSup:
                    F += (
                        inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + inner(S,sym(grad(v)))*dx
                        - inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Di/self.Ra) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form) #try 1/avg(h)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )

            elif self.thermal_conv == "natural_Gr":
                F += (
                    #----------  Old form --------------------#
                    - self.advect * inner(outer(u,u), grad(v)) * dx
                    + self.advect * dot(v('+')-v('-'), uflux_int_('+')-uflux_int_('-'))*dS  #Upwinding
                    #-----------New form ---------------#
                    - inner(theta*g, v) * dx
                    + (1./(self.Pr * sqrt(self.Gr))) * inner(th_flux, grad(theta_)) * dx
                    + self.Di * inner((theta + self.Theta)*dot(g, u), theta_) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        (1./sqrt(self.Gr)) * inner(G,sym(grad(v)))*dx
                        - (1./sqrt(self.Gr)) * self.nu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
                        - (1./sqrt(self.Gr)) * self.nu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
                        + (1./sqrt(self.Gr)) * 2. * self.nu * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
                        - (self.Di/sqrt(self.Gr)) * inner(inner(G,sym(grad(u))),theta_) * dx
                        )
                elif self.formulation_LTup:
                    F += (
                        inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + (1./sqrt(self.Gr)) * inner(G,sym(grad(v)))*dx
                        + (1./sqrt(self.Gr)) * sigma * inner(jmp_penalty, 2*avg(outer(v,n))) * dS
                        - (1./sqrt(self.Gr)) * inner(avg(G), 2*avg(outer(v, n))) * dS
                        - (self.Di/sqrt(self.Gr)) * inner(inner(G,sym(grad(u))),theta_) * dx
                        )
                    if self.fluxes == "ip":
                        S_rh = self.problem.const_rel(-L)
                        F -= (1./sqrt(self.Gr)) * inner(avg(S_rh), 2*avg(outer(v,n))) * dS
                elif self.formulation_TSup:
                    F += (
                        (1./sqrt(self.Gr)) * inner(S,sym(grad(v)))*dx
                        - (1./sqrt(self.Gr)) * inner(avg(ST), 2*avg(outer(u, n))) * dS
                        - (1./sqrt(self.Gr)) * inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Di/sqrt(self.Gr)) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * (1./sqrt(self.Gr)) * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            (1./sqrt(self.Gr)) *sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )
                elif self.formulation_LTSup:
                    F += (
                        inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + (1./sqrt(self.Gr)) * inner(S,sym(grad(v)))*dx
                        - (1./sqrt(self.Gr)) * inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Di/sqrt(self.Gr)) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form) #try 1/avg(h)
                        F += (
                            (1./sqrt(self.Gr)) * sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )
            elif self.thermal_conv == "forced":
                """
                For the forced convection regime I choose a characteristic velocity, and so the Peclet and
                Reynolds numbers will appear in the formulation

                """
                if not("Br" in list(self.problem.const_rel_params.keys())): self.Br = Constant(0.)
                F += (
                    #----------  Old form --------------------#
                    - self.Re * self.advect * inner(outer(u,u), grad(v)) * dx
                    + self.Re * self.advect * dot(v('+')-v('-'), uflux_int_('+')-uflux_int_('-'))*dS  #Upwinding
                    + (1./self.Pe) * inner(th_flux, grad(theta_)) * dx
                    )
                if self.formulation_Tup:
                    F += (
                        inner(G,sym(grad(v)))*dx
                        - self.nu * inner(avg(2*sym(grad(u))), 2*avg(outer(v, n))) * dS
                        - self.nu * inner(avg(2*sym(grad(v))), 2*avg(outer(u, n))) * dS
                        + 2. * self.nu * sigma/avg(h) * inner(2*avg(outer(u,n)), 2*avg(outer(v,n))) * dS
                        - (self.Br/self.Pe) * inner(inner(G,sym(grad(u))), theta_) * dx
                        )
                elif self.formulation_LTup:
                    jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                    F += (
                        inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + inner(G,sym(grad(v)))*dx
                        + sigma * inner(jmp_penalty, 2*avg(outer(v,n))) * dS
                        - inner(avg(G), 2*avg(outer(v, n))) * dS
                        - (self.Br/self.Pe) * inner(inner(G,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ip":
                        S_rh = self.problem.const_rel(-L)
                        F -= inner(avg(S_rh), 2*avg(outer(v,n))) * dS
                elif self.formulation_TSup:
                    F += (
                        inner(S,sym(grad(v)))*dx
                        - inner(avg(ST), 2*avg(outer(u, n))) * dS
                        - inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Br/self.Pe) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )
                elif self.formulation_LTSup:
                    F += (
                        inner(L, LT) * dx
                        + inner(2*avg(outer(u,n)), avg(LT)) * dS
                        + inner(S,sym(grad(v)))*dx
                        - inner(avg(S), 2*avg(outer(v, n))) * dS
                        - inner(G,ST) * dx
                        - (self.Br/self.Pe) * inner(inner(S,sym(grad(u))), theta_) * dx
                        )
                    if self.fluxes == "ldg":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                        )
                    elif self.fluxes == "mixed":
                        jmp_penalty = self.ip_penalty_jump(1./avg(h), U_jmp, form=penalty_form) #try 1/avg(h)
                        F += (
                            sigma * inner(jmp_penalty, 2*avg(outer(v, n))) * dS
                            - (sigma_*avg(h)) * inner(2*avg(S[i,j]*n[j]),avg(ST[i,j]*n[j])) * dS
                        )

        #For BCs
        def a_bc(u, v, bid, g_D, form_="cr"):
            U_jmp_bdry = outer(u - g_D, n)
            jmp_penalty_bdry = self.ip_penalty_jump(1./h, U_jmp_bdry, form=form_) #Try with 1/h?
            if self.formulation_LSup:
                abc = -inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid)
            elif self.formulation_Lup:
                abc = sigma*inner(jmp_penalty_bdry, outer(v,n))*ds(bid)
                if self.fluxes == "ldg":
                    abc -= inner(outer(v,n), G)*ds(bid)
                elif self.fluxes == "ip":
                    abc -= inner(outer(v,n), S_rh)*ds(bid)
            elif self.formulation_Sup:
                abc = -inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid)
            elif self.formulation_up:
                abc = -inner(outer(v,n),2*self.nu*sym(grad(u)))*ds(bid) - inner(outer(u-g_D,n),2*self.nu*sym(grad(v)))*ds(bid) + 2.*self.nu*(sigma/h)*inner(v,u-g_D)*ds(bid)
            elif self.formulation_Tup:
                if self.thermal_conv == "natural_Ra":
                    abc = self.Pr * ( -inner(outer(v,n),2*self.nu*sym(grad(u)))*ds(bid) - inner(outer(u-g_D,n),2*self.nu*sym(grad(v)))*ds(bid) + 2.*self.nu*(sigma/h)*inner(v,u-g_D)*ds(bid) )
                elif self.thermal_conv == "natural_Gr":
                    abc = (1./sqrt(self.Gr)) * (-inner(outer(v,n),2*self.nu*sym(grad(u)))*ds(bid) - inner(outer(u-g_D,n),2*self.nu*sym(grad(v)))*ds(bid) + 2.*self.nu*(sigma/h)*inner(v,u-g_D)*ds(bid) )
                else:
                    abc = -inner(outer(v,n),2*self.nu*sym(grad(u)))*ds(bid) - inner(outer(u-g_D,n),2*self.nu*sym(grad(v)))*ds(bid) + 2.*self.nu*(sigma/h)*inner(v,u-g_D)*ds(bid)
            elif self.formulation_TSup:
                if self.thermal_conv == "natural_Ra":
                    abc = self.Pr * (-inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid) )
                elif self.thermal_conv == "natural_Gr":
                    abc = (1./sqrt(self.Gr)) * (-inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid))
                else:
                    abc = -inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid)
            elif self.formulation_LTup:
                if self.thermal_conv == "natural_Ra":
                    abc = self.Pr * sigma * inner(jmp_penalty_bdry, outer(v,n))*ds(bid)
                    if self.fluxes == "ldg":
                        abc -= self.Pr * inner(outer(v,n), G)*ds(bid)
                    elif self.fluxes == "ip":
                        abc -= self.Pr * inner(outer(v,n), S_rh)*ds(bid)
                elif self.thermal_conv == "natural_Gr":
                    abc = (1./sqrt(self.Gr)) * sigma * inner(jmp_penalty_bdry, outer(v,n))*ds(bid)
                    if self.fluxes == "ldg":
                        abc -= (1./sqrt(self.Gr)) * inner(outer(v,n), G)*ds(bid)
                    elif self.fluxes == "ip":
                        abc -= (1./sqrt(self.Gr)) * inner(outer(v,n), S_rh)*ds(bid)
                else:
                    abc = sigma*inner(jmp_penalty_bdry, outer(v,n))*ds(bid)
                    if self.fluxes == "ldg":
                        abc -= inner(outer(v,n), G)*ds(bid)
                    elif self.fluxes == "ip":
                        abc -= inner(outer(v,n), S_rh)*ds(bid)
            elif self.formulation_LTSup:
                if self.thermal_conv == "natural_Ra":
                    abc = self.Pr * ( -inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid) )
                elif self.thermal_conv == "natural_Gr":
                    abc = (1./sqrt(self.Gr)) * ( -inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid))
                else:
                    abc = -inner(outer(v,n),S)*ds(bid) + sigma*inner(outer(v,n), jmp_penalty_bdry)*ds(bid)

            return abc

        def c_bc(u, v, bid, g_D, advect):
            #--------- Old form ---------------------#
            if g_D is None:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u
            else:
                uflux_ext = 0.5*(inner(u,n)+abs(inner(u,n)))*u + 0.5*(inner(u,n)-abs(inner(u,n)))*g_D
            c_term = advect * dot(v,uflux_ext)*ds(bid)
            #---------- New form ---------------------#
            #if g_D is None:
            #     c_term = 0.5 * advect * dot(u, n) * dot(u, v) * ds
            #else:
                #c_term = 0.5 * advect * dot(u, n) * dot(u, v) * ds(bid) - 0.5 * advect * dot(g_D, n) * dot(u, v) * ds(bid)
            return c_term
#            return None

        exterior_markers = list(self.mesh.exterior_facets.unique_markers)
        for bc in self.bcs:
            if "DG" in str(bc._function_space) or "CG" in str(bc._function_space):
                continue
            g_D = bc.function_arg
            bid = bc.sub_domain
            exterior_markers.remove(bid)
            F += a_bc(u, v, bid, g_D, form_=penalty_form)
            F += c_bc(u, v, bid, g_D, self.advect)
        for bid in exterior_markers:
            F += c_bc(u, v, bid, None, self.advect)
        return F

    def get_transfers(self):
        V = self.Z.sub(self.velocity_id)
        dgtransfer = DGInjection()
        transfers = {VectorElement("DG", V.mesh().ufl_cell(), V.ufl_element().degree()): (dgtransfer.prolong, restrict, dgtransfer.inject)}
        return transfers

    def configure_patch_solver(self, opts):
        opts["patch_pc_patch_sub_mat_type"] = "seqdense"
        opts["patch_sub_pc_factor_mat_solver_type"] = "petsc"
        opts["pc_python_type"] = "firedrake.ASMStarPC"
        opts["pc_star_backend"] = "tinyasm"

    def distribution_parameters(self):
        return {"partition": True, "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}

class RTSolver(HDivSolver):

    def function_space(self, mesh, k):
        eleth = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.tdim == 2:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1)
            eleL = VectorElement("DG", mesh.ufl_cell(), k-1, dim=3)
        else:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=5)
            eleL = VectorElement("DG", mesh.ufl_cell(), k-1, dim=6)
        eleu = FiniteElement("RT", mesh.ufl_cell(), k, variant="integral")
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k-1)
        if self.formulation_Sup:
            Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))
        elif self.formulation_Lup:
            Z = FunctionSpace(mesh, MixedElement([eleL,eleu,elep]))
        elif self.formulation_LSup:
            Z = FunctionSpace(mesh, MixedElement([eleL,eles,eleu,elep]))
        elif self.formulation_up:
            Z = FunctionSpace(mesh, MixedElement([eleu,elep]))
        elif self.formulation_TSup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eles,eleu,elep]))
        elif self.formulation_Tup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eleu,elep]))
        return Z


class BDMSolver(HDivSolver):

    def function_space(self, mesh, k):
        eleth = FiniteElement("CG", mesh.ufl_cell(), k)
        if self.tdim == 2:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1)
            eleL = VectorElement("DG", mesh.ufl_cell(), k-1, dim=3)
        else:
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=5)
            eles = VectorElement("DG", mesh.ufl_cell(), k-1, dim=6)
        eleu = FiniteElement("BDM", mesh.ufl_cell(), k, variant="integral")
        elep = FiniteElement("Discontinuous Lagrange", mesh.ufl_cell(), k-1)
        if self.formulation_Sup:
            Z = FunctionSpace(mesh, MixedElement([eles,eleu,elep]))
        elif self.formulation_Lup:
            Z = FunctionSpace(mesh, MixedElement([eleL,eleu,elep]))
        elif self.formulation_LSup:
            Z = FunctionSpace(mesh, MixedElement([eleL,eles,eleu,elep]))
        elif self.formulation_up:
            Z = FunctionSpace(mesh, MixedElement([eleu,elep]))
        elif self.formulation_TSup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eles,eleu,elep]))
        elif self.formulation_Tup:
            Z = FunctionSpace(mesh, MixedElement([eleth,eleu,elep]))
        elif self.formulation_LTup:
            Z = FunctionSpace(mesh, MixedElement([eleL,eleth,eleu,elep]))
        elif self.formulation_LTSup:
            Z = FunctionSpace(mesh, MixedElement([eleL,eleth,eles,eleu,elep]))
        return Z
