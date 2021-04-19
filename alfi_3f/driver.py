from alfi_3f.solver import ScottVogeliusSolver, TaylorHoodSolver, P1P1Solver, P1P0Solver #TODO: Check
from mpi4py import MPI
from firedrake.petsc import PETSc
from firedrake import *
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def get_default_parser():
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nref", type=int, default=1)
    parser.add_argument("--baseN", type=int, default=16)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--stabilisation-weight-u", type=float, default=None)
    parser.add_argument("--stabilisation-weight-t", type=float, default=None)
    parser.add_argument("--solver-type", type=str, default="almg",
                        choices=["lu", "allu", "almg", "aljacobi", "alamg", "simple"])#, "lu-reg", "lu-p1"])
    parser.add_argument("--patch", type=str, default="macro",
                        choices=["star", "macro"])
    parser.add_argument("--patch-composition", type=str, default="additive",
                        choices=["additive", "multiplicative"])
    parser.add_argument("--mh", type=str, default="bary",
                        choices=["uniform", "bary", "uniformbary"])
    parser.add_argument("--stabilisation-type-u", type=str, default=None,
                        choices=["none", "burman", "gls", "supg"])
    parser.add_argument("--stabilisation-type-t", type=str, default=None,
                        choices=["none", "burman"])
    parser.add_argument("--linearisation", type=str, default="newton",
                        choices=["newton", "picard", "kacanov"]) #kacanov=full Picard #TODO: "regularised"
    parser.add_argument("--thermal-conv", type=str, default="none",
                        choices=["none", "natural_Ra", "natural_Ra2", "natural_Gr", "forced"])
    parser.add_argument("--discretisation", type=str, required=True,
                        choices=["sv","th","p1p1","p1p0"]) #Want: hdiv-ldg
    parser.add_argument("--gamma", type=float, default=1e4)
    parser.add_argument("--clear", dest="clear", default=False,
                        action="store_true")
    parser.add_argument("--time", dest="time", default=False,
                        action="store_true")
    parser.add_argument("--mkl", dest="mkl", default=False,
                        action="store_true")
    parser.add_argument("--checkpoint", dest="checkpoint", default=False,
                        action="store_true")
    parser.add_argument("--paraview", dest="paraview", default=False,
                        action="store_true")
    parser.add_argument("--const-rel-output", dest="const_rel_output", default=False,
                        action="store_true")
    parser.add_argument("--restriction", dest="restriction", default=False,
                        action="store_true")
    parser.add_argument("--no-convection", dest="no_convection", default=False,
                        action="store_true")
    parser.add_argument("--rebalance", dest="rebalance", default=False,
                        action="store_true")
    parser.add_argument("--high-accuracy", dest="high_accuracy", default=False,
                        action="store_true")
    parser.add_argument("--low-accuracy", dest="low_accuracy", default=False,
                        action="store_true")
    parser.add_argument("--smoothing", type=int, default=None)
    parser.add_argument("--cycles", type=int, default=None)
    return parser

def visualisation(solver, file_, time):  #FIXME: What should we do here?
    raise NotImplementedError
#    if solver.formulation_Sup:
#        (S_,u,p) = solver.z.split()
#        k = solver.z.sub(1).ufl_element().degree()
#    elif solver.formulation_up:
#        (u,p) = solver.z.split()
#        k = solver.z.sub(0).ufl_element().degree()
#        S = solver.problem.const_rel(sym(grad(u)))
#    elif solver.formulation_TSup:
#        (theta,S_,u,p) = solver.z.split()
#        k = solver.z.sub(2).ufl_element().degree()
#    elif solver.formulation_Tup:
#        (theta,u,p) = solver.z.split()
#        k = solver.z.sub(1).ufl_element().degree()
#        S = solver.problem.const_rel(sym(grad(u)))
#
#    if solver.formulation_Sup or solver.formulation_LSup or solver.formulation_TSup:
#        if solver.tdim == 2:
#            (S_1,S_2) = split(S_)
#            S = as_tensor(((S_1,S_2),(S_2,-S_1)))
#        else:
#            (S_1,S_2,S_3,S_4,S_5) = split(S_)
#            S = as_tensor(((S_1,S_2,S_3),(S_2,S_5,S_4),(S_3,S_4,-S_1-S_5)))
#    if solver.formulation_LSup:
#        if solver.tdim == 2:
#            (D_1,D_2) = split(D_)
#            D = as_tensor(((D_1,D_2),(D_2,-D_1)))
#        else:
#            (D_1,D_2,D_3,D_4,D_5) = split(D_)
#            D = as_tensor(((D_1,D_2,D_3),(D_2,D_5,D_4),(D_3,D_4,-D_1-D_5)))
#    else:
#        D = sym(grad(u))
#
#    SS = project(S,TensorFunctionSpace(solver.z.ufl_domain(),"DG",k-1))
#    DD = project(D,TensorFunctionSpace(solver.z.ufl_domain(),"DG",k-1))
#    u.rename("Velocity")
#    p.rename("Pressure")
#    SS.rename("Stress")
#    file_.write(solver.visprolong(u),solver.visprolong(theta),time=time)  #Try just velocity for now
#    file_.write(solver.visprolong(u),solver.visprolong(p),solver.visprolong(SS),solver.visprolong(DD),time=time)


#def const_relation_output(solver,directory):#FIXME: This is going to take more work... is it worth it?
#    file_= directory+"const_rel-"
#    for param_str in solver.problem.const_rel_params.keys():
#        file_ += param_str + "-%f-"%(float(getattr(solver, param_str)))
#    if solver.form_Sup:
#        (S_,u,p) = solver.z.split()
#        (S_1,S_2) = split(S_)
#        S = as_tensor(((S_1,S_2),(S_2,-S_1)))
#        k = solver.z.sub(0).ufl_element().degree()
#    else:
#        (u,p) = solver.z.split()
#        S = solver.problem.const_rel(sym(grad(u)))
#        k = solver.z.sub(0).ufl_element().degree()-1
#                                                                          #TODO: Allow different elements
#    gradsym_norm_squared = project(inner(sym(grad(u)), sym(grad(u))), FunctionSpace(solver.z.ufl_domain(),"DG",2*k))
#    gradsym_norm_loc = np.sqrt(np.absolute(gradsym_norm_squared.vector().get_local()))
#    stress_norm_squared = project(inner(S, S), FunctionSpace(solver.z.ufl_domain(),"DG",2*k))    #Question:Another way of computing this?
#    stress_norm_loc = np.sqrt(np.absolute(stress_norm_squared.vector().get_local()))
#    np.savetxt(file_+".out",(stress_norm_loc, gradsym_norm_loc))
#
#    t = np.arange(0.0001, 5, 0.005)  #TODO: This works for Bingham. We should make it general...
#    s0 = 2.*float(solver.nu)*t + float(solver.tau)
#
#    fig, ax = plt.subplots()
#
#    aq = plt.figure(1)
#    plt.clf()
#
#    plt.vlines(x=0,ymin=0.,ymax=float(solver.tau),color='r')
#    plt.plot(t,s0,color='r')
#    plt.scatter(gradsym_norm_loc,stress_norm_loc,marker = "D",s=1.5)
#
#    plt.xlim(0, 4.6)
#    plt.ylim(-0.01, 3.)
#
#    plt.rc('font', size=18)
#    plt.rc('axes', titlesize=20)
#
#    plt.xlabel(r'$|\mathbf{D}|$')
#    plt.ylabel(r'$|\mathbf{S}|$')
#
#    ax.grid()
#    aq.tight_layout()
#    ax.tick_params(labelsize=13)
#
#    fig.savefig(file_+".png")


def get_solver(args, problem, hierarchy_callback=None):
    solver_t = {"sv": ScottVogeliusSolver,
                "th": TaylorHoodSolver,
                "p1p1": P1P1Solver,
                "p1p0": P1P0Solver}[args.discretisation]

    solver = solver_t(
        problem,
        solver_type=args.solver_type,
        stabilisation_type_u=args.stabilisation_type_u,
        stabilisation_type_t=args.stabilisation_type_t,
        linearisation=args.linearisation,
        nref=args.nref,
        k=args.k,
        gamma=args.gamma,
        patch=args.patch,
        use_mkl=args.mkl,
        supg_method="shakib",
        stabilisation_weight_u=args.stabilisation_weight_u,
        hierarchy=args.mh,
        patch_composition=args.patch_composition,
        thermal_conv=args.thermal_conv,
        restriction=args.restriction,
        smoothing=args.smoothing,
        cycles=args.cycles,
        rebalance_vertices=args.rebalance,
        high_accuracy=args.high_accuracy,
        low_accuracy=args.low_accuracy,
        hierarchy_callback=hierarchy_callback,
        no_convection=args.no_convection,
        exactly_div_free = True if args.discretisation in ["sv", "hdiv-ldg"] else False
    )
    return solver


def performance_info(comm, solver):
        if comm.rank == 0:
            print(BLUE % "Some performance info:")
        events = ["MatMult", "MatSolve", "PCSetUp", "PCApply", "PCPATCHSolve", "PCPATCHApply", "KSPSolve_FS_0",  "KSPSolve_FS_Low", "KSPSolve", "SNESSolve", "ParLoopExecute", "ParLoopCells", "SchoeberlProlong", "SchoeberlRestrict", "inject", "prolong", "restriction", "MatFreeMatMult", "MatFreeMatMultTranspose", "DMPlexRebalanceSharedPoints", "PCPatchComputeOp", "PCPATCHScatter"]
        perf = dict((e, PETSc.Log.Event(e).getPerfInfo()) for e in events)
        perf_reduced = {}
        for k, v in perf.items():
            perf_reduced[k] = {}
            for kk, vv in v.items():
                perf_reduced[k][kk] = comm.allreduce(vv, op=MPI.SUM) / comm.size
        perf_reduced_sorted = [(k, v) for (k, v) in sorted(perf_reduced.items(), key=lambda d: -d[1]["time"])]
        if comm.rank == 0:
            for k, v in perf_reduced_sorted:
                print(GREEN % (("%s:" % k).ljust(30) + "Time = % 6.2fs, Time/1kdofs = %.2fs" % (v["time"], 1000*v["time"]/solver.Z.dim())))
            time = perf_reduced_sorted[0][1]["time"]
            print(BLUE % ("% 5.1fs \t % 4.2fs \t %i" % (time, 1000*time/solver.Z.dim(), solver.Z.dim())))


def run_solver(solver, args, const_rel_params, predictors = {}):
    """
    args contains the arguments for the solver
    const_rel_params is a dictionary of the form {"nu": nus} where nus is a list
    predictors is a dictionary of the form {"eps": "secant"}
    """
    if args.time:
        PETSc.Log.begin()
    problemsize = solver.Z.dim()
    outdir = "output/%i/" % problemsize
    chkptdir = "checkpoint/%i/" % problemsize
    if args.clear:
        shutil.rmtree(chkptdir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)
    comm = solver.mesh.mpi_comm()
    comm.Barrier()
    if args.paraview:
        file_parav = outdir + "solution-"

        for param_str, params_ in const_rel_params.items():
            file_parav += "-" + param_str + "-%f"%(float(params_[-1]))
        file_parav += ".pvd"
        pvdf = File(file_parav)
    if args.checkpoint:
        os.makedirs(chkptdir, exist_ok=True)
    if args.const_rel_output:
        os.makedirs(outdir + "const_rel/", exist_ok=True)
    results = {}

    #Set all the initial parameters
    for param_str, param_list in const_rel_params.items():
        getattr(solver, param_str).assign(param_list[0])

    #Solve over all the parameters
    counter = 0

    for param_str in const_rel_params.keys():
        for param in const_rel_params[param_str][counter:]:
            getattr(solver, param_str).assign(param)

            #Create a string with the current parameters for stdout
            string_of_parameters="solution-"
            for param_str_ in const_rel_params.keys():
                string_of_parameters+=param_str_+"-%s"%(float(getattr(solver, param_str_)))


            try:
#                raise Exception#########
                #Store last computed solution in case we use secant continuation next
                prev = solver.z.copy(deepcopy=True)

                #Load solution for this parameter if already computed
                with DumbCheckpoint(chkptdir + string_of_parameters, mode=FILE_READ) as checkpoint:
                    checkpoint.load(solver.z, name="up_"+string_of_parameters)
                #Prepare param_last and param_last2 in case we use secant prediction next
                getattr(solver, param_str+"_last2").assign(getattr(solver, param_str+"_last"))
                getattr(solver, param_str+"_last").assign(getattr(solver, param_str))
                solver.z_last.assign(prev)

            except:
                predictor = predictors.get(param_str,"trivial")
                (z, info_dict) = solver.solve(param_str, const_rel_params.keys(),predictor)
                results[(param_str,param)] = info_dict
                if args.checkpoint:
                    with DumbCheckpoint(chkptdir + string_of_parameters, mode=FILE_UPDATE) as checkpoint:
                        checkpoint.store(solver.z, name="up_"+string_of_parameters)
            counter = 1
        if args.paraview:
            visualisation(solver, pvdf, float(getattr(solver,param_str)))#param

    if args.const_rel_output:
        const_relation_output(solver,outdir+"const_rel/")

    if comm.rank == 0:
        for pars in results:
            print(results[pars])

    if args.time:
        performance_info(comm, solver)
    return results
