from pprint import pprint
from mms2d import CarreauMMS_up, CarreauMMS_Sup, CarreauMMS_Lup, CarreauMMS_LSup
from mms3d import CarreauMMS3D_up, CarreauMMS3D_Sup, CarreauMMS3D_Lup, CarreauMMS3D_LSup
from alfi_3f import get_default_parser, get_solver, run_solver
import os

from firedrake import *
import numpy as np
#import inflect
#eng = inflect.engine()

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True,
                    choices=[2, 3])
parser.add_argument("--fields", type=str, default="Sup",
                        choices=["Sup", "up", "Lup", "LSup"])
parser.add_argument("--variation", type=str, default="nu",
                        choices=["nu", "r"])
args, _ = parser.parse_known_args()

if args.fields in ["up", "Sup"] and args.discretisation in ["bdm1p0", "rt1p0"]:
    args.variation = "nu"

if args.variation == "r":
    #Constitutive parameters (to study variations in r)
    nu_s = [1.0]
    nu = Constant(nu_s[0])
    epss = [1e-4]
    eps = Constant(epss[0])
    r_s = []   ##The errors will be computed for the last element in each list
    #Shear-thickenning
    #r_s1 = [2.0]; r_s.append(r_s1)
    #r_s2 = [2.7]; r_s.append(r_s2)
    #r_s3 = [3.5]; r_s.append(r_s3)
    ##Shear-thinning
    r_s1 = [2.0]; r_s.append(r_s1)
    r_s2 = [1.8]; r_s.append(r_s2)
    r_s3 = [1.6]; r_s.append(r_s3)
    r = Constant(r_s[0][0])

    const_params_list = [{"nu": nu_s, "eps": epss, "r": r_s[0]}]
    for rr in r_s[1:]:
        const_params_list.append({"nu": [nu_s[-1]], "eps": [epss[-1]], "r": rr})
    params_to_check = {"r": [a[-1] for a in r_s]}
    print("params_to_check: ", params_to_check)

elif args.variation == "nu":
    #Constitutive parameters (to study variations in nu)
    epss = [1e-4]
    eps = Constant(epss[0])
    #Power-law exponent
    if args.fields in ["up", "Sup"] and args.discretisation in ["bdm1p0", "rt1p0"]:
        r_s = [2.0]
    else:
        r_s = [2.0]#, 1.8] #1.8,2.5
    r = Constant(r_s[0])
    #Viscosities
    nu_s = []
    Res1 = [1]; nu_s1 = [1./re for re in Res1]; nu_s.append(nu_s1)
    Res2 = [9, 10]; nu_s2 = [1./re for re in Res2]; nu_s.append(nu_s2)
    Res3 = [50, 90, 100]; nu_s3 = [1./re for re in Res3]; nu_s.append(nu_s3)
    Res4 = [400, 500, 900, 1000]; nu_s4 = [1./re for re in Res4]; nu_s.append(nu_s4)
    nu = Constant(nu_s[0][0])

    const_params_list = [{"eps": epss, "r": r_s, "nu": nu_s[0]}]
    for nu_ in nu_s[1:]:
        const_params_list.append({"eps": [epss[-1]], "r": [r_s[-1]], "nu": nu_})
    params_to_check = {"nu": [a[-1] for a in nu_s]}
    print("params_to_check: ", params_to_check)


var_problem = {
    2: {
        "up": CarreauMMS_up,
        "Sup": CarreauMMS_Sup,
        "Lup": CarreauMMS_Lup,
        "LSup": CarreauMMS_LSup,
    },
    3: {
        "up": CarreauMMS3D_up,
        "Sup": CarreauMMS3D_Sup,
        "Lup": CarreauMMS3D_Lup,
        "LSup": CarreauMMS3D_LSup,
    }
}[args.dim][args.fields]
problem_ = var_problem(baseN=args.baseN, r=r, nu=nu, eps=eps)

results = {}
for Ra in list(params_to_check.values())[0]:
    results[Ra] = {}
    for s in ["velocity", "velocitygrad", "pressure", "stress", "quasi-norm", "divergence", "relvelocity", "relvelocitygrad", "relpressure", "relstress", "relquasi-norm"]:
        results[Ra][s] = []
comm = None
hs = []
for nref in range(1, args.nref+1):
    args.nref = nref
    #Reset parameter to the initial value
    setattr(problem_, list(params_to_check.keys())[0], Constant(list(params_to_check.values())[0][0]))
    solver_ = get_solver(args, problem_)
#    solver_.gamma.assign(0.)

#    #Solve a Stokes problem first and use as initial guess
#    problem_.interpolate_initial_guess(solver_.z)
#    solver_.no_convection = True
#    results0 = run_solver(solver_, args, {"r": [2.0]})
#    solver_.no_convection = args.no_convection

    mesh = solver_.mesh
    h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh))
    with h.dat.vec_ro as w:
        hs.append((w.max()[1], w.sum()/w.getSize()))
    # h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh)).vector().max()
    # hs.append(h)
    comm = mesh.comm

    for n in range(len(const_params_list)):
        Ra = list(params_to_check.values())[0][n]
        #Make sure the appropriate parameters are used in the CR (for some reason we need this here...)
        setattr(problem_, list(params_to_check.keys())[0], Constant(Ra))
        setattr(solver_, list(params_to_check.keys())[0], Constant(Ra))

        solver_output = run_solver(solver_, args, const_params_list[n])
        z = solver_.z
        Z = z.function_space()
        if args.fields == "up":
            u, p = z.split()
            S = problem_.const_rel(sym(grad(u)))
        elif args.fields == "Sup":
            SS, u, p = z.split()
            S = solver_.stress_to_matrix(SS)
        elif args.fields == "Lup":
            _, u, p = z.split()
            S = problem_.const_rel(sym(grad(u)))
        elif args.fields == "LSup":
            _, SS, u, p = z.split()
            S = solver_.stress_to_matrix(SS)
        S_F = problem_.quasi_norm(sym(grad(u)))
        pintegral = assemble(p*dx)
        area = assemble(Constant(1., mesh)*dx)
        p.assign(p - Constant(pintegral/area))

        u_ = problem_.exact_velocity(Z)
        p_ = problem_.exact_pressure(Z)
        S_ = problem_.exact_stress(Z)
        S_F_exact = problem_.quasi_norm(sym(grad(u_)))
        pressureintegral = assemble(p_*dx)

        r_exp = float(solver_.r)
        r_exp_conj = r_exp/(r_exp - 1.)

        uerr = pow(assemble((pow((u[i]-u_[i])*(u[i]-u_[i]),r_exp/2.))*dx),1./r_exp)
        u__norm = pow(assemble((pow(u_[i]*u_[i],r_exp/2.))*dx),1./r_exp)
        ugraderr = pow(assemble(pow((Dx(u[i] - u_[i],j))*(Dx(u[i] - u_[i],j)),r_exp/2.)*dx),1./r_exp)
        u__gradnorm = pow(assemble(pow((Dx(u_[i],j))*(Dx(u_[i],j)),r_exp/2.)*dx),1./r_exp)
        perr = pow(assemble(pow(abs(inner(p-p_,p-p_)),r_exp_conj/2.)*dx),1./r_exp_conj)
        p__norm = pow(assemble(pow(abs(inner(p_,p_)),r_exp_conj/2.)*dx),1./r_exp_conj)
        stresserr =  pow(assemble((pow((S[i,j] - S_[i,j])*(S[i,j] - S_[i,j]), r_exp_conj/2.))*dx), 1./r_exp_conj)
        stress__norm =  pow(assemble((pow(S_[i,j]*S_[i,j], r_exp_conj/2.))*dx), 1./r_exp_conj)
        qnorm_error =  pow(assemble((pow((S_F[i,j] - S_F_exact[i,j])*(S_F[i,j] - S_F_exact[i,j]), r_exp_conj/2.))*dx), 1./r_exp_conj)
        qnorm_norm =  pow(assemble((pow(S_F_exact[i,j]*S_F_exact[i,j], r_exp_conj/2.))*dx), 1./r_exp_conj)
        div_error = norm(div(u))

        results[Ra]["velocity"].append(uerr)
        results[Ra]["velocitygrad"].append(ugraderr)
        results[Ra]["pressure"].append(perr)
        results[Ra]["stress"].append(stresserr)
        results[Ra]["quasi-norm"].append(qnorm_error)
        results[Ra]["relvelocity"].append(uerr/u__norm)
        results[Ra]["relvelocitygrad"].append(ugraderr/u__gradnorm)
        results[Ra]["relpressure"].append(perr/p__norm)
        results[Ra]["relstress"].append(stresserr/stress__norm)
        results[Ra]["relquasi-norm"].append(qnorm_error/qnorm_norm)
        results[Ra]["divergence"].append(div_error)
        if comm.rank == 0:
            print("|div(u_h)| = ", div_error)
            print("p_exact * dx = ", pressureintegral)
            print("p_approx * dx = ", pintegral)

if comm.rank == 0:
    for Ra in list(params_to_check.values())[0]:
        print("Results for %s ="%list(params_to_check.keys())[0], Ra)
        print("|u-u_h|", results[Ra]["velocity"])
        print("convergence orders:", convergence_orders(results[Ra]["velocity"]))
        print("(broken) W^{1,r}: |u-u_h|", results[Ra]["velocitygrad"])
        print("convergence orders:", convergence_orders(results[Ra]["velocitygrad"]))
        print("|p-p_h|", results[Ra]["pressure"])
        print("convergence orders:", convergence_orders(results[Ra]["pressure"]))
        print("Lr' |stress-stress_h|", results[Ra]["stress"])
        print("convergence orders:", convergence_orders(results[Ra]["stress"]))
        print("Lr' |Fstress-Fstress_h|", results[Ra]["quasi-norm"])
        print("convergence orders:", convergence_orders(results[Ra]["quasi-norm"]))
    print("gamma =", args.gamma)
    print("h =", hs)

##Test plot
#u.rename("Velocity")
#p.rename("Pressure")
##u_ee = interpo(problem_.exact_velocity(Z), VectorFunctionSpace(mesh, "CG", 3))
#u_ee = Function(VectorFunctionSpace(mesh, "CG", 3)).interpolate(problem_.exact_velocity(Z))
#u_ee.rename("exact_vel")
#File("plot_test/plot_alfi3f.pvd").write(u,p,u_ee)


#    for Ra in params_to_check:
#        print("%%Ra = %i" % Ra)
#        print("\\pgfplotstableread[col sep=comma, row sep=\\\\]{%%")
#        print("hmin, havg, error_v, error_vgrad, error_p, error_th, error_thgrad, error_str, relerror_v, relerror_vgrad, relerror_p, relerror_th, relerror_thgrad, relerror_str, div\\\\")
#        for i in range(len(hs)):
#            print(",".join(map(str, [hs[i][0], hs[i][1], results[Ra]["velocity"][i], results[Ra]["velocitygrad"][i], results[Ra]["pressure"][i], results[Ra]["stress"][i], results[Ra]["quasi-norm"][i], results[Ra]["relvelocity"][i], results[Ra]["relvelocitygrad"][i], results[Ra]["relpressure"][i], results[Ra]["relstress"][i], results[Ra]["relquasi-norm"][i], results[Ra]["divergence"][i]])) + "\\\\")
##        def numtoword(num):
##            return eng.number_to_words(num).replace(" ", "").replace("-","")
#        name = "gamma" \
#            + args.discretisation.replace("0", "zero")
#        print("}\\%s" % name)
