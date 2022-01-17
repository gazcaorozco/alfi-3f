## python mms.py --dim 2 --rheol newtonian --zdamping 1.0 --mh bary --patch macro --solver-type lu --gamma 0.0 --nref 2 --k 2 --linearisation newton --discretisation synovialSV --scalar-conv forced2
from firedrake import *

from alfi_3f import get_default_parser, get_solver, run_solver
from alfi_3f.other_models.synovial import SynovialSVSolver
from mms2d import SynovialMMS2D
from mms3d import SynovialMMS3D

import os
from pprint import pprint
import numpy as np
#import inflect
#eng = inflect.engine()

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True,
                    choices=[2, 3])
parser.add_argument("--fields", type=str, default="cup",
                        choices=["cup"])
parser.add_argument("--rheol", type=str, default="newtonian",
                        choices=["synovial", "power-law", "newtonian"])
parser.add_argument("--variation", type=str, default="Pe",
                       choices=["alpha", "Re", "Pe"])
parser.add_argument("--zdamping", type=float, default=1.0)
args, _ = parser.parse_known_args()

assert args.scalar_conv == "forced2", "This model only works with 'forced2' scalar_conv"
advect = Constant(1.)
if args.no_convection: advect.assign(0.)
zdamping = args.zdamping

#Rheological parameters
if args.variation == "Pe":
    #Common to all models
    betas = [1e-4]; beta = Constant(betas[0])
    epss = [0.001]; eps = Constant(epss[0])
    Re_s = [1.0]; Re = Constant(Re_s[0])
    Pe_s = []
    Pe_s1 = [1]; Pe_s.append(Pe_s1)
    Pe_s2 = [100]; Pe_s.append(Pe_s2)
    Pe_s3 = [1000]; Pe_s.append(Pe_s3)
    Pe_s4 = [1e4]; Pe_s.append(Pe_s4)
    Pe_s5 = [1e5]; Pe_s.append(Pe_s5)
    Pe = Constant(Pe_s[0][0])
    if args.rheol == "synovial":
        alphas = [0.0, 1.0, 2.0]; alpha = Constant(alphas[0]) #Newtonian alpha=0
        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]; alpha = Constant(alphas[0]) #Newtonian alpha=0

    elif args.rheol == "power-law":
        alphas = [2.0, 1.6]; alpha = Constant(alphas[0]) #Newtonian alpha=2

    else:
        #Dummy parameters
        alphas = [2.0]; alpha = Constant(alphas[0])
    const_params_list = [{"beta": betas, "eps": epss, "Re": Re_s, "alpha": alphas, "Pe": Pe_s[0]}]
#    for Pe_ in Pe_s[1:]:
    for Pe_ in Pe_s[1:2]:
        const_params_list.append({"beta": [betas[-1]], "eps": [epss[-1]], "Re": [Re_s[-1]], "alpha": [alphas[-1]], "Pe": Pe_})
    params_to_check = {"Pe": [a[-1] for a in Pe_s]}

else: #TODO: Variation in other parameters
    raise NotImplementedError
print("We'll compute the erros for: ", params_to_check)

if args.dim == 2:
    if args.fields == "cup":
        problem_ = SynovialMMS2D(advect=advect, rheol=args.rheol, Pe=Pe, Re=Re, alpha=alpha, beta=beta, eps=eps, zdamping=zdamping)
    else:
        raise NotImplementedError
else:
    if args.fields == "cup":
        problem_ = SynovialMMS3D(advect=advect, rheol=args.rheol, Pe=Pe, Re=Re, alpha=alpha, beta=beta, eps=eps, zdamping=zdamping)
    else:
        raise NotImplementedError

results = {}
for Ra in list(params_to_check.values())[0]:
    results[Ra] = {}
    for s in ["velocity", "velocitygrad", "pressure", "concentration", "concentrationgrad", "stress", "divergence", "relvelocity", "relvelocitygrad", "relpressure", "relconcentration", "relconcentrationgrad", "relstress"]:
        results[Ra][s] = []
comm = None
hs = []
for nref in range(1, args.nref+1):
    args.nref = nref
    solver_ = get_solver(args, problem_)
#    solver_.gamma.assign(0.)
#    problem_.interpolate_initial_guess(solver_.z)
#    solver_.no_convection = True
#    results0 = run_solver(solver_,args, {"r": [2.0]})
#    solver_.no_convection = False
    mesh = solver_.mesh
    h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh))
    with h.dat.vec_ro as w:
        hs.append((w.max()[1], w.sum()/w.getSize()))
    # h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh)).vector().max()
    # hs.append(h)
    comm = mesh.comm

    for n in range(len(const_params_list)):
        Ra = list(params_to_check.values())[0][n]
        print("Ra ---   ",Ra)

        solver_output = run_solver(solver_, args, const_params_list[n])
        z = solver_.z
        Z = z.function_space()
        if args.fields == "cup":
            conc, u, p = z.split()
            S = problem_.const_rel(sym(grad(u)), conc)
            (conc_, u_, p_) = problem_.exact_solution(Z)
            S_ = problem_.const_rel(sym(grad(u_)), conc_)

        else:
            raise NotImplementedError

        veldiv = norm(div(u))
        pressureintegral = assemble(p_ * dx)
        pintegral = assemble(p*dx)

        if problem_.rheol == "power-law":
            r_exp = float(solver_.alpha)
            r_exp_conj = r_exp/(r_exp - 1.)
        else:    #TODO: How do we actually measure the errors in the synovial case?
            r_exp = 2.0
            r_exp_conj = 2.0
#        i,j = indices(2)

        uerr = pow(assemble((pow((u[i]-u_[i])*(u[i]-u_[i]),r_exp/2.))*dx),1./r_exp)
        u__norm = pow(assemble((pow(u_[i]*u_[i],r_exp/2.))*dx),1./r_exp)
        ugraderr = pow(assemble(pow((Dx(u[i] - u_[i],j))*(Dx(u[i] - u_[i],j)),r_exp/2.)*dx),1./r_exp)
        u__gradnorm = pow(assemble(pow((Dx(u_[i],j))*(Dx(u_[i],j)),r_exp/2.)*dx),1./r_exp)
        perr = pow(assemble(pow(abs(inner(p-p_,p-p_)),r_exp_conj/2.)*dx),1./r_exp_conj)
        p__norm = pow(assemble(pow(abs(inner(p_,p_)),r_exp_conj/2.)*dx),1./r_exp_conj)
        concerr = norm(conc_-conc)
        conc__norm = norm(conc_)
        gradconcerr = norm(grad(conc_ - conc))
        conc__gradnorm = norm(grad(conc_))
        stresserr =  pow(assemble((pow((S[i,j] - S_[i,j])*(S[i,j] - S_[i,j]), r_exp_conj/2.))*dx), 1./r_exp_conj)
        stress__norm =  pow(assemble((pow(S_[i,j]*S_[i,j], r_exp_conj/2.))*dx), 1./r_exp_conj)

        results[Ra]["velocity"].append(uerr)
        results[Ra]["velocitygrad"].append(ugraderr)
        results[Ra]["pressure"].append(perr)
        results[Ra]["concentration"].append(concerr)
        results[Ra]["concentrationgrad"].append(gradconcerr)
        results[Ra]["stress"].append(stresserr)
        results[Ra]["relvelocity"].append(uerr/u__norm)
        results[Ra]["relvelocitygrad"].append(ugraderr/u__gradnorm)
        results[Ra]["relpressure"].append(perr/p__norm)
        results[Ra]["relconcentration"].append(concerr/conc__norm)
        results[Ra]["relconcentrationgrad"].append(gradconcerr/conc__gradnorm)
        results[Ra]["relstress"].append(stresserr/stress__norm)
        results[Ra]["divergence"].append(veldiv)
        if comm.rank == 0:
            print("|div(u_h)| = ", veldiv)
            print("p_exact * dx = ", pressureintegral)
            print("p_approx * dx = ", pintegral)

if comm.rank == 0:
    str_power_law_W = "W^{1,r}: " if args.rheol == "power-law" else "H1: "
    str_power_law_L = "L^{r'}: " if args.rheol == "power-law" else "L2: "
    for Ra in list(params_to_check.values())[0]:
        print("Results for %s ="%list(params_to_check.keys())[0], Ra)
        print("|u-u_h|", results[Ra]["velocity"])
        print("convergence orders:", convergence_orders(results[Ra]["velocity"]))
        print(str_power_law_W+"|u-u_h|", results[Ra]["velocitygrad"])
        print("convergence orders:", convergence_orders(results[Ra]["velocitygrad"]))
        print(str_power_law_L+"|p-p_h|", results[Ra]["pressure"])
        print("convergence orders:", convergence_orders(results[Ra]["pressure"]))
        print(str_power_law_L+"|stress-stress_h|", results[Ra]["stress"])
        print("convergence orders:", convergence_orders(results[Ra]["stress"]))
        print("L2: |c-c_h|", results[Ra]["concentration"])
        print("convergence orders:", convergence_orders(results[Ra]["concentration"]))
        print("H1: |c-c_h|", results[Ra]["concentrationgrad"])
        print("convergence orders:", convergence_orders(results[Ra]["concentrationgrad"]))
    print("gamma =", args.gamma)
    print("h =", hs)
