from pprint import pprint
from poiseuille import OldroydB_Poiseuille
from alfi_3f import get_default_parser, get_solver, run_solver
import os

from firedrake import *
import numpy as np

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
args, _ = parser.parse_known_args()

#Constitutive parameters 
nu_s = [1.0]
nu = Constant(nu_s[0])
G_s = [0.0, 0.5, 1.0]
G = Constant(G_s[0])
taus = []
taus_1 = [1.0]; taus.append(taus_1)
tau = Constant(taus[0][0])

const_params_list = [{"G": G_s, "tau": taus[0], "nu": nu_s}]
params_to_check = {"tau": [a[-1] for a in taus]}
print("params_to_check: ", params_to_check)

problem_ = OldroydB_Poiseuille(baseN=args.baseN, nu=nu, tau=tau, G=G)

results = {}
for Ra in list(params_to_check.values())[0]:
    results[Ra] = {}
    for s in ["velocity", "velocitygrad", "pressure", "stress", "divergence", "relvelocity", "relvelocitygrad", "relpressure", "relstress"]:
        results[Ra][s] = []
comm = None
hs = []
for nref in range(1, args.nref+1):
    args.nref = nref
    solver_ = get_solver(args, problem_)
    solver_.gamma.assign(0.)

    mesh = solver_.mesh
    h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh))
    with h.dat.vec_ro as w:
        hs.append((w.max()[1], w.sum()/w.getSize()))
    # h = Function(FunctionSpace(mesh, "DG", 0)).interpolate(CellSize(mesh)).vector().max()
    # hs.append(h)
    comm = mesh.comm

    for n in range(len(const_params_list)):
        Ra = list(params_to_check.values())[0][n]

        solver_output = run_solver(solver_, args, const_params_list[n])
        z = solver_.z
        Z = z.function_space()
        SS, u, p = z.split()
        S = solver_.stress_to_matrix(SS)
        pintegral = assemble(p*dx)
        #area = assemble(Constant(1., mesh)*dx)
        #p.assign(p - Constant(pintegral/area))

        u_ = problem_.exact_vel(Z)
        p_ = problem_.exact_pressure(Z)
        S_ = problem_.exact_stress(Z)
        pressureintegral = assemble(p_*dx)

        uerr = norm(u - u_)
        u__norm = norm(u_)
        ugraderr = norm(grad(u-u_))
        u__gradnorm = norm(grad(u_))
        perr = norm(p - p_)
        p__norm = norm(p_)
        stresserr = norm(SS - S_)
        stress__norm = norm(S_)
        div_error = norm(div(u))

        results[Ra]["velocity"].append(uerr)
        results[Ra]["velocitygrad"].append(ugraderr)
        results[Ra]["pressure"].append(perr)
        results[Ra]["stress"].append(stresserr)
        results[Ra]["relvelocity"].append(uerr/u__norm)
        results[Ra]["relvelocitygrad"].append(ugraderr/u__gradnorm)
        results[Ra]["relpressure"].append(perr/p__norm)
        results[Ra]["relstress"].append(stresserr/stress__norm)
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
        print("(broken) H1: |u-u_h|", results[Ra]["velocitygrad"])
        print("convergence orders:", convergence_orders(results[Ra]["velocitygrad"]))
        print("|p-p_h|", results[Ra]["pressure"])
        print("convergence orders:", convergence_orders(results[Ra]["pressure"]))
        print("L2 |stress-stress_h|", results[Ra]["stress"])
        print("convergence orders:", convergence_orders(results[Ra]["stress"]))
    print("gamma =", args.gamma)
    print("h =", hs)
