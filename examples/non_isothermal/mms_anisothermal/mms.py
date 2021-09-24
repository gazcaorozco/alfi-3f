#python mms.py --patch macro --mh bary --k 2 --dim 2 --fields Tup --discretisation sv --thermal-conv natural_Ra --temp-dependent viscosity-conductivity --gamma 0.0 --solver-type lu --cycles 1 --smoothing 5 --stabilisation-type-u none --fluxes ip --nref 3

from pprint import pprint
from mms2d import OBCavityMMS_Tup, OBCavityMMS_TSup, OBCavityMMS_LTup, OBCavityMMS_LTSup
#from mms3d import OBCavityMMS3D_up, OBCavityMMS3D_Sup
from alfi_3f import get_default_parser, get_solver, run_solver
import os
from firedrake import *
import numpy as np
import inflect
eng = inflect.engine()

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
#parser.add_argument("--baseN", type=int, default=40)
parser.add_argument("--dim", type=int, required=True,
                    choices=[2, 3])
parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup", "LTup", "LTSup"])
parser.add_argument("--temp-dependent", type=str, default="viscosity-conductivity",
                        choices=["none","viscosity","viscosity-conductivity"])
args, _ = parser.parse_known_args()

Ra_s = [1,1000,10000]#,20000]
Ra_s = [1]#,1000,5000, 7500]#, 8500, 10000]#,20000]
Ra = Constant(Ra_s[0])
Pr_s = [1.]
Pr = Constant(Pr_s[0])
Di_s = [0.]#, 0.3]
Di = Constant(Di_s[0])
r_s = [2.0, 2.7]#, 3.5]
r_s = [2.0, 2.3, 2.7]#, 3.5]
r_s = [2.0]
#r_s = [2.0, 1.8, 1.6]
if args.discretisation in ["bdm1p0", "rt1p0"] and args.fields == "Tup": r_s = [2.0]
r = Constant(r_s[0])
continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Ra": Ra_s}

problem_cl = {
    2: {
        "Tup": OBCavityMMS_Tup,
        "TSup": OBCavityMMS_TSup,
        "LTup": OBCavityMMS_LTup,
        "LTSup": OBCavityMMS_LTSup,
    },
    3: {
        "Tup": None,
        "TSup": None,
        "LTup": None,
        "LTSup": None,
    }
}[args.dim][args.fields]

problem_ = problem_cl(temp_dependent=args.temp_dependent, baseN=args.baseN, Pr=Pr, Ra=Ra, r=r, Di=Di)

results = {}
for Ra in [Ra_s[-1]]:
    results[Ra] = {}
    for s in ["velocity", "velocitygrad", "pressure", "temperature", "temperaturegrad", "stress", "quasi-norm", "divergence", "relvelocity", "relvelocitygrad", "relpressure", "reltemperature", "reltemperaturegrad", "relstress", "relquasi-norm"]:
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

    for Ra in [Ra_s[-1]]:
        problem_.Ra.assign(Ra)

        solver_output = run_solver(solver_, args, continuation_params)
        z = solver_.z
        Z = z.function_space()

        (theta_, u_, p_) = problem_.exact_solution(Z)
        S_ = problem_.exact_stress(Z)
        if args.fields == "Tup":
            theta, u, p = z.split()
            S = problem_.const_rel(sym(grad(u)), theta)
        elif args.fields == "TSup":
            theta, SS, u, p = z.split()
            S = solver_.stress_to_matrix(SS)
        elif args.fields == "LTup":
            L_, theta, u, p = z.split()
            (L_1, L_2, L_3) = split(L_)
            L = as_tensor(((L_1,L_2),(L_2,L_3)))
            S = problem_.const_rel(sym(grad(u))+L, theta)
        elif args.fields == "LTSup":
            _, theta, SS, u, p = z.split()
            S = solver_.stress_to_matrix(SS)
        S_F_exact = problem_.quasi_norm(sym(grad(u_)))
        S_F = problem_.quasi_norm(sym(grad(u)))

        veldiv = norm(div(u))
        pressureintegral = assemble(p_ * dx)
        pintegral = assemble(p*dx)
        #?????????????????????????????
#        print("norm(L)= ",norm(L))

        r_exp = float(solver_.r)
        r_exp_conj = r_exp/(r_exp - 1.)
#        i,j = indices(2)

        uerr = pow(assemble((pow((u[i]-u_[i])*(u[i]-u_[i]),r_exp/2.))*dx),1./r_exp)
        u__norm = pow(assemble((pow(u_[i]*u_[i],r_exp/2.))*dx),1./r_exp)
        ugraderr = pow(assemble(pow((Dx(u[i] - u_[i],j))*(Dx(u[i] - u_[i],j)),r_exp/2.)*dx),1./r_exp)
        u__gradnorm = pow(assemble(pow((Dx(u_[i],j))*(Dx(u_[i],j)),r_exp/2.)*dx),1./r_exp)
        perr = pow(assemble(pow(abs(inner(p-p_,p-p_)),r_exp_conj/2.)*dx),1./r_exp_conj)
        p__norm = pow(assemble(pow(abs(inner(p_,p_)),r_exp_conj/2.)*dx),1./r_exp_conj)
        thetaerr = norm(theta_-theta)
        theta__norm = norm(theta_)
        gradthetaerr = norm(grad(theta_ - theta))
        theta__gradnorm = norm(grad(theta_))
        stresserr =  pow(assemble((pow((S[i,j] - S_[i,j])*(S[i,j] - S_[i,j]), r_exp_conj/2.))*dx), 1./r_exp_conj)
        stress__norm =  pow(assemble((pow(S_[i,j]*S_[i,j], r_exp_conj/2.))*dx), 1./r_exp_conj)
        qnorm_error =  pow(assemble((pow((S_F[i,j] - S_F_exact[i,j])*(S_F[i,j] - S_F_exact[i,j]), r_exp_conj/2.))*dx), 1./r_exp_conj)
        qnorm_norm =  pow(assemble((pow(S_F_exact[i,j]*S_F_exact[i,j], r_exp_conj/2.))*dx), 1./r_exp_conj)

        results[Ra]["velocity"].append(uerr)
        results[Ra]["velocitygrad"].append(ugraderr)
        results[Ra]["pressure"].append(perr)
        results[Ra]["temperature"].append(thetaerr)
        results[Ra]["temperaturegrad"].append(gradthetaerr)
        results[Ra]["stress"].append(stresserr)
        results[Ra]["quasi-norm"].append(qnorm_error)
        results[Ra]["relvelocity"].append(uerr/u__norm)
        results[Ra]["relvelocitygrad"].append(ugraderr/u__gradnorm)
        results[Ra]["relpressure"].append(perr/p__norm)
        results[Ra]["reltemperature"].append(thetaerr/theta__norm)
        results[Ra]["reltemperaturegrad"].append(gradthetaerr/theta__gradnorm)
        results[Ra]["relstress"].append(stresserr/stress__norm)
        results[Ra]["relquasi-norm"].append(qnorm_error/qnorm_norm)
        results[Ra]["divergence"].append(veldiv)
        if comm.rank == 0:
            print("|div(u_h)| = ", veldiv)
            print("p_exact * dx = ", pressureintegral)
            print("p_approx * dx = ", pintegral)

## FOr tests...
theta_exact = Function(FunctionSpace(z.ufl_domain(), "CG", 1)).interpolate(theta_)
#u_exact = Function(FunctionSpace(z.ufl_domain(), FiniteElement("BDM",z.ufl_domain(),1,variant="integral"))).interpolate(u_)
u_exact = Function(u.function_space()).interpolate(u_)
p_exact = Function(FunctionSpace(z.ufl_domain(), "DG", 0)).interpolate(p_)
File("exact_mms.pvd").write(theta_exact, u_exact, p_exact)
File("computed_mms.pvd").write(theta, u, p)
#??????????????????????

if comm.rank == 0:
    for Ra in [Ra_s[-1]]:
        print("Results for Ra =", Ra)
        print("|u-u_h|", results[Ra]["velocity"])
        print("convergence orders:", convergence_orders(results[Ra]["velocity"]))
        print("|p-p_h|", results[Ra]["pressure"])
        print("convergence orders:", convergence_orders(results[Ra]["pressure"]))
        print("|theta-theta_h|", results[Ra]["temperature"])
        print("convergence orders:", convergence_orders(results[Ra]["temperature"]))
        print("|stress-stress_h|", results[Ra]["stress"])
        print("convergence orders:", convergence_orders(results[Ra]["stress"]))
        print("Lr' |Fstress-Fstress_h|", results[Ra]["quasi-norm"])
        print("convergence orders:", convergence_orders(results[Ra]["quasi-norm"]))
    print("gamma =", args.gamma)
    print("h =", hs)

#    for Ra in [Ra_s[-1]]:
#        print("%%Ra = %i" % Ra)
#        print("\\pgfplotstableread[col sep=comma, row sep=\\\\]{%%")
#        print("hmin, havg, error_v, error_vgrad, error_p, error_th, error_thgrad, error_str, relerror_v, relerror_vgrad, relerror_p, relerror_th, relerror_thgrad, relerror_str, div\\\\")
#        for i in range(len(hs)):
#            print(",".join(map(str, [hs[i][0], hs[i][1], results[Ra]["velocity"][i], results[Ra]["velocitygrad"][i], results[Ra]["pressure"][i], results[Ra]["temperature"][i], results[Ra]["temperaturegrad"][i], results[Ra]["stress"][i], results[Ra]["relvelocity"][i], results[Ra]["relvelocitygrad"][i], results[Ra]["relpressure"][i], results[Ra]["reltemperature"][i], results[Ra]["reltemperaturegrad"][i], results[Ra]["relstress"][i], results[Ra]["divergence"][i]])) + "\\\\")
#        def numtoword(num):
#            return eng.number_to_words(num).replace(" ", "").replace("-","")
#        name = "Ra" + numtoword(int(Ra)) \
#            + "gamma" + numtoword(int(args.gamma)) \
#            + args.discretisation.replace("0", "zero")
#        print("}\\%s" % name)
