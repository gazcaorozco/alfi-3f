#python mms.py --dim 2 --fields Tup --gamma 0.0 --temp-dependent viscosity-conductivity --non-dimensional rayleigh1 --discretisation th --mh uniform --patch star --solver-type lu --k 2 --nref 2
from pprint import pprint
from mms2d import OBCavityMMS_up, OBCavityMMS_Sup
from mms3d import OBCavityMMS3D_up, OBCavityMMS3D_Sup
from implcfpc import get_default_parser, get_solver, run_solver
import os
from firedrake import *
import numpy as np
import inflect
eng = inflect.engine()

convergence_orders = lambda x: np.log2(np.array(x)[:-1] / np.array(x)[1:])

parser = get_default_parser()
parser.add_argument("--dim", type=int, required=True,
                    choices=[2, 3])
parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup"])
parser.add_argument("--temp-dependent", type=str, default="viscosity-conductivity",
                        choices=["none","viscosity","viscosity-conductivity"])
parser.add_argument("--non-dimensional", type=str, default="rayleigh1",
                        choices=["rayleigh1"])
args, _ = parser.parse_known_args()

Ra_s = [1,1000,10000]#,20000]
Ra = Constant(Ra_s[0])
Pr_s = [1.]
Pr = Constant(Pr_s[0])
Di_s = [0., 0.3]
Di = Constant(Di_s[0])
r_s = [2.0, 2.7, 3.5]
#r_s = [2.0]
#r_s = [2.0, 1.8, 1.6]
r = Constant(r_s[0])
continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Ra": Ra_s}

if args.dim == 2:
    if args.fields == "Tup":
        problem_ = OBCavityMMS_up(baseN=args.baseN, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Ra=Ra, r=r, Di=Di)
    else:
        problem_ = OBCavityMMS_Sup(baseN=args.baseN, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Ra=Ra, r=r, Di=Di)
else:
    if args.fields == "Tup":
        problem_ = OBCavityMMS3D_up(baseN=args.baseN, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Ra=Ra, r=r, Di=Di)
    else:
        problem_ = OBCavityMMS3D_Sup(baseN=args.baseN, temp_dependent=args.temp_dependent, non_dimensional=args.non_dimensional, Pr=Pr, Ra=Ra, r=r, Di=Di)
    

results = {}
for Ra in [Ra_s[-1]]:
    results[Ra] = {}
    for s in ["velocity", "velocitygrad", "pressure", "temperature", "temperaturegrad", "stress", "divergence", "relvelocity", "relvelocitygrad", "relpressure", "reltemperature", "reltemperaturegrad", "relstress"]:
        results[Ra][s] = []
comm = None
hs = []
for nref in range(1, args.nref+1):
    args.nref = nref
    solver_ = get_solver(args, problem_)
    solver_.gamma.assign(0.)
    problem_.interpolate_initial_guess(solver_.z)
    solver_.no_convection = True
    results0 = run_solver(solver_,args, {"r": [2.0]})
    solver_.no_convection = False
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
        if args.fields == "Tup":
            theta, u, p = z.split()
            S = problem_.const_rel(sym(grad(u)), theta)
            (theta_, u_, p_) = problem_.exact_solution(Z)
            S_ = problem_.const_rel(sym(grad(u_)), theta_)
            
        else:
            theta, SS, u, p = z.split()
            if solver_.exactly_div_free:
                (S_1,S_2) = split(SS)
                S = as_tensor(((S_1,S_2),(S_2,-S_1)))
            else:
                (S_1,S_2,S_3) = split(SS)
                S = as_tensor(((S_1,S_2),(S_2,S_3)))

            (theta_, S_, u_, p_) = problem_.exact_solution(Z)


        veldiv = norm(div(u))
        pressureintegral = assemble(p_ * dx)
        pintegral = assemble(p*dx)

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

        results[Ra]["velocity"].append(uerr)
        results[Ra]["velocitygrad"].append(ugraderr)
        results[Ra]["pressure"].append(perr)
        results[Ra]["temperature"].append(thetaerr)
        results[Ra]["temperaturegrad"].append(gradthetaerr)
        results[Ra]["stress"].append(stresserr)
        results[Ra]["relvelocity"].append(uerr/u__norm)
        results[Ra]["relvelocitygrad"].append(ugraderr/u__gradnorm)
        results[Ra]["relpressure"].append(perr/p__norm)
        results[Ra]["reltemperature"].append(thetaerr/theta__norm)
        results[Ra]["reltemperaturegrad"].append(gradthetaerr/theta__gradnorm)
        results[Ra]["relstress"].append(stresserr/stress__norm)
        results[Ra]["divergence"].append(veldiv)
        if comm.rank == 0:
            print("|div(u_h)| = ", veldiv)
            print("p_exact * dx = ", pressureintegral)
            print("p_approx * dx = ", pintegral)

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
    print("gamma =", args.gamma)
    print("h =", hs)

    for Ra in [Ra_s[-1]]:
        print("%%Ra = %i" % Ra)
        print("\\pgfplotstableread[col sep=comma, row sep=\\\\]{%%")
        print("hmin, havg, error_v, error_vgrad, error_p, error_th, error_thgrad, error_str, relerror_v, relerror_vgrad, relerror_p, relerror_th, relerror_thgrad, relerror_str, div\\\\")
        for i in range(len(hs)):
            print(",".join(map(str, [hs[i][0], hs[i][1], results[Ra]["velocity"][i], results[Ra]["velocitygrad"][i], results[Ra]["pressure"][i], results[Ra]["temperature"][i], results[Ra]["temperaturegrad"][i], results[Ra]["stress"][i], results[Ra]["relvelocity"][i], results[Ra]["relvelocitygrad"][i], results[Ra]["relpressure"][i], results[Ra]["reltemperature"][i], results[Ra]["reltemperaturegrad"][i], results[Ra]["relstress"][i], results[Ra]["divergence"][i]])) + "\\\\")
        def numtoword(num):
            return eng.number_to_words(num).replace(" ", "").replace("-","")
        name = "Ra" + numtoword(int(Ra)) \
            + "gamma" + numtoword(int(args.gamma)) \
            + args.discretisation.replace("0", "zero")
        print("}\\%s" % name)
