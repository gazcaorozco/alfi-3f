#Temperature-dependent viscosity
#mpiexec -n 4 python viscoplastic_cavity2d.py --discretisation sv --mh bary --patch macro --restriction --solver-type almg --cycles 4 --smoothing 4 --k 2 --nref 1 --high-accuracy --non-isothermal viscosity --fields TSup --gamma 100000 --thermal-conv forced
from firedrake import *
from alfi_3f import *

from coolingBingham_problem import BinghamCoolingChannel_Tup, BinghamCoolingChannel_LTup, BinghamCoolingChannel_TSup, BinghamCoolingChannel_LTSup

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup", "LTup", "LTSup"])
    parser.add_argument("--non-isothermal", type=str, default="none",
                        choices=["none","viscosity","yield-stress"])
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    if args.discretisation in ["bdm", "rt"]: assert args.baseN == 0, "Don't forget to start with the finer mesh when using HDiv formulations by setting --baseN 0"

    #Reynolds number
    Res = [1.]
    Re = Constant(Res[0])

    #Peclet number 
    Pe_s = [1.,10.]
    Pe_s = [1.]
    Pe = Constant(Pe_s[0])

    #Bingham number (at the inlet)
#    Bn_s = [0.]
    if args.non_isothermal == "yield-stress":
        Bn_s = [0.,0.5,1.,1.5]
    else:
        Bn_s = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6,6.5,7.] #Works with Br=0
        Bn_s = [0.,0.5,1.,1.5,2.,2.5,3.,3.5,4.] #For use with Br=0.15
#        Bn_s = [0.0,0.1,0.5,1.]#Tests
    Bn = Constant(Bn_s[0])

    #Brinkman number
    if args.non_isothermal == "yield-stress":
        Br_s = [0.]
    else:
        Br_s = [0.]
#        Br_s = [0.,0.1]
    Br = Constant(Br_s[0])

    #Regularisation 
    zepss_0 = [1., 5.]
    if args.non_isothermal == "yield-stress":
        if args.fields in ["Tup", "LTup"]:
            zeps = [6.,7.,8.,9.,10,11.,12.,25,40,80] + list(range(120, 10000+30, 30))#allu works with 40 but takes a while...
        else:
            zeps = [6.,7.,200,500] + list(range(1000,50000 + 500, 500))
            zeps = [6.,7.,10.,100,125,150,175,200,220,240,260,280,300,340,380,420,460,500,600,700,800,900] + list(range(1000, 50000 + 250, 250))
            zeps = [6.,7.,10.,100,125,150,175,200,220,240,260,280,300,340,380,420,460,500,600,700,800,900] + [1000]
    else:
#        zepss_0 = [1., 1.5, 2., 2.5, 3., 5.]  #For use with Br=0.15 
        if args.fields in ["Tup", "LTup"]:
            zeps = [6.,7.,8.,9.,10,11.,12.,25,40,80] + list(range(120, 10000+40, 40))#Works with allu
            zeps = [6.,7.,8.,9.,10,11.,12.,17,25,30,35,40,46,53,60,80] + list(range(120, 10000+40, 40))#Works with allu
        else:
            zeps = [6., 7., 20, 30, 40, 50, 70, 100, 120, 150, 170, 200, 350, 500] + list(range(1000, 100000 + 1000, 1000))
#            zeps = [5.25,5.5, 6.,7.,10,12,17,25,100,200,1000] + list(range(2000,100000 + 2000, 2000)) #For use with Br=0.15
            zeps = [6., 7., 20, 30, 40, 50, 70, 100, 120, 150, 170, 200, 350, 500]#, 1000]# + list(range(1000, 100000 + 1000, 1000))

    epss_0 = [1./z_ for z_ in zepss_0]
    epss = [1./z_ for z_ in zeps]
    eps = Constant(epss_0[0])

    #Temperature drop
    if args.fields == "yield-stress":
    #    temp_drop_s = [1.]
        temp_drop_s = [10.]
    else:
    #    temp_drop_s = [1.]
        temp_drop_s = [10.]
    temp_drop = Constant(temp_drop_s[0])

#---------- For tests------------------------------------------#
    Res = [1.]; Re = Constant(Res[0])
#    Pe_s = [1.]; Pe = Constant(Pe_s[0])
#    Bn_s = [0.,0.2,0.35,0.5]; Bn = Constant(Bn_s[0])
#    Bn_s = [0.]; Bn = Constant(Bn_s[0])
    Br_s = [0.]; Br = Constant(Br_s[0])
#    epss_0 = [1.]; eps = Constant(epss_0[0])
    temp_drop_s = [2, 5]; temp_drop = Constant(temp_drop_s[0])
#--------------------------------------------------------------#

    problem_cl = {
        "Tup": BinghamCoolingChannel_Tup,
        "TSup": BinghamCoolingChannel_TSup,
        "LTup": BinghamCoolingChannel_LTup,
        "LTSup": BinghamCoolingChannel_LTSup,
    }[args.fields]

    problem_ = problem_cl(args.baseN, args.non_isothermal, Re=Re, Pe=Pe, Bn=Bn, Br=Br, eps=eps, temp_drop=temp_drop)
    solver_ = get_solver(args, problem_)
    problem_.interpolate_initial_guess(solver_.z)

    continuation_params = {"Re": Res,"Br": Br_s,"Pe": Pe_s,"Bn": Bn_s,"temp_drop": temp_drop_s,"eps": epss_0}
    continuation_params2 = {"Re": [Res[-1]],"Br": [Br_s[-1]],"Pe": [Pe_s[-1]],"Bn": [Bn_s[-1]], "temp_drop": [temp_drop_s[-1]],"eps": epss}
    results = run_solver(solver_, args, continuation_params)
    results2 = run_solver(solver_, args, continuation_params2, {"eps": "secant"})

    #Quick visualisation
    if args.plots:
        k = solver_.z.sub(problem_.vel_id).ufl_element().degree()
        if args.fields in ["Tup","LTup"]:
            if args.fields == "Tup":
                theta, u, p = solver_.z.split()
            else:
                L_, theta, u, p = solver_.z.split()
                (L_1, L_2) = split(L_)
                L = as_tensor(((L_1,L_2),(L_2,-L_1)))
            S = problem_.const_rel(sym(grad(u)), theta)
        else:
            if args.fields == "TSup":
                theta, S_, u, p = solver_.z.split()
            else:
                L_, theta, S_, u, p = solver_.z.split()
                (L_1, L_2) = split(L_)
                L = as_tensor(((L_1,L_2),(L_2,-L_1)))
            (S_1, S_2) = split(S_)
            S = as_tensor(((S_1,S_2),(S_2,-S_1)))
        D = sym(grad(u))
        symgrad_u = D if args.fields in ["Tup","TSup"] else D + L
        SS = interpolate(S,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        DD = interpolate(symgrad_u,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        u_in = problem_.bingham_poiseuille(solver_.z.ufl_domain(), solver_.Bn)
        u_inflow = project(u_in,VectorFunctionSpace(solver_.z.ufl_domain(),"CG",k))
        u.rename("Velocity")
        p.rename("Pressure")
        theta.rename("Temperature")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        u_inflow.rename("Inlet velocity")
        string = "_"+args.fields+"_"
        if args.non_isothermal == "viscosity":
            string += "_visc"
        elif args.non_isothermal == "yield-stress":
            string += "_ystress"
        string += "_tdrop%s"%(temp_drop_s[-1])
        string += "_Bn%s"%(Bn_s[-1])
        string += "_Br%s"%(Br_s[-1]) 
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        File("output_plots/z%s.pvd"%string).write(DD,SS,u,theta,p,u_inflow)
