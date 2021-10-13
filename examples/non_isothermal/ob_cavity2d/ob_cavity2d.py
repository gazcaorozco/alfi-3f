#mpiexec -n 4 python ob_cavity2d.py --discretisation sv --mh bary --patch macro --restriction --baseN 10 --gamma 10000 --temp-bcs left-right --stabilisation-weight-u 5e-3 --solver-type almg --thermal-conv natural_Ra --fields Tup --temp-dependent viscosity-conductivity --stabilisation-type-u burman --stabilisation_type_t none --k 2 --nref 2 --cycles 2 --smoothing 4
from firedrake import *
from alfi_3f import *

from ob_problem2d import OBCavity2D_Tup, OBCavity2D_LTup, OBCavity2D_TSup, OBCavity2D_LTSup

from firedrake.petsc import PETSc
PETSc.Sys.popErrorHandler()

if __name__ == "__main__":

    parser = get_default_parser()
    parser.add_argument("--diagonal", type=str, default="left",
                        choices=["left", "right", "crossed"])
    parser.add_argument("--fields", type=str, default="Tup",
                        choices=["Tup", "TSup", "LTup", "LTSup"])
    parser.add_argument("--temp-bcs", type=str, default="left-right",
                        choices=["left-right","down-up"])
    parser.add_argument("--temp-dependent", type=str, default="none",
                        choices=["none","viscosity","viscosity-conductivity"])
    parser.add_argument("--unstructured", dest="unstructured", default=True,
                        action="store_true")
    parser.add_argument("--plots", dest="plots", default=False,
                        action="store_true")
    args, _ = parser.parse_known_args()

    assert args.thermal_conv in ["natural_Ra"], "You need to select natural convection 'natural_Ra'"

    #Prandtl number
    Pr_s = [1.,10.]
    Pr_s = [1.]
#    Pr_s = [1.,0.5,0.1,0.01]
#    Pr_s = [1.,1000,2500,5000,10000]
    Pr = Constant(Pr_s[0])

    #Rayleigh number
    Ra_s = [1,2500] + list(range(5000,50000000 + 2500,2500)) #To test how high can it get (very inefficient)
    Ra_s = [1, 350, 700, 2500, 5000] + list(range(10000, 100000 + 5000,5000)) + list(range(100000+10000, 1000000+10000, 10000)) #Newtonian
    #Ra_s = [1,2500,5000] + list(range(10000, 20000 + 5000,5000)) #For Power-law (shear thinning)
    #Ra_s = [1, 350,700,1250,2500,5000] + list(range(10000, 100000 + 5000,5000)) #For Power-law (shear thickenning)
    Ra = Constant(Ra_s[0])

    #Power-law
    r_s = [2.0]
    #r_s = [2.0,2.3,2.6,2.7]#,3.5]
    #r_s = [2.0,1.9,1.8,1.7,1.6]
    #r_s = [2.0,1.8,1.6] #For power-law
    if args.discretisation in ["bdm", "rt"] and args.fields == "Tup": r_s = [2.0]
    r = Constant(r_s[0])

    #Dissipation number
    Di_s = [0.]
#    Di_s = [0.,0.2,0.4,0.6,0.8,1.,1.3,1.5,1.7,1.9,2.]
#    Di_s = [0.,0.6,1.3]#,2.]  #For Navier-Stokes
    Di = Constant(Di_s[0])

    problem_cl = {
        2: {
            "Tup": OBCavity2D_Tup,
            "TSup": OBCavity2D_TSup,
            "LTup": OBCavity2D_LTup,
            "LTSup": OBCavity2D_LTSup,
        },
        3: {
            "Tup": None,
            "TSup": None,
            "LTup": None,
            "LTSup": None,
        }
    }[2][args.fields] #TODO: Add 3D problem classes

    problem_ = problem_cl(args.baseN, temp_bcs=args.temp_bcs, temp_dependent=args.temp_dependent, diagonal=args.diagonal, unstructured=args.unstructured, Pr=Pr, Ra=Ra, r=r, Di=Di)
    solver_ = get_solver(args, problem_)

    problem_.interpolate_initial_guess(solver_.z)
    solver_.no_convection = True
    results0 = run_solver(solver_,args, {"r": [2.0]})
    solver_.no_convection = False

    continuation_params = {"r": r_s,"Pr": Pr_s,"Di": Di_s,"Ra": Ra_s}
    results = run_solver(solver_, args, continuation_params)

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

        nu_temp = problem_.viscosity(theta)
        D = sym(grad(u))
        SS = interpolate(S,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        DD = interpolate(D+L,TensorFunctionSpace(solver_.z.ufl_domain(),"DG",k-1))
        nu_temp_ = interpolate(nu_temp, FunctionSpace(solver_.z.ufl_domain(),"CG",k-1))
        u.rename("Velocity")
        p.rename("Pressure")
        theta.rename("Temperature")
        nu_temp_.rename("Viscosity_Temp")
        SS.rename("Stress")
        DD.rename("Symmetric velocity gradient")
        string = "_"+args.fields+"_"
        string += args.temp_bcs
        if args.temp_dependent == "viscosity":
            string += "_visc"
        elif args.temp_dependent == "viscosity-conductivity":
            string += "_visccond"
        string += "_Ra%s"%(Ra_s[-1])
        string += "_Pr%s"%(Pr_s[-1])
        string += "_Di%s"%(Di_s[-1])
        string += "_r%s"%(r_s[-1])
        string += "_k%s"%(args.k)
        string += "_nref%s"%(args.nref)
        string += "stabu_%s"%(args.stabilisation_type_u)
        string += "stabt_%s"%(args.stabilisation_type_t)
        string += "_%s"%(args.solver_type)

        File("plots/z%s.pvd"%string).write(DD,SS,u,theta)
