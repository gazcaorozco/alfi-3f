from firedrake import *
from firedrake.dmhooks import get_appctx, get_function_space
from firedrake.mg.utils import get_level

class Stabilisation(object):

    def __init__(self, Z, wind_id, state=None, field_id=None, weight=None):
        self.Z = Z
        self.V = Z.sub(wind_id)
        self.mesh = Z.ufl_domain()
        self.wind_id = wind_id
        self.field_id = wind_id if field_id is None else field_id
        if state is None:
            self.wind = Function(self.V, name="Wind")
            self.separate_wind = True
        else:
            self.wind = split(state)[self.wind_id]
            self.separate_wind = False
        self.weight = Constant(weight) if weight is not None else None

    def update(self, w):   #TODO: If we use a coefficient depending e.g. on S, this needs to be changed
        if not self.separate_wind:
            return

        if isinstance(w, Function):
            self.wind.assign(w)
        else:
            with self.wind.dat.vec_wo as wind_v:
                w.copy(wind_v)

        # Need to update your coarse grid mates, too.
        # There's probably a better way to do this.
        dm = self.V.dm
        wind = self.wind

        while get_level(get_function_space(dm).mesh())[1] > 0:
            cdm = dm.coarsen()

            cctx = get_appctx(cdm)
            if cctx is None:
                break
            cwind = cctx.F._cache['coefficient_mapping'][wind]
            inject(wind, cwind)

            dm = cdm
            wind = cwind

class BurmanStabilisation(Stabilisation):

    def __init__(self, *args, h=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.weight is None:
            self.weight = Constant(3e-3) # as chosen in doi:10.1016/j.apnum.2007.11.001
            # stream line diffusion type
            # self.weight = Constant(4e-3) #
        if h is None:
            if self.mesh.topological_dimension() == 3:
                self.h = FacetArea(self.mesh)**0.5  # go from area to length
            else:
                self.h = FacetArea(self.mesh)
        else:
            self.h = h

    def form(self, u, v):
        mesh = self.mesh
        n = FacetNormal(mesh)
        h = self.h
        # beta = avg(facet_avg(sqrt(dot(self.wind, n)**2+1e-10)))
        beta = avg(facet_avg(sqrt(inner(self.wind, self.wind)+1e-10)))
        return 0.5 * self.weight * avg(h)**2 * beta * dot(jump(grad(u), n), jump(grad(v), n))*dS

class SUPG(Stabilisation):

    def __init__(self, Pe, *args, magic=1.0, h=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.Pe = Pe
        self.magic = magic
        self.h = h or CellSize(self.mesh)
        if self.weight is None:
            tdim = self.mesh.topological_dimension()
            self.weight = Constant(0.1) if tdim == 3 else Constant(1)

    def coefficient(self):
        raise NotImplementedError

    def form(self, LL, v, dx):
        beta = self.coefficient()
        w = self.wind
        return self.weight * beta * inner(LL, dot(grad(v), w)) * dx

    def form_gls(self, LL, LL_, dx):
        beta = self.coefficient()
#        w = self.wind
        return self.weight * beta * inner(LL, LL_) * dx

class ShakibHughesZohanSUPG(SUPG):
    """
    Stabilisation described by (3.58) in

    @article{shakib1991,
        doi = {10.1016/0045-7825(91)90041-4},
        year = 1991,
        volume = {89},
        number = {1-3},
        pages = {141--219},
        author = {F. Shakib and T. J. R. Hughes and Z. Johan},
        title = {A new finite element formulation for computational fluid dynamics: {X}. {T}he compressible {E}uler and {N}avier-{S}tokes equations},
        journal = {Computer Methods in Applied Mechanics and Engineering}
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def coefficient(self):
        w = self.wind
        nu = 1.0/self.Pe   #Maybe use a better name? Re - momentum, Pe - scalar advection-diffusion...
        h = self.h
        beta = ((4.0*dot(w, w)/(h**2)) + self.magic*(4.0*nu/h**2)**2 )**(-0.5)
        return beta

class TurekSUPG(SUPG):

    """
    Stabilisation described in

    @book{turek1999,
      Author = {S. Turek},
      Title = {Efficient Solvers for Incompressible Flow Problems: An Algorithmic and Computational Approach},
      series = {Lecture Notes in Computational Science and Engineering},
      Publisher = {Springer},
      Year = {2013},
      ISBN = {3642635733},
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        SUPG.__init__(self, *args, **kwargs)
        self.w_avg = Constant(0.0)
        self.domain_measure = assemble(Constant(1.0)*dx(domain=self.mesh))

    def update(self, w):
        SUPG.update(self, w)
        nrm = assemble(sqrt(inner(self.wind, self.wind))*dx)
        self.w_avg.assign(nrm/self.domain_measure)

    def coefficient(self):
        Re = self.Pe
        w = self.wind
        h = self.h

        Re_tau = cell_avg(sqrt(inner(w, w))) * h * Re
        w_avg = self.w_avg
        beta = self.magic * h * 2. * Re_tau / ((w_avg) * (1. + Re_tau))
        return beta
