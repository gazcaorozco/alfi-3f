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
