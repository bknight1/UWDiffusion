import sympy
import underworld3 as uw
from underworld3.systems.solvers import SNES_Poisson
from underworld3.function import expression

class SNES_Diffusion(SNES_Poisson):
    """
    Diffusion Equation Solver with multi-step time integration.

    Supports arbitrary order multistep schemes (BDF, Adams-Moulton)
    for time discretization and history tracking.
    """

    def __init__(
        self,
        mesh,
        u_Field,
        order=1,
        degree=2,
        theta=0.5,
        verbose=False,
    ):
        super().__init__(mesh, 
                         u_Field, 
                         verbose=verbose, 
                         degree=degree)
        
        self.order = order
        self.theta = theta

        # Eulerian DDt for time history of u
        self.DuDt = uw.systems.Eulerian_DDt(
            mesh,
            u_Field,
            vtype=uw.VarType.SCALAR,
            degree=u_Field.degree,
            continuous=u_Field.continuous,
            varsymbol=u_Field.symbol,
            verbose=verbose,
            bcs=self.essential_bcs,
            order=order,
            smoothing=0.0,
        )


        self.kappa_star = [
            uw.function.expression(
                rf'\upkappa^{"*"*(i+1)}_{u_Field.name}',
                0.
            )
            for i in range(order)
        ]

        ### source term
        self.S = sympy.Matrix.zeros(1, 1)

        self.flux_history = [sympy.Matrix([[0] * mesh.dim]) for _ in range(order)]

    def update_kappa(self):
        ### copy down the chain
        if self.order > 1:
            for i in range(self.order):
                self.kappa_star[i+1] = self.kappa_star[i]
        ### update the first one
        self.kappa_star[0].sym = self.constitutive_model.diffusivity.sym
        

    def update_history_terms(self):
        self.update_kappa()

        for i in range(self.order):
            # Update historical flux
            self.flux_history[i] = self.DuDt.psi_star[i].jacobian() * self.kappa_star[i]

    def adams_moulton_flux(self, flux, flux_history, order=None):
        # Adams-Moulton coefficients for up to third order
        if order is None:
            order = self.order
        with sympy.core.evaluate(False):
            if order == 1:
                return (flux + flux_history[0]) / 2
            elif order == 2:
                return (5*flux + 8*flux_history[0] - flux_history[1]) / 12
            elif order == 3:
                return (9*flux + 19*flux_history[0] - 5*flux_history[1] + flux_history[2]) / 24
            else:
                raise NotImplementedError("Order > 3 not implemented for Adams-Moulton flux")

    def solve(
        self,
        dt,
        zero_init_guess=True,
        verbose=False,
    ):

        # if not self.constitutive_model._solver_is_setup:
        #     self.is_setup = False
        #     self.update_history_terms()

        # print( self.kappa_star[0].sym )

        # print(self.is_setup)

        # if not self.is_setup:
        #     self._setup_pointwise_functions(verbose)
        #     self._setup_discretisation(verbose)
        #     self._setup_solver(verbose)

            
        
        # Update source term using BDF time stepping
        self.f = - (sympy.simplify(self.DuDt.bdf(order=self.order)) / dt) + self.S

        # Construct current flux
        flux = self.DuDt._psi_meshVar.jacobian() * self.constitutive_model.diffusivity

        # Adams-Moulton flux term
        self.flux = self.adams_moulton_flux(flux, self.flux_history, order=self.order)

        # Call parent solve
        super().solve(zero_init_guess=zero_init_guess, verbose=verbose)

        # Update time history for next step
        self.DuDt.update_post_solve()
        self.update_history_terms()
