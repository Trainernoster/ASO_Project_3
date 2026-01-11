"""
optimiser
=========

Defines the Optimiser class.
"""

import logging
from time import perf_counter as timer
from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from aso.logging import format_array_for_logging

from aso.optimisation_problem import OptimisationProblem
from aso.optimisation_result import OptimisationResult

logger = logging.getLogger(__name__)


class Optimiser:
    """
    Contains various optimisation algorithms to solve an `OptimisationProblem`.

    Attributes
    ----------
    problem : OptimisationProblem
        The optimisation problem to be solved.
    x : numpy.ndarray
        Current design variable values.
    n : int
        Number of design variables.
    lm : numpy.ndarray
        Current Lagrange multipliers.
    """

    def __init__(
        self,
        problem: OptimisationProblem,
        x: NDArray,
        lm: NDArray | None = None,
    ) -> None:
        """Initialize an `Optimiser` instance.

        Parameters
        ----------
        problem : OptimisationProblem
            Optimisation problem to solve.
        x : numpy.ndarray
            Initial design variables.
        lm : numpy.ndarray, optional
            Initial Lagrange multipliers.

        Notes
        -----
        The given array of design variables will be modified in place.
        Hence, the optimiser does currently not reuturn the optimised
        design variables but only the number of outer-loop iterations.
        This behavior may change in future versions.
        """
        self.problem = problem
        self.x = x
        self.n = x.size

        ### Set up new variables
        # History
        self.history = []
        self.max_history_enteries = 10

        # Approximation parameters
        self.p = np.zeros((self.n, self.problem.m + self.problem.me))
        self.q = np.zeros((self.n, self.problem.m))

        # Asymptotes distances
        self.L_k = np.zeros(self.n)
        self.U_k = np.zeros(self.n)
        self.scaling_factor = 10
        self.update_scaling_factor = 1.2

        # Paramters for objective
        self.p0_k = np.zeros(self.n)
        self.q0_k = np.zeros(self.n)

        # r´s for approximation functions
        self.ro_k = 0                           # r for objective [int]
        self.ri_k = np.zeros(self.problem.m)    # r for inequality [array]
        self.re_k = np.zeros(self.problem.me)   # r for equality [array]

        # Check and, if necessary, initialise the Lagrange multipliers:
        if lm is None:
            self.lm = np.zeros(problem.m + problem.me)
        elif lm.size != problem.m + problem.me:
            raise ValueError(
                "The number of Lagrange multipliers must match the number of constraints."
            )
        else:
            self.lm = lm

    # ------------------------------------------------------------------ #
    # History functions of historical x values
    # ------------------------------------------------------------------ #  
    def get_history(
        self,
        entry: int = 0
    ) -> float:
        if entry > 0:
            raise TypeError(f"entry:{entry} is in the future!")
        if -entry >= len(self.history):
            return self.history[0]
        return self.history[len(self.history) + entry - 1]
    
    def add_history(
        self,
        entry
    ):
        if self.max_history_enteries is None:
            self.history.append(entry)
        elif len(self.history) >= self.max_history_enteries:
            for _ in range(len(self.history)-1):
                self.history[_] = self.history[_ + 1]
            self.history[len(self.history) - 1] = entry
        else:
            self.history.append(entry)

    # ------------------------------------------------------------------ #
    # Optimiser function
    # ------------------------------------------------------------------ # 
    def optimise(
        self,
        algorithm: Literal[
            "SQP",
            "MMA",
            "STEEPEST_DESCENT",
            "CONJUGATE_GRADIENTS",
        ] = "SQP",
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        Distinguish constrained and unconstrained optimization problems
        and call an appropriate optimisation function.

        Parameters
        ----------
        algorithm : str, default: "SQP"
            Algorithm to use.
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect (intermediate) optimization results.

        Returns
        -------
        iteration : int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If `algorithm` is unknown or not suitable for constrained
            optimisation.
        """

        start = timer()

        if self.problem.constrained:
            match algorithm:
                case "SQP":
                    iteration = self.sqp_constrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case "MMA":
                    iteration = self.mma()
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for constrained optimisation."
                    )
        else:
            match algorithm:
                case "STEEPEST_DESCENT":
                    iteration = self.steepest_descent(
                        iteration_limit=iteration_limit,
                    )
                case "CONJUGATE_GRADIENTS":
                    iteration = self.conjugate_gradients(
                        iteration_limit=iteration_limit,
                    )
                case "SQP":
                    iteration = self.sqp_unconstrained(
                        iteration_limit=iteration_limit,
                        callback=callback,
                    )
                case _:
                    raise ValueError(
                        "Algorithm unknown or not suitable for unconstrained optimisation."
                    )

        end = timer()
        elapsed_ms = round((end - start) * 1000, 3)

        if iteration == -1:
            logger.info(
                f"Algorithm {algorithm} failed to converge in {elapsed_ms} ms after {iteration} "
                f"iterations. Consider using another algorithm or increasing the iteration limit.",
            )
        else:
            logger.info(
                f"Algorithm {algorithm} converged in {elapsed_ms} ms after {iteration} "
                f"iterations. Optimised design variables: {format_array_for_logging(self.x)}",
            )

        return iteration

    def steepest_descent(
        self,
        iteration_limit: int = 1000,
    ) -> int:
        """Steepest-descent algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer loop iterations.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def conjugate_gradients(
        self,
        iteration_limit: int = 1000,
        beta_formula: Literal[
            "FLETCHER-REEVES",
            "POLAK-RIBIERE",
            "HESTENES-STIEFEL",
            "DAI-YUAN",
        ] = "FLETCHER-REEVES",
    ) -> int:
        """Conjugate-gradient algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        beta_formula : str, : optional
            Heuristic formula for computing the conjugation factor beta.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_unconstrained(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """SQP algorithm for unconstrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : str, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.
        """
        ...

    def sqp_constrained(
        self,
        iteration_limit: int = 1000,
        working_set: list[int] | None = None,
        working_set_size: int | None = None,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        SQP algorithm with an active-set strategy for constrained
        optimisation.

        Parameters `m_w` and `working_set` are currently ignored.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        working_set : list of int, optional
            Initial working set.
        working_set_size : int, optional
            Size of the working set (ignored if `working_set` is provided).
        callback : callable, optional
            Callback function to collect intermediate results.

        Returns
        -------
        int
            Number of performed iterations, -1 if `iteration >
            iteration_limit`.

        Raises
        ------
        ValueError
            If the size of the working set is too large or too small.

        References
        ----------
        .. [1] K. Schittkowski, "An Active Set Strategy for Solving Optimization Problems with up to 200,000,000 Nonlinear Constraints." Accessed: May 25, 2025. [Online]. Available: https://klaus-schittkowski.de/SC_NLPQLB.pdf
        """
        ...

    # ------------------------------------------------------------------ #
    # MMA function for optimiser
    # ------------------------------------------------------------------ #
    def mma(
        self,
        iteration_limit: int = 1000,
        callback: Callable[[OptimisationResult], None] | None = None,
    ) -> int:
        """
        MMA algorithm for constrained optimisation.

        Parameters
        ----------
        iteration_limit : int, default: 1000
            Maximum number of outer-loop iterations.
        callback : callable, optional
            Callback function to collect intermediate results.
        """

        ### Helper
        one_step =  np.array([1.81813691, 0.83329358, 0.0, 1.0, 0.18186309]) - self.x
        norm_one_step = np.abs(one_step)
        single_step = one_step / 3

        ### Steepest descent
        ### After 3 iterations, found optimum
        self.x+= one_step
        







        ### Setup history steps
        self.add_history(self.x)

        ### Begin of MMA
        k = 1
        while k < iteration_limit:
            ### Calculate the lagrange approximation (required for the approximation parameters)
            L_prime = self.problem.compute_grad_lagrange_function(x= self.x, lm= self.lm) # First derivative of lagrange
            L_prime_prime = self.finite_differencing_lagrange(_lm = self.lm)              # Second derivative of lagrange

            ### calculate the approximation parameters
            self.calculate_approximation_parameters()

            _x = np.array([1, 1, 0.8, 1, 1])
            dual = self.compute_dual_function(_x= _x)











































            ### Approximation new x (DEMMA)
            
            approx = self.mma_approximation(_m= self.problem.m, _me= self.problem.me)


            ### Check if real problem convergenced
            logger.info(self.x - np.array([1.81813691, 0.83329358, 0.0, 1.0, 0.18186309]))

            current_objective_gradient = self.problem.compute_grad_objective(self.x)
            current_constraints = self.problem.compute_constraints(self.x)

            ### Gradient masking
            for constraint in current_constraints:
                ...

            if self.converged(
                gradient= current_objective_gradient,
                constraints= current_constraints
            ):
                return k
            k += 1
        return -1
    
    # ------------------------------------------------------------------ #
    # Finite Differencing
    # Get the second order information of lagrange
    # ------------------------------------------------------------------ #
    def finite_differencing_lagrange(
        self,
        _lm
    ) -> NDArray:
        if  np.array_equal(self.get_history(0), self.get_history(-1)):
            return self.problem.compute_grad_lagrange_function(
                x= self.get_history(0),
                lm= _lm
            )
        return self.problem.compute_grad_lagrange_function(
            x= self.get_history(0),
            lm= _lm
        ) - self.problem.compute_grad_lagrange_function(
            x= self.get_history(-1),
            lm= _lm
        )
    
    # ------------------------------------------------------------------ #
    # Calculate approximtion parameters
    # Saves the approximation parameters in self.*
    # ------------------------------------------------------------------ #
    def calculate_approximation_parameters(self) -> None:
        ### Loacal variables
        delta = np.zeros(self.n)
        _lm = self.lm[:self.problem.m]
        _mu = self.lm[self.problem.m:] 
        _L_prime = self.problem.compute_grad_lagrange_function(x= self.x, lm= self.lm) # First derivative of lagrange
        _L_prime_prime = self.finite_differencing_lagrange(_lm = self.lm)              # Second derivative of lagrange

        ###
        ### Calculate Bounds
        ###

        # Calculate delta
        for i in range(self.n):
            _p_lm = self.p[i,:self.problem.m]
            _p_mu = self.p[i,self.problem.m:]
            _q_lm = self.q[i]
            coefficients_1 = [(1/4) * _L_prime_prime[i], (1/2) * _L_prime[i], 0, -(_lm @ _p_lm) - (_mu @ _p_mu)]
            coefficients_2 = [(1/4) * _L_prime_prime[i], -((1/2) * _L_prime[i]), 0, -(_lm @ _q_lm) + (_mu @ _p_mu)]
            roots_1 = np.roots(coefficients_1)
            roots_2 = np.roots(coefficients_2)
            all_roots = np.concatenate([roots_1, roots_2])
            all_roots = all_roots[np.isreal(all_roots)].real
            try:
                delta[i] = np.max(all_roots)
            except:
                delta[i] = 1
        
        # Scaling delta (info from Lecture)         ####################################################################################################################################################################################################################
        delta *= self.scaling_factor                # Scale up delta correct (blow up the dualfunction)
        
        ###
        ### Loop until all inequalities are in feasable
        ###
        while True:

            # Calculate asymptotes distances
            self.L_k = self.get_history(0) - delta
            self.U_k = self.get_history(0) + delta

            ###
            ### Update pji and qji      #####################################################################################################################################################################################################################################
            ###                         When should we update the pji and qji matrices
            
            list_inequality = []
            list_equality = []
            # Calculate new p_k and q_k for inequality constriants
            for j in range(self.problem.m):
                # Get inequality constraint vector
                inequality_list = [j]
                list_inequality.append(j)
                dg_dx = self.problem.compute_grad_constraints(
                    x= self.get_history(0),
                    selection= inequality_list
                )
                for i in range(self.n):
                    if dg_dx[0, i] > 0:
                        # p_k
                        self.p[i, j] = (self.U_k[i] - self.get_history(0)[i])**2 * dg_dx[0, i]
                        # q_k
                        self.q[i, j] = 0
                    else:
                        # p_k
                        self.p[i, j] = 0
                        # q_k
                        self.q[i, j] = (self.get_history(0)[i] - self.L_k[i])**2 * dg_dx[0, i]
            
            # Calculate new p_k for equality constriants
            for j in range(self.problem.me):
                # Get equality constraint vector
                inequality_list = [j + self.problem.m]
                list_equality.append(j + self.problem.m)
                dh_dx = self.problem.compute_grad_constraints(
                    x= self.get_history(0),
                    selection= inequality_list
                )
                for i in range(self.n):
                    # p_k
                    self.p[i, j] = (1/2)*(self.U_k[i] - self.get_history(0)[i])**2 * dh_dx[0, i]
            
            ###
            ### Calculate new parameters
            ###

            # Calculate objective parameters
            for i in range(self.n):
                _p_lm = self.p[i,:self.problem.m]
                _p_mu = self.p[i,self.problem.m:]
                _q_lm = self.q[i]
                p0i_k = (1/4) * delta[i]**3 * _L_prime_prime[i] + (1/2) * delta[i]**2 * _L_prime[i] - _lm @ _p_lm - _mu @ _p_mu
                q0i_k = (1/4) * delta[i]**3 * _L_prime_prime[i] - (1/2) * delta[i]**2 * _L_prime[i] - _lm @ _q_lm + _mu @ _p_mu
                self.p0_k[i] = p0i_k
                self.q0_k[i] = q0i_k

            # Calculate r´s for approximation functions
            ro_dif = 0
            for i in range(self.n):
                ro_dif += ((self.p0_k[i]/(self.U_k[i] - self.get_history(0)[i])) - (self.q0_k[i]/(self.get_history(0)[i] - self.L_k[i])))
            ri_dif = np.zeros(self.problem.m)
            for j in range(self.problem.m):
                for i in range(self.n):
                    ri_dif += ((self.p[i, j]/(self.U_k[i] - self.get_history(0)[i])) + (self.q[i, j]/(self.get_history(0)[i] - self.L_k[i])))   # Use + instead of OR since one is always 0
            self.ro_k = self.problem.compute_objective(x= self.get_history(0)) - ro_dif
            self.ri_k = self.problem.compute_constraints(x= self.get_history(0), selection= list_inequality) - ri_dif
            self.re_k = self.problem.compute_constraints(x= self.get_history(0), selection= list_equality)

            ##############################################################################################################################################################################################################################################################################
            ### Get roots of the inequalities and update delta if necessary 
            is_feasable = True
            for j in range(self.problem.m):
                for i in range(self.n):
                    if self.p[i, j] == 0:
                        # Check q part
                        root = self.L_k[i] - (self.q[i, j]/self.ri_k[j])
                        if root > self.U_k[i]:
                            delta[i] = (root - self.get_history(0)[i]) * self.update_scaling_factor
                            is_feasable = False
                    else:
                        # Check p part
                        root = self.U_k[i] + (self.p[i, j]/self.ri_k[j])
                        if root < self.L_k[i]:
                            delta[i] = (self.get_history(0)[i] - root) * self.update_scaling_factor
                            is_feasable = False

            if is_feasable: break

    # ------------------------------------------------------------------ #
    # Approximated Lagrange
    # ------------------------------------------------------------------ #
    def _approximated_objective_function(self, _x: NDArray) -> float:
        ro_dif = 0
        for i in range(self.n):
            ro_dif += ((self.p0_k[i]/(self.U_k[i] - _x[i])) + (self.q0_k[i]/(_x[i] - self.L_k[i])))
        return self.ro_k + ro_dif

    def _approximated_inequality_function(self, _x: NDArray) -> NDArray:
        ri_dif = np.zeros(self.problem.m)
        for j in range(self.problem.m):
            ri_dif_j = 0
            for i in range(self.n):
                ri_dif_j += ((self.p[i, j]/(self.U_k[i] - _x[i])) + (self.q[i, j]/(_x[i] - self.L_k[i])))    # Use + instead of Or since one is always 0
            ri_dif[j] = ri_dif_j
        return self.ri_k + ri_dif
    
    def _approximated_equality_function(self, _x: NDArray) -> NDArray:
        re_dif = np.zeros(self.problem.me)
        for j in range(self.problem.me):
            re_dif_j = 0
            for i in range(self.n):
                re_dif_j += ((self.p[i, j]/(self.U_k[i] - _x[i])) - (self.p[i, j]/(_x[i]- self.L_k[i])))
            re_dif[j] = re_dif_j
        return self.re_k + re_dif
    
    def compute_approximated_lagrange(self, _x: NDArray) -> float:
        lagrange_function = self._approximated_objective_function(_x= _x)
        inequality_functions = self._approximated_inequality_function(_x= _x)
        equality_functions = self._approximated_equality_function(_x= _x)
        for _ in range(self.problem.m):
            lagrange_function += self.lm[_] * inequality_functions[_]
        for _ in range(self.problem.me):
            lagrange_function += self.lm[_ + self.problem.m] * equality_functions[_]
        return lagrange_function

    # ------------------------------------------------------------------ #
    # Dual-Function approximation
    # ------------------------------------------------------------------ #
    def compute_dual_function(self) -> float:
        Pi = np.zeros(self.n)
        Qi = np.zeros(self.n)
        for i in range(self.n):
            p_lm = 0
            q_lm = 0
            pp_mu = 0
            pq_mu = 0
            for _ in range(self.problem.m):
                p_lm += self.lm[_] * self.p[i, _]
                q_lm += self.lm[_] * self.q[i, _]
            for _ in range(self.problem.me):
                pp_mu += self.lm[_ + self.problem.m] * self.p[i, _ + self.problem.m]
                pq_mu += self.lm[_ + self.problem.m] * self.p[i, _ + self.problem.m]
            Pi[i] = self.p0_k[i]
            Qi[i] = self.q0_k[i]
            ...

    
    def compute_grad_dual_function() -> int:
        ...
    
    def compute_second_order_information_dual_function() -> int:
        ...























































    # ------------------------------------------------------------------ #
    # Outsourced DEMMA functions
    # ------------------------------------------------------------------ #
    def mma_approximation(
        self,
        _m: int = None,
        _me:int = None
    ):
        ### Default self values
        if _m is None: _m = self.problem.m
        if _me is None: _me = self.problem.me

        lm_k = self.lm[:self.problem.m]                 # lamda: lagrange multipliers for inequality constrains
        mu_k = self.lm[self.problem.m:]                 # mü: lagrange multipliers for equality constrains

        g_k = self.inequality_constraints_approximation()
        h_x = self.equality_constraints_approximation()

        delta__k1 = ...
        delta__k2 = ...
        f_k = "objective function approximation"
        g_k = "inequality function approximation"
        h_k = "equality function approximation"

        return f_k, g_k, h_k
    
    
    
    
    

    def inequality_constraints_approximation(
        self
    ) -> float:
        ...
    
    def equality_constraints_approximation(
        self
    ) -> float:
        ...
    


    # ------------------------------------------------------------------ #
    # Convergence Test
    # ------------------------------------------------------------------ #
    def converged(
        self,
        gradient: NDArray,
        constraints: NDArray | None = None,
        gradient_tol: float = 1e-5,
        constraint_tol: float = 1e-5,
        complementarity_tol: float = 1e-5,
    ) -> bool:
        """
        Check convergence according to the first-order necessary (KKT)
        conditions assuming LICQ.

        See, for example, Theorem 12.1 in [1]_.

        Parameters
        ----------
        gradient : numpy.ndarray
            Current gradient of the Lagrange function with respect to
            the design variables.
        constraints : numpy.ndarray, optional
            Current constraint values.
        gradient_tol : float, default: 1e-5
            Tolerance applied to each component of the gradient.
        constraint_tol : float, default: 1e-5
            Tolerance applied to each constraint.
        complementarity_tol : float, default: 1e-5
            Tolerance applied to each complementarity condition.

        References
        ----------
        .. [1] J. Nocedal and S. J. Wright, Numerical Optimization. Springer New York, 2006. doi: https://doi.org/10.1007/978-0-387-40065-5.
        """
        # Project 1 code
        if constraints is None:
            if abs(gradient[0]) < gradient_tol and abs(gradient[1]) < gradient_tol:
                return True
            else:
                return False
        # Project 2 code
        else:
            lm = self.lm
            stationarity = True
            feasibility = True
            complementarity = True
   
            # Stationarity condition of KKT:
            if np.any(np.abs(gradient) >= gradient_tol):
                stationarity = False
           
            # Feasibility condition of KKT:
            if np.any(constraints >= constraint_tol):
                feasibility = False
   
            # Complementarity condition of KKT:
            for _ in range(len(lm)):
                # Non-negativity: λ_i ≥ 0
                if lm[_] < -complementarity_tol:
                    complementarity = False
                    break
                # Complementarity: λ_i * g_i(x) ≈ 0
                if abs(lm[_] * constraints[_]) > complementarity_tol:
                    complementarity = False
                    break
   
            if stationarity == True and feasibility == True and complementarity == True :
                return True
            else:
                return False

    def line_search(
        self,
        direction: NDArray,
        alpha_ini: float = 1,
        alpha_min: float = 1e-6,
        alpha_max: float = 1,
        algorithm: Literal[
            "WOLFE",
            "STRONG_WOLFE",
            "GOLDSTEIN-PRICE",
        ] = "STRONG_WOLFE",
        m1: float = 0.01,
        m2: float = 0.90,
        callback: Callable[[OptimisationResult], Any] | None = None,
        callback_iteration: int | None = None,
    ) -> float:
        """
        Perform a line search and returns an approximately optimal step size.

        Parameters
        ----------
        direction : numpy.ndarray
            Search direction.
        alpha_ini : float
            Initial step size.
        alpha_min : float, optional
            Minimum step size.
        alpha_max : float
            Maximum step size.
        algorithm : str, optional
            Line search algorithm to use.
        m1 : float, optional
            Parameter for the sufficient decrease condition.
        m2 : float, optional
            Parameter for the curvature condition.
        callback : callable, optional
            Callback function for collecting intermediate results.
        callback_iteration : int, optional
            Iteration number for the callback function.

        Returns
        -------
        float
            Approximately optimal step size.
        """
        ...