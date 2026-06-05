from __future__ import print_function
from typing import List, TypedDict
import numpy as np
from ortools.linear_solver import pywraplp  # type: ignore
from .optimization_problem import OptimizationProblem
from .status_checks import has_errors


class _OptionalSolutionKeys(TypedDict, total=False):
    # Present only on an optimal solve.
    variable: List[float]
    constraint: List[float]
    objective_value: float


class UnscoredSolution(_OptionalSolutionKeys):
    """What :class:`MiniOrtoolsSolver` produces from one scenario.

    ``solve_status`` is always present; the decision vector and its derived
    fields appear only when the solve reached optimality. It carries no
    feasibility probability — the solver does not produce one.
    """

    solve_status: int


class ScoredSolution(UnscoredSolution):
    """An Unscored Solution after the Scoring stage has added its feasibility."""

    feasibility_probability: float


def is_optimal(solution: "UnscoredSolution") -> bool:
    """Whether a solve reached optimality (and so carries a decision vector)."""
    return solution["solve_status"] == 0


def feasibility(solution: "ScoredSolution") -> float:
    """The feasibility probability of a scored solution."""
    return solution["feasibility_probability"]


def score(solution: "UnscoredSolution", probability: float) -> "ScoredSolution":
    """The one transform that turns an Unscored Solution into a Scored one."""
    return {**solution, "feasibility_probability": probability}


def select_solver(optimization_problem: OptimizationProblem) -> str:
    """Pick the right OR-Tools backend for a problem.

    Pure LPs (all-continuous) get GLOP, OR-Tools' fast simplex solver. Problems
    with integer variables need a MIP backend, so they get SCIP.
    """
    if getattr(optimization_problem, "integer_variables", None):
        return "SCIP"
    return "GLOP"


def solver_available(name: str) -> bool:
    """Whether an OR-Tools backend can be created in this build.

    Any name OR-Tools recognizes works, including commercial solvers (GUROBI,
    CPLEX, XPRESS) when OR-Tools is built against them and a license is present.
    """
    return pywraplp.Solver.CreateSolver(name) is not None


class MiniOrtoolsSolver:
    "Class that translate a Optimization problem to Ortools framework, solve and retrieve solution"

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        solver_selection: "str | None" = None,
        print_log: bool = False,
    ):
        self.status: List[str] = []
        self.problem: OptimizationProblem = optimization_problem
        # None -> auto-select (GLOP for LP, SCIP for MILP). An explicit name is
        # used verbatim, so an unknown name still surfaces a solver-creation error.
        self.solver_selected: "str | None" = solver_selection
        self.print_log: bool = print_log
        self.__validate_optimization_problem()

    def __validate_optimization_problem(self):
        if not isinstance(self.problem, OptimizationProblem):
            self.status.append("[ERROR] Optimization problem validation failed")
            return
        self.status.append("[OK] Optimization problem validation succeeded")
        self.__start_optimization_chain()

    def __start_optimization_chain(self):
        if has_errors(self.problem.status):
            self.status.append("[ERROR] Optimization problem creation failed")
            return
        self.status.append("[OK] Optimization problem creation succeeded")
        self.__create_solver()
        self.__solve()

    def __create_solver(self):
        if self.solver_selected is None:
            self.solver_selected = select_solver(self.problem)
        self.solver = pywraplp.Solver.CreateSolver(self.solver_selected)
        if self.solver is None:
            self.status.append("[ERROR] Solver creation failed")
            return
        self.status.append("[OK] Solver creation succeeded")
        self.__create_variables()
        self.__create_constraints()
        self.__create_objective_function()

    def __create_variables(self):
        n_var = len(self.problem.coefficient.objective)
        integer = set(self.problem.integer_variables)
        infinity = self.solver.infinity()
        self.variables = [
            self.solver.IntVar(0, infinity, "x{}".format(str(id)))
            if id in integer
            else self.solver.NumVar(0, infinity, "x{}".format(str(id)))
            for id in range(n_var)
        ]
        self.status.append("[OK] Variables creation succeeded")
        self.status.append(
            "[INFO] Number of variables: {}".format(self.solver.NumVariables())
        )

    def __create_constraints(self):
        self.constraints = []
        constraint_matrix = np.asarray(
            self.problem.coefficient.constraint, dtype=float
        )
        rhs = np.asarray(self.problem.coefficient.rhs, dtype=float).reshape(-1)
        infinity = self.solver.infinity()
        for i in range(constraint_matrix.shape[0]):
            constraint = self.solver.Constraint(-infinity, float(rhs[i]))
            for j, coefficient in enumerate(constraint_matrix[i]):
                if coefficient != 0.0:  # unset coefficients default to 0
                    constraint.SetCoefficient(self.variables[j], float(coefficient))
            self.constraints.append(constraint)
        self.status.append("[OK] Constraints creation succeeded")
        self.status.append(
            "[INFO] Number of constraints: {}".format(self.solver.NumConstraints())
        )

    def __create_objective_function(self):
        self.objective = self.solver.Objective()
        coefficients = np.asarray(
            self.problem.coefficient.objective, dtype=float
        ).reshape(-1)
        for j, coefficient in enumerate(coefficients):
            if coefficient != 0.0:
                self.objective.SetCoefficient(self.variables[j], float(coefficient))
        self.objective.SetMinimization()
        self.status.append("[OK] Objective function creation succeeded")

    def __solve(self):
        if has_errors(self.status):
            self.status.append("[ERROR] Solving process failed")
            return
        self.solve_status = self.solver.Solve()
        self.status.append("[OK] Solving process succeeded")
        self.__retrieve_solution()

    def __retrieve_solution(self):
        # The solver produces an Unscored Solution: solve_status always, the
        # decision vector and its derived fields only when optimal. Scoring
        # (apply_quality_measure) is what later adds feasibility_probability.
        self.solution: UnscoredSolution = {"solve_status": self.solve_status}
        if self.solve_status == self.solver.OPTIMAL:
            self.__retrieve_optimal_solution()
        if self.print_log:
            # Debug trace, not part of the solution contract; kept off the dict.
            self.log: List[str] = self.status + self.problem.status

    def __retrieve_optimal_solution(self):
        variables = [x.solution_value() for x in self.variables]
        x = np.asarray(variables, dtype=float)
        constraint_matrix = np.asarray(
            self.problem.coefficient.constraint, dtype=float
        )
        rhs = np.asarray(self.problem.coefficient.rhs, dtype=float).reshape(-1)
        objective_coefficient = np.asarray(
            self.problem.coefficient.objective, dtype=float
        ).reshape(-1)
        # Per-constraint slack g_i(x) = A_i . x - b_i (the constraint function
        # from the problem form A x - b <= 0). Used as a clustering feature
        # downstream. (Previously this erroneously computed A_i . x - b_i*sum(x).)
        constraints = (constraint_matrix @ x - rhs).tolist()
        objective_value = float(objective_coefficient @ x)
        self.solution["variable"] = variables
        self.solution["constraint"] = constraints
        self.solution["objective_value"] = objective_value
