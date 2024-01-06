from __future__ import print_function
from numbers import Number
from typing import List, TypedDict
import pandas as pd
from ortools.linear_solver import pywraplp  # type: ignore
from .optimization_problem import OptimizationProblem


class Solution(TypedDict):
    variable: List[Number]
    constraint: List[Number]
    objective_value: Number
    solve_status: str
    feasibility_probability: float
    log: str


class MiniOrtoolsSolver:
    "Class that translate a Optimization problem to Ortools framework, solve and retrieve solution"

    def __init__(
        self,
        optimization_problem: OptimizationProblem,
        solver_selection: str = "SCIP",
        print_log: bool = False,
    ):
        self.status: List[str] = []
        self.problem: OptimizationProblem = optimization_problem
        self.solver_selected: str = solver_selection
        self.print_log: bool = print_log
        self.__validate_optimization_problem()

    def __validate_optimization_problem(self):
        if not isinstance(self.problem, OptimizationProblem):
            self.status.append("[ERROR] Optimization problem validation failed")
            return
        self.status.append("[OK] Optimization problem validation succeeded")
        self.__start_optimization_chain()

    def __start_optimization_chain(self):
        if any("[ERROR]" in status for status in self.problem.status):
            self.status.append("[ERROR] Optimization problem creation failed")
            return
        self.status.append("[OK] Optimization problem creation succeeded")
        self.__create_solver()
        self.__solve()

    def __create_solver(self):
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
        self.variables = [
            self.solver.NumVar(0, self.solver.infinity(), "x{}".format(str(id)))
            for id in range(n_var)
        ]
        self.status.append("[OK] Variables creation succeeded")
        self.status.append(
            "[INFO] Number of variables: {}".format(self.solver.NumVariables())
        )

    def __create_constraints(self):
        self.constraints = []
        coefficient = self.problem.coefficient.constraint
        rhs = self.problem.coefficient.rhs
        for index, row in coefficient.iterrows():
            constraint = self.solver.Constraint(
                -self.solver.infinity(), float(rhs.iloc[index])
            )
            for index_single_coefficient, single_coefficient in row.items():
                constraint.SetCoefficient(
                    self.variables[index_single_coefficient], float(single_coefficient)
                )
            self.constraints.append(constraint)
        self.status.append("[OK] Constraints creation succeeded")
        self.status.append(
            "[INFO] Number of constraints: {}".format(self.solver.NumConstraints())
        )

    def __create_objective_function(self):
        self.objective = self.solver.Objective()
        coefficients = self.problem.coefficient.objective
        for index_single_coefficient, single_coefficient in coefficients.iterrows():
            self.objective.SetCoefficient(
                self.variables[index_single_coefficient], float(single_coefficient)
            )
        self.objective.SetMinimization()
        self.status.append("[OK] Objective function creation succeeded")

    def __solve(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Solving process failed")
            return
        self.solve_status = self.solver.Solve()
        self.status.append("[OK] Solving process succeeded")
        self.__retrieve_solution()

    def __retrieve_solution(self):
        self.solution = Solution()
        if self.solve_status == self.solver.OPTIMAL:
            self.__retrieve_optimal_solution()
        self.solution.solve_status = self.solve_status
        if self.print_log:
            self.solution.log = self.status + self.problem.status

    def __retrieve_optimal_solution(self):
        variables = [x.solution_value() for x in self.variables]
        coefficient = self.problem.coefficient["constraint"]
        rhs = self.problem.coefficient["rhs"]
        objective_coefficient = self.problem.coefficient["objective"]
        constraints = [
            self.__evaluate_equation(row - float(rhs.iloc[index]), variables)
            for index, row in coefficient.iterrows()
        ]
        objective_value = self.__evaluate_equation(objective_coefficient, variables)
        self.solution["variable"] = variables
        self.solution["constraint"] = constraints
        self.solution["objective_value"] = float(objective_value)

    def __evaluate_equation(self, coefficients_values, variable_values) -> float:
        equation_value = 0.0
        if type(variable_values) == pd.Series:
            for index, value in variable_values.items():
                equation_value += value * coefficients_values.iloc[index]
        else:
            for index, value in enumerate(variable_values):
                equation_value += value * coefficients_values.iloc[index]
        return equation_value
