import pytest
import numpy as np
from sirom.optimization_problem import OptimizationProblem
from sirom.mini_ortools_solver import MiniOrtoolsSolver, select_solver

c_value = np.array([-3, -4])  # [x,y]
A_value = np.matrix([
    [1, 2], 
    [-3, 1], 
    [1, -1], 
    [-1, 0], 
    [0, -1]
])  # [x,y]
b_value = np.array([14, 0, 2, 0, 0])


optimization_problem = OptimizationProblem(c_value, A_value, b_value)


def test_mini_ortools_solver_optimization_problem_broken():
    opt_problem = OptimizationProblem("", A_value, b_value)
    mini_ortool = MiniOrtoolsSolver(opt_problem)
    assert "[ERROR] Optimization problem creation failed" in mini_ortool.status


def test_mini_ortools_solver_wrong_solver_selection():
    mini_ortool = MiniOrtoolsSolver(optimization_problem, "SIROM")
    assert "[ERROR] Solver creation failed" in mini_ortool.status


def test_mini_ortools_solver_rejecting_optimization_problem():
    mini_ortool = MiniOrtoolsSolver("")
    assert "[ERROR] Optimization problem validation failed" in mini_ortool.status


def test_mini_ortools_solver_validating_optimization_problem():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert "[OK] Optimization problem validation succeeded" in mini_ortool.status


def test_mini_ortools_solver_creating_variables():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert "[OK] Variables creation succeeded" in mini_ortool.status


def test_mini_ortools_solver_creating_constraints():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert "[OK] Constraints creation succeeded" in mini_ortool.status


def test_mini_ortools_solver_solution_objective_value():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    # approx: GLOP is a floating-point simplex (returns -33.999...).
    assert mini_ortool.solution["objective_value"] == pytest.approx(-34.0)


def test_mini_ortools_solver_solution_solve_status():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert mini_ortool.solution["solve_status"] == 0


def test_solution_constraint_is_the_slack():
    # At the optimum x ~ [6, 4], the per-constraint slack A_i.x - b_i is
    # [0, -14, 0, -6, -4] for this problem.
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert mini_ortool.solution["constraint"] == pytest.approx(
        [0.0, -14.0, 0.0, -6.0, -4.0], abs=1e-6
    )


def test_solve_emits_no_future_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        mini_ortool = MiniOrtoolsSolver(optimization_problem)
        assert mini_ortool.solution["solve_status"] == 0
        assert mini_ortool.solution["objective_value"] == pytest.approx(-34.0)


def test_select_solver_glop_for_continuous():
    assert select_solver(optimization_problem) == "GLOP"


def test_select_solver_scip_for_integer():
    problem = OptimizationProblem(c_value, A_value, b_value, integer_variables=[0])
    assert select_solver(problem) == "SCIP"


def test_default_solver_is_glop_and_optimum_unchanged():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert mini_ortool.solver_selected == "GLOP"
    assert mini_ortool.solution["objective_value"] == pytest.approx(-34.0)


def test_integer_problem_solves_with_integer_values():
    # max x + y (as min -x - y) over a box; forcing integrality must still solve
    # and return whole numbers.
    c = np.array([-1, -1])
    A = np.matrix([[1, 0], [0, 1]])
    b = np.array([2, 2])
    problem = OptimizationProblem(c, A, b, integer_variables=[0, 1])
    mini_ortool = MiniOrtoolsSolver(problem)
    assert mini_ortool.solver_selected == "SCIP"
    assert mini_ortool.solution["solve_status"] == 0
    assert all(
        abs(v - round(v)) < 1e-6 for v in mini_ortool.solution["variable"]
    )
