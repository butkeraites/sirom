import pytest
from sirom.code.optimization_problem import OptimizationProblem
from sirom.code.mini_ortools_solver import MiniOrtoolsSolver

c_value = [-3,-4] #[x,y]
A_value = [[1,2],[-3,1],[1,-1],[-1,0],[0,-1]] #[x,y]
b_value = [14,0,2,0,0]
optimization_problem = OptimizationProblem(c_value, A_value, b_value)

def test_mini_ortools_solver_optimization_problem_broken():
    opt_problem = OptimizationProblem("",A_value,b_value)
    mini_ortool = MiniOrtoolsSolver(opt_problem)
    assert "[ERROR] Optimization problem creation failed" in mini_ortool.status

def test_mini_ortools_solver_wrong_solver_selection():
    mini_ortool = MiniOrtoolsSolver(optimization_problem, 'SIROM')
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
    assert mini_ortool.solution['objective_value'] == -34.0

def test_mini_ortools_solver_solution_solve_status():
    mini_ortool = MiniOrtoolsSolver(optimization_problem)
    assert mini_ortool.solution['solve_status'] == 0