import pytest
from sirom.code.optimization_problem import OptimizationProblem

c_value = [3,1] #[x,y]
A_value = [[1,1],[1,0],[0,1],[-1,0],[0,-1]] #[x,y]
b_value = [2,1,2,0,0]

def test_optimization_problem_success():
    opt_problem = OptimizationProblem(c_value,A_value,b_value)
    assert "[OK] Optimization problem creation succeeded" in opt_problem.status

def test_optimization_problem_failed():
    opt_problem = OptimizationProblem("",A_value,b_value)
    assert "[ERROR] Optimization problem creation failed" in opt_problem.status

def test_optimization_problem_objective_coeficient_empty():
    opt_problem = OptimizationProblem("",A_value,b_value)
    assert "[ERROR] Undefined objective coeficient" in opt_problem.status

def test_optimization_problem_constraint_coeficient_empty():
    opt_problem = OptimizationProblem(c_value,"",b_value)
    assert "[ERROR] Undefined constraint coeficient" in opt_problem.status

def test_optimization_problem_rhs_coeficient_empty():
    opt_problem = OptimizationProblem(c_value,A_value,"")
    assert "[ERROR] Undefined rhs coeficient" in opt_problem.status

def test_optimization_problem_dimension_validation_error():
    opt_problem = OptimizationProblem("",A_value,b_value)
    assert "[ERROR] Dimension can not be evaluated" in opt_problem.status

def test_optimization_problem_dimension_inconsistency_error():
    opt_problem = OptimizationProblem([1],A_value,b_value)
    assert "[ERROR] Dimension inconsistency detected" in opt_problem.status