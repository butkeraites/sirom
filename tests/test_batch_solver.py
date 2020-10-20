import pytest
from sirom.code.batch_solver import ProblemsBucket

c_value = [3,1] #[x,y]
lb_A_value = [[1,1],[1,0],[0,1],[-1,0],[0,-1]] #[x,y]
ub_A_value = [[2,2],[2,1],[1,2],[-1,0],[0,-1]] #[x,y]
lb_b_value = [2,1,2,0,0]
ub_b_value = [3,2,3,0,0]

def test_batch_solver_success():
    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value)
    assert "[OK] Optimization batch creation succeeded" in opt_problem_batch.status

def test_batch_solver_failed():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Optimization batch creation failed" in opt_problem.status

def test_batch_solver_without_objective_coefficient():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Failed acquiring objective coefficient" in opt_problem.status