import pytest
from sirom.code.batch_solver import ProblemsBucket

number_of_scenarios = 10

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

def test_batch_solver_undefined_objective_coefficient():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Undefined objective coefficient" in opt_problem.status

def test_batch_solver_undefined_lb_constraint_coefficient():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Undefined lb_constraint coefficient" in opt_problem.status

def test_batch_solver_undefined_ub_constraint_coefficient():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Undefined ub_constraint coefficient" in opt_problem.status

def test_batch_solver_undefined_lb_rhs_coefficient():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Undefined lb_rhs coefficient" in opt_problem.status

def test_batch_solver_undefined_ub_rhs_coefficient():
    opt_problem = ProblemsBucket("","","","","")
    assert "[ERROR] Undefined ub_rhs coefficient" in opt_problem.status

def test_batch_solver_failed_acquiring_objective_coefficient():
    opt_problem = ProblemsBucket({'oi' : ""},"","","","")
    assert "[ERROR] Failed acquiring objective coefficient" in opt_problem.status

def test_batch_solver_failed_acquiring_lb_constraint_coefficient():
    opt_problem = ProblemsBucket("",{'oi' : ""},"","","")
    assert "[ERROR] Failed acquiring lb_constraint coefficient" in opt_problem.status

def test_batch_solver_failed_acquiring_ub_constraint_coefficient():
    opt_problem = ProblemsBucket("","",{'oi' : ""},"","")
    assert "[ERROR] Failed acquiring ub_constraint coefficient" in opt_problem.status

def test_batch_solver_failed_acquiring_lb_rhs_coefficient():
    opt_problem = ProblemsBucket("","","",{'oi' : ""},"")
    assert "[ERROR] Failed acquiring lb_rhs coefficient" in opt_problem.status

def test_batch_solver_failed_acquiring_ub_rhs_coefficient():
    opt_problem = ProblemsBucket("","","","",{'oi' : ""})
    assert "[ERROR] Failed acquiring ub_rhs coefficient" in opt_problem.status

def test_batch_solver_undefined_number_of_scenarios():
    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, "")
    assert "[ERROR] Undefined number of scenarios" in opt_problem_batch.status

def test_batch_solver_float_number_of_scenarios():
    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, 100.0)
    assert "[ERROR] Failed acquiring number of scenarios" in opt_problem_batch.status

def test_batch_solver_negative_number_of_scenarios():
    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, -1)
    assert "[ERROR] Failed acquiring number of scenarios" in opt_problem_batch.status

def test_batch_solver_scenarios_size():
    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios)
    assert (len(opt_problem_batch.coefficient['scenarios_constraint']) == number_of_scenarios) & \
            (len(opt_problem_batch.coefficient['scenarios_rhs'])== number_of_scenarios)

def test_batch_solver_solve():
    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios)
    opt_problem_batch.solve()