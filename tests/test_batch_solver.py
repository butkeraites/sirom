import pytest
from sirom.batch_solver import ProblemsBucket

number_of_scenarios = 10

c_value = [3, 1]  # [x,y]
lb_A_value = [[1, 1], [1, 0], [0, 1], [-1, 0], [0, -1]]  # [x,y]
ub_A_value = [[2, 2], [2, 1], [1, 2], [-1, 0], [0, -1]]  # [x,y]
lb_b_value = [2, 1, 2, 0, 0]
ub_b_value = [3, 2, 3, 0, 0]


def test_batch_solver_success():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value
    )
    assert "[OK] Optimization batch creation succeeded" in opt_problem_batch.status


def test_batch_solver_failed():
    opt_problem = ProblemsBucket("", "", "", "", "")
    assert "[ERROR] Optimization batch creation failed" in opt_problem.status


def test_batch_solver_undefined_objective_coefficient():
    opt_problem = ProblemsBucket("", "", "", "", "")
    assert "[ERROR] Undefined objective coefficient" in opt_problem.status


def test_batch_solver_undefined_lb_constraint_coefficient():
    opt_problem = ProblemsBucket("", "", "", "", "")
    assert "[ERROR] Undefined lb_constraint coefficient" in opt_problem.status


def test_batch_solver_undefined_ub_constraint_coefficient():
    opt_problem = ProblemsBucket("", "", "", "", "")
    assert "[ERROR] Undefined ub_constraint coefficient" in opt_problem.status


def test_batch_solver_undefined_lb_rhs_coefficient():
    opt_problem = ProblemsBucket("", "", "", "", "")
    assert "[ERROR] Undefined lb_rhs coefficient" in opt_problem.status


def test_batch_solver_undefined_ub_rhs_coefficient():
    opt_problem = ProblemsBucket("", "", "", "", "")
    assert "[ERROR] Undefined ub_rhs coefficient" in opt_problem.status


def test_batch_solver_failed_acquiring_objective_coefficient():
    opt_problem = ProblemsBucket({"oi": ""}, "", "", "", "")
    assert "[ERROR] Failed acquiring objective coefficient" in opt_problem.status


def test_batch_solver_failed_acquiring_lb_constraint_coefficient():
    opt_problem = ProblemsBucket("", {"oi": ""}, "", "", "")
    assert "[ERROR] Failed acquiring lb_constraint coefficient" in opt_problem.status


def test_batch_solver_failed_acquiring_ub_constraint_coefficient():
    opt_problem = ProblemsBucket("", "", {"oi": ""}, "", "")
    assert "[ERROR] Failed acquiring ub_constraint coefficient" in opt_problem.status


def test_batch_solver_failed_acquiring_lb_rhs_coefficient():
    opt_problem = ProblemsBucket("", "", "", {"oi": ""}, "")
    assert "[ERROR] Failed acquiring lb_rhs coefficient" in opt_problem.status


def test_batch_solver_failed_acquiring_ub_rhs_coefficient():
    opt_problem = ProblemsBucket("", "", "", "", {"oi": ""})
    assert "[ERROR] Failed acquiring ub_rhs coefficient" in opt_problem.status


def test_batch_solver_undefined_number_of_scenarios():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, ""
    )
    assert "[ERROR] Undefined number of scenarios" in opt_problem_batch.status


def test_batch_solver_float_number_of_scenarios():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, 100.0
    )
    assert "[ERROR] Failed acquiring number of scenarios" in opt_problem_batch.status


def test_batch_solver_negative_number_of_scenarios():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, -1
    )
    assert "[ERROR] Failed acquiring number of scenarios" in opt_problem_batch.status


def test_batch_solver_scenarios_size():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios
    )
    assert (
        len(opt_problem_batch.coefficient.scenarios_constraint)
        == number_of_scenarios
    ) & (len(opt_problem_batch.coefficient.scenarios_rhs) == number_of_scenarios)


def test_batch_solver_solve():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios
    )
    opt_problem_batch.solve()


def test_batch_solver_default_number_of_clusters():
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios
    )
    assert opt_problem_batch.number_of_clusters == 3


def test_batch_solver_custom_number_of_clusters_is_stored():
    opt_problem_batch = ProblemsBucket(
        c_value,
        lb_A_value,
        ub_A_value,
        lb_b_value,
        ub_b_value,
        number_of_scenarios,
        number_of_clusters=5,
    )
    assert opt_problem_batch.number_of_clusters == 5


def test_batch_solver_parallel_matches_serial():
    # Solving the same scenarios serially (n_jobs=1) and in parallel (n_jobs=4)
    # must produce identical, in-order results.
    serial = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value,
        number_of_scenarios=8, n_jobs=1,
    )
    parallel = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value,
        number_of_scenarios=8, n_jobs=4,
    )
    # Reuse the same sampled scenarios for both so the comparison is exact.
    parallel.coefficient.scenarios_constraint = serial.coefficient.scenarios_constraint
    parallel.coefficient.scenarios_rhs = serial.coefficient.scenarios_rhs

    serial.solve()
    parallel.solve()

    assert [r.get("objective_value") for r in serial.results] == [
        r.get("objective_value") for r in parallel.results
    ]


def test_batch_solver_quality_measure_feasibility_extremes():
    # x=0 satisfies every constraint (A*0 = 0 <= b) for all scenarios -> 1.0;
    # a huge x violates the upper-bound rows for all scenarios -> 0.0.
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios
    )
    opt_problem_batch.results = [
        {"solve_status": 0, "variable": [0.0, 0.0]},
        {"solve_status": 0, "variable": [1000.0, 1000.0]},
    ]
    opt_problem_batch.apply_quality_measure(number_of_scenarios=10)
    assert opt_problem_batch.results[0]["feasibility_probability"] == 1.0
    assert opt_problem_batch.results[1]["feasibility_probability"] == 0.0


def test_batch_solver_quality_measure_skips_non_optimal():
    # A non-optimal result (no "variable" key) must not crash the quality
    # measure; it should be scored as never feasible instead.
    opt_problem_batch = ProblemsBucket(
        c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios
    )
    opt_problem_batch.results.append({"solve_status": 2})  # INFEASIBLE, no variable
    opt_problem_batch.apply_quality_measure(number_of_scenarios=5)
    assert opt_problem_batch.results[-1]["feasibility_probability"] == 0.0
