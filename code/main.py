from batch_solver import ProblemsBucket


def test_optimization():
    c_value = [-3,-4] #[x,y]
    lb_A_value = [[1,1],[1,0],[0,1],[-1,0],[0,-1]] #[x,y]
    ub_A_value = [[2,2],[2,1],[1,2],[-1,0],[0,-1]] #[x,y]
    lb_b_value = [2,1,2,0,0]
    ub_b_value = [3,2,3,0,0]

    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios=100)
    opt_problem_batch.solve()
    opt_problem_batch.cluster_and_selection()
    opt_problem_batch.solve_cluster_tree()
    opt_problem_batch.apply_quality_measure(number_of_scenarios=100)
    #TODO: IMPROVE EFFICIENCY, AND CREATE PARETO FRONTIER, EXPORT  CSV

test_optimization()