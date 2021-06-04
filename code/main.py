from sirom.code.batch_solver import ProblemsBucket
from sirom.code.cluster_tree import ClusterTree

information = {
    'name': 'Node test.'
}

def test_optimization():
    number_of_scenarios = 10
    c_value = [-3,-4] #[x,y]
    lb_A_value = [[1,1],[1,0],[0,1],[-1,0],[0,-1]] #[x,y]
    ub_A_value = [[2,2],[2,1],[1,2],[-1,0],[0,-1]] #[x,y]
    lb_b_value = [2,1,2,0,0]
    ub_b_value = [3,2,3,0,0]

    opt_problem_batch = ProblemsBucket(c_value,lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios)
    opt_problem_batch.solve()
    opt_problem_batch.cluster_and_selection()

def test_cluster_tree():
    def create_test_childs(tree,clones):
        nodes = tree.get_all_nodes()
        for node in nodes:
            if not tree.tree_nodes[node]['child_nodes']:
                tree.create_nodes(node,clones)

    new_tree = ClusterTree(information)
    create_test_childs(new_tree, [{'information': 'child 1'}, {'information': 'child 2'}])
    create_test_childs(new_tree, [{'information': 'child 1'}, {'information': 'child 2'}])
    create_test_childs(new_tree, [{'information': 'child 1'}, {'information': 'child 2'}])
    print('------')
    new_tree.from_root_to_leafs()

test_optimization()