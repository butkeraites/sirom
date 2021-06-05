from numpy.core import numeric
from numpy.core.fromnumeric import mean
import numpy as np

import pandas as pd

from datetime import date
import time

from sklearn.cluster import KMeans
from smt.sampling_methods import LHS

from sirom.code.optimization_problem import OptimizationProblem
from sirom.code.mini_ortools_solver import MiniOrtoolsSolver
from sirom.code.cluster_tree import ClusterTree

class ProblemsBucket:
    'Class that will generate and storage all instances used to solve a problem like:'
    'min c * x : S.t. [lb_A, ub_A] * x - [lb_b, ub_b] <= 0'
    def __init__(self, c_value, lb_A_value, ub_A_value, lb_b_value, ub_b_value, number_of_scenarios = 100):
        self.status = []
        self.results = []
        self.coefficient = {}
        self.number_of_scenarios = -1
        self.__coefficient_validation(c_value,"objective")
        self.__coefficient_validation(lb_A_value,"lb_constraint")
        self.__coefficient_validation(ub_A_value,"ub_constraint")
        self.__coefficient_validation(lb_b_value,"lb_rhs")
        self.__coefficient_validation(ub_b_value,"ub_rhs")
        self.__number_of_scenarios_validation(number_of_scenarios)
        self.__dimension_validation()
        self.__problem_integrity_validation()

    def __coefficient_validation(self, coefficient, identification):
        if coefficient :
            try:
                self.coefficient[identification] = pd.DataFrame(coefficient)
                self.status.append("[OK] Successfuly acquired {} coefficient".format(identification))
            except:
                self.status.append("[ERROR] Failed acquiring {} coefficient".format(identification))
        else:
            self.status.append("[ERROR] Undefined {} coefficient".format(identification))

    def __dimension_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Dimension can not be evaluated")
        else:
            objective_coefficients, objective_equations = self.coefficient["objective"].shape
            lb_constraint_equations, lb_constraint_coefficients = self.coefficient["lb_constraint"].shape
            ub_constraint_equations, ub_constraint_coefficients = self.coefficient["ub_constraint"].shape
            lb_rhs_coefficients, lb_rhs_dimension = self.coefficient["lb_rhs"].shape
            ub_rhs_coefficients, ub_rhs_dimension = self.coefficient["ub_rhs"].shape
            if  objective_coefficients     == lb_constraint_coefficients \
            and lb_constraint_coefficients == ub_constraint_coefficients \
            and lb_constraint_equations    == ub_constraint_equations \
            and ub_constraint_equations    == ub_rhs_coefficients \
            and lb_rhs_dimension           == ub_rhs_dimension \
            and lb_rhs_coefficients        == ub_rhs_coefficients \
            and objective_equations        == 1 \
            and ub_rhs_dimension           == 1:
                self.status.append("[OK] Optimization batch creation succeeded")
            else:
                self.status.append("[ERROR] Dimension inconsistency detected")

    def __number_of_scenarios_validation(self,number_of_scenarios):
        if number_of_scenarios:
            if isinstance(number_of_scenarios,int) and (number_of_scenarios >= 0):
                self.status.append("[OK] Successfuly acquired number of scenarios")
                self.number_of_scenarios = number_of_scenarios
            else:
                self.status.append("[ERROR] Failed acquiring number of scenarios")
        else:
            self.status.append("[ERROR] Undefined number of scenarios")

    def __problem_integrity_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Optimization batch creation failed")
        else:
            self.status.append("[OK] Optimization batch creation succeeded")
            self.__generate_all_coefficients()
    
    def __generate_coefficients(self, number_of_scenarios):
        def generate_constraint_coefficient():
            lb_constraint_equations, lb_constraint_coefficients = self.coefficient["lb_constraint"].shape
            xlimits = np.array([[0.0, 1.0]]*(lb_constraint_equations * lb_constraint_coefficients))
            sampling = LHS(xlimits=xlimits)
            print('[{}] Coefficient generation'.format(date.today()))
            tic = time.time()
            scenarios_delta = sampling(number_of_scenarios)
            toc = time.time()
            print('[{}] Duration: {}'.format(date.today(),toc-tic))
            scenarios_delta_reshaped = np.array(scenarios_delta.transpose())
            scenarios_delta_reshaped = scenarios_delta_reshaped.reshape(number_of_scenarios, lb_constraint_equations, lb_constraint_coefficients)
            scenarios_constraint = []
            for scenario_delta_reshaped in range(number_of_scenarios):
                scenarios_constraint.append(self.coefficient["lb_constraint"] + (self.coefficient["ub_constraint"]-self.coefficient["lb_constraint"])*scenarios_delta_reshaped[scenario_delta_reshaped])
            return scenarios_constraint
        
        def generate_rhs_coefficient():
            lb_rhs_equations, lb_rhs_coefficients = self.coefficient["lb_rhs"].shape
            xlimits = np.array([[0.0, 1.0]]*(lb_rhs_equations * lb_rhs_coefficients))
            sampling = LHS(xlimits=xlimits)
            print('[{}] RHS generation'.format(date.today()))
            tic = time.time()
            scenarios_delta = sampling(number_of_scenarios)
            toc = time.time()
            print('[{}] Duration: {}'.format(date.today(),toc-tic))
            scenarios_delta_reshaped = np.array(scenarios_delta.transpose())
            scenarios_delta_reshaped = scenarios_delta_reshaped.reshape(number_of_scenarios, lb_rhs_equations, lb_rhs_coefficients)
            scenarios_rhs = []
            for scenario_delta_reshaped in range(number_of_scenarios):
                scenarios_rhs.append(self.coefficient["lb_rhs"] + (self.coefficient["ub_rhs"]-self.coefficient["lb_rhs"])*scenarios_delta_reshaped[scenario_delta_reshaped])
            return scenarios_rhs

        return generate_constraint_coefficient(), generate_rhs_coefficient()

    def __generate_all_coefficients(self):
        self.coefficient['scenarios_constraint'], self.coefficient['scenarios_rhs'] = self.__generate_coefficients(self.number_of_scenarios)
    
    def solve(self):
        c_value = np.array(self.coefficient["objective"])
        print('[{}] Solve process started'.format(date.today()))
        for scenario in range(self.number_of_scenarios):
            tic = time.time()
            A_value = np.matrix(self.coefficient['scenarios_constraint'][scenario])
            b_value = np.array(self.coefficient['scenarios_rhs'][scenario])
            optimization_problem = OptimizationProblem(c_value,A_value,b_value)
            mini_ortool = MiniOrtoolsSolver(optimization_problem)
            toc = time.time()
            print('[{}] Scenario {} | Duration: {}'.format(date.today(),scenario,toc-tic))
            self.results.append(mini_ortool.solution)

    def cluster_and_selection(self):
        def calculate_wcss(points_coordinates):
            kmeans = KMeans(n_clusters=1, random_state=0).fit(points_coordinates)
            return kmeans.inertia_

        def close_nodes(nodes):
            max_number_of_points = {
                'id': -1,
                'number_of_points': 0
            }
            max_wcss = {
                'id': -1,
                'wcss': 0
            }
            for node in range(len(nodes)):
                if nodes[node]['number_of_points'] > max_number_of_points['number_of_points']:    
                    max_number_of_points = {
                        'id': node,
                        'number_of_points': nodes[node]['number_of_points']
                    }
                if nodes[node]['wcss'] > max_wcss['wcss']:
                    max_wcss = {
                        'id': node,
                        'wcss': nodes[node]['wcss']
                    }
            for node in range(len(nodes)):
                if (node != max_number_of_points['id']) & \
                    (node != max_wcss['id']):
                    nodes[node]['replicate'] = False
            return nodes

        def nodes_can_be_divided(cluster_tree):
            all_nodes = cluster_tree.get_all_nodes()
            for node in all_nodes:
                if cluster_tree.tree_nodes[node]['data']['replicate']:
                    return True
            return False

        def divide_nodes(cluster_tree, number_of_clusters):
            parent_nodes = cluster_tree.get_all_nodes()
            for parent_node_id in parent_nodes:
                parent_node = cluster_tree.tree_nodes[parent_node_id]['data']
                if parent_node['replicate']:
                    nodes = []
                    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(parent_node['points_coordinates'])
                    for node in range(number_of_clusters):
                        res = [x for x, z in enumerate(kmeans.labels_) if z == node]
                        if len(res) <= number_of_clusters:
                            nodes.append({
                                'replicate': False,
                                'points_coordinates': parent_node['points_coordinates'][res],
                                'points_ids': [parent_node['points_ids'][ids] for ids in res],
                                'number_of_points': len(res),
                                'wcss': calculate_wcss(parent_node['points_coordinates'][res])
                            })
                        else:
                            nodes.append({
                                'replicate': True,
                                'points_coordinates': parent_node['points_coordinates'][res],
                                'points_ids': [parent_node['points_ids'][ids] for ids in res],
                                'number_of_points': len(res),
                                'wcss': calculate_wcss(parent_node['points_coordinates'][res])
                            })
                    nodes = close_nodes(nodes)
                    cluster_tree.create_nodes(parent_node_id,nodes)
                    cluster_tree.tree_nodes[parent_node_id]['data']['replicate'] = False


        if self.results :
            number_of_clusters = 3
            root_node = []
            for result in self.results:
                if result['solve_status'] == 0:
                    root_node.append([result['objective_value']]+result['constraint'])
            self.cluster_tree = ClusterTree({
                'replicate': True,
                'points_ids': [point_id for point_id in range(len(root_node))],
                'points_coordinates': np.matrix(root_node),
                'number_of_points': len(root_node),
                'wcss': calculate_wcss(np.matrix(root_node))
            })
            print('[{}] Cluser and Selection started'.format(date.today()))
            while nodes_can_be_divided(self.cluster_tree):
                tic = time.time()
                divide_nodes(self.cluster_tree, number_of_clusters)
                toc = time.time()
                print('[{}] Duration: {}'.format(date.today(),toc-tic))
            

    def solve_cluster_tree(self):
        def solve_optimization_problem(scenarios):
            c_value = np.array(self.coefficient["objective"])    
            
            A_value = -1
            
            print('[{}] Node optimization started'.format(date.today()))
            tic = time.time()
            for scenario in scenarios:
                if isinstance(A_value, int):
                    A_value = np.matrix(self.coefficient['scenarios_constraint'][scenario])
                    b_value = np.array(self.coefficient['scenarios_rhs'][scenario])
                else:
                    A_value = np.concatenate((A_value,np.matrix(self.coefficient['scenarios_constraint'][scenario])), axis=0)
                    b_value = np.concatenate((b_value,np.array(self.coefficient['scenarios_rhs'][scenario])), axis=0)
            optimization_problem = OptimizationProblem(c_value,A_value,b_value)
            mini_ortool = MiniOrtoolsSolver(optimization_problem)
            toc = time.time()
            print('[{}] Duration: {}'.format(date.today(),toc-tic))
            return mini_ortool.solution

        all_nodes = self.cluster_tree.get_all_nodes()
        for node in all_nodes:
            selected_scenarios = self.cluster_tree.tree_nodes[node]['data']['points_ids']
            self.cluster_tree.tree_nodes[node]['problem'] = solve_optimization_problem(selected_scenarios)
            self.results.append(self.cluster_tree.tree_nodes[node]['problem'])
    
    def apply_quality_measure(self, number_of_scenarios):
        scenarios_constraint, scenarios_rhs = self.__generate_coefficients(number_of_scenarios)
        print('[{}] Quality measure application started'.format(date.today()))
        for result in self.results:
            tic = time.time()
            result_feasibility = []
            for scenario in range(len(scenarios_constraint)):
                constraints_evaluation = pd.DataFrame(np.dot(scenarios_constraint[scenario], result['variable']))-scenarios_rhs[scenario]
                if constraints_evaluation.max().item() > 0:
                    result_feasibility.append(0)
                else:
                    result_feasibility.append(1)
            mean_result_feasibility = mean(result_feasibility)
            toc = time.time()
            print('[{}] Quality measurement evaluated: {} - Elapsed time: {}'.format(date.today(), mean_result_feasibility, toc-tic))
            result['feasibility_probability'] = mean_result_feasibility