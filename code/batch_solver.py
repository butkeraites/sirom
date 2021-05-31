import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
import lhsmdu

from sirom.code.optimization_problem import OptimizationProblem
from sirom.code.mini_ortools_solver import MiniOrtoolsSolver

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
    
    def __generate_all_coefficients(self):
        def generate_constraint_coefficient():
            lb_constraint_equations, lb_constraint_coefficients = self.coefficient["lb_constraint"].shape
            scenarios_delta = lhsmdu.sample(lb_constraint_equations * lb_constraint_coefficients, self.number_of_scenarios) # Latin Hypercube Sampling of two variables
            scenarios_delta_reshaped = np.array(scenarios_delta.transpose())
            scenarios_delta_reshaped = scenarios_delta_reshaped.reshape(self.number_of_scenarios, lb_constraint_equations, lb_constraint_coefficients)
            self.coefficient['scenarios_constraint'] = []
            for scenario_delta_reshaped in range(self.number_of_scenarios):
                self.coefficient['scenarios_constraint'].append(self.coefficient["lb_constraint"] + (self.coefficient["ub_constraint"]-self.coefficient["lb_constraint"])*scenarios_delta_reshaped[scenario_delta_reshaped])
        
        def generate_rhs_coefficient():
            lb_rhs_equations, lb_rhs_coefficients = self.coefficient["lb_rhs"].shape
            scenarios_delta = lhsmdu.sample(lb_rhs_equations * lb_rhs_coefficients, self.number_of_scenarios) # Latin Hypercube Sampling of two variables
            scenarios_delta_reshaped = np.array(scenarios_delta.transpose())
            scenarios_delta_reshaped = scenarios_delta_reshaped.reshape(self.number_of_scenarios, lb_rhs_equations, lb_rhs_coefficients)
            self.coefficient['scenarios_rhs'] = []
            for scenario_delta_reshaped in range(self.number_of_scenarios):
                self.coefficient['scenarios_rhs'].append(self.coefficient["lb_rhs"] + (self.coefficient["ub_rhs"]-self.coefficient["lb_rhs"])*scenarios_delta_reshaped[scenario_delta_reshaped])

        generate_constraint_coefficient()
        generate_rhs_coefficient()
    
    def solve(self):
        c_value = np.array(self.coefficient["objective"])
        for scenario in range(self.number_of_scenarios):
            A_value = np.matrix(self.coefficient['scenarios_constraint'][scenario])
            b_value = np.array(self.coefficient['scenarios_rhs'][scenario])
            optimization_problem = OptimizationProblem(c_value,A_value,b_value)
            mini_ortool = MiniOrtoolsSolver(optimization_problem)
            self.results.append(mini_ortool.solution)

    def cluster_and_selection(self):
        if self.results :
            self.cluster_tree = []
            root_node = []
            for result in self.results:
                if result['solve_status'] == 0:
                    root_node.append([result['objective_value']]+result['constraint'])
            self.cluster_tree.append({
                'node_status': 'open',
                'replicate': True,
                'points': np.arange(len(root_node)),
                'coordinates': np.matrix(root_node)
            })
            kmeans = KMeans(n_clusters=3, random_state=0).fit(self.cluster_tree[0]['coordinates'])
            #TODO: CONTINUE TO CALCULATE WCSS FOR EACH CLUSTER AND CONTINUE SIROM
        else:
            self.status.append("[ERROR] Solve process must be succsessfully executed first.")
