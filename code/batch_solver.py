from numbers import Number
from typing import TypedDict

from numpy.core.fromnumeric import mean
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans  # type: ignore
from smt.sampling_methods import LHS  # type: ignore

from datetime import date
import time

from .cluster_tree import ClusterTree, Leaf, RootData
from .mini_ortools_solver import MiniOrtoolsSolver, Solution
from .optimization_problem import OptimizationProblem


class Coefficients(TypedDict):
    objective: pd.DataFrame
    lb_constraint: pd.DataFrame
    ub_constraint: pd.DataFrame
    lb_rhs: pd.DataFrame
    ub_rhs: pd.DataFrame
    scenarios_constraint: list[pd.DataFrame]
    scenarios_rhs: list[pd.DataFrame]


class ProblemsBucket:
    """Class that will generate and storage all instances used to solve a problem like:
    min c * x : S.t. [lb_A, ub_A] * x - [lb_b, ub_b] <= 0"""

    def __init__(
        self,
        c_value: list[Number],
        lb_A_value: list[list[Number]],
        ub_A_value: list[list[Number]],
        lb_b_value: list[Number],
        ub_b_value: list[Number],
        number_of_scenarios: int = 100,
    ):
        self.status: list[str] = []
        self.results: list[Solution] = []
        self.number_of_scenarios: int = -1
        c_validated = self.__coefficient_validation(c_value, "objective")
        lb_A_validated = self.__coefficient_validation(lb_A_value, "lb_constraint")
        ub_A_validated = self.__coefficient_validation(ub_A_value, "ub_constraint")
        lb_b_validated = self.__coefficient_validation(lb_b_value, "lb_rhs")
        ub_b_validated = self.__coefficient_validation(ub_b_value, "ub_rhs")
        self.__set_coefficient(
            c_validated, lb_A_validated, ub_A_validated, lb_b_validated, ub_b_validated
        )
        self.__number_of_scenarios_validation(number_of_scenarios)
        self.__dimension_validation()
        self.__problem_integrity_validation()

    def __set_coefficient(
        self,
        c_validated,
        lb_A_validated,
        ub_A_validated,
        lb_b_validated,
        ub_b_validated,
    ):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Coefficient cannot be defined")
            return
        self.coefficient = Coefficients(
            c_validated, lb_A_validated, ub_A_validated, lb_b_validated, ub_b_validated
        )

    def __coefficient_validation(
        self, coefficient: list[Number] | list[list[Number]], identification: str
    ) -> pd.DataFrame:
        if not coefficient:
            self.status.append(
                "[ERROR] Undefined {} coefficient".format(identification)
            )
        try:
            df_coefficient = pd.DataFrame(coefficient)
            self.status.append(
                "[OK] Successfuly acquired {} coefficient".format(identification)
            )
            return df_coefficient
        except:
            self.status.append(
                "[ERROR] Failed acquiring {} coefficient".format(identification)
            )
            return pd.DataFrame()

    def __dimension_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Dimension can not be evaluated")
            return
        objective_coefficients, objective_equations = self.coefficient.objective.shape
        (
            lb_constraint_equations,
            lb_constraint_coefficients,
        ) = self.coefficient.lb_constraint.shape
        (
            ub_constraint_equations,
            ub_constraint_coefficients,
        ) = self.coefficient.ub_constraint.shape
        lb_rhs_coefficients, lb_rhs_dimension = self.coefficient.lb_rhs.shape
        ub_rhs_coefficients, ub_rhs_dimension = self.coefficient.ub_rhs.shape
        if not (
            objective_coefficients == lb_constraint_coefficients
            and lb_constraint_coefficients == ub_constraint_coefficients
            and lb_constraint_equations == ub_constraint_equations
            and ub_constraint_equations == ub_rhs_coefficients
            and lb_rhs_dimension == ub_rhs_dimension
            and lb_rhs_coefficients == ub_rhs_coefficients
            and objective_equations == 1
            and ub_rhs_dimension == 1
        ):
            self.status.append("[ERROR] Dimension inconsistency detected")
            return
        self.status.append("[OK] Optimization batch creation succeeded")

    def __number_of_scenarios_validation(self, number_of_scenarios: int):
        if not number_of_scenarios:
            self.status.append("[ERROR] Undefined number of scenarios")
            return
        if not (isinstance(number_of_scenarios, int) and (number_of_scenarios >= 0)):
            self.status.append("[ERROR] Failed acquiring number of scenarios")
            return
        self.number_of_scenarios = number_of_scenarios
        self.status.append("[OK] Successfuly acquired number of scenarios")

    def __problem_integrity_validation(self):
        if any("[ERROR]" in status for status in self.status):
            self.status.append("[ERROR] Optimization batch creation failed")
            return
        self.__generate_all_coefficients()
        self.status.append("[OK] Optimization batch creation succeeded")

    def __generate_coefficients(
        self, number_of_scenarios: int
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        def generate_constraint_coefficient() -> list[pd.DataFrame]:
            (
                lb_constraint_equations,
                lb_constraint_coefficients,
            ) = self.coefficient.lb_constraint.shape
            xlimits = np.array(
                [[0.0, 1.0]] * (lb_constraint_equations * lb_constraint_coefficients)
            )
            sampling = LHS(xlimits=xlimits)
            print("[{}] Coefficient generation".format(date.today()))
            tic = time.time()
            scenarios_delta = sampling(number_of_scenarios)
            toc = time.time()
            print("[{}] Duration: {}".format(date.today(), toc - tic))
            scenarios_delta_reshaped = np.array(scenarios_delta.transpose())
            scenarios_delta_reshaped = scenarios_delta_reshaped.reshape(
                number_of_scenarios, lb_constraint_equations, lb_constraint_coefficients
            )
            scenarios_constraint: list[pd.DataFrame] = []
            for scenario_delta_reshaped in range(number_of_scenarios):
                scenarios_constraint.append(
                    pd.DataFrame(
                        self.coefficient.lb_constraint
                        + (
                            self.coefficient.ub_constraint
                            - self.coefficient.lb_constraint
                        )
                        * scenarios_delta_reshaped[scenario_delta_reshaped]
                    )
                )
            return scenarios_constraint

        def generate_rhs_coefficient() -> list[pd.DataFrame]:
            lb_rhs_equations, lb_rhs_coefficients = self.coefficient.lb_rhs.shape
            xlimits = np.array([[0.0, 1.0]] * (lb_rhs_equations * lb_rhs_coefficients))
            sampling = LHS(xlimits=xlimits)
            print("[{}] RHS generation".format(date.today()))
            tic = time.time()
            scenarios_delta = sampling(number_of_scenarios)
            toc = time.time()
            print("[{}] Duration: {}".format(date.today(), toc - tic))
            scenarios_delta_reshaped = np.array(scenarios_delta.transpose())
            scenarios_delta_reshaped = scenarios_delta_reshaped.reshape(
                number_of_scenarios, lb_rhs_equations, lb_rhs_coefficients
            )
            scenarios_rhs = []
            for scenario_delta_reshaped in range(number_of_scenarios):
                scenarios_rhs.append(
                    self.coefficient.lb_rhs
                    + (self.coefficient.ub_rhs - self.coefficient.lb_rhs)
                    * scenarios_delta_reshaped[scenario_delta_reshaped]
                )
            return scenarios_rhs

        coefficients: tuple[list[pd.DataFrame], list[pd.DataFrame]] = (
            generate_constraint_coefficient(),
            generate_rhs_coefficient(),
        )
        return coefficients

    def __generate_all_coefficients(self):
        (
            self.coefficient.scenarios_constraint,
            self.coefficient.scenarios_rhs,
        ) = self.__generate_coefficients(self.number_of_scenarios)

    def solve(self):
        c_value = np.array(self.coefficient.objective)
        print("[{}] Solve process started".format(date.today()))
        for scenario in range(self.number_of_scenarios):
            tic = time.time()
            A_value = np.matrix(self.coefficient.scenarios_constraint[scenario])
            b_value = np.array(self.coefficient.scenarios_rhs[scenario])
            optimization_problem = OptimizationProblem(c_value, A_value, b_value)
            mini_ortool = MiniOrtoolsSolver(optimization_problem)
            toc = time.time()
            print(
                "[{}] Scenario {} | Duration: {}".format(
                    date.today(), scenario, toc - tic
                )
            )
            self.results.append(mini_ortool.solution)

    def cluster_and_selection(self):
        def calculate_wcss(points_coordinates) -> float:
            kmeans = KMeans(n_clusters=1, random_state=0).fit(points_coordinates)
            return kmeans.inertia_

        def close_nodes(nodes):
            max_number_of_points = {"id": -1, "number_of_points": 0}
            max_wcss = {"id": -1, "wcss": 0}
            for node in range(len(nodes)):
                if (
                    nodes[node]["number_of_points"]
                    > max_number_of_points["number_of_points"]
                ):
                    max_number_of_points = {
                        "id": node,
                        "number_of_points": nodes[node]["number_of_points"],
                    }
                if nodes[node]["wcss"] > max_wcss["wcss"]:
                    max_wcss = {"id": node, "wcss": nodes[node]["wcss"]}
            for node in range(len(nodes)):
                if (node != max_number_of_points["id"]) & (node != max_wcss["id"]):
                    nodes[node]["replicate"] = False
            return nodes

        def nodes_can_be_divided(cluster_tree):
            all_nodes = cluster_tree.get_all_nodes()
            for node in all_nodes:
                if cluster_tree.tree_nodes[node]["data"]["replicate"]:
                    return True
            return False

        def divide_nodes(cluster_tree: ClusterTree, number_of_clusters):
            parent_nodes = cluster_tree.get_all_nodes()
            for parent_node_id in parent_nodes:
                parent_node: RootData = cluster_tree.tree_nodes[parent_node_id]["data"]
                if not parent_node["replicate"]:
                    continue
                nodes: list[RootData] = []
                kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(
                    parent_node["points_coordinates"]
                )
                for node in range(number_of_clusters):
                    res = [x for x, z in enumerate(kmeans.labels_) if z == node]
                    points_coordinates = parent_node["points_coordinates"][res]
                    new_node = RootData(
                        replicate=len(res) > number_of_clusters,
                        points_coordinates=points_coordinates,
                        points_ids=[parent_node["points_ids"][ids] for ids in res],
                        number_of_points=len(res),
                        wcss=calculate_wcss(points_coordinates),
                    )
                    nodes.append(new_node)
                nodes = close_nodes(nodes)
                cluster_tree.create_nodes(parent_node_id, nodes)
                cluster_tree.tree_nodes[parent_node_id]["data"]["replicate"] = False

        if not self.results:
            return

        number_of_clusters = 3
        root_node = []
        for result in self.results:
            if result["solve_status"] == 0:
                root_node.append([result["objective_value"]] + result["constraint"])
        self.cluster_tree = ClusterTree(
            {
                "replicate": True,
                "points_ids": [point_id for point_id in range(len(root_node))],
                "points_coordinates": np.asarray(root_node),
                "number_of_points": len(root_node),
                "wcss": calculate_wcss(np.asarray(root_node)),
            }
        )
        print("[{}] Cluser and Selection started".format(date.today()))
        while nodes_can_be_divided(self.cluster_tree):
            tic = time.time()
            divide_nodes(self.cluster_tree, number_of_clusters)
            toc = time.time()
            print("[{}] Duration: {}".format(date.today(), toc - tic))

    def solve_cluster_tree(self):
        def solve_optimization_problem(scenarios: list[int]):
            c_value = np.array(self.coefficient.objective)

            def get_A_and_b_values(scenarios):
                scenario = scenarios[0]
                A_value = np.matrix(self.coefficient.scenarios_constraint[scenario])
                b_value = np.array(self.coefficient.scenarios_rhs[scenario])

                if len(scenarios) > 1:
                    for scenario in scenarios[1:]:
                        A_value = np.concatenate(
                            (
                                A_value,
                                np.matrix(
                                    self.coefficient.scenarios_constraint[scenario]
                                ),
                            ),
                            axis=0,
                        )
                        b_value = np.concatenate(
                            (
                                b_value,
                                np.array(self.coefficient.scenarios_rhs[scenario]),
                            ),
                            axis=0,
                        )

                return A_value, b_value

            print("[{}] Node optimization started".format(date.today()))
            tic = time.time()
            if scenarios:
                A_value, b_value = get_A_and_b_values(scenarios)

            optimization_problem = OptimizationProblem(c_value, A_value, b_value)
            mini_ortool = MiniOrtoolsSolver(optimization_problem)
            toc = time.time()
            print("[{}] Duration: {}".format(date.today(), toc - tic))
            return mini_ortool.solution

        all_nodes = self.cluster_tree.get_all_nodes()
        for node in all_nodes:
            leaf = self.cluster_tree.tree_nodes[node]
            data = leaf["data"]
            selected_scenarios = data["points_ids"]
            self.cluster_tree.tree_nodes[node].problem = solve_optimization_problem(
                selected_scenarios
            )
            self.results.append(self.cluster_tree.tree_nodes[node].problem)

    def apply_quality_measure(self, number_of_scenarios: int):
        scenarios_constraint, scenarios_rhs = self.__generate_coefficients(
            number_of_scenarios
        )
        print("[{}] Quality measure application started".format(date.today()))
        for result in self.results:
            tic = time.time()
            result_feasibility = []
            for scenario in range(len(scenarios_constraint)):
                constraints_evaluation = (
                    pd.DataFrame(
                        np.dot(
                            scenarios_constraint[scenario], np.array(result["variable"])
                        )
                    )
                    - scenarios_rhs[scenario]
                )
                if constraints_evaluation.max().item() > 0:
                    result_feasibility.append(0)
                else:
                    result_feasibility.append(1)
            mean_result_feasibility = mean(result_feasibility)
            toc = time.time()
            print(
                "[{}] Quality measurement evaluated: {} - Elapsed time: {}".format(
                    date.today(), mean_result_feasibility, toc - tic
                )
            )
            result["feasibility_probability"] = float(mean_result_feasibility)
