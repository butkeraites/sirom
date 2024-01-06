from typing import Dict, List, TypedDict
import uuid

import numpy as np

from code.mini_ortools_solver import Solution


class RootData(TypedDict):
    replicate: bool
    points_ids: List[int]
    points_coordinates: np.ndarray
    number_of_points: int
    wcss: float


class Leaf(TypedDict):
    data: RootData
    child_nodes: List[uuid.UUID]


class SolvedLeaf(Leaf):
    problem: Solution


class ClusterTree:
    "Class that will be used to represent the cluster tree"

    class Node:
        "Class that will be used to compose every node on cluster tree"

        def __init__(self, data: RootData):
            self.id = uuid.uuid4()
            self.data = data

    def __init__(self, root_data: RootData):
        self.tree_nodes: Dict[uuid.UUID, Leaf] = {}
        root = self.Node(root_data)
        self.root_node = root.id
        self.create_leaf(root)

    def create_leaf(self, leaf: Node):
        self.tree_nodes[leaf.id] = {"data": leaf.data, "child_nodes": []}

    def create_nodes(self, parent_id: uuid.UUID, child_dataset: List[RootData]):
        for child_data in child_dataset:
            child = self.Node(child_data)
            self.tree_nodes[parent_id]["child_nodes"].append(child.id)
            self.create_leaf(child)

    def get_all_nodes(self) -> List[uuid.UUID]:
        return list(self.tree_nodes.keys())

    def get_child_nodes(self, parent_id: uuid.UUID) -> List[uuid.UUID]:
        return self.tree_nodes[parent_id]["child_nodes"]

    def from_root_to_leafs(self):
        def go_to_child(parent: uuid.UUID):
            print("-----------------------")
            print("Node: " + str(parent))
            print("Data: ")
            print(self.tree_nodes[parent]["data"])
            print("-----------------------")
            if self.tree_nodes[parent]["child_nodes"]:
                for child in self.tree_nodes[parent]["child_nodes"]:
                    go_to_child(child)
            else:
                print("I am a leaf")
                print("+++++++++++")

        go_to_child(self.root_node)
