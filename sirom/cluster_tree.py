from __future__ import annotations

from typing import Dict, List, Optional, TypedDict
import uuid

import numpy as np
from sklearn.cluster import KMeans  # type: ignore

from sirom.mini_ortools_solver import UnscoredSolution


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
    problem: UnscoredSolution


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

    @classmethod
    def build(
        cls, root_points, number_of_clusters: int
    ) -> Optional["ClusterTree"]:
        """Build a cluster tree from the root point set and subdivide it.

        The tree owns its own shaping: it seeds a replicable root from
        ``root_points`` (one point per optimal scenario solution), then
        repeatedly splits the splittable nodes (KMeans into
        ``number_of_clusters``) until none remain. Returns ``None`` when there
        are no points to cluster.
        """
        points = np.asarray(root_points)
        if points.size == 0:
            return None
        tree = cls(
            {
                "replicate": True,
                "points_ids": list(range(len(points))),
                "points_coordinates": points,
                "number_of_points": len(points),
                "wcss": cls._calculate_wcss(points),
            }
        )
        while tree._nodes_can_be_divided():
            tree._divide_nodes(number_of_clusters)
        return tree

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

    # --- subdivision algorithm (private; driven by build) -------------------

    @staticmethod
    def _calculate_wcss(points_coordinates) -> float:
        kmeans = KMeans(n_clusters=1, random_state=0).fit(points_coordinates)
        return kmeans.inertia_

    @staticmethod
    def _close_nodes(nodes: List[RootData]) -> List[RootData]:
        # Keep only the node with the most points and the one with the highest
        # WCSS splittable; mark every other sibling done. This is the selection
        # heuristic that decides where the tree keeps growing.
        max_number_of_points = {"id": -1, "number_of_points": 0}
        max_wcss = {"id": -1, "wcss": 0.0}
        for node in range(len(nodes)):
            if nodes[node]["number_of_points"] > max_number_of_points["number_of_points"]:
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

    def _nodes_can_be_divided(self) -> bool:
        for node in self.get_all_nodes():
            if self.tree_nodes[node]["data"]["replicate"]:
                return True
        return False

    def _divide_nodes(self, number_of_clusters: int) -> None:
        for parent_node_id in self.get_all_nodes():
            parent_node: RootData = self.tree_nodes[parent_node_id]["data"]
            if not parent_node["replicate"]:
                continue
            parent_count = parent_node["number_of_points"]
            nodes: List[RootData] = []
            kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(
                parent_node["points_coordinates"]
            )
            for node in range(number_of_clusters):
                res = [x for x, z in enumerate(kmeans.labels_) if z == node]
                if not res:
                    # KMeans can leave a cluster empty when points are
                    # duplicated; skip it so wcss isn't computed on an empty
                    # set (which would crash the run).
                    continue
                points_coordinates = parent_node["points_coordinates"][res]
                new_node = RootData(
                    # Only keep splitting when the cluster has more than k points
                    # AND the split was productive (it shrank the set). When
                    # duplicated points all collapse into one cluster the child
                    # would equal its parent; stopping there prevents an infinite
                    # subdivision loop.
                    replicate=(len(res) > number_of_clusters)
                    and (len(res) < parent_count),
                    points_coordinates=points_coordinates,
                    points_ids=[parent_node["points_ids"][ids] for ids in res],
                    number_of_points=len(res),
                    wcss=self._calculate_wcss(points_coordinates),
                )
                nodes.append(new_node)
            nodes = self._close_nodes(nodes)
            self.create_nodes(parent_node_id, nodes)
            self.tree_nodes[parent_node_id]["data"]["replicate"] = False
