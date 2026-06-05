import numpy as np

from sirom.cluster_tree import ClusterTree


def test_build_empty_returns_none():
    assert ClusterTree.build([], number_of_clusters=3) is None


def test_build_small_instance_seeds_root():
    # As in the pipeline, the root carries one point per optimal scenario.
    points = [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]]
    tree = ClusterTree.build(points, number_of_clusters=2)
    assert tree is not None
    root = tree.tree_nodes[tree.root_node]["data"]
    assert root["number_of_points"] == 4
    assert root["wcss"] >= 0.0
    assert len(tree.get_all_nodes()) >= 1


def test_build_divides_spread_points():
    rng = np.random.RandomState(0)
    # Two well-separated clusters of points so KMeans actually splits.
    points = np.vstack(
        [rng.normal(0.0, 0.1, size=(8, 3)), rng.normal(10.0, 0.1, size=(8, 3))]
    ).tolist()
    tree = ClusterTree.build(points, number_of_clusters=2)
    assert tree is not None
    # The root plus at least one round of children.
    assert len(tree.get_all_nodes()) > 1
    # Every node's points_ids index back into the original point set.
    for node in tree.get_all_nodes():
        ids = tree.tree_nodes[node]["data"]["points_ids"]
        assert all(0 <= i < len(points) for i in ids)


def test_build_wcss_nonnegative_everywhere():
    rng = np.random.RandomState(1)
    points = rng.normal(0.0, 1.0, size=(12, 2)).tolist()
    tree = ClusterTree.build(points, number_of_clusters=3)
    assert tree is not None
    for node in tree.get_all_nodes():
        assert tree.tree_nodes[node]["data"]["wcss"] >= 0.0
