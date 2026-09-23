from typing import List, Dict, Optional

import torch

from tabicl.prior.graph_lib._base import RandomTransformer, Context, FeatureSpec
from tabicl.prior.graph_lib._node_function import RandomNodeFunction


class RandomGraphFunction(RandomTransformer):
    """
    Samples a dataset by propagating data through a graph.
    """
    def __init__(self, context: Context, dag: List[List[int]], node_feature_specs: List[Dict[str, FeatureSpec]],
                 force_physics_nodes: Optional[List[bool]] = None):
        """
        :param context: Context.
        :param dag: Graph (list of parent node idxs for each node).
        :param node_feature_specs: feature specs with feature names for each node.
        :param force_physics_nodes: Optional per-node boolean list. If a node's entry is True, that node's
            random function is biased towards using a physics-informed formula (see RandomNodeFunction /
            RandomFunction's force_physics). Has no effect on nodes that produce a y feature, which are
            always kept physics-free regardless of this flag (see RandomNodeFunction). Defaults to all-False,
            i.e. unchanged behavior from before this parameter existed.
        """
        super().__init__(context=context)
        self.dag = dag
        self.node_feature_specs = node_feature_specs
        self.force_physics_nodes = force_physics_nodes if force_physics_nodes is not None else [False] * len(dag)

    def _fit(self, n_samples: int):
        self.nodes_ = [
            RandomNodeFunction(self.context, feature_specs=feature_specs, force_physics=self.force_physics_nodes[node_idx])
            for node_idx, feature_specs in enumerate(self.node_feature_specs)
        ]
        # for efficiency, prune nodes whose values don't need to be computed
        self.should_compute_ = [False for _ in range(len(self.nodes_))]
        for node_idx in reversed(range(len(self.nodes_))):
            if len(self.node_feature_specs[node_idx]) >= 1:
                self.should_compute_[node_idx] = True
            if self.should_compute_[node_idx]:  # could have been set by successors or by itself
                for parent in self.dag[node_idx]:
                    self.should_compute_[parent] = True

    def _transform(self, n_samples: int) -> Dict[str, torch.Tensor]:
        n_nodes = len(self.node_feature_specs)
        node_values = [None for _ in range(n_nodes)]
        features = dict()
        for node_idx in range(len(self.node_feature_specs)):
            if self.should_compute_[node_idx]:
                node_values[node_idx], out_features = self.nodes_[node_idx](
                    [node_values[parent] for parent in self.dag[node_idx]], n_samples
                )
                features = features | out_features
        return features
