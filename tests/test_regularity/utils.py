from dataclasses import dataclass
from typing import List

import numpy as np

from rgr.constants.types import get_dtype_idx


@dataclass
class MockSzemeredi:
    N: int
    k: int
    epsilon: float
    classes: np.ndarray
    classes_cardinality: int


def _init_szemeredi(n_nodes: int, n_partitions: int) -> MockSzemeredi:
    reg = MockSzemeredi(n_nodes,
                        n_partitions,
                        0.285,
                        np.empty((n_nodes,), dtype=get_dtype_idx()),
                        0)

    return reg


def _create_mock_partition(reg: MockSzemeredi, partitions: List[np.ndarray]) -> None:
    for c, indices in enumerate(partitions):
        reg.classes[indices] = c

    reg.classes_cardinality = partitions[1].size


def create_reg_szemeredi(n_nodes: int,
                         n_partitions: int,
                         partitions: List) -> MockSzemeredi:
    reg = _init_szemeredi(n_nodes, n_partitions)

    _create_mock_partition(reg, partitions)

    return reg
