import numpy as np
import pytest
from sgt.constants.types import get_dtype_adj

from sgt.graph.complete import Complete
from sgt.graph.cycle import Cycle
from sgt.graph.graph import Graph

DTYPE_ADJ = get_dtype_adj()

