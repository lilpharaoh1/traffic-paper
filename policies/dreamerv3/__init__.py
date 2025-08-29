"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from policies.dreamerv3.dreamerv3 import DreamerV3Traffic, DreamerV3TrafficConfig

__all__ = [
    "DreamerV3Traffic",
    "DreamerV3TrafficConfig",
]
