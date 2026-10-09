"""Smoke tests: the package imports and pure-python pieces work."""
from fedknob.utils.seeding import set_seed


def test_seeding_runs():
    set_seed(123)
    set_seed(0)
