"""Definitions both workflows may depend on, owned by this repository.

Nothing here may import from a research checkout: the Flask process must run
from this repository alone. The WholeBody release worker is the only component
that loads research code, and it does so in its own subprocess.
"""
