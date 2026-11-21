"""Definitions both workflows may depend on, owned by this repository.

Nothing here may import from a research checkout: the Flask process must run
from this repository alone. The model is served by the forecast API in ``api/``,
a separate process the demo calls over HTTP.
"""
