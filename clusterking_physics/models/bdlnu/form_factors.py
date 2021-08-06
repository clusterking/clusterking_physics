# 3rd
from flavio.physics.bdecays.formfactors.b_p import bcl
from flavio.parameters import default_parameters
from functools import lru_cache as cache

cache_size = 1000

default_central_values = default_parameters.get_central_all()


@cache(maxsize=cache_size)
def fplus(q2):
    return bcl.ff_isgurwise("B->D", q2, default_central_values, 4.8, n=3)["f+"]


@cache(maxsize=cache_size)
def fzero(q2):
    return bcl.ff_isgurwise("B->D", q2, default_central_values, 4.8, n=3)["f0"]


@cache(maxsize=cache_size)
def fT(q2):
    return bcl.ff_isgurwise("B->D", q2, default_central_values, 4.8, n=3)["fT"]
