# std
import numpy as np
from functools import lru_cache as cache

# 3rd party
from wilson import Wilson
from numba import jit

# ours
from clusterking_physics.models.bdlnu.form_factors import fplus, fzero, fT
from clusterking_physics.models.bdlnu.inputs import *

# todo: make pycharm ignore name convention pylinting in this file

#  kinematic variables
#  q2,  El ,   thetal

## Limits:

#  thetal [0,pi]

#  q2 [   inputs['mtau']^2,  (inputs['mB'] -inputs['mD'])^2 ]

#  El  [     inputs['mtau']^2/(  2 * sqrt(q2) ),     sqrt(q2)/2   ] for  w1   and [  0,  inputs['mtau']^2/(2 sqrt(q2))  ]  for w2


@jit(nopython=True)
def Klambda(a, b, c):
    # even though the formula is positive definite, use max to enforce this
    # even when rounding errors occurr (else problems with sqrt later)
    return max(0, a**2 + b**2 + c**2 - 2 * (a * b + a * c + b * c))


# Improves speed and allows us to compile more


@jit(nopython=True)
def kvec(q2):
    return 1 / (2 * mB) * np.sqrt(Klambda(mB**2, mD**2, q2))


##  23


@cache(maxsize=1000)
def cvl_bctaunutau(w):
    return w.match_run(mb, "WET", "flavio")["CVL_bctaunutau"]


@cache(maxsize=1000)
def cvr_bctaunutau(w):
    return w.match_run(mb, "WET", "flavio")["CVR_bctaunutau"]


@cache(maxsize=1000)
def csl_bctaunutau(w):
    return w.match_run(mb, "WET", "flavio")["CSL_bctaunutau"]


@cache(maxsize=1000)
def csr_bctaunutau(w):
    return w.match_run(mb, "WET", "flavio")["CSR_bctaunutau"]


@cache(maxsize=1000)
def ct_bctaunutau(w):
    return w.match_run(mb, "WET", "flavio")["CT_bctaunutau"]


def H0(w: Wilson, q2, El):
    return (
        (1 + cvl_bctaunutau(w) + cvr_bctaunutau(w))
        * 2
        * mB
        * kvec(q2)
        / np.sqrt(q2)
        * fplus(q2)
    )


def Ht(w: Wilson, q2, El):
    return (
        (1 + cvl_bctaunutau(w) + cvr_bctaunutau(w))
        * (mB**2 - mD**2)
        / (np.sqrt(q2))
        * fzero(q2)
    )


def HS(w: Wilson, q2, El):
    return (
        (csr_bctaunutau(w) + csl_bctaunutau(w))
        * (mB**2 - mD**2)
        / (mb - mc)
        * fzero(q2)
    )


##


def Hpm(w: Wilson, q2, El):
    return ct_bctaunutau(w) * (2j * mB * kvec(q2)) / (mB + mD) * fT(q2)


##


def H0t(w: Wilson, q2, El):
    return ct_bctaunutau(w) * (2j * mB * kvec(q2)) / (mB + mD) * fT(q2)


#    32


def Icalzero(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)

    return _Icalzero(q2, H0val, Hpmval, H0tval)


@jit(nopython=True)
def _Icalzero(q2, H0val, Hpmval, H0tval):
    return (
        mtau * np.sqrt(q2) * np.absolute(H0val) ** 2
        + 4 * mtau * np.sqrt(q2) * np.absolute(Hpmval + H0tval) ** 2
        + 2j * mtau**2 * H0val * np.conjugate(Hpmval + H0tval)
        - 2j * q2 * np.conjugate(H0val) * (Hpmval + H0tval)
    )


def IcalzeroI(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    HSval = HS(w, q2, El)
    Htval = Ht(w, q2, El)

    return _IcalzeroI(q2, H0val, Hpmval, H0tval, HSval, Htval)


@jit(nopython=True)
def _IcalzeroI(q2, H0val, Hpmval, H0tval, HSval, Htval):

    return -np.sqrt(q2) * np.conjugate(H0val) * (
        mtau * Htval + np.sqrt(q2) * HSval
    ) - 2j * mtau * np.conjugate(Hpmval + H0tval) * (
        mtau * Htval + np.sqrt(q2) * HSval
    )


Icalp = 0
Icalm = 0


## 30


def Gamma00p(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    # HSval = HS(w, q2, El)

    return np.absolute(2j * np.sqrt(q2) * (Hpmval + H0tval) - mtau * H0val) ** 2


def Gammat0p(w: Wilson, q2, El):
    # H0val = H0(w, q2, El)
    # Hpmval = Hpm(w, q2, El)
    # H0tval = H0t(w, q2, El)
    HSval = HS(w, q2, El)
    Htval = Ht(w, q2, El)

    return np.absolute(mtau * Htval + np.sqrt(q2) * HSval) ** 2


def GammaI0p(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    HSval = HS(w, q2, El)
    Htval = Ht(w, q2, El)

    return 2 * np.real(
        (2j * np.sqrt(q2) * (Hpmval + H0tval) - mtau * H0val)
        * np.conjugate(mtau * Htval + np.sqrt(q2) * HSval)
    )


Gammapp = 0

Gammamp = 0


def Gamma0m(w: Wilson, q2, El):
    H0val = H0(w, q2, El)
    Hpmval = Hpm(w, q2, El)
    H0tval = H0t(w, q2, El)
    # HSval = HS(w, q2, El)
    # Htval = Ht(w, q2, El)

    return np.absolute(np.sqrt(q2) * H0val - 2j * mtau * (Hpmval + H0tval)) ** 2


Gammapm = 0

Gammamm = 0


#    A2  and A3, A4


def Ical0(w: Wilson, q2, El):
    return 2 * np.real(2 * IcalzeroI(w, q2, El))


def Ical1(w: Wilson, q2, El):
    return 2 * np.real(2 * Icalzero(w, q2, El))


def Gammap0(w: Wilson, q2, El):
    return 2 * Gammat0p(w, q2, El)


def Gammam0(w: Wilson, q2, El):
    return 2 * Gamma0m(w, q2, El)


def Gammam2(w: Wilson, q2, El):
    return -2 * Gamma0m(w, q2, El)


def Gammap2(w: Wilson, q2, El):
    return 2 * Gamma00p(w, q2, El)


def Gammam1(w: Wilson, q2, El):
    return 0.0


def Gammap1(w: Wilson, q2, El):
    return 2 * GammaI0p(w, q2, El)
