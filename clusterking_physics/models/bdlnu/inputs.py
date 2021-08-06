""" Input parameters/physical constants. """

import flavio
from flavio.parameters import default_parameters as par


# inputs now taken from flavio except for running quark masses #  (since
# we just need them at one scale it is more efficient to write them directly)

# Note: For performance reasons, this shouldn't be a dictionary
GF = par.get_central("GF")
Vcb = par.get_central("Vcb")
new = 1
mB = par.get_central("m_B+")
mD = par.get_central("m_D0")
Btaul = flavio.sm_prediction("BR(tau->mununu)")
mtau = par.get_central("m_tau")
mb = 4.09206  # Running quark mass (MSbar) at the scale 4.8 GeV
mc = 0.962098  # Running quark mass (MSbar) at the scale 4.8 GeV
tauBp = par.get_central("tau_B+")
