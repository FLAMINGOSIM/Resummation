Resummation
===========

A Python implementation of the resummation model, which can be used to compute matter power suppression signals from halo baryon fractions. This version of the resummation model is described in van Daalen et al. (2026, VD26): https://ui.adsabs.harvard.edu/abs/2026MNRAS.545f2086V/abstract

Installation
------------
The easiest way to install the package is to use pip:
```
pip3 install resummation
```

Notes on usage
--------------
Creating a new instance of Resummation initializes the model to a set of DMO power spectra. See docstring of __init__ for initialization parameters.

After intializing the model, baryon fractions (or retained mass fractions directly) should be set before calling run_resummation(). Optionally, stellar fractions and a collective retained mass fraction for the combined 5xR500c regions can be set as well (see sections 2.2.3 and 3.6 of VD26, respectively). The model does not need to be reinitialized in order to provide predictions for new sets of baryon fractions: simply set new baryon fractions on the existing object and call run_resummation() again.

To set mean halo baryon fractions, use either set_fret_from_fb() (when providing regular baryon fractions), or set_fret_from_fbc() (when providing corrected baryon fractions). Alternatively, use set_fret() to set retained mass fractions directly. See the documentation of these functions for more information.

Author
------
+ Marcel van Daalen (@daalen on GitHub, @MPvanDaalen on PyPI)
