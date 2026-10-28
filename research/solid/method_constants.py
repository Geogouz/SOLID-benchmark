"""The constants of the SOLID method, the same at every site, as the confirmatory round confirmed them.

The design process wrote them into its records, the presence decision research/artifacts/solid/selections/
bern-presence-law-decision.json, the budget check selections/budget-check.json and the method record
selections/method.json of research/solid/confirm_round.py, which the public repository does not hold. A reader's run
(SOLID_READER=1) takes them from here: profiles.method_record returns the method below, and the compositions of
profiles.py and compose_unified_profiles.py give the tracked profiles of research/artifacts/solid/profiles from the
shipped fits with these values.

What the method fixes:
- the presence law, a Gaussian law of a link's success against its distance, with the quantile 0.9, the factor 0.75 and
  the width 0.15 of the presence decision, whose reach every scenario estimates by maximum likelihood from its own
  attempt outcomes (reach "ml", the replacement ml_reach that the round adopted, runtime/engine/src/presence.rs);
- the anchor only budgets at twice the base budgets of research/solid/deployment.py, the grid step over the square root
  of two and 2 x 512 = 1024 joint samples, with the engine's seed 20260918;
- the link model R0, fitted with the options of the fit files of suffix _r0, none: a Student t packet core of 3 degrees
  of freedom, a Gaussian persistent offset and no false early detection, with no link component removed;
- in the cooperative modes 64 starts, and the first stage of the frozen campaign fits, not derived from the link model;
- the engine's law of raw two way ranging timings, fitted on the TWR-CLOUD-Bern training table (CalibrationModel's
  default in runtime/engine/src/calibration.rs), which converts raw timings within 0 to 200 ns.
"""

# the presence law of the presence decision and the reach the method estimates it with
PRESENCE_LAW = dict(shape="gaussian", quantile=0.9, factor=0.75, width=0.15)
REACH = "ml"
# the factor of the anchor only budgets over the base budgets of deployment.py, a power of two: the grid step over the
# square root of two and twice the 512 joint samples, 1024 (deployment.scale_budgets)
BUDGET = 2
SEED = 20260918
# the degrees of freedom of the Student t packet core and the shape of the persistent offset of the link model
NU = 3.0
OFFSET_SHAPE = "gaussian"
# the starts of the cooperative solver
STARTS = 64
# the engine's law of raw two way ranging timings and the interval of raw timings it converts, which the production
# calibration of the data folder holds as well (fit_bern.FULL_CALIBRATION)
CALIBRATION = dict(range_slope_cm_per_ns=28.512859035713586, range_offset_cm=19.57939566609116, scale_slope=0.02635621175286423,
                   scale_intercept_cm=4.649664346296255, scale_min_cm=8.0, scale_max_cm=55.0)
TIMINGS_NS = dict(runtime_tof_floor_ns=0.0, runtime_tof_max_ns=200.0)

# The method as the method record of the confirmatory round states it: the suffix of the fit files of its link model,
# the link components it removed by a refit, none, its switches, the reach replacement alone, and its budget factor, the
# fit options of the fits of that suffix, none, and the fit files the evaluation composes from, under
# research/artifacts/solid/fits.
SUFFIX = "_r0"
FIT_OPTIONS = ()
SPEC = dict(suffix=SUFFIX, reduced=[], changes=["ml_reach"], budget=BUDGET)
FITS = dict(tiers="tiers_layout_fit_r0.json", bregar="bregar_layout_fits_r0.json", idlab="idlab_link_fit_r0.json", bern="bern_mixed_fit_nu3_r0.json")
