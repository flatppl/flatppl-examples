#!/usr/bin/env python3
"""Evaluate histfactory.hs3 with ROOT/RooFit (the reference HistFactory engine).

Loads the HS3 document via ROOT's native reader (``RooJSONFactoryWSTool``),
builds the RooFit likelihood, and reports the log-density at a few parameter
points. This is the oracle the sibling ``histfactory.flatppl`` is checked
against: FlatPPL's ``logdensityof`` reproduces ROOT's parameter-dependent
log-likelihood (the two differ only by a parameter-independent constant — ROOT's
extended binned NLL drops the per-bin ``log(n!)`` term that the Poisson pmf in
FlatPPL keeps).

Requires ROOT >= 6.30 with RooFit JSON support (e.g. ``conda install -c conda-forge root``).
Run:  python histfactory_root.py
"""

import os
import sys
from pathlib import Path

import ROOT

HS3 = str(Path(__file__).resolve().with_name("histfactory.hs3"))

# Parameters of the model (HS3 `parameter_points: default_values`).
PARAMS = ["mu", "syst1", "syst2", "syst3", "mcstat_0", "mcstat_1"]
DEFAULT = dict(mu=1.0, syst1=0.0, syst2=0.0, syst3=0.0, mcstat_0=1.0, mcstat_1=1.0)


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ws = ROOT.RooWorkspace("ws")
    if not ROOT.RooJSONFactoryWSTool(ws).importJSON(HS3):
        raise SystemExit(f"failed to import {HS3}")

    pdf = ws.pdf("model_channel1")  # RooProdPdf: main Poisson * constraints
    data = ws.data("observed_channel1")  # the binned observation
    # Constraint nominals are global observables (held fixed during evaluation).
    glob = ROOT.RooArgSet(*[ws.var(f"nom_{p}") for p in PARAMS if ws.var(f"nom_{p}")])
    nll = pdf.createNLL(data, ROOT.RooFit.GlobalObservables(glob))

    def logpdf(**point):
        for name, value in point.items():
            ws.var(name).setVal(value)
        return -nll.getVal()  # log-likelihood (up to a parameter-independent constant)

    base = logpdf(**DEFAULT)
    print("ROOT log-likelihood (up to const):")
    print(f"  @ default_values {DEFAULT}")
    print(f"      logL = {base:.10f}")
    pert = dict(DEFAULT, mu=1.5, syst1=0.5, mcstat_0=1.1)
    print(f"  @ {pert}")
    print(f"      logL = {logpdf(**pert):.10f}")
    print(f"  Δ(logL) default→perturbed = {logpdf(**pert) - base:.10f}")

    # Skip Python/ROOT teardown: cppyy's dealloc of the RooFit objects can
    # segfault on some builds, and the results above are already printed.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
