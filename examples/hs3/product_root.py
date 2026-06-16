#!/usr/bin/env python3
"""Evaluate product.hs3 with ROOT/RooFit.

HS3 paper A.2: a ``product_dist`` of two ``gaussian_dist`` (g1, g2) over the
SAME observable ``x``, normalized to a single density, scored against 10
unbinned toy observations. The normalized pointwise product of two Gaussians is itself a Gaussian (with
1/sigma*^2 = 1/sigma1^2 + 1/sigma2^2), so this is a proper (non-extended)
density. At the default point that closed form gives Sigma logL = -13.9458491571
over the 10 toy entries; ROOT reports -13.9458508897, agreeing to its numeric
normalization-integral precision (RooIntegrator1D, ~1e-6).

This is the oracle the sibling ``product.flatppl`` is checked against: FlatPPL's
``logdensityof(likelihoodof(iid(prod, 10), toy), theta)`` reproduces the log-
likelihood. FlatPPL evaluates the Gaussian-product normalizer in closed form, so
it returns the EXACT analytic value (-13.9458491571); ROOT integrates the
normalizer numerically (RooIntegrator1D), hence the ~1e-6 difference above.

Requires ROOT >= 6.30 with RooFit JSON support.
Run:  python product_root.py
"""
import os
import sys
from pathlib import Path

import ROOT

HS3 = str(Path(__file__).resolve().with_name("product.hs3"))

# Free parameters (HS3 `parameter_points: default_values`, minus the observable x).
DEFAULT = dict(mu1=0.0, sigma1=1.0, mu2=1.0, sigma2=2.0)


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ws = ROOT.RooWorkspace("ws")
    if not ROOT.RooJSONFactoryWSTool(ws).importJSON(HS3):
        raise SystemExit(f"failed to import {HS3}")

    pdf = ws.pdf("prod")        # RooProdPdf(g1, g2), normalized over x
    data = ws.data("toy")       # 10 unbinned entries
    nll = pdf.createNLL(data)

    def logpdf(**point):
        for name, value in point.items():
            ws.var(name).setVal(value)
        return -nll.getVal()    # Σ_i log p(x_i | θ)

    base = logpdf(**DEFAULT)
    print(f"  logL @ default {DEFAULT}")
    print(f"      logL = {base:.10f}")
    pert = dict(DEFAULT, mu1=0.5)
    print(f"  @ {pert}")
    print(f"      logL = {logpdf(**pert):.10f}")
    print(f"  Δ(logL) default→perturbed = {logpdf(**pert) - base:.10f}")

    sys.stdout.flush()
    os._exit(0)  # skip cppyy teardown (can segfault); results already printed


if __name__ == "__main__":
    main()
