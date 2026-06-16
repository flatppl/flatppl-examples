#!/usr/bin/env python3
"""Evaluate gaussian.hs3 with ROOT/RooFit.

Loads the HS3 document (the HS3-paper single-Gaussian measurement: a
``gaussian_dist`` for observable ``x`` with mean ``mu`` and fixed ``sigma``,
one observation x = 1.27) via ROOT's native reader and reports the
log-likelihood as a function of ``mu``. This is the oracle the sibling
``gaussian.flatppl`` is checked against; here ROOT and FlatPPL agree on the
*absolute* log-density (a plain normalized Gaussian — no extended/constant
offset), e.g. logL(mu=0) = -1.7253885332.

Requires ROOT >= 6.30 with RooFit JSON support.
Run:  python gaussian_root.py
"""
import os
import sys
from pathlib import Path

import ROOT

HS3 = str(Path(__file__).resolve().with_name("gaussian.hs3"))


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ws = ROOT.RooWorkspace("ws")
    if not ROOT.RooJSONFactoryWSTool(ws).importJSON(HS3):
        raise SystemExit(f"failed to import {HS3}")

    pdf = ws.pdf("gauss_x")              # RooGaussian(x | mu, sigma)
    data = ws.data("obs_gaussian_channel")
    nll = pdf.createNLL(data)            # single observation x = 1.27

    def logpdf(mu):
        ws.var("mu").setVal(mu)
        return -nll.getVal()             # log p(x = 1.27 | mu, sigma = 1)

    for mu in (0.0, 0.5, 1.27):
        print(f"  logL @ mu={mu:<4} = {logpdf(mu):.10f}")

    sys.stdout.flush()
    os._exit(0)  # skip cppyy teardown (can segfault); results already printed


if __name__ == "__main__":
    main()
