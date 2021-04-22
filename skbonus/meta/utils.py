"""Utilities to help with mEBMs."""

import numpy as np
import pandas as pd


def fake_std(mebm):
    return [np.zeros_like(dom) for dom in mebm.domains_]


def get_fake_hist_edges(mebm, f, n=10):
    xmin = mebm.domains_[f][0]
    xmax = mebm.domains_[f][-1]
    return np.linspace(xmin, xmax, num=n+1)


def get_fake_hist_counts(mebm, f, n=10):
    xmin = mebm.domains_[f][0]
    xmax = mebm.domains_[f][-1]
    return 20 + np.linspace(0, n-1, num=n)
