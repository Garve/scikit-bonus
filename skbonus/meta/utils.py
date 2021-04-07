"""Utilities to help with mEBMs."""

import numpy as np
import pandas as pd

from interpret.glassbox.ebm.ebm import EBMExplanation


def fake_std(mebm):
    return [np.zeros_like(dom) for dom in mebm.domains_]


def feature_groups(mebm):
    return [[i] for i in range(mebm.n_features_in_)]


def get_fake_hist_edges(mebm, f, n=10):
    xmin = mebm.domains_[f][0]
    xmax = mebm.domains_[f][-1]
    return np.linspace(xmin, xmax, num=n+1)


def get_fake_hist_counts(mebm, f, n=10):
    xmin = mebm.domains_[f][0]
    xmax = mebm.domains_[f][-1]
    return 20 + np.linspace(0, n-1, num=n)


def make_selector(X):
    return pd.DataFrame({
        'Name': X.columns,
        'Type': 'continuous',
        '# Unique': X.nunique(),
        '% Non-zero': (X != 0).mean(),
    })


def explain_global(mebm, feature_names, feature_importances, selector, name=None):
    """ Provides global explanation for model.
    Args:
        name: User-defined explanation name.
    Returns:
        An explanation object,
        visualizing feature-value pairs as horizontal bar chart.
    """

    lower_bound = np.inf
    upper_bound = -np.inf
    for feature_group_index, _ in enumerate(feature_groups(mebm)):
        errors = fake_std(mebm)[feature_group_index]
        scores = mebm.outputs_[feature_group_index]

        lower_bound = min(lower_bound, np.min(scores - errors))
        upper_bound = max(upper_bound, np.max(scores + errors))

    bounds = (lower_bound, upper_bound)

    # Add per feature graph
    data_dicts = []
    feature_list = []
    density_list = []
    for feature_group_index, feature_indexes in enumerate(
        feature_groups(mebm)
    ):
        model_graph = mebm.outputs_[feature_group_index]

        # NOTE: This uses stddev. for bounds, consider issue warnings.
        errors = fake_std(mebm)[feature_group_index]

        if len(feature_indexes) == 1:
            # hack. remove the 0th index which is for missing values
            model_graph = model_graph[1:]
            errors = errors[1:]

            bin_labels = mebm.domains_[feature_indexes[0]]

            scores = list(model_graph)
            upper_bounds = list(model_graph + errors)
            lower_bounds = list(model_graph - errors)
            density_dict = {
                "names": get_fake_hist_edges(mebm, feature_indexes[0]),
                "scores": get_fake_hist_counts(mebm, feature_indexes[0]),
            }

            feature_dict = {
                "type": "univariate",
                "names": bin_labels,
                "scores": scores,
                "scores_range": bounds,
                "upper_bounds": upper_bounds,
                "lower_bounds": lower_bounds,
            }
            feature_list.append(feature_dict)
            density_list.append(density_dict)

            data_dict = {
                "type": "univariate",
                "names": bin_labels,
                "scores": model_graph,
                "scores_range": bounds,
                "upper_bounds": model_graph + errors,
                "lower_bounds": model_graph - errors,
                "density": {
                    "names": get_fake_hist_edges(mebm, feature_indexes[0]),
                    "scores": get_fake_hist_counts(mebm, feature_indexes[0]),
                },
            }

            data_dicts.append(data_dict)
        else:
            raise Exception("Interactions greater than 2 not supported.")

    overall_dict = {
        "type": "univariate",
        "names": feature_names,
        "scores": feature_importances,
    }
    internal_obj = {
        "overall": overall_dict,
        "specific": data_dicts,
        "mli": [
            {
                "explanation_type": "ebm_global",
                "value": {"feature_list": feature_list},
            },
            {"explanation_type": "density", "value": {"density": density_list}},
        ],
    }

    return EBMExplanation(
        "global",
        internal_obj,
        feature_names=feature_names,
        feature_types=['continuous'] * mebm.n_features_in_,
        name=name,
        selector=selector,
    )
