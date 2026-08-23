#!/usr/bin/env python

# SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ
#
# SPDX-License-Identifier: GPL-3.0-or-later

import warnings
from typing import Callable, Literal

import matplotlib as mpl
import pandas as pd


def deprecateOptions(
    module_name: str,
    options_name: str,
    deprecations: dict[str, str],
) -> None:
    """
    Allows to depracted module global options

    Parameters
    ----------
    module_name :
        name of module in which we deprecated settings
    options_name :
        name of the options object within this modules that now holds the settings
    deprecations :
        mapping from the old settings name to the new one

    Note
    ----
    This function is needed only temporarily and as soon as all module level options
    are removed, it should also go.

    Example
    -------
    How to use the function in a module

    deprecateOptions(
        module_name=__name__,
        options_name="history_options",
        deprecations={"AGGREGATIONS": "aggregations", "AGGREGATION": "aggregation"},
    )
    """

    import sys
    import warnings
    from types import ModuleType
    from typing import Any

    module = sys.modules[module_name]

    def warn(name: str) -> None:
        warnings.warn(
            f"{module_name}.{name} is deprecated and will be removed in SaQC 2.11. "
            f"Use {__name__}.{options_name}.{deprecations[name]} instead.",
            FutureWarning,
            stacklevel=3,
        )

    class _ConfigModule(ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name in deprecations:
                warn(name)
                return getattr(
                    getattr(sys.modules[__name__], options_name), deprecations[name]
                )

            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

        def __setattr__(self, name: str, value: Any) -> None:
            if name in deprecations:
                warn(name)
                setattr(
                    getattr(sys.modules[__name__], options_name),
                    deprecations[name],
                    value,
                )
                return

            super().__setattr__(name, value)

    module.__class__ = _ConfigModule


class HistoryOptions:
    def __init__(self):
        self.aggregations: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            "last": lambda x: x.ffill(axis=1).iloc[:, -1],
            "max": lambda x: x.max(axis=1),
            "min": lambda x: x.min(axis=1),
        }
        self.aggregation: Literal["last", "min", "max"] = "last"


class FieldTargetOptions:
    def __init__(self):
        self.existing_target = "overwrite"

    @property
    def existing_target(self):
        return self._existing_target

    @existing_target.setter
    def existing_target(self, value: Literal["overwrite", "append"]):
        if value not in {"overwrite", "append"}:
            raise ValueError(
                "only 'overwrite' and 'append' are supported values for 'existing_target'"
            )
        if value == "overwrite":
            warnings.warn(
                "Overwriting existing targets in field-target flagging workflows is deprecated "
                "and will be removed in SaQC 3.0. To opt into the new append behavior, set "
                "saqc.options.field_target.existing_target = 'append'.",
                FutureWarning,
                stacklevel=2,
            )
        self._existing_target = value


class PlottingOptions:
    def __init__(self):
        self.backend = mpl.get_backend()
        self.test_mode = False
        # default color cycle for flags markers (seaborn color palette)
        self.marker_cycle = [
            (0.00784313725490196, 0.24313725490196078, 1.0),
            (1.0, 0.48627450980392156, 0.0),
            (0.10196078431372549, 0.788235294117647, 0.2196078431372549),
            (0.9098039215686274, 0.0, 0.043137254901960784),
            (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
            (0.6235294117647059, 0.2823529411764706, 0.0),
            (0.9450980392156862, 0.2980392156862745, 0.7568627450980392),
            (0.6392156862745098, 0.6392156862745098, 0.6392156862745098),
            (1.0, 0.7686274509803922, 0.0),
            (0.0, 0.8431372549019608, 1.0),
        ]
        # default color cycle for plot colors (many-in-one-plots)
        self.plot_cycle = [(0, 0, 0)] + self.marker_cycle
        # default data plot configuration (color kwarg only effective for many-to-one-plots)
        self.plot_kwargs = {"alpha": 0.8, "linewidth": 1, "color": self.plot_cycle}
        # default figure configuration
        self.fig_kwargs = {"figsize": (16, 9)}
        # default flags markers configuration
        self.scatter_kwargs = {
            "marker": ["s", "D", "^", "o", "v"],
            "color": self.marker_cycle,
            "alpha": 0.7,
            "zorder": 10,
            "edgecolors": "black",
            "s": 70,
        }


class OptimizationOptions:
    def __init__(self):
        self.seed = None


class ParsingOptions:
    def __init__(self):
        self.allowed_series_methods = {
            "abs",
            "add",
            "aggregate",
            # "apply",
            "asfreq",
            "autocorr",
            "bfill",
            "between",
            "between_time",
            "clip",
            "combine",
            "combine_first",
            "corr",
            "count",
            "cummax",
            "cummin",
            "cumprod",
            "cumsum",
            "diff",
            "div",
            "dot",
            "eq",
            "expanding",
            "ffill",
            "fillna",
            "ge",
            # "groupby",
            "gt",
            "isna",
            "isnull",
            "kurt",
            "le",
            "lt",
            "map",
            "mask",
            "max",
            "mean",
            "median",
            "min",
            "mod",
            "mul",
            "ne",
            "nlargest",
            "notna",
            "notnull",
            "nsmallest",
            "pct_change",
            "pow",
            "prod",
            "product",
            "quantile",
            "replace",
            "resample",
            "rolling",
            "round",
            "sem",
            "skew",
            "std",
            "sub",
            "sum",
            "transform",
            "truediv",
            "var",
            "where",
        }


history = HistoryOptions()
plotting = PlottingOptions()
optimization = OptimizationOptions()
parsing = ParsingOptions()
field_target = FieldTargetOptions()
