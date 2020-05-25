#!/usr/bin/env python

from operator import itemgetter
from inspect import signature, Parameter, _VAR_POSITIONAL, _VAR_KEYWORD
from typing import Tuple, Dict, Generator, Any, Set

import numpy as np
import pandas as pd


class Func:
    """
    This class is basically extends functool.partial` and in
    fact, most of the constructor implementation is taken
    directly from the python standard lib. Not messing with the
    partial class directly proved to be an easier aproach though.

    Besides the implementation of a partial like functionality
    `Func` provides a couple of properties/methods used to check
    the passed arguments before the actual function call happens.
    """
    def __init__(self, *args, **kwargs):
        if len(args) < 1:
            raise TypeError("'Func' takes at least one argument")
        func, *args = args
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if isinstance(func, Func):
            args = func.args + args
            kwargs = {**func.kwargs, **kwargs}
            func = func.func

        self._signature = signature(func)
        # NOTE:
        # bind_partial comes with a validity check, so let's use it
        self._signature.bind_partial(*args, **kwargs)

        self.__name__ = func.__name__
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__name__}, {self.args}, {self.kwargs})"

    def __call__(self, *args, **kwargs):
        keywords = {**self.kwargs, **kwargs}
        return self.func(*self.args, *args, **keywords)

    @property
    def _parameters(self) -> Generator[Tuple[str, Parameter], None, None]:
        """
        yield all 'normal' parameters and their names, skipping
        VAR_POSITIONALs (*args) and VAR_KEYWORDs (**kwargs) as
        the don't help evaluating the correctness of the passed
        arguments.
        """
        for k, v in self._signature.parameters.items():
            if v.kind in (_VAR_POSITIONAL, _VAR_KEYWORD):
                continue
            yield k, v

    @property
    def parameters(self) -> Tuple[str]:
        """
        return the names of all parameters, i.e. positional
        and keyword arguments without varargs
        """
        return tuple(map(itemgetter(0), self._parameters))

    @property
    def optionals(self) -> Tuple[str]:
        """
        return the names of all optional parameters without varargs
        """
        return tuple(k for k, v in self._parameters if v.default is not Parameter.empty)

    def _getPositionals(self):
        """
        return the names of all positional parameters without varargs
        """
        return tuple(k for k, v in self._parameters if v.default is Parameter.empty)

    positionals = property(_getPositionals)

    def addGlobals(self, globs: Dict[str, Any]):
        """
        Add the given key-value pairs to the function's global
        scope. We abuse the __globals__ mechanism mainly to
        make certain other functions (dis-)available within the
        'Func' body.
        """
        self.func.__globals__.update(globs)
        return self

    def getUnbounds(self) -> Set[str]:
        """
        returns all the names of all unbound variables,
        i.e. not yet `partialed` parameters
        """
        return set(self.positionals[len(self.args):]) - set(self.kwargs.keys())


class RegisterFunc(Func):

    """
    This class acts as a simple wrapper around all registered
    functions. Currently its sole purpose is to inject additional
    call arguments
    """
    def __call__(self, *args, **kwargs):
        # NOTE:
        # injecting the function name into the
        # keywords is sort of hacky
        kwargs = {"func_name": self.__name__, **kwargs}
        return super().__call__(*args, **kwargs)


class SaQCFunc(Func):

    """
    This class represents all test-, process and horminzation functions
    provided through `SaQC`. Every call to an `SaQC` object will be wrapped
    with all its non-dynamic arguments.

    `SaQCFunc`s are callable and expose the signature `data`, `field` and
    `flagger`
    """

    # NOTE:
    # we should formalize the function interface somehow, somewhere
    _injectables = ("data", "field", "flagger")

    def __init__(self, *args, plot=False, lineno=None, expression=None, **kwargs):
        super().__init__(*args, **kwargs)

        unbound = self.getUnbounds()
        if unbound:
            raise TypeError(f"missing required arguments: {', '.join(unbound)}")

        self.plot = plot
        self.lineno = lineno
        self.expr = expression

    def _getPositionals(self) -> Tuple[int]:
        """
        Returns all positional (i.e. non-optional arguments)
        without the `data`, `field` and `flagger`
        """
        positionals = super()._getPositionals()
        return tuple(k for k in positionals if k not in self._injectables)

    positionals = property(_getPositionals)

    def __call__(self, data, field, flagger):
        # NOTE:
        # when assigning new variables to `data`, the respective
        # field is missing in `flags`, so we add it if necessary in
        # order to keep the columns from `data` and `flags` in sync
        if field not in flagger.getFlags():
            flagger = flagger.merge(flagger.initFlags(data=pd.Series(name=field)))

        # NOTE: replace flagged values by nan
        mask = flagger.isFlagged()
        data_in = data.copy()
        data_in[mask] = np.nan

        data_result, flagger_result = self.func(data_in, field, flagger, *self.args, **self.kwargs)

        # NOTE: reinject the masked values
        data_result.aloc[mask] = data[mask]

        return data_result, flagger_result


# NOTE:
# the global SaQC function store,
# will be filled by calls to register
FUNC_MAP : Dict[str, RegisterFunc] = {}


def register(func):
    name = func.__name__
    func = RegisterFunc(func)
    FUNC_MAP[name] = func
    return func
