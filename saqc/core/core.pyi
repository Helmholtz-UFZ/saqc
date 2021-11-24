import numpy as np
import pandas as pd
from typing import Sequence, Union, Tuple, Hashable, Any, Optional, Callable
from typing_extensions import Literal
from dios import DictOfSeries
from saqc.constants import UNFLAGGED, BAD
from saqc.core.flags import Flags
from saqc.core.translator import Translator
from saqc.core.register import FunctionWrapper
from scipy.spatial.distance import pdist
from saqc.lib.types import (
    FreqString,
    CurveFitter,
    LinkageString,
    InterpolationString,
    ExternalFlag,
    GenericFunction,
)

class SaQC:
    def __init__(
        self,
        data=None,
        flags=None,
        scheme: str | TranslationScheme = "float",
        copy: bool = True,
    ): ...
    def _construct(self, **attributes) -> SaQC: ...
    def _validate(self, reason=None): ...
    @property
    def attrs(self) -> dict[Hashable, Any]: ...
    @attrs.setter
    def attrs(self, value: dict[Hashable, Any]) -> None: ...
    @property
    def dataRaw(self) -> DictOfSeries: ...
    @property
    def flagsRaw(self) -> Flags: ...
    @property
    def data(self) -> pd.DataFrame: ...
    @property
    def flags(self) -> pd.DataFrame: ...
    @property
    def result(self) -> SaQCResult: ...
    def _expandFields(
        self,
        regex: bool,
        multivariate: bool,
        field: str | Sequence[str],
        target: str | Sequence[str] = None,
    ) -> Tuple[List[str], List[str]]: ...
    def _wrap(self, func: FunctionWrapper): ...
    def _callFunction(
        self, function: Callable, field: str | Sequence[str], *args: Any, **kwargs: Any
    ) -> SaQC: ...
    def __getattr__(self, key): ...
    def copy(self, deep=True): ...
    def __copy__(self): ...
    def __deepcopy__(self, memodict=None): ...
    def _initTranslationScheme(
        self, scheme: str | TranslationScheme
    ) -> TranslationScheme: ...
    def _initData(self, data, copy: bool) -> DictOfSeries: ...
    def _initFlags(self, flags, copy: bool) -> Flags: ...
    def flagMissing(
        self,
        field: str | Sequence[str],
        flag: float = BAD,
        to_mask: float = UNFLAGGED,
        target: str | Sequence[str] = None,
        **kwargs
    ) -> SaQC: ...
    def assignChangePointCluster(
        self,
        field: str | Sequence[str],
        stat_func: Callable[[np.ndarray, np.ndarray], float],
        thresh_func: Callable[[np.ndarray, np.ndarray], float],
        window: str | Tuple[str, str],
        min_periods: int | Tuple[int, int],
        closed: Literal["right", "left", "both", "neither"] = "both",
        reduce_window: str = None,
        reduce_func: Callable[[np.ndarray, np.ndarray], int] = lambda x, _: int(
            x.argmax()
        ),
        model_by_resids: bool = False,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def fitPolynomial(
        self,
        field: str | Sequence[str],
        window: Union[int, str],
        order: int,
        min_periods: int = 0,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def flagDriftFromReference(
        self,
        field: str | Sequence[str],
        reference: str,
        freq: FreqString,
        thresh: float,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        target=None,
        flag: float = BAD,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def flagDriftFromScaledNorm(
        self,
        field: str | Sequence[str],
        set_1: Sequence[str],
        set_2: Sequence[str],
        freq: FreqString,
        spread: float,
        frac: float = 0.5,
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: pdist(
            np.array([x, y]), metric="cityblock"
        )
        / len(x),
        method: LinkageString = "single",
        target: str = None,
        flag: float = BAD,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def correctDrift(
        self,
        field: str | Sequence[str],
        maintenance_field: str,
        model: Callable[..., float],
        cal_range: int = 5,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def correctRegimeAnomaly(
        self,
        field: str | Sequence[str],
        cluster_field: str,
        model: CurveFitter,
        tolerance: Optional[FreqString] = None,
        epoch: bool = False,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def correctOffset(
        self,
        field: str | Sequence[str],
        max_jump: float,
        spread: float,
        window: FreqString,
        min_periods: int,
        tolerance: Optional[FreqString] = None,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def assignRegimeAnomaly(
        self,
        field: str | Sequence[str],
        cluster_field: str,
        spread: float,
        method: LinkageString = "single",
        metric: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.abs(
            np.nanmean(x) - np.nanmean(y)
        ),
        frac: float = 0.5,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def forceFlags(
        self,
        field: str | Sequence[str],
        flag: float = BAD,
        target: str | Sequence[str] = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def clearFlags(
        self,
        field: str | Sequence[str],
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def flagUnflagged(
        self,
        field: str | Sequence[str],
        flag: float = BAD,
        target: str | Sequence[str] = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def flagManual(
        self,
        field: str | Sequence[str],
        mdata: Union[pd.Series, pd.DataFrame, DictOfSeries, list, np.ndarray],
        method: Literal[
            "left-open", "right-open", "closed", "plain", "ontime"
        ] = "left-open",
        mformat: Literal["start-end", "mflag"] = "start-end",
        mflag: Any = 1,
        flag: float = BAD,
        target: str | Sequence[str] = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def processGeneric(
        self,
        field: str | Sequence[str],
        func: GenericFunction,
        target: str | Sequence[str] = None,
        flag: float = UNFLAGGED,
        to_mask: float = UNFLAGGED,
        **kwargs
    ) -> SaQC: ...
    def flagGeneric(
        self,
        field: str | Sequence[str],
        func: GenericFunction,
        target: Union[str, Sequence[str]] = None,
        flag: float = BAD,
        to_mask: float = UNFLAGGED,
        **kwargs
    ) -> SaQC: ...
    def interpolateByRolling(
        self,
        field: str | Sequence[str],
        window: Union[str, int],
        func: Callable[[pd.Series], float] = np.median,
        center: bool = True,
        min_periods: int = 0,
        flag: float = UNFLAGGED,
        target: str | Sequence[str] = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def interpolateInvalid(
        self,
        field: str | Sequence[str],
        method: InterpolationString,
        order: int = 2,
        limit: int = 2,
        downgrade: bool = False,
        flag: float = UNFLAGGED,
        target: str | Sequence[str] = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def flagMVScores(
        self,
        field: str | Sequence[str],
        trafo: Callable[[pd.Series], pd.Series] = lambda x: x,
        alpha: float = 0.05,
        n: int = 10,
        func: Callable[[pd.Series], float] = np.sum,
        iter_start: float = 0.5,
        partition: Optional[Union[int, FreqString]] = None,
        partition_min: int = 11,
        stray_range: Optional[FreqString] = None,
        drop_flagged: bool = False,
        thresh: float = 3.5,
        min_periods: int = 1,
        target: str = None,
        flag: float = BAD,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def flagCrossStatistic(
        self,
        field: str | Sequence[str],
        thresh: float,
        method: Literal["modZscore", "Zscore"] = "modZscore",
        flag: float = BAD,
        target: str = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def shift(
        self,
        field: str | Sequence[str],
        freq: str,
        method: Literal["fshift", "bshift", "nshift"] = "nshift",
        freq_check: Optional[Literal["check", "auto"]] = None,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def resample(
        self,
        field: str | Sequence[str],
        freq: str,
        func: Callable[[pd.Series], float] = np.mean,
        method: Literal["fagg", "bagg", "nagg"] = "bagg",
        maxna: Optional[int] = None,
        maxna_group: Optional[int] = None,
        maxna_flags: Optional[int] = None,
        maxna_group_flags: Optional[int] = None,
        flag_func: Callable[[pd.Series], float] = max,
        freq_check: Optional[Literal["check", "auto"]] = None,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def concatFlags(
        self,
        field: str | Sequence[str],
        target: str,
        method: Literal[
            "inverse_fagg",
            "inverse_bagg",
            "inverse_nagg",
            "inverse_fshift",
            "inverse_bshift",
            "inverse_nshift",
            "inverse_interpolation",
        ],
        freq: Optional[str] = None,
        drop: Optional[bool] = False,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def calculatePolynomialResidues(
        self,
        field: str | Sequence[str],
        window: Union[str, int],
        order: int,
        min_periods: int = 0,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def calculateRollingResidues(
        self,
        field: str | Sequence[str],
        window: Union[str, int],
        func: Callable[[pd.Series], float] = np.mean,
        min_periods: int = 0,
        center: bool = True,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def roll(
        self,
        field: str | Sequence[str],
        window: Union[str, int],
        func: Callable[[pd.Series], float] = np.mean,
        min_periods: int = 0,
        center: bool = True,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def assignKNNScore(
        self,
        field: str | Sequence[str],
        target: str,
        n: int = 10,
        func: Callable[[pd.Series], float] = np.sum,
        freq: Union[float, str] = np.inf,
        min_periods: int = 2,
        method: Literal["ball_tree", "kd_tree", "brute", "auto"] = "ball_tree",
        metric: str = "minkowski",
        p: int = 2,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def copyField(
        self,
        field: str | Sequence[str],
        target: str,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def maskTime(
        self,
        field: str | Sequence[str],
        mode: Literal["periodic", "mask_field"],
        mask_field: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        closed: bool = True,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def plot(
        self,
        field: str | Sequence[str],
        path: Optional[str] = None,
        max_gap: Optional[FreqString] = None,
        stats: bool = False,
        history: Optional[Literal["valid", "complete", "clear"]] = "valid",
        xscope: Optional[slice] = None,
        phaseplot: Optional[str] = None,
        stats_dict: Optional[dict] = None,
        store_kwargs: Optional[dict] = None,
        to_mask: float = np.inf,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...
    def transform(
        self,
        field: str | Sequence[str],
        func: Callable[[pd.Series], pd.Series],
        freq: Optional[Union[float, str]] = None,
        target: str | Sequence[str] = None,
        flag: ExternalFlag = None,
        to_mask: ExternalFlag = None,
        **kwargs
    ) -> SaQC: ...

class SaQCResult:
    def __init__(
        self, data: DictOfSeries, flags: Flags, attrs: dict, scheme: TranslationScheme
    ): ...
    def _validate(self): ...
    @property
    def data(self) -> pd.DataFrame: ...
    @property
    def flags(self) -> pd.DataFrame: ...
    @property
    def dataRaw(self) -> DictOfSeries: ...
    @property
    def flagsRaw(self) -> Flags: ...
    @property
    def columns(self) -> DictOfSeries: ...
    def __getitem__(self, key): ...
    def __repr__(self): ...
