<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog
## Unreleased
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.5.0...develop)
### Added
- `flagGeneric`: target broadcasting
- `SaQC`: automatic translation of incoming flags
- Option to change the flagging scheme after initialization
- `SaQC`: support for selection, slicing and setting of items by use of subscription on SaQC objects (e.g. `qc[key]` and `qc[key] = value`).
   Selection works with single keys, collections of keys and string slices (e.g. `qc["a":"f"]`).  Values can be SaQC objects, pd.Series, 
   Iterable of Series and dict-like with series values.
### Changed
### Removed
### Fixed
- `Flags`: add meta entry to imported flags
- group operations were overwriting existing flags
- `SaQC._construct` : was not working for inherit classes (used hardcoded `SaQC` to construct a new instance).
### Deprecated

## [2.5.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.4.1) - 2023-06-22
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.4.1...v2.5.0)
### Added
- WMO standard mean aggregations
- Function selection via strings for most function-expecting parameters
- `SaQC.plot`:
  - enable multivariate plots
  - keyword `plot_kwargs` to pass matplotlib related arguments
- CLI:
  - `--version` to print the SaQC version
  - `-ll` as a shorthand for `--log-level`
  - `--json-field` to use a non-root element of a json file.
  - basic json support for CLI config files, which are detected by `.json`-extension.
- `SaQC.flagScatterLowpass`: option to select function based on string names.
- Checks and unified error message for common function inputs.
### Changed
- Require pandas >= 2.0
- `SaQC.flagUniLOF` and `SaQC.assignUniLOF`: changed parameter `fill_na` to type `bool`.
- `SaQC.plot`:
   - changed default color for single variables to `black` with `80% transparency`
   - added seperate legend for flags
### Removed
- `SaQC.plot`: option to plot with complete history (`history="complete"`)
- Support for Python 3.8
### Fixed
- `SaQC.assignChangePointCluster` and `SaQC.flagChangePoints`: A tuple passed `min_period`
   was only recognised if `window` was also a tuple.
- `SaQC.propagateFlags` was overwriting existing flags
### Deprecated
- `SaQC.andGroup` and `SaQC.orGroup`: option to pass dictionaries to `group`.
- `SaQC.plot`:
  - `phaseplot` in favor of usage with `mode="biplot"`
  - `cyclestart` in favor of usage with `marker_kwargs`
- `SaQC.flagStatLowPass` in favor of `SaQC.flagScatterLowpass`

## [2.4.1](https://git.ufz.de/rdm-software/saqc/-/tags/v2.4.1) - 2023-06-22
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.4.0...v.2.4.1)
### Added
### Changed
- pin pandas to versions >= 2.0
### Removed
- removed deprecated `DictOfSeries.to_df`
### Fixed
### Deprecated

## [2.4.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.4.0) - 2023-04-25
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.3.0...v2.4.0)
### Added
- Methods `logicalAnd` and `logicalOr`
- `Flags` support slicing and column selection with `list` or a `pd.Index`.
- Expose the `History` via `SaQC._history`
- Config function `cv` (coefficient of variation)
### Changed
- Rename `interplateInvalid` to `interpolate`
- Rename `interpolateIndex` to `align`
- Rewrite of `dios.DictOfSeries`
### Removed
- Parameter `limit` from `align`
- Parameter `max_na_group_flags`, `max_na_flags`, `flag_func`, `freq_check` from `resample`
### Fixed
- `func` arguments in text configurations were not parsed correctly
- fail on duplicated arguments to test methods
- `reample` was not writing meta entries
- `flagByScatterLowpass` was overwriting existing flags
- `flagUniLOF` and `flagLOF` were overwriting existing flags
### Deprecated
- Deprecate `flagMVScore` parameters: `partition` in favor of `window`, `partition_min` in favor of `min_periods`, `min_periods` in favor of `min_periods_r`
- Deprecate `interpolate`, `linear` and `shift` in favor of `align`
- Deprecate `roll` in favor of `rolling`
- Deprecate `DictOfSeries.to_df` in favor of `DictOfSeries.to_pandas`
## [2.3.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.3.0) - 2023-01-17
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.2.1...v2.3.0)
### Added
- add option to not overwrite existing flags to `concatFlags`
- add option to pass existing axis object to `plot`
- python 3.11 support
- added Local Outlier Factor functionality
### Changed
- Remove all flag value restrictions from the default flagging scheme `FloatTranslator`
- Renamed `TranslationScheme.forward` to `TranslationScheme.toInternal`
- Renamed `TranslationScheme.backward` to `TranslationScheme.toExternal`
- Changed default value of the parameter `limit` for `SaQC.interpolateIndex` and `SaQC.interpolateInvalid` to ``None``
- Changed default value of the parameter ``overwrite`` for ``concatFlags`` to ``False``
- Deprecate ``transferFlags`` in favor of ``concatFlags``
### Removed
- python 3.7 support
### Fixed
- Error for interpolations with limits set to be greater than 2 (`interpolateNANs`)
- Error when fitting polynomials to irregularly sampled data (`fitPolynomial`)

## [2.2.1](https://git.ufz.de/rdm-software/saqc/-/tags/v2.2.1) - 2022-10-29
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.2.0...v2.2.1)
### Added
- data label to `plot` legend
### Changed
- `dfilter` default value inference to respect the function default value of `plot`
### Removed
### Fixed
- functions not handling `target` failed to overwrite existing variables

## [2.2.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.2.0) - 2022-10-28
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.1.0...v2.2.0)
### Added
- translation of `dfilter`
- new generic function `clip`
- parameter `min_periods` to `SaQC.flagConstants`
- function `fitLowpassFilter`
- tracking interpolation routines in `History`
### Changed
- test function interface changed to `func(saqc: SaQC, field: str | Sequence[str], *args, **kwargs)`
- lib function `butterFilter` returns `NaN` for too-short series
- `dfilter` default value precedence order
### Removed
- `closed` keyword in `flagJumps`
### Fixed
- fixed undesired behavior in `flagIsolated` for not harmonized data
- fixed failing translation of `dfilter`-defaults
- fixed unbound recursion error when interpolating with order-independent methods in `interpolateIndex`
- fixed not working min_periods condition if `window=None` in `assignZScore`
- fixed Exception occuring when fitting polynomials via `polyfit` to harmonized data, containing all-NaN gaps wider than the polynomial fitting window size.
- fixed bug in function parameter checking
- fixed bug one-off bug in `flagJumps`

## [2.1.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.1.0) - 2022-06-14
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.0.1...v2.1.0)
### Added
- documentation of global keywords
- generic documentation module `docurator.py`
- documentation of flagging constants
- `pyproject.toml`
- new function `progagateFlags`
- include function typehints in parameter documentation
- `label` parameter to the generic function `isflagged`
### Changed
- `flagOffsets` parameters `thresh` and `thresh_relative` are optional
- corrected false notion of the term *residual* (replace all occurences of *residue* by *residual*)
- `FILTER_NONE` and `FILTER_ALL` are top level constants (imported in `saqc.__init__`)
- renamed `maskTime` to `selectTime`
- `SaQC.data` returns `dios.DictOfSeries`
- `SaQC.flags` returns `dios.DictOfSeries` or `pd.DataFrame`
- `SaQC.data` and `SaQC.flags` are not mutated by function calls
- renamed `History.max` to `History.squeeze`
- renamed parameter `freq` of function flagByStray to `window`
- `DmpScheme`: set `DFILTER_DEFAULT` to 1 in order to not mask the flag 'OK'
### Removed
- data accessors `SaQC.result`, `SaQC.data_raw`, `SaQC.flags_raw`
### Fixed
- `flagOffset` failure on falsy `thresh`
- `flagCrossStatistics` failure on unaligned input variables
- `plot` data loss when using *dfilter* kwarg
- `correctDrift`: failure on single value intervals
- `concatFlags`: information loss by appending squeezed histories
- `interpolateInvalid`: replace flags by interpolated values
- `resample`: pass resampling function to  `history.appy()`
- `tools.seasonalMask`: mask swapping with `include_bounds=True`
- `flagGeneric`:
  - fixed inconsistent history meta writing
  - fixed handling of existing flags
- `proGeneeric`: fixed inconsistent history meta writing
- `docs`: removed documentation of data/flags parameters from automatic sphinx doc

## [2.0.1](https://git.ufz.de/rdm-software/saqc/-/tags/v2.0.1) - 2021-12-20
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.0.0...v2.0.1)
### Added
- CLI now accepts remote configuration and data files as URL
- new function `transferFlags`
- improved error messages from `flagGeneric` and `processGeneric`
- new `ax_kwargs` keyword to `SaQC.plot` function
### Changed
- generate documentation from the `develop` branch
- doctest is now ran upon push to the `develop` branch, failing doc snippets cause CI-pipeline to fail
- renamed function `flagCrossStatistic` to `flagCrossStatistics`
### Removed
- removed function `flagDriftFromScaledNorm`
- removed `stats` keywords and functionality from `SaQC.plot` function
### Fixed
- RDM/UFZ logos:
  - use the English versions of the respective images
  - use full URLs instead of the repository local URLs in `README.md`
- fix code snippets in `README.md`
- fix version confusion
- `copyField`: fix misleading error message
- `flagGeneric`: fix failure on empty data
- existing `target` variables led to function calls on `target` instead of `field`
- the functions `flagDriftFromNorm`, `flagDriftFromReference`, `flagCrossStatistics` and `flagMVScores` now properly support the field-target workflow
- `field` was not masked for resampling functions
- allow custom registered functions to overwrite built-ins.

## [2.0.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.0.0) - 2021-11-25
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v1.5.0...v2.0.0)

This release marks the beginning of a new release cycle. Basically the entire system got reworked between versions 1.4 and 2.0, a detailed changelog is not recoverable and/or useful.
