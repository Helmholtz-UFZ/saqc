<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

## Unreleased
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.3.0...develop)
### Added
### Changed
### Removed
### Fixed

## [2.3.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.3.0) - 2023-01-17
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.2.1...v2.3.0)
### Added
- add option to not overwrite existing flags to `concatFlags`
- add option to pass existing axis object to `plot`
- python 3.11 support
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
