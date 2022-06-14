<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

This changelog starts with version 2.0.0. Basically all parts of the system, including the format of this changelog, have been reworked between the releases 1.4 and 2.0. Preceding the major breaking release 2.0, the maintenance of this file was rather sloppy, so we won't provide a detailed change history for early versions.


## Unreleased
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.1.0...develop)
### Added
### Changed
### Removed
### Fixed

## [2.1.0](https://git.ufz.de/rdm-software/saqc/-/tags/v2.0.1) - 2022-06-14
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
