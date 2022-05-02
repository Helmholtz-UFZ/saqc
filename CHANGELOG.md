<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

This changelog starts with version 2.0.0. Basically all parts of the system, including the format of this changelog, have been reworked between the releases 1.4 and 2.0. Preceding the major breaking release 2.0, the maintenance of this file was rather sloppy, so we won't provide a detailed change history for early versions.


## Unreleased
[List of commits](https://git.ufz.de/rdm-software/saqc/-/compare/v2.0.1...develop)

### Added
- global keywords documentation resource added
- generic documentation module `docurator.py` added to `lib`
- flagging constants documentation resource added
- `pytest.ini`: to setup default path and markers for pytest
### Changed
- documentation pipeline changed to base on methods decorators
- `flagOffsets` parameters `thresh` and `thresh_relative` now both are optional
- flags concatenation tasks (for squeezed and explicit histories) are now all channeled through the function `concatFlags`
- corrected false notion of *residual* concept (old notion: *residue* got replaced by *residual*)
- constants `FILTER_NONE` and `FILTER_ALL` are now imported to `saqc.__init__`
- renamed `maskTime` to `selectTime`
- `.gitlab-ci.py`: always run all pytest-tests in CI/CD pipelines
- `.gitlab-ci.py`: use reports to enable `Tests` in CI/CD pipeline results
- `procGeneric`: changed default `flag` value to `np.nan`

### Removed
### Fixed
- `flagOffset` bug with zero-valued threshold
- `flagCrossStatistics` bug with unaligned input variables
- `plot` fixed data loss when using *dfilter* kwarg
- `DmpScheme`: set `DFILTER_DEFAULT` to 1 in order to not mask the flag 'OK'
- `correctDrift`: fixed bug when correcting single value intervals
- `concatFlags`: fixed bug in context of squeezed history appending (UNTOUCHED vs UNFLAGGED information now doesnt get lost)
- `interpolateInvalid`: Fix: replacement of flags for interpolated values now works
- `resample`: resampling func now actually gets passed on to `history.appy()`
- `tools.seasonalMask`: fixed bug that swaps the entire mask upon `include_bounds=True`
- `flagGeneric`:
  - fixed inconsistent history meta writing
  - fixed handling of existing flags
- `proGeneeric`: fixed inconsistent history meta writing

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
