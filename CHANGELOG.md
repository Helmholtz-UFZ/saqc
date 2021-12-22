# Changelog

This changelog starts with version 2.0.0. Basically all parts of the system, including the format of this changelog, have been reworked between the releases 1.4 and 2.0. Preceding the major breaking release 2.0, the maintenance of this file was rather sloppy, so we won't provide a detailed change history for early versions.


## [Unreleased]
### Added
### Changed
### Removed
### Fixed

## [2.0.1] - 2021-12-20
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

## [2.0.0] - 2021-11-25
This release marks the beginning of a new release cycle. Basically the entire system got reworked between versions 1.4 and 2.0, a detailed changelog is not recoverable and/or useful.
