# Changelog

This changelog starts with version 2.0.0. Basically all parts of the system, including the format of this changelog, have been reworked between the releases 1.4 and 2.0. Preceding the major breaking release 2.0, the maintenance of this file was rather sloppy, so we won't provide a detailled change history for early versions.


## [Unreleased]
### Added
- The CLI now accepts remote configuration files given by an URL
- function `transferFlags` added 
### Changed
- Documentaion now is generated from the develop branch
- Doctest is now ran upon push to the develop branch (failing doc snippets cause CI-pipeline to fail)
- Rename function `flagCrossStatistic` to `flagCrossStatistics`
### Removed
- removed function `flagDriftFromScaledNorm`
### Fixed
- RDM/UFZ ogos:
  - use the english versions of the respective images
  - use full urls instead of the repo local urls in README.md
- Fix the README.md code snippets
- Fix version confusion
- `copyField`: fix missleading error message
- `flagGeneric`: fix failure on empty data
- Existing `target` variables led to function calls on `target` instead of `field`
- The functions `flagDriftFromNorm`, `flagDriftFromReference`, `flagCrossStatistics` and `flagMVScores` now properly support the field-target workflow
- `field` was not masked for resampling functions

## [2.0.0] - 2021-11-25
This release marks the beginning of a new release cycle. Basically the entire system got reworked between versions 1.4 and 2.0, a detailed changelog is not recoverable and/or useful.
