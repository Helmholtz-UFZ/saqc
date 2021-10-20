# 1.1

## Features
- register is importable from the top level module 
- flagIsolated now respects time gaps in addition to value numbers
- Make the comparator argument to isflagged available from the config


## Bugfixes
- Fixed missing constant lookup in the evaluator
- Preserve untouched/checked variables and don't remove them from the data input

 
## Refactorings
--

## Breaking Changes
-- 

# 1.2

## Features
- Python 3.8 support
- exe: added the dmp flagger option
- exe: use nodata argument as nodata-representation in output
- flagging functions: implemented flagging function aiming to flag invalid value raises in a given time range
- anaconda support

## Bugfixes
- pass the harmonization function names to the flagger
- variables not listed in the varname column of the configuration file
  were not available in generic tests
- Harmonization by interpolation, now will no longer insert a BAD-flagged but propperly interpolated value between two frequency alligned meassurements, that are seperated exactly by a margin of two times the frequency (instead, BAD flagged NaN gets inserted - as expected)
- Fixed "not a frequency" - bug, occuring when trying to aggregate values to a 1-unit-frequency (1 Day, 1 Hour, ...)

## Refactorings
- configuration reader rework

## Breaking Changes
-- 

# 1.3

## Features
- spike detection test `spikes_flagRaise`
- spike detection test `spikes_oddWater`
- generic processing function `procGeneric` 

## Bugfixes
- configuration: certain whitespace patterns broke the configuration parsing
- configuration: multiple tests in one configuration row were not parsed correctly
- reader: variables only available within the flagger were not transformed correctly

## Refactorings
- Improved logging

## Breaking Changes
- configuration: quoted variable names are handled as regular expressions
- functions: renamed many test functions to a uniform naming scheme


# 1.4

## Features
- added the data processing module `proc_functions`
- `flagCrossValidation` implemented
- CLI: added support for parquet files

## Bugfixes
- `spikes_flagRaise` - overestimation of value courses average fixed
- `spikes_flagRaise` - raise check window now closed on both sides

## Refactorings
- renamed `spikes_oddWater` to `spikes_flagMultivarScores`
- added STRAY auto treshing algorithm to `spikes_flagMultivarScores`
- added "unflagging" - postprocess to `spikes_flagMultivarScores`
- improved and extended masking

## Breaking Changes
- register is now a decorator instead of a wrapper

# 1.5

coming soon ...

## Features

## Bugfixes

## Refactorings

## Breaking Changes
