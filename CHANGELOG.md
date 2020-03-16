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

# 1.3

coming soon...

## Features

## Bugfixes
- configuration: certain whitespace patterns broke the configuration parsing

## Refactorings
