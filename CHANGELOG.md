# 1.1

## Features
- register is importable from the top level module 
- lagIsolated now respects time gaps in addition to value numbers
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

## Bugfixes
- pass the harmonization function names to the flagger
- variables not listed in the varname column of the configuration file
  were not available in generic tests

## Refactorings
- configuration reader rework