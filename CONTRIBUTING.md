# Development Environment
We recommend a virtual python environment for development, a more detailed description of the setup process can be found in the [docs](https://rdm-software.pages.ufz.de/saqc/getting_started/InstallationGuide.html#set-up-a-virtual-environment).

# Testing
SaQC comes with an extensive test suite based on [pytest](https://docs.pytest.org/en/latest/).
In order to run all tests execute `python -m pytest .`, for faster iteration a test run with 
`python -m pytest --ignore test/lib test` is usually enough.

# Coding conventions

## Naming

### Code
We implement the following naming conventions:
- Classes: CamelCase
- Functions: camelCase
- Variables/Arguments: snake_case

### Argument names in public functions signatures

first, its not necessary to have *talking* arg-names, in contrast to variable names in 
code. This is, because one always must read the documentation. To use and parameterize a function,
just by guessing the meaning of the argument names and not read the docs, 
will almost never work. thats why, we dont have the obligation to make names (very) 
talkative.

second, because of the nature of a function (to have a *simple* way to use complex code), 
its common to use simple and short names. This means, to omit any *irrelevant* information. 

For example if we have a function that fit a polynomial on some data with three arguments.
Lets say we have:
 - the data input, 
 - a threshold that defines a cutoff point for a calculation on a polynomial and
 - a third argument. 

one could name the args `data, poly_cutoff_threshold, ...`, but much better names would 
be `data, thresh, ...`, because a caller dont need the extra information, 
stuffed in the name. 
If the third argument is also some kind of threshold, 
one can use `data, cutoff, thresh`, because the *thresh-* information of the `cutoff` 
parameter is not crucial and the caller knows that this is a threshold from the docstring.

third, underscores give a nice feedback if one doing wrong or over complex. 
No underscore is fine, one underscore is ok, if the information is *really necessary* (see above), 
but if one use two or more underscores, one should think of a better naming, 
or omit some information. 
Sure, seldom but sometimes it is necessary to use 2 underscores, but we consider it as bad style.
Using 3 or more underscores, is not allowed unless have write an reasoning and get it
signed by at least as many core developers as underscores one want to use.


In short the naming should *give a very, very rough idea* of the purpose of the argument, 
but not *explain* the usage or the purpose. 
It is not a shame to name a parameter just `n` or `alpha` etc. if for example the algorithm 
(from the paper etc.) name it alike. 


### Test Functions
- testnames: [testmodule_]flagTestName
 
## Formatting
We use [black](https://black.readthedocs.io/en/stable/) in its default settings.
Within the `saqc` root directory run `black .`.

## Imports
Only absolute imports are accepted.


# Development Workflow
## Repository Structure

- `master` - branch:
  + Stable and usually protected.
  + Regular merges from `develop`, these merges are tagged and increasing at least the minor version.
  + Irregular merges from `develop` in case of critical bugs. Such merges increase at least the patch level.
  + Merges into `master` usually lead to a PyPI release.
- `develop` - branch:
  + The main development branch, no hard stability requirements/guarantees.
  + Merges into `develop` should mostly follow a [Merge Request Workflow](#merge-request-workflow), minor changes can however be committed directly. Such minor changes include:
    * Typos and white space changes
    * Obvious bug in features implemented by the committing developer
    
    
## Merge Request Workflow
- Most changes to `saqc` are integrated by merge requests from a feature branch into `develop`
- All merge requests need to be reviewed by at least one other core developer (currently @palmb, @luenensc and @schaefed).
- We implement the following Gitlab based review process:
  + The author assigns the Merge Request to one of the core developers. The reviewer should review the request within one week,
    large requests may of course lead to longer review times.
  + Reviewer and Author discuss any issues using the Gitlab code review facilities:
    * In case all concerns are resolved, the reviewer approves the Merge Request and assigns it back to the author.
    * In case reviewer and author can't resolve their discussion, the Merge Request should be assigned to another reviewer.
      The new reviewer is now in charge to come to a decision, by either approving, closing or going into another review iteration.
  + The author of an approved Merge Request:
    * has the right and the duty to merge into the `develop` branch, any occurring conflicts need to be addressed by the author,
    * is always highly encouraged to provide a summary of the changes introduced with the Merge Request in its description upon integration. This recommandation becomes an obligation in case of interface modification or changes to supported and/or documented workflows.


## Release Cycle
- We employ a release cycle of roughly 4 weeks.
- To avoid the avoid the integration of untested and/or broken changes, the merge window closes one week before the intended
  release date. Commits to `develop` after the merge window of a release closes need to be integrated during the subsequent release
  cycle
- The release cycle is organized by Gitlab Milestones, the expiration date of a certain milestone indicates the end of the 
  related merge window, the actual merge into `master` and the accompanying release is scheduled for the week after the
  milestones expiration date. 
- Issues and Merge Requests can and should be associated to these milestone as this help in the organization of review activities.
