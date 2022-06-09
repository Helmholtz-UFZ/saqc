<!--
SPDX-FileCopyrightText: 2021 Helmholtz-Zentrum für Umweltforschung GmbH - UFZ

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Development Environment
We recommend a virtual python environment for development, a more detailed description of the setup process can be found in the [docs](https://rdm-software.pages.ufz.de/saqc/gettingstarted/InstallationGuide.html#set-up-a-virtual-environment).

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

First, in contrast to variable names in code, it is not necessary to have *talking* function argument names. 
A user is always expected to have had acknowledged the documentation. Using and parameterizing a function,
just by guessing the meaning of the argument names, without having read the documentation, 
will almost never work. That is the reason, why we dont have the obligation to make names (very) 
talkative.

Second, from the nature of a function to deliver a *simple* way of using complex code, it follows, that simple and short names are to prefer. This means, the encoding of *irrelevant* informations in names should be omitted. 

For example, take a function of three arguments, that fits a polynomial to some data.
Lets say we have:
 - the data input, 
 - a threshold, that defines a cutoff point for a calculation on a polynomial and
 - a third argument. 

One could name the corresponding arguments: `data, poly_cutoff_threshold, ...`. However, much better names would 
be: `data, thresh, ...`, because a caller that is aware of a functions documentation doesnt need the extra information, 
encoded in the name. 
If the third argument is also some kind of threshold, 
one can use `data, cutoff, thresh`, because the *thresh-* information of the `cutoff` 
parameter is not crucial, and the caller knows that this is a threshold from having studied the docstring, anyways.

Third, underscores give a nice implicit feedback, on whether one is doing wrong or getting over complex in the naming behavior. 
To have no underscore, is just fine. Having one underscore, is ok, if the encoded information appended through the underscore is *really necessary* (see above). 
If one uses two or more underscores, one should think of a better naming or omit some information. 
Sure, although it is seldom, it might sometimes be necessary to use two underscores, but still the usage of two underscores is considered bad style.
Using three or more underscores is not allowed unless having issued a exhaustive and accepted (by at least one core developer per underscore) reasoning.


In short, the naming should *give a very, very rough idea* of the purpose of the argument, 
but not *explain* the usage or the purpose. 
It is not a shame to name a parameter just `n` or `alpha` etc., if, for example, the algorithm 
(from the paper etc.) names it alike. 


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
