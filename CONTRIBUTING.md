# Development Environment
We recommend a virtual python environment for development. The setup process is described in detail in our [GettingStarted](docs/GettingStarted.md).

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

### Test Functions
- testnames: [testmodule_]flagTestName
 
## Formatting
We use (black)[https://black.readthedocs.io/en/stable/] with a line length if 120 characters.
Within the `SaQC` root directory run `black -l 120`.

## Imports
Only absolute imports are accepted.


# Development Workflow
## Repository Structure

- `master` - branch:
  + Stable and usually protected.
  + Regular merges from `develop` according to the [release cycle](#release-cycle). These merges get a tag, increasing at least the minor version.
  + Irregular merges from `develop` in case if critical bugs. Such merges increase at least the patch level.
  + Merges into `master` usually lead to a PyPI release
- `develop` - branch:
  + The main development branch, no hard stability requirements/guarantees
  + Merges into `develop` should mostly follow a Merge Request Workflow, minor changes can however be committed directly. Such minor changes include:
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
