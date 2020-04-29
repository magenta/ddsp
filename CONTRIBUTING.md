# How to Contribute

We'd love to accept your patches and contributions to this project.
DSP can be subtle to get completely right, so we particularly appreciate the
contributions of those with expertise in signal processing to help fix any
mistakes we may have made 😄.

# Versioning

We'll do our best to keep the verison updated. This repo contains two code bases
which makes versioning a bit tricky. The core code base `ddsp/` and a more
experimental training code base `ddsp/training/` that is used for active
research. We will thus adopt the following scheme for incrementing version:

`vMajor.Minor.Revision`

* Major: Breaking change in `ddsp/`
* Minor: New feature in `ddsp/`, breaking change in `training/`
* Revision: New feature in `training/`, minor bug fix anywhere

## Code Design Goals
As much as we can, we would like the DDSP library to be approachable,
well-tested, well-documented, and full of useful examples. Thus, PRs that add
new functionality should be accompanined with ample documentation and tests to
help newcomers understand a typical use case, and guard against silent failures
from breaking changes in the future. Please follow the existing doc/testing
style when you can.

To ensure a consistent style, new code should follow the [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html)
and will need to pass a google style linter before acceptance. While this can
add a little work up front, and occasionally make things more verbose, it helps
reduce mental overhead and makes the code more readable.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Please be sure to test your code by running `pytest` and `pylint` before
submitting a pull request for review. The easiest way to do this is to run
`sh ci-test.sh` from the base directory. Note that code cannot be merged until
these tests pass on [Travis](https://travis-ci.org/magenta/ddsp).


## Getting Started

If you're looking for a way to contribute, but not sure where to start, you
could:

* Add some documentation to an existing function.
* Add a missing test to improve coverage.
* Add type hints to functions in a new file.
* Add a new colab tutorial or demo, covering a typical use case or showing something cool.
* Respond to a bug or feature request in the github [Issues](github.com/magenta/ddsp/issues).
* Add a new signal `Processor` and corresponding test.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.


## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
