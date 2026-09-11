# Release process

This page describes the release automation checked into this repository.
Maintainers prepare and validate a concrete version, then use the configured
GitHub workflow to build distributions and publish them. Installing or building
the documentation does not publish a release.

## Understand the artifacts

| Artifact | Contents | Important consequence |
| --- | --- | --- |
| Wheel | Python package, required compiled extension, distribution metadata | Must match the target Python and platform |
| Source distribution | Python and C sources, headers, tests, examples, documentation sources, tools | Installing requires a compiler; generated figure assets are omitted |
| Documentation site | Rendered Markdown, assets, search index | Describes the checkout from which it was built |

`MANIFEST.in` defines source-distribution inclusion. The C files and headers
must ship, while binaries left in a developer's tree must not. Regenerate figures
before building a complete site from an unpacked source archive.

## Prepare the version and release notes

`quadrivium.__version__` in `quadrivium/__init__.py` supplies the package version
through `pyproject.toml`. Choose the next version according to the compatibility
impact. A behavior correction can still affect users who depend on earlier
results; explain those changes explicitly.

Move the relevant **Unreleased** notes in `CHANGELOG.md` to a dated release
section, update comparison links, and describe any migration steps. Update
version examples such as the installation pin when appropriate. The generated
site changelog should always match the root file.

```bash
python tools/gen_docs.py
python tools/gen_figures.py
```

Review the generated reference and figures after regeneration. A figure can
change because of a numerical fix or a rendering environment change; establish
which before accepting it.

## Validate the candidate checkout

Use a clean development environment with `.[dev]` installed:

```bash
python -m pytest -q
QUADRIVIUM_NO_ACCEL=1 python -m pytest -q
python tools/gen_docs.py --check
python tools/gen_figures.py --check
python -m mkdocs build --strict
python -m build
python -m twine check --strict dist/*
```

Inspect the built site and distribution contents. Confirm that the wheel has the
compiled extension and that the source archive includes every required C
source/header. Keep old artifacts out of the set selected for publishing.

Install the candidate wheel into a fresh environment and run a smoke test from
outside the source tree. This catches accidental imports from the checkout:

```bash
python -m venv /tmp/quadrivium-release-check
/tmp/quadrivium-release-check/bin/python -m pip install dist/quadrivium-*.whl
cd /tmp
/tmp/quadrivium-release-check/bin/python -c "import quadrivium as qd; print(qd.__file__); print(qd.__version__); print(qd.accel.show_config()); assert qd.brent(lambda x: x*x-2, 0, 2).converged"
```

The wheel glob should select the single candidate appropriate to that
interpreter. In a multi-platform artifact directory, select the exact wheel.
Also install and test the source archive in a separate scratch environment.

## Understand the release workflow

`.github/workflows/release.yml` accepts either a pushed `v*` tag or a manual
workflow dispatch. A manual run selects `testpypi` or `pypi`; it is a publishing
action, not merely a build preview.

| Job | Responsibility |
| --- | --- |
| `build` | Validate tag/version agreement for tag events, run tests, build/check the sdist, smoke-test its installation |
| `wheels` | Build and test platform/interpreter wheels using the workflow's cibuildwheel matrix |
| `publish-testpypi` | Publish a manual TestPyPI run after build and wheel jobs complete |
| `publish-pypi` | Publish a release tag or manually selected PyPI run; produce attestations and a GitHub Release for tags |

Read the workflow's current matrix before a release; it is the source of truth
for which interpreters and platforms are built. The package's source metadata
and the release automation need to agree.

## Configure publishing and documentation

The workflow uses OIDC Trusted Publishing. Repository, workflow filename, and
environment names must match the configured publisher. The checked-in workflow
uses `pypi` and `testpypi` environments for their respective indexes.

Confirm those account settings before triggering a release; a workflow file
alone does not demonstrate that external account configuration is complete.
No API token is needed by the publishing jobs as written.

The separate `.github/workflows/docs.yml` builds and deploys the site to GitHub
Pages for its configured triggers. It checks generated references and runs a
strict site build. Its trigger is not automatically equivalent to every release
tag, so verify documentation publication independently of package publication.

## Publish a reviewed candidate

Commit the reviewed source, release notes, and generated assets. The tag must
be exactly `v` followed by the packaged version. For example, if the candidate
version were `1.2.0`, the release commands would be:

```bash
git tag -a v1.2.0 -m "Quadrivium 1.2.0"
git push origin v1.2.0
```

These commands trigger publication through the release workflow. Confirm the
commit and version before using them. A manual TestPyPI run can rehearse the
publishing path with its own publisher configuration.

After publication, verify the uploaded version, expected artifact set, a fresh
installation, release notes, and the documentation site. A successful build is
only one stage of that verification.

## Diagnose a failed release

| Failure | Investigation |
| --- | --- |
| Tag/version mismatch | Compare the tag suffix with `quadrivium.__version__` at the tagged commit |
| Extension missing | Inspect wheel contents, C-source inclusion, and isolated installation logs |
| One wheel fails | Read that platform's compile/test log; do not infer success from other runners |
| Publishing authorization fails | Check the configured OIDC publisher against owner, repository, workflow, and environment |
| Generated docs are stale | Regenerate from the intended checkout and inspect the diff |
| Figure check differs | Compare the recorded rendering environment and numerical source changes |
| Package published but docs old | Inspect the separate Pages workflow and its triggers |

Treat publication as a persistent external action. If an artifact has already
been distributed, preserve the relationship between its version, source tag,
and release notes; prepare a corrective release rather than silently rewriting
that history. Coordinate any index-side correction with the maintainer's
release procedure.
