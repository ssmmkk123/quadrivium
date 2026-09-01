# Release process

How a version of quadrivium reaches PyPI. This page is for maintainers; if you
are installing the package, see [Installation](installation.md).

Releases are automated: pushing a `v*` tag builds the distributions, checks
them, publishes to PyPI over OIDC with no stored credentials, and creates a
GitHub Release with the artifacts and their build attestations attached.

## One-time setup

### 1. Claim the name on PyPI

The distribution name is `quadrivium`, which was unregistered when the project
was named. Names are first come, first served, so check it is still free
(<https://pypi.org/project/quadrivium/> should be a 404) and claim it with the
first upload. Publishing to a project someone else owns fails with
`403 Forbidden`; the fix is to change `name` in `pyproject.toml` and the
install commands through the documentation, not to retry.

### 2. Configure Trusted Publishing

The project does not exist on PyPI until the first upload, so add a **pending**
publisher: *Your projects → Publishing → Add a pending publisher*. PyPI creates
the project the first time that workflow uploads to it.

| Field | Value |
| --- | --- |
| PyPI project name | `quadrivium` |
| Owner | `ssmmkk123` |
| Repository | `quadrivium` |
| Workflow | `release.yml` |
| Environment | `pypi` |

After the first release the same entry appears under the project's own
*Publishing* settings, where later changes are made.

### 3. Create the GitHub environment

In the repository settings, create an environment named `pypi`. Adding a
required reviewer to it means every publish waits for a human approval, which
is worth the extra click.

No API token is stored anywhere. The workflow authenticates with a
short-lived OIDC identity issued to that specific workflow in that specific
repository.

### 4. Turn on GitHub Pages

The documentation deploys itself from `.github/workflows/docs.yml`, but the
first deployment needs Pages enabled: repository *Settings → Pages → Build and
deployment → Source: GitHub Actions*. After that, every push to `main` that
touches `docs/`, `mkdocs.yml`, the package, or the changelog rebuilds the site
at <https://ssmmkk123.github.io/quadrivium/>.

If the site will live somewhere else, change `site_url` in `mkdocs.yml`, the
`Documentation` URL in `pyproject.toml`, and the links in `README.md`.

## Cutting a release

### 1. Decide the version

[Semantic versioning](https://semver.org): patch for a fix that changes no
API, minor for new methods or new optional arguments, major for anything that
could break an existing call. A change to what a method *returns* numerically —
a corrected convergence rate, a different default — is at least a minor
release, and belongs in the changelog under *Fixed* or *Changed* with the
before-and-after stated.

### 2. Update the version and changelog

The version has one source of truth:

```python
# quadrivium/__init__.py
__version__ = "1.2.0"
```

`pyproject.toml` reads it from there at build time, so nothing else needs
editing. Move the `Unreleased` entries in `CHANGELOG.md` under the new version
with today's date, add the comparison link at the bottom, and regenerate the
copy the site serves:

```bash
python tools/gen_docs.py
```

### 3. Check everything locally

```bash
python -m unittest discover -s tests     # the full suite, all green
python tools/gen_docs.py --check         # generated pages match the code
mkdocs build --strict                    # no broken links or missing pages
python -m build                          # sdist + wheel into dist/
python -m twine check dist/*             # metadata renders on PyPI
```

Then install the built wheel into a clean environment and import it from
somewhere other than the source tree, which is the check that catches a
missing subpackage in the wheel:

```bash
python -m venv /tmp/relcheck
/tmp/relcheck/bin/python -m pip install dist/quadrivium-*.whl
cd /tmp && /tmp/relcheck/bin/python -c "import quadrivium; print(quadrivium.__version__)"
```

### 4. Rehearse on TestPyPI (optional)

The release workflow can be run manually from the Actions tab with
`target: testpypi`, which publishes to <https://test.pypi.org> instead. Install
from there with the real index still available for dependencies:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ quadrivium
```

TestPyPI needs its own trusted publisher, configured the same way, with the
environment named `testpypi`.

### 5. Tag and push

```bash
git commit -am "Release 1.2.0"
git tag -a v1.2.0 -m "quadrivium 1.2.0"
git push origin main --follow-tags
```

The tag must match the version in `quadrivium/__init__.py`; the workflow checks
this and stops if they disagree, which is what prevents a mislabelled release.

### 6. Watch it land

The workflow builds, tests the wheel on a clean runner, publishes, and creates
the GitHub Release. Then:

- confirm the new version at <https://pypi.org/project/quadrivium/> and that
  the README renders,
- `pip install quadrivium==1.2.0` in a scratch environment,
- check the documentation site rebuilt at
  <https://ssmmkk123.github.io/quadrivium/>.

## If something goes wrong

**A release cannot be replaced.** PyPI permanently reserves a version number:
deleting a release does not free it. If a published version is broken, yank it
(which hides it from new installs while leaving existing pins working) and
publish a fix as the next patch version.

**The tag was wrong.** If the workflow failed before publishing, delete the tag
locally and remotely, fix the problem, and re-tag:

```bash
git tag -d v1.2.0
git push --delete origin v1.2.0
```

Once a version is on PyPI, this is no longer an option — go forward instead.

**The upload was rejected with 403.** Either the trusted publisher does not
match (check owner, repository, workflow filename, and environment name
exactly) or the project name belongs to someone else.

## What ships

The wheel contains the `quadrivium` package and nothing else. The sdist adds
the test suite, the examples, the documentation sources, and `tools/`, so a
downstream packager can build and validate the release from it alone:

```bash
tar tzf dist/quadrivium-*.tar.gz | head -20
python -m unittest discover -s tests      # runs from an unpacked sdist
```
