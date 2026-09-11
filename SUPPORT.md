# Support

Start with the [getting-started guide](docs/getting-started.md) for imports,
arrays, callbacks, results, and tolerances. The [documentation home](docs/index.md)
links to task guides; the [API reference](docs/api/index.md) supplies complete
signatures and class interfaces. For unexpected results, check the
[FAQ](docs/faq.md) and [known limitations](docs/limitations.md).

## Ask a usage question

Use GitHub Discussions if enabled, or open an issue clearly marked as a usage
question. State the problem you are trying to solve, what you have tried, and
which part of the result or interface is unclear. Include a small runnable
example with its imports and data.

## Report a numerical or installation problem

Include:

- Quadrivium version and source commit for unreleased work;
- Python version, operating system, and `qd.accel.show_config()` output;
- input shapes, dtypes, method options, tolerances, and seeds;
- the actual result, status, message, or complete error traceback;
- expected behavior and how you established it independently.

For installation errors, include the relevant build log and `sys.executable`.
For interoperability issues, include the external library's version. NumPy is
not required for ordinary Quadrivium execution, so a NumPy version is relevant
only when the issue involves it.

Check whether the documentation and installed package describe the same
revision: this checkout includes unreleased APIs. See [installation](docs/installation.md)
for import-path and environment diagnostics.

## Private concerns

Use [Security](SECURITY.md) for vulnerabilities and the
[Code of Conduct](CODE_OF_CONDUCT.md) for conduct concerns. Avoid placing private
or sensitive details in a public issue. Development procedures are in
[Contributing](CONTRIBUTING.md).
