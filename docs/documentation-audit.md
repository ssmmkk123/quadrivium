# Documentation audit and scope map

This audit inventories handwritten documentation, classifies page roles, and identifies coverage gaps for subpackages and cross-cutting topics.

## Scope inventory

### Root-level docs

| File | Classification | Purpose |
| --- | --- | --- |
| `README.md` | overview + reference | Project summary, installation, high-level usage, doc links |
| `CONTRIBUTING.md` | policy/process | Contribution workflow, tests, docs regeneration expectations |
| `SECURITY.md` | policy/process | Security reporting process |
| `SUPPORT.md` | policy/process | Support and issue routing |
| `CHANGELOG.md` | reference | Release history |

### Handwritten docs under `docs/`

| Page | Classification |
| --- | --- |
| `index.md` | overview |
| `installation.md` | tutorial/reference |
| `getting-started.md` | tutorial |
| `design.md` | reference |
| `limitations.md` | reference |
| `examples.md` | tutorial |
| `faq.md` | reference/decision guide |
| `contributing.md` | policy/process |
| `release-process.md` | policy/process |
| `guides/core.md` | decision guide |
| `guides/linalg.md` | decision guide |
| `guides/rootfind.md` | decision guide |
| `guides/interpolate.md` | decision guide |
| `guides/approx.md` | decision guide |
| `guides/diff.md` | decision guide |
| `guides/integrate.md` | decision guide |
| `guides/ode.md` | decision guide |
| `guides/pde.md` | decision guide |
| `guides/optimize.md` | decision guide |
| `guides/transforms.md` | decision guide |
| `guides/stochastic.md` | decision guide |
| `guides/special.md` | decision guide |
| `api/index.md` and `api/*.md` | API index/reference (generated) |

## Subpackage coverage gap map

| Subpackage | Current strength | Gap before this overhaul |
| --- | --- | --- |
| `core` | conventions and shared types are documented | lacked visual map of result record anatomy |
| `linalg` | rich method coverage and comparisons | lacked quick visual chooser and convergence figures |
| `rootfind` | scalar/system/polynomial guidance present | lacked robustness and trajectory visuals |
| `interpolate` | broad interpolation method guidance | lacked Runge/smoothness visual anchors |
| `approx` | fitting/minimax narrative present | lacked comparative error-envelope figures |
| `diff` | AD vs finite/spectral guidance present | lacked truncation-roundoff tradeoff visuals |
| `integrate` | quadrature families and pitfalls explained | lacked error-order slope and adaptivity visuals |
| `ode` | stiffness and method families well described | lacked stability-region and drift-summary visuals |
| `pde` | PDE family guidance and pitfalls present | lacked CFL/stability and conservation visuals |
| `optimize` | broad optimization map present | lacked path/region and local-vs-global visuals |
| `transforms` | broad transform coverage present | lacked spectrum-domain interpretation visuals |
| `stochastic` | generators/sampling/MCMC/SDE present | lacked diagnostics visuals (ACF/ESS) |
| `special` | wide function coverage present | lacked identity-verification plot examples |

## Cross-cutting coverage gaps

| Topic | Existing coverage | Gap addressed |
| --- | --- | --- |
| Performance | discussed in getting-started/README | needed more visual cost-accuracy framing |
| Stability | described in narrative examples | needed explicit boundary/region figures |
| Failure modes | discussed in limitations/pitfalls | needed clear “use X instead of Y” visuals |
| Reproducibility | `rng=` and deterministic examples documented | needed reusable checklist figure and standards |

## Outcome target

- Every guide has explicit visual evidence sections.
- Core pages include orientation figures that reduce time-to-first-correct-method.
- Asset management and quality gates keep visuals reliable and non-drifting.
