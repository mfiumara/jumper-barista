# H1: posture mirror/reference regression

This is test/reference hygiene, not M4R or A4 motion acceptance. Simulation only.
No policy, geometry, hardware constant, torque model, or runtime behavior changed.

## Provenance and scope

- Card: `t_06ac2aba`; branch: `wt/m4r-posture`.
- Exact base: `c1b1d30be375f6464675c53ab8ad5e864c8a352d`.
- Final full SHA and clean status are in the post-commit `verification.json`
  and the card handoff (a commit cannot contain its own SHA).
- Changed paths: `tests/test_posture.py` (reference semantics, analytic controls,
  its 13 inherited lint diagnostics) and this evidence note. No other source edits.
- Independent exact-SHA review is pre-created as `t_31304b41`.

The existing test demanded exact sign reversal after reflecting the footprint but
not its asymmetric HOME reference. At the base, its 10-degree case returned
`-0.17621266841888428` rad against `-0.17453292519943295`, a signed difference of
`-0.0016797432194513295` rad. The original 0.2-mm/6e-5-rad comment was stale.

## Mathematical expectation

Let `n` be the centered reference, `p = R(-t)n`, and `Q` reflect y and exchange
LF/RF, LM/RM, LR/RR. Then `Qp = R(t)Qn`. Define

- `A = sum(n dot Qn)`;
- `B = sum(n cross Qn)`;
- `b = -atan2(B, A)`.

The estimator using the **unchanged** reference reads `-t + b` (modulo the
principal-angle branch), not `-t`. Reflecting **both** footprint and reference
instead gives `-t`. A symmetric reference gives `b = 0`.

Scalar double-precision calculation directly from the nominal coordinates gives
`A = 0.2461804301535`, `B = 0.00041351324666666567`, and
`b = -0.001679714579699413` rad. At 10 degrees the analytic expectation is
`-0.17621263977913237`; the production float32 estimate differs by only
`-2.863975190336987e-08` rad. Full values at -15, 0, 10, and 45 degrees are in
`analytic.json`.

The synthetic control starts from `(x, +/-1)` for `x = 1, 0, -1`, then shears
`x` by `k*y`. Its closed-form oracle is `A = 10 - 6*k²`, `B = 12*k`:

| Shear k | Bias b (rad) |
|---|---:|
| 0 | 0 |
| -0.25 | 0.3021452235394375 |
| 0.25 | -0.3021452235394375 |

All three are checked at four angles with a translated footprint. Tests verify
unmirrored angle recovery, fixed-reference bias, and mirrored-reference sign
reversal. The nonzero-shear controls explicitly reject naive negation. Monkeypatch
replaces only the reference lookup during each test; no cached tensor is mutated.
The independent `posture4` sign-table assertion is retained. The nominal absolute
tolerance is tightened from 1e-4 to 1e-6 rad, with relative tolerance disabled;
synthetic float64 controls use 1e-12 rad, also with relative tolerance disabled.

This does **not** make the real robot mirror-symmetric. The runtime augmentation's
ideal sign table remains unchanged; arbitrary deformations of an asymmetric robot
are not thereby certified exactly equivariant. Changing training or hardware
semantics would require a separate card.

## Scoped lint rationale

- Two inactive E402 `noqa` directives removed; no new suppression introduced.
- Percent formatting replaced by the same four-decimal f-string formatting.
- The local export fixture's mutable posture default becomes `None`, allocating
  the same dictionary when omitted. Explicit `{}` still reaches the refusal test.
- Seven mutable class-fixture dictionaries receive `ClassVar` annotations. This
  preserves their deliberate shared lifetime; no fixture storage is relocated.
- Two local import blocks are sorted/separated; imports and assertions remain.

## /poteto-mode activation receipt

Explicit activation came from the card body. Before edits, this run read:

- `/Users/mattiafiumara/.claude/skills/poteto-mode/SKILL.md`;
- `/Users/mattiafiumara/.claude/skills/pstack/SKILL.md`;
- `/Users/mattiafiumara/.claude/skills/pstack/playbooks/bug-fix.md`.

These are the globally installed portable skills, followed through native Hermes
tools, not an invented shell command. The checkout and exact anchor were verified
clean first. Checklist:

1. Reproduce: original focused suite at clean base, 1 failed / 62 passed.
2. Isolate: compare fixed vs reflected reference; independently derive scalar and
   synthetic shear oracles. A read-only delegated mathematical review agreed;
   its claims were checked by local execution, not treated as proof.
3. Fix: smallest test-only correction plus the explicitly owned lint diagnostics.
   Implementation remains with the card's sole writer; no delegated source writes.
4. Verify: rerun the same suite, original base source, analytic controls, scoped
   Ruff, complete diff and no-runtime/hardware-change checks.
5. Failing-before-fix evidence: the immutable base already contains the failing
   test, so no redundant failing-test commit was made.
6. Publication skipped: pushes/PRs are prohibited. Independent review is the
   pre-created board child, not a second same-card review lane.

## Reproduction and real outputs

Working directory:
`/Users/mattiafiumara/repos/jumper-barista/.worktrees/t_06ac2aba`.

Evidence directory: `out/barista/m4r_h1/` under that worktree.
Interpreter P: `/Users/mattiafiumara/repos/jumper-barista/.worktrees/t_2c59bb36/.venv/bin/python`.

The existing Python 3.12.13 dependency environment was borrowed read-only, with
`PYTHONPATH=$PWD:$PWD/rl`, `PYTHONDONTWRITEBYTECODE=1`, `RUFF_NO_CACHE=true`.
Actual module roots for tasks, mjlab, mjrl and posture.state were asserted by
inspection to be this worktree and recorded in `verification.json`. No pip install
or environment/toolchain repair was performed. System Python is 3.9.6 and was used
only for the standard-library setup detector. The detector reported native:cpu,
missing CycloneDDS and per-checkout NPU headers; this is not a full setup-gates
pass or deploy-provenance repair. Native CPU environment tests ran successfully.

`P out/barista/m4r_h1/verify.py` exits 0 and records argv, cwd, child exit codes,
source hashes and logs in `verification.json`. Its checks are:

| Command (P is the interpreter above) | Exit | Result |
|---|---:|---|
| `git show <base>:tests/test_posture.py` | 0 | Exact original source checked against detached base checkout |
| `P -m pytest tests/test_posture.py -q -p no:cacheprovider` in detached base checkout | 1 | 1 failed, 62 passed; original mirror mismatch |
| `P -m ruff check --stdin-filename tests/test_posture.py -` with exact base source | 1 | Same 13 inherited diagnostics |
| `P -m pytest tests/test_posture.py -q -p no:cacheprovider` | 0 | 79 passed, 19 warnings |
| `P -m pytest tests/test_posture.py -q -k mirror -p no:cacheprovider` | 0 | 17 passed, 62 deselected, 1 warning |
| `P -m ruff check tests/test_posture.py --output-format concise` | 0 | All checks passed! |
| `git diff --check <base>` | 0 | Empty |
| `git diff --exit-code <base> -- assets tasks scenes rl deploy` | 0 | Empty; no hardware/runtime changes |

Initial clean-base logs are `before-tests.log` and `before-ruff.log`; the repeatable
comparison is `baseline-posture.log` / `final-posture.log`. `complete-diff.log`
contains the reviewed patch. The final evidence archive is attached to this card.

The repeat baseline uses a detached worktree created by
`git worktree add --detach out/barista/m4r_h1_base <base>` (exit 0), with its own
cwd and `PYTHONPATH=<base-worktree>:<base-worktree>/rl`. Its exact HEAD, clean
status and actual import roots are recorded separately. An initial attempt to
rerun the entire original file after extracting it under `out/` caused two extra
`FileNotFoundError` failures: the export tests locate `scripts/` relative to their
own file. That attempt is not a valid baseline comparison; its log and manifest
are retained as `relocated-baseline-invalid*`. The original clean-base run was
unaffected. The runner now verifies the proper layout and checks the actual
`1 failed, 62 passed` result, not merely an expected nonzero exit code.

Known warnings: inherited TorchScript deprecation and MuJoCo attachment timestep/
integrator warnings. No tests skipped or xfailed. No full-suite or whole-repository
lint claim. No new policy replay, render, torque audit or motion validation was
performed for this test-only repair; existing focused tests do exercise native CPU
environments. M4R/A4, torque/render evidence and all other hygiene remain gated.
