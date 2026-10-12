# Native prerequisite checkpoint (M4R-H2b)

Task `t_88ae5e12`, base `db258afa9749d18f8a6250b0d75faad153ad9af6`.
Initial observation on macOS arm64, 2026-10-11. This is a blocked setup checkpoint,
not a successful build, provenance verification, or M4R acceptance.

## Resume checkpoint — 2026-10-12

Resume HEAD: `141c6218782781cd53f0982cbdbd2403b92bb7b2`; tracked tree
was clean. Mattia's acceptance recorded on this card resolves the licence gate
below, but restricts acquisition to `deploy/fsm/vendor/rknpu2/fetch.sh` only.
The unchanged script lists both the header and runtime at lines 25–28 and
downloads each missing entry at lines 39–67. It has no header-only option.
Running it would violate the card's explicit prohibition on runtime download;
bypassing it would violate the newly recorded acquisition restriction.

Required decision: authorize header-only acquisition using the exact URL and
SHA256 already pinned in that script, without fetching the runtime. Alternatively,
authorize a separately scoped header-only script option. Neither change is
assumed here. The licence need not be accepted again.

Poteto-mode activation receipt: read
`~/.claude/skills/poteto-mode/SKILL.md`, its canonical
`~/.claude/skills/pstack/SKILL.md`, and `pstack/playbooks/session-pickup.md`
in full. Applied the portable method directly; no CLI invocation, profile change,
or substitute agent process is required. Session-pickup checklist:

- Recovered the exact card, prior checkpoint and downstream review dependency.
- Read the recent acceptance and compared it with the actual fetch script.
- Verified clean tracked state and exact resume HEAD with Git (exit 0).
- Reconstructed the invariant: header-only, local, pinned, no runtime download.
- Falsified the assumption that the approved script can fetch only the header.
- Stopped the first unproven acquisition step at the conflicting authorization.
- Preserved this updated checkpoint; no costly build was repeated.

No delegation was needed for this bounded scope check. No source download,
build, link/load probe, or provenance test was executed during this resume.
Completeness and negative controls remain unverified. The historical discovery
and first-attempt results below are retained, not represented as new execution.

## Licence gate — stop before SDK download

The public pinned licence at
https://raw.githubusercontent.com/airockchip/rknn-toolkit2/v2.3.2/LICENSE
states:

> BY OPENING OR USING THIS FILE, RECEIVER HEREBY ACKNOWLEDGES AND AGREES

It then sets out warranty disclaimers and the receiver's responsibility to obtain
necessary third-party licences. The task explicitly requires stopping when terms
require user acceptance. Mattia must decide whether to accept the pinned terms
for the intended local header use; reading the terms here does not assert acceptance
on Mattia's behalf. No header or runtime was fetched. Do not run `fetch.sh`: it
also downloads the runtime, which this card prohibits.

The licence text was retrieved with `curl -fLsS` (exit 0). Its SHA256 is
`d846f57d942c7dfdca7b8b54f9e8bb39e1e226790dc4f5ee205d6fd678961720`.
Local receipt: `out/toolchain/receipts/rockchip-v2.3.2-LICENSE.txt` (ignored).
The expected header SHA256 remains unverified because no header was downloaded:
`c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a`.

## Discovery and activation receipt

- Read `CHARTER.md`, repository rules, setup-env skill, the H2 `REPORT.txt`,
  the unchanged provenance test, and existing Rockchip fetch/licence documentation.
- `git rev-parse HEAD`: base above; `git status --short`: empty before edits.
- `uname -m`: `arm64` (exit 0).
- `python3 .claude/skills/setup-env/scripts/detect.py`: exit 0; system Python
  3.9.6 is out of range, `idlc`/headers absent, Rockchip header missing.
  No suggested global installation was executed.
- Existing read-only interpreter
  `/Users/mattiafiumara/repos/jumper-barista/.worktrees/t_2c59bb36/.venv/bin/python
  --version`: `Python 3.12.13` (exit 0). No environment was modified.
- `/poteto-mode` activation was NOT achieved: `command -v pstack` found no command;
  `~/.claude/commands/poteto-mode.md` was absent; filename searches under
  `~/.claude` and `~/.agents` returned no match. A bounded filename search under
  `~/.hermes` found copies in another profile, which were not loaded or changed.
  The supervisor must supply an approved activation entry point for this profile;
  no activation receipt is fabricated.
- `git ls-remote https://github.com/eclipse-cyclonedds/cyclonedds.git
  refs/tags/11.0.1 refs/tags/11.0.1^{}`: exit 0, tag maps to
  `e54e991f75a3e67f8e628da3171122e36ea5b872`. This is remote tag discovery,
  not a downloaded-source hash or licence audit. No CycloneDDS source was fetched.

## Resume contract (not executed)

After explicit gate resolution and mode activation, use only task-owned
`out/toolchain` directories. Inspect CycloneDDS source licences at the exact
commit above before building. Configure Release, native arm64, IDLC ON, testing
and examples OFF; build with at most two jobs. No sudo, global install, shell
profile changes, or Linux `ldconfig` on macOS.

Fetch only the pinned Rockchip header after acceptance, verify its expected hash,
and never commit or redistribute it. Set `PATH` and `CYCLONEDDS_HOME` only in
child environments. Use an isolated `CARGO_TARGET_DIR`, `CARGO_BUILD_JOBS=1`,
and `CARGO_NET_OFFLINE=true`. Record real `idlc` version, installed header/dylib
hashes, and a native link/load probe before calling the toolchain ready.

Run the unchanged default-feature test:
`python -m pytest tests/test_deploy_sources.py::test_the_provenance_dates_everything_the_controller_is_built_from -q`.
No skip, mock, hidden Cargo, or feature reduction is allowed. It must reach the
hook-presence assertion at line 109, completeness at 114, and negative controls
at 117. If a subsequent failure needs source changes, preserve it and hand off.

Current stage: licence review only. No build, link/load probe, or test was run
in this attempt. Previous H2's missing-IDLC failure is not a result of this run.
No motion, replay, torque recertification, deployment, or publication was performed.
