# Native prerequisite execution (M4R-H2b)

Task `t_88ae5e12`, base `db258afa9749d18f8a6250b0d75faad153ad9af6`.
Initial observation on macOS arm64, 2026-10-11. Continuation `t_dfaf7b4a`
ran on 2026-10-12 from `ba9184d56688b7fe0eafc4f6e672c52d80c114ee`.

## Current result — bounded execution complete, provenance FAIL

The manager's final answer authorizes direct pinned-header-only acquisition.
Mattia's licence acceptance stands; neither gate needs asking again.
Only `rknn_api.h` was fetched, verified and placed in the ignored vendor path.
No runtime was downloaded and the SDK is not committed or redistributed.

CycloneDDS source HEAD was verified as
`e54e991f75a3e67f8e628da3171122e36ea5b872` (11.0.1); source/licence
hashes are in `SOURCES.md`. Configure, build (two jobs), install, `idlc -v`,
architecture/dependency inspection, native probe compile and run all exited 0.
Actual version: `idlc (Eclipse Cyclone DDS) 11.0.1`.
The probe printed `CycloneDDS native link/load and QoS allocation succeeded
(no participant).` It opens no DDS participant or robot bus.

Installed prefix: `out/toolchain/cyclonedds-install` in this worktree.
All installed executables/dylibs inspected are Mach-O arm64. CMake discovered
existing Homebrew OpenSSL; `libddsc` depends on `/opt/homebrew/opt/openssl@3/lib/`
`libssl.3.dylib` and `libcrypto.3.dylib`. No Homebrew command or global install
was run. This prefix is machine-local, not a self-contained distributable.

- `libddsc.11.0.1.dylib` SHA256:
  `dcf70f0c92154e7499dac03fff2a3977dc40b178ead6b5157e19b89472ec3e39`.
- Installed `include/dds/dds.h` SHA256:
  `34ef07e510d2743ed75c86861297267c78271b6f30abccabf41e139862e90257`.
- Rockchip header SHA256:
  `c48e11a6f41b451a5fd1e4ad774ea60252d3d94f78bee9b21ea3d21b21deba9a`.

The unchanged default-feature provenance test exited **1**, `1 failed in 27.44s`.
Cargo build returned 0 (the test passed its line 96 assertion). Hook-presence
at line 109 passed. Completeness at line 114 failed on these generated inputs:

```
out/toolchain/cargo-target/debug/build/mjrl-fsm-e4860612e6df1d20/out/idl_bindings.rs
out/toolchain/cargo-target/debug/build/mjrl-fsm-e4860612e6df1d20/out/task_hooks.rs
```

Negative controls at line 117 were **not reached**. `_uncovered` currently
excludes only `deploy/fsm/target` and paths outside the repository (line 52).
The required isolated target is inside this worktree, so generated files enter
the source comparison. Smallest handoff: supervisor/reviewer must scope a
generated-build-input classification fix or explicitly revise the target-location
contract. No test/Rust/source-list edit, relocated target, or second test attempt
was used to turn this failure green. Successful native build/load is not successful
provenance, full prerequisite acceptance, or M4R acceptance.

### Reproduce (local authorization only)

From this worktree root, first clone and verify source and read its licences:

```sh
git clone --depth 1 --branch 11.0.1 https://github.com/eclipse-cyclonedds/cyclonedds.git out/toolchain/cyclonedds-source
git -C out/toolchain/cyclonedds-source rev-parse HEAD
git -C out/toolchain/cyclonedds-source archive --format=tar HEAD -o ../receipts/cyclonedds-source.tar
```

Create `out/toolchain/receipts` before the archive command. Compare source and
licence hashes with `SOURCES.md` before the build. Fetch only the header URL
recorded there with `curl -fLsS --max-time 60`, to the ignored destination plus
`.part`; verify its SHA256 before renaming to `include/rknn_api.h`. Do not run
`fetch.sh`. Existing valid source/header can be reused without downloading again.

```sh
python3 barista/native_prerequisites.py build
python3 barista/native_prerequisites.py verify
python3 barista/native_prerequisites.py test --python /Users/mattiafiumara/repos/jumper-barista/.worktrees/t_2c59bb36/.venv/bin/python
```

The runner uses only stdlib; test interpreter is existing Python 3.12.13 borrowed
read-only. System Python measured 3.14.7 (outside simulator support range), used
only for the stdlib runner/detector. No venv was modified. PATH, CYCLONEDDS_HOME,
offline Cargo, single Cargo job, isolated target/temp and checkout PYTHONPATH are
child-only. Exact commands/exits are in `out/toolchain/receipts/commands.jsonl`;
all output logs, installed header/dylib hashes and test environment are beside it.
`REPORT.txt` binds final SHA and text-log hashes. No full environment/gates,
simulation/replay/torque validation, deployment or hardware claim is made.

Poteto activation: read alias, canonical pstack skill and session-pickup playbook
in full; applied the portable method directly. Recovered card/authorization,
checked clean resume SHA, reconstructed header-only invariant, falsified prior
missing-prerequisite state by real build/probe/test, and stopped at the first
out-of-scope source fix. No delegation was needed for this serialized bounded lane;
independent review remains the existing downstream card. No CLI/profile change.

## Historical resume checkpoint — superseded by final manager approval

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

## Historical licence gate — resolved before continuation

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

## Historical resume contract (now executed above)

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

At that historical checkpoint the stage was licence review only; no build,
link/load probe or test had run. Current continuation results are above.
No motion, replay, torque recertification, deployment, or publication was performed.
