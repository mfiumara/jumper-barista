# M4R physical-interface representation contract, version 1

Status: supervisor-adopted measurement contract for card `t_570b4632`, based on the
independent D3 review at `c1b1d30be375f6464675c53ab8ad5e864c8a352d`. This is not M4R
motion acceptance. Implementation base: `674b04fec565f4c728f46c2a9ad56815cc050ae4`.
Independent exact-SHA review belongs to child `t_30acb44f`.

The binding source is `attachments/t_0bd129cc/REVIEW.txt`, lines 46–54 on the
`jumper-barista` board, SHA256
`ef20f1743040aef95638375b047a4a9d0eb56b5f5b97d3ef53cc0ea9df0a662a`.
The card explicitly adopts that proposal; the source report itself remains unchanged.

## Adopted text

The intentional grasp consists of exactly two physical interface pairs:

- F: LF finger grip insert against bar_top_grip_collar_a.
- P: LF palm grip insert against bar_top_grip_collar_b.

For reporting of intentional physical interfaces only, F is represented by the exact geom set {robot/LF_finger_grip_insert_link_meshcol, robot/LF_finger_grip_insert_link_visual} paired only with bar_top_grip_collar_a; P is represented by {robot/LF_palm_grip_insert_link_meshcol, robot/LF_palm_grip_insert_link_visual} paired only with bar_top_grip_collar_b. These map four geometric pairs to two physical interfaces. This is duplicate-representation accounting, not a whole-claw waiver or a hardware/mesh modification.

Only those four exact geometric pair keys are exempt from the >=0.005 m non-grasp clearance requirement. The visual aliases never establish grip; the original two collision pairs must provide physical opposing contact/retention, with penetration depths and forces reported and reviewed for physical plausibility. This mapping introduces no permission for unbounded penetration or lost grasp.

Every other robot-to-external-pitcher pair remains in scope with >=0.005 m clearance at every spiral physics substep, including both inserts against the opposite collar, bar, shelf, struts and all other pitcher parts; all other palm/finger bodies, pads, tips and collision pieces; display_module; and all remaining robot geoms, including their visuals. The original external-pitcher membership (including visual spout, excluding only the established internal water proxies) is unchanged. There is no inheritance by body, prefix, parent, contact proximity or shared material, and no additional alias inferred at runtime.

Keep an unrestricted all-pair minimum alongside the explicitly mapped non-grasp minimum. Retain the two original collision-pair identities separately in the physical-contact report. Every non-exempt actual robot/pitcher contact remains checked; measured contact depth takes precedence and must not be replaced by a positive distance or hidden by aliasing. In particular, finger-insert/safety-shelf contact remains forbidden and fails this requirement. Preserve existing brewer/environment checks, run_m4 gates, geometry, contact masks/friction/solver settings and all hardware thresholds. Before any future implementation is accepted, exact-name and negative-case tests must prove that shelf, opposite-collar, palm-body and display pairs remain non-exempt.

This proposal requires explicit supervisor authorization, versioned evidence identifying the adopted mapping and an independent final exact-SHA review. No existing replay becomes accepted through this proposal alone.

## Reporting implementation

`barista/verification/m4_pair_clearance.py` reports contract version
`m4r-two-interface-v1`. `INTENTIONAL_GRASP_PAIRS` retains the two original collision
identities. `INTENTIONAL_INTERFACE_PAIRS` explicitly associates each with its one
visual alias; `EXEMPT_GEOM_PAIRS` contains exactly the resulting four pairs.
No body/prefix lookup expands that list. An unknown name receives no exemption;
a missing name lookup or ambiguous duplicate name fails rather than adding aliases.

Existing unrestricted, non-grasp and non-exempt actual-contact minima remain.
`physical_collision_contacts` separately reports each original collision pair's
contact-record count, deepest measured distance and time, maximum normal force per
contact record, and force-sample count. A missing force callback leaves force null,
not zero. Counts accumulate contact records across observations, not durations or
retention certificates. `run_m4.py` supplies `mj_contactForce`'s normal component
through the new callback. Visual aliases supply neither grip counts nor forces.
This report does not assess whether contact depth, opposing force, or retention is
physically acceptable; that still needs independent motion review. There is no new
acceptance boolean or altered original gate.

## Bounded verification

On macOS arm64, Python 3.12.13, MuJoCo 3.11.0, native:cpu, offscreen CGL:

- Focused `tests/test_barista_m4r.py`: 57 passed.
- Combined M4R, release and grip-attachment tests: 68 passed.
- Changed Python files: Ruff passed.
- Original `run_m4.py` AST matches the base exactly after removing only the new
  force-reading helper and the one reporting callback keyword. Thus all original
  gates, thresholds, distance methods, geometry selection and motion code are intact.
- Preserved c1b1d30 NPZ SHA256:
  `fb1e56bd63c1a0b01ad5f7f64c685abe84acf096284e460a7c2f1705b85d4b94`.
- Reconstructed qpos frames 1757 and 1808, approximately 35.141 and 36.161 s,
  with `mj_forward`, not `mj_step`. Each checks 1602 robot/external-pitcher pairs;
  exactly four are exempt. Respectively 23 and 22 remaining pairs fail 5 mm.
- At 35.141 s, palm-body visual/collar_b remains -4.565034361 mm by the confirmed
  convex-distance metric. Finger-insert/shelf actual contact is -0.375275933 mm.
- At 36.161 s, palm-body visual/collar_b remains -4.486663873 mm; the actual
  finger-insert/shelf contact is -0.384778271 mm. Neither is exempt.

Artifacts and the reproducible verification script are under
`out/barista/interface-map/`. The pair table retains each measurement method;
negative visual convex distances alone do not prove nonconvex material overlap.
The labelled offscreen image is a diagnostic, not visual proof of clearance.

Saved NPZ states carry qpos, not original dynamic contact forces. The diagnostic
forces use zero qvel/control and the existing film model's defaults; they exercise
reporting only and are explicitly labelled as reconstructed. The model emits its
existing timestep/integrator attachment warnings; no dynamics uses those defaults
in this check. No new replay, sweep, torque audit, retention or hardware claim is
made. Payload, the servo curve, 1.2 N m continuous budget and 300 ms peak-duration
limit remain unchanged. Water remains a particle/visual approximation; everything
here is simulation only. No new third-party asset or licence claim is introduced.
