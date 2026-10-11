//! `jumper.barista` in the controller: the `jumper.five_foot` hook, unchanged.
//!
//! The barista task is the five-foot world (same observations, same 16-joint
//! action, same carried left-front claw) with an audited servo runtime and the
//! coffee-bar scene. Its controls are therefore the five-foot task's, and the
//! hook that answers them -- the claw closed on `claw_left` / `claw_right` and
//! the arm presets -- is that task's own. Re-exporting it, rather than copying
//! it, keeps one implementation of the squeeze limit and the mirror.
//!
//! Simulation-only project: the bundled mode is the five-foot stance with the
//! claw carried. The spiral pour itself is the reference replay
//! `barista/verification/run_m4.py`; it is not a mode of this controller.

pub use super::task_jumper__five_foot::build;
