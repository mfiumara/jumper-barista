# Jumper Barista: the café film (Act 2)

Mattia's brief from 2026-10-11. It builds on the accepted pour at `287861a`. Everything in
`CHARTER.md` still applies: real hardware limits, simulation-only labelling, gates.

## Character

Jumper is an **Italian hipster barista**: moustache, Italian outfit (for example a flat cap
or beret, an apron or waistcoat, maybe round glasses), a cute voice with an Italian accent.
Keep it affectionate and charming, not a caricature.

## Story beats (target 30–45 s, plus a 15–20 s vertical cut)

1. **Shift start.** Jumper stands on a low back counter behind the main bar and jumps up onto
   the bar. *Stretch goal:* the jump policy rises about 0.16 m on flat ground, so landing on a
   ledge needs new work and is not guaranteed. **Fallback:** a celebratory jump on the bar
   itself.
2. **Idle.** It does a silly dance (reuse the existing dance policies) while waiting.
3. **Customer arrives.** We hear the customer's voice. Jumper stops dancing, turns or looks up
   at the customer and listens. The customer is the camera (point of view at bar height); there
   is no 3D human.
4. **Takes the order.** Real speech-to-text (whisper.cpp / mlx-whisper, local) on the
   customer's audio, then an intent parser. The subtitles show the actual transcript.
5. **Confirms.** Jumper says something like *"Order noted — I prepare it now!"* in the cute
   Italian voice, and the eye displays change expression.
6. **Turns 90° and walks along the bar** to the pour-over station (reuse the existing walking
   policy), then aligns and blends into the accepted pour's start pose.
7. **The spiral pour** (accepted M4), polished: warm café lighting, a visible water stream, a
   coffee-coloured bed, a V60 that reads as a V60.
8. Optional button at the end: a proud gesture or bow.

## Voices

- The customer and Jumper both use synthetic voices from **ElevenLabs or fish.audio**.
- **Gate:** these need Mattia's account and API key, and they cost credits. Before any
  generation, check the account tier, commercial/social-use terms, attribution and cost
  (jumper-design `docs/providers.md` gives the preflight). Never print or commit keys. If no
  key is configured, block and ask Mattia to add it to the engineer profile `.env` himself.
- STT runs on the real generated customer audio. Do not hand-write the transcript.

## Outfit (design gate)

Follow the jumper-design workflow: present **2–3 distinct outfit concepts** (renders or
concept images, clearly labelled as concepts) and **wait for Mattia's explicit pick** before
generating geometry. The outfit is display-only `.skin/3` / `visual_overrides` plus accessory
meshes. It must not change collision or physics, and must not foul the claw or legs during the
grasp and pour. Record licences in `SOURCES.md`.

## Music and sound

Use royalty-free or CC0 music and SFX (the café ambience, the pour) only, with licence
evidence in `SOURCES.md`.

## Milestones

| # | Milestone | Target |
|---|---|---|
| A1 | Café scene: back counter + bar + brew station; walking route; spawn and collision checks | 2026-10-14 |
| A2 | Outfit concepts presented to Mattia (gate), then the chosen skin built and rendered | concepts by 10-13, skin by 10-17 |
| A3 | Voice pipeline: provider preflight (gate), customer + Jumper lines, STT transcript, eye expressions | 2026-10-16 |
| A4 | Motion chain: (jump) → dance → stop/look → turn 90° → walk → align → pour, as one continuous sim run within the torque budget | 2026-10-19 |
| A5 | Film: horizontal 30–45 s + vertical 15–20 s, music, subtitles, small simulation note | 2026-10-22 |
| A6 | Independent review; Mattia's taste review (gate) | 2026-10-24 |
