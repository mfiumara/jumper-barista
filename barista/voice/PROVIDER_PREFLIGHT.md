# Voice provider preflight (A3a, no spend)

Checked 2026-10-11 (CEST) by the jumper-barista engineer, following jumper-design
`docs/providers.md` ("Mandatory service preflight before spending credits", design repo
commit `918d192`). **Nothing was generated and nothing was spent.** Prices, tiers and
promotions below are what the official pages showed on that date; recheck before buying.
Unknown values are written as unknown.

## Key presence (names only, values never read out)

Checked in the environment and in `~/.hermes/profiles/jumper-barista-engineer/.env` by
variable name only (a grep that prints present/absent; the file was not printed):

| variable | documented by | present |
|---|---|---|
| `ELEVENLABS_API_KEY` | ElevenLabs API docs | **absent** |
| `FISH_API_KEY` | Fish Audio SDK/API docs | **absent** |
| `FISH_AUDIO_API_KEY` (card's name; also accepted) | — | **absent** |

The `.env` file exists; no variable whose name contains `ELEVEN`, `FISH` or `XI_` is in
it. With no key there is no account to query, so **account tier, remaining credits and
API entitlement are unknown**. No account call was made.

## What we would generate

Default script (`script.json`): 3 lines, plus the optional greeting and the `exclaim`
alternative = 152 characters / 154 UTF-8 bytes. With ~10 takes per line (voice choice,
accent, timing): about **1,520 characters**. Voice-design previews come on top (credit cost
per preview: unknown).

## ElevenLabs

| item | finding | source |
|---|---|---|
| Plans | Free $0 (10k credits/mo); **Starter $6/mo (30k credits; first month $1 "until Oct 18")**; Creator $22 (121k, $11 first month); Pro $99; Scale $299; Business $990 | elevenlabs.io/pricing |
| Credit cost | Multilingual v2: 1 credit per character; Flash/Turbo 0.5–1 credit/char via API. API pay-as-you-go list: v2 Multilingual / v3 $0.08 per 1k chars; v4 $0.08 (promo $0.022 until Oct 12); Flash/Turbo $0.04 | elevenlabs.io/pricing, elevenlabs.io/pricing/api |
| Commercial use | Terms §(c): Free Users "may only use the Services for non-commercial purposes"; Paid Users "may use the Services for commercial purposes", subject to the Prohibited Use Policy. Starter is the first plan listing "Commercial License" | elevenlabs.io/terms-of-use (Last Updated 09 October 2026), elevenlabs.io/pricing |
| Attribution | Free plan / signed-out: must attribute by putting "elevenlabs.io" or "11.ai" in the title; paid-plan content: commercial, indefinitely, attribution "subject to the agreement". Content generated outside a paid subscription "always requires attribution" | docs billing page elevenlabs.io/docs/overview/administration/billing.mdx (fetched); help article "Can I publish the content I generate on the platform?" (help.elevenlabs.io returned 403 to a script; text taken from the search-engine copy of that page) |
| Social media | Allowed as publishing; a competition entry on X/YouTube should be treated as commercial (prizes, promotion) → needs a paid plan | inference from the terms above |
| Voice licensing | Use a stock (default) voice or a Voice Design voice made from a text prompt. **No cloning** (Instant/Professional Voice Cloning are not needed and not to be used); the Prohibited Use Policy (Last Updated 09 October 2026) forbids "cloning, synthesizing, or impersonating the voice ... of any living individual without their explicit and informed consent". Community Voice Library voices have a separate Voice Library Addendum, which was not verified (the URL tried returned 404) → avoid them | elevenlabs.io/terms-of-use (Supplemental Terms list), elevenlabs.io/use-policy |
| Fit | Multilingual v2 (29 languages) / v3 (70+) can voice English text with an Italian voice, which suits the "cute Italian accent" brief (inference; quality to be judged on real takes in A3b) | elevenlabs.io/pricing/api |
| Our cost | ~1.5k credits of 30k on Starter → **$1 (promo, if bought by Oct 18) or $6** for one month; cancel afterwards (paid-period content stays commercially licensed per the docs) | above |

## Fish Audio

| item | finding | source |
|---|---|---|
| App plans | Free $0 (8k credits/mo, ≤500 chars/generation); **Plus $15/mo monthly or $11/mo billed annually ($132)**, 250k credits, Voice Design, "Commercial use allowed"; Pro $100/mo ($75 annual) | fish.audio/plan/ |
| API | Pay-as-you-go, no subscription: `s2.1-pro` / `s2-pro` **$15 per million UTF-8 bytes**; `s2.1-pro-free` **$0** "for testing, prototyping, development, and smaller businesses, without TTFA or DPA guarantees". Minimum top-up: unknown | docs.fish.audio pricing-and-rate-limits, docs.fish.audio/features/text-to-speech |
| Commercial use | Terms: use is "internal, personal, non-commercial" unless "a user of Paid Services". Plan FAQ: free users personal/non-commercial only; "Premium subscribers can use verified voices (that you own) for commercial purposes". Whether `s2.1-pro-free` output counts as Paid Services: **unknown** | fish.audio/terms/ (Effective date August 18, 2024), fish.audio/plan/ |
| Attribution | No attribution clause found in the terms | fish.audio/terms/ |
| Voice licensing | Commercial use is tied to voices you own: a Voice Design voice (Plus) or your own recording. The public community voice models include clones of real people → **do not use them** | fish.audio/plan/ FAQ |
| Our cost | API: 1,540 bytes × $15/M ≈ **$0.03** (plus unknown minimum top-up); app: one month Plus **$15** | above |

## Recommendation

**ElevenLabs Starter, one month, stock or Voice Design voices, no cloning.** The commercial
licence and the "no attribution on paid plans" position are stated plainly in the current
terms and docs, its multilingual models give an Italian-accented English voice for Jumper,
and our ~1.5k characters fit in Starter's 30k credits many times over. The price is $1 if
bought by 2026-10-18 (first-month promotion shown today), otherwise $6. Fish Audio's API is
cheaper per character, but its commercial status for the free model, its minimum top-up and
its voice-ownership rule are unclear or need Plus ($15) anyway.

For Mattia (gate):

1. Approve the provider and a cap: **ElevenLabs Starter, ≤ $6 + tax, one month, ≤ 10k
   credits for this card**; voices: stock or Voice Design only; no cloning of any real
   person.
2. Subscribe himself and add `ELEVENLABS_API_KEY` to the engineer profile `.env` himself.
3. A3b then makes one read-only `GET /v1/user/subscription` call to record the tier and
   credits (no generation), and only then runs
   `python -m barista.voice.pipeline --provider elevenlabs --allow-spend`.

If no key is added, the film can ship with the `placeholder-local` voices only as a draft;
they are macOS system voices and their use in a published video was not checked here.
