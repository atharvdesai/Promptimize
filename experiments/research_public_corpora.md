# Public real-prompt corpora research (Nov 7, 2026)

Goal: find real (non-synthetic) developer prompts to use as independent test data for the
decision gate, and as a candidate source for more diverse training examples. Each candidate
checked for license, consent/provenance, and what hygiene is required before any use.
Nothing has been downloaded or used yet.

## Verdict summary

| Corpus | What it is | License | Consent / provenance | Access | Verdict |
|---|---|---|---|---|---|
| CodeChat (SEAL) | 587,568 real developer-LLM coding conversations (v2.0), 30+ languages | ODC-BY (inherits WildChat) | Derived from WildChat service-consented logs | HuggingFace `Suzhen/CodeChat`, ungated | **Primary candidate** |
| WildChat-1M | 1M real user-ChatGPT conversations | ODC-BY (retroactive from ImpACT, 2024-06-26) | Users consented via notice when using the free WildChat service | HF `allenai/WildChat-1M`, gated (account + terms) | Fallback if CodeChat insufficient |
| ClarQ (ACL 2020) | StackExchange posts + clarification questions + answers | Stated CC BY 4.0; upstream StackExchange is CC BY-SA (mismatch, see risks) | Public forum posts | Google Drive link from repo | Candidate for clarification-question pairs |
| DevGPT (NAIST-SE) | Developer-ChatGPT conversations harvested from public share links in GitHub issues/PRs/commits and HN | **No explicit license** (GitHub API: license null, no LICENSE file) | Public share pages; implicit sharing | GitHub / Zenodo | Hold - license gap |
| Stack Exchange dump | All SO/SE questions; posts closed "needs details or clarity" are moderator-labeled ambiguous coding requests | CC BY-SA 4.0 | Platform ToS, public posts | archive.org dump (very large) | Strong labels, heavy lift; hold unless needed |

## Per-corpus notes

### CodeChat (primary candidate)
- Paper/repo: github.com/Software-Evolution-Analytics-Lab-SEAL/CodeChat (repo MIT; the *dataset* is ODC-BY per HF card).
- HF API check (Nov 7): tags `license:odc-by`, gated=false, 2,485 downloads.
- Real coding workflows: developer prompts + LLM responses with code snippets.
- Before use: extract **user prompt turns only** (never model responses - those are GPT-3.5/4
  outputs and model-output training restrictions apply); dedupe internally and against our
  420-row dataset (fuzzy); screen prompts for PII/secrets; record ODC-BY attribution in the
  dataset README; keep a manifest of source IDs.
- Limitation to record: prompts came from people talking to ChatGPT, not to a
  clarification-gating agent; distribution differs from real gate inputs. Still the closest
  public source of real developer phrasing.

### WildChat-1M (fallback)
- ODC-BY; non-toxic version only (toxic removed 2024-07-22).
- PII-adjacent fields present: hashed IPs, request headers, state/country. If ever used,
  drop those fields at ingestion.
- Gated - requires accepting HF terms with an account.

### ClarQ
- Real StackExchange posts (technical: unix, askubuntu, tex, etc.) each paired with the
  clarification question asked and the answer received - direct evidence of what humans ask
  when a request is unclear.
- License wrinkle: authors state CC BY 4.0, but StackExchange content is CC BY-SA; treat any
  redistribution of derivatives as BY-SA. For internal evaluation-only use with attribution
  this is workable; do not fold it into a redistributed dataset without resolving this.
- Access is a Google Drive folder linked from the repo README.

### DevGPT (hold)
- Genuinely relevant content, but the repo carries no license (GitHub API `license: null`)
  and share-page harvesting has an OpenAI ToS question. Do not use until the authors clarify
  licensing or an alternative with clear rights covers the need.

### Stack Exchange dump (hold)
- Posts closed as "needs details or clarity" are real moderator-labeled ambiguous coding
  questions - arguably the best label source - but the dump is tens of GB and needs
  filtering. CC BY-SA 4.0 is clear. Parked unless CodeChat proves insufficient.

## Hygiene plan (applies to any corpus adopted)
1. User prompts only; never model responses, for license and contamination reasons.
2. License + attribution recorded in `data/README.md` and the trial entry.
3. Dedupe within corpus and against our 420-row dataset (fuzzy near-duplicate check).
4. PII/secret scan on extracted prompts before committing anything derived.
5. Any derived test set gets frozen labels before scoring, same as synthetic probes.
6. Counts and filtering decisions logged in `experiments/trials.jsonl` like any other trial.

## Sources (fetched Nov 7, 2026)
- https://raw.githubusercontent.com/NAIST-SE/DevGPT/main/README.md (provenance method)
- https://api.github.com/repos/NAIST-SE/DevGPT (license: null)
- https://huggingface.co/datasets/allenai/WildChat-1M (ODC-BY, consent model, PII fields)
- https://huggingface.co/datasets/Suzhen/CodeChat + https://huggingface.co/api/datasets/Suzhen/CodeChat (odc-by, ungated, WildChat-derived)
- https://raw.githubusercontent.com/vaibhav4595/ClarQ/master/README.md (CC BY 4.0, StackExchange source)
- https://aclanthology.org/2020.acl-main.651/ (ClarQ paper)

## Review constraints (Nov 7, 2026 - applied before any download or use)

1. Verify upstream collection/consent, exact subset provenance and license scope before
   downloading or using CodeChat or WildChat. An ODC-BY database license does not by itself
   settle rights in individual conversations or privacy.
2. User-only prompts can still contain private or third-party data, and prompts alone may
   omit conversation context needed for valid labels - keep enough context to label
   honestly, exclude the rest.
3. Start with a bounded, minimum-necessary sample for evaluation before any larger use.
4. Exclude sensitive content entirely (category-level exclusion), not just regex scanning.
5. Retain attribution and exact source revisions for anything used.
6. Never upload raw conversations to paid models by default.
7. ClarQ is not usable for training until its license mismatch (stated CC BY vs upstream
   CC BY-SA) is resolved.
8. Treat every corpus as a candidate, not verified clean, until these checks pass.
9. Probe family 2 stays diagnostic-only framing; ongoing total budget ($10 cap) unchanged.

## Verification results (Nov 8, 2026, ~1 AM) - desk research, nothing downloaded

### WildChat upstream collection/consent: VERIFIED (paper arxiv 2405.01470)
- Two-step explicit consent: "User Consent for Data Collection, Use, and Sharing" plus a
  secondary confirmation; chat access only after consenting (paper Appendix B, Figures 8-9).
- PII anonymized with Microsoft Presidio + Spacy NER + custom rules across names, phones,
  emails, credit cards, URLs in 10 languages. IPs hashed; headers reduced to browser +
  accept-language; toxic conversations removed from the current release.
- License ODC-BY (retroactive since 2024-06-26).

### CodeChat derivation chain: VERIFIED for v1 (paper arxiv 2509.10402v1)
- Filter: WildChat conversations whose LLM turns contain triple-backtick code blocks -
  82,845 conversations (9.9% of WildChat), 311,161 turns, 26,085 users (distinct hashed IPs).
- "All other metadata from WildChat is inherited without changes" - NO additional PII
  screening by CodeChat; it relies on WildChat's automated scrub.
- v2.0 (587,568 conversations, current HF main @1681793a6cfb27b68c1f285bd1e78588132bc1c1)
  is NOT paper-documented; only the card describes it. Exact revision to pin if used:
  main @1681793a6cfb27b68c1f285bd1e78588132bc1c1 (no tags/branches pin v1 separately).

### What this does and does not settle
- Settled: a real consent mechanism existed upstream; the derivation filter is simple and
  documented (v1); full conversation context is retained (68% multi-turn), so labels can
  use context rather than prompt-only text.
- Not settled: automated PII scrubbing is not proof of cleanliness (pasted private code or
  secrets can survive NER); ODC-BY is a database license and does not by itself settle
  rights in individual conversations; v2's exact construction is undocumented.
- Standing rules for any use: user turns only (assistant turns are OpenAI outputs - never
  training input); bounded minimum-necessary EVALUATION sample first; category-level
  sensitive-content exclusion; attribution + exact revision recorded; no raw conversations
  to paid models; internal evaluation only - no redistribution of derived sets.
- Verdict: CodeChat passes desk-level checks as a CANDIDATE for a bounded evaluation
  sample. Training use stays parked pending the license-scope question above.
