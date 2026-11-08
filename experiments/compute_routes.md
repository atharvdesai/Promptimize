# T4 compute-route verification (Nov 8, 2026, ~1 AM)

Official `laya-train --freeze-encoder` full training is OOM-blocked in this 2 GB sandbox
(measured in T4: SIGKILL while loading the torch model). Three routes checked against
what is actually reachable under existing authority; no new subscriptions or commitments made.

## Route 1: Google Colab free tier (user's existing Google account)
- Official FAQ (research.google.com/colaboratory/faq.html, fetched Nov 8): free GPU access,
  resources "not guaranteed and not unlimited", limits fluctuate; free tier prioritizes
  users actively programming in a notebook. Training in a notebook is the intended use.
- Browser check (Nov 8 ~1 AM): cloud browser is signed OUT of Google on this profile.
  Profile remembers atharvdesai@gmail.com; vault "Google login" entry exists, but Google
  rejected the stored password: "Your password was changed 3 days ago" - the vault Google
  password is STALE (changed ~Nov 5). Sign-in stopped there; no retries attempted.
- STATUS: BLOCKED on stale vault password. Unblock = user re-saves the Google login in the
  vault (or signs in once in a shared session). Then a T4 allocation smoke test
  (open notebook, connect runtime, nvidia-smi, disconnect) can verify the route end-to-end.
- Expected effort after unblock: install laya, upload 332-row train CSV + labels, run
  laya-train --freeze-encoder on a T4, checkpoint to the user's Drive. $0.
- Risk to note: free runtimes can terminate mid-run; training must checkpoint and resume.

## Route 2: Kaggle free GPU (2xT4, ~30 GPU-hours/week per current guides)
- Browser check (Nov 8 ~1 AM): no signed-in Kaggle session on this profile; no Kaggle
  account found in user context. GPU access on Kaggle requires account phone verification.
- STATUS: BLOCKED on account creation - a new commitment that needs the user's decision.
  Not created.

## Route 3: User's Mac (MacBook Air M3, per user context)
- The LAYA training repo ships an Apple Silicon MPS script (see research_laya_training.md).
  Hardware fit (unified memory for model + optimizer) is unverified; no remote access path.
- STATUS: requires the user to run the script locally. Steps are ready in the guide;
  hardware fit unknown until tried.

## Recommendation
Cheapest path with least user friction: user re-saves the Google login in the vault, then
Colab route gets a GPU smoke test and, if it allocates, the real T4 run with checkpointing.
Kaggle only if Colab fails and the user wants the account. Mac script as the offline fallback.

## Sources
- https://research.google.com/colaboratory/faq.html (fetched Nov 8, 2026)
- Kaggle GPU quota: current third-party guides (kaggle.com/docs/efficient-gpu-usage among
  search results; not re-verified against a signed-in account)
- Sign-in state checks: live cloud browser, Nov 8, 2026 ~1 AM
