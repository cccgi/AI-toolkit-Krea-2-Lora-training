# Why ai-toolkit-perceptual, not vanilla ai-toolkit

This project trains on
[`BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual`](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual),
a fork of [`ostris/ai-toolkit`](https://github.com/ostris/ai-toolkit),
rather than the vanilla upstream repository.

## What to verify before relying on this choice

Forks drift. Before depending on this recommendation for a new
project, verify against the fork's current commit history and open
issues on both repos:

1. Whether the specific fixes below have since been merged upstream
   into `ostris/ai-toolkit` (in which case vanilla ai-toolkit may now
   be sufficient).
2. Whether the fork has fallen behind upstream on unrelated features
   or security fixes.

## What this fork provided that was needed

This project used `ai-toolkit-perceptual` because, at the time of this
research, it carried fixes and features required to get Krea-2 RAW
character LoRA training working reliably that were not present (or
not stable) in the vanilla upstream branch being tracked at the time:

- Face-mask (`mask_path` / `mask_min_value`) regional loss weighting
  support wired through the Krea-2 model's training path -- this is
  the single highest-impact setting found in this project's research
  (see `docs/KREA2_FINDINGS.md`), and needed to actually apply
  correctly to Krea-2's single-stream MMDiT architecture, not just to
  Flux-family models.
- Dataset latent/text-embedding caching fixes for the Krea-2
  architecture, avoiding intermittent Windows-specific path and
  caching errors seen on vanilla checkouts during this research.

## If you want to try vanilla ai-toolkit instead

Vanilla `ostris/ai-toolkit` may work directly for your use case,
especially if the fixes above have since been merged upstream. To try
it: replace the `REPO_URL` in `setup/00_install_ai_toolkit_perceptual.bat`
with `https://github.com/ostris/ai-toolkit.git`, run the smoke test
(`RUN_ME_train_lora.bat` always runs a 1-step smoke test before the
full run), and check for `mask_path` support errors in the console
output. If the smoke test fails with a `mask_path` or `mask_min_value`
related `KeyError`/`ValueError`, that is a signal you need this fork
(or the upstream fix, once/if merged).
