# HeyDonna single-Spark preparation

Fork baseline: MiaAI-Lab commit `09d4424be2b777818471b9bba8c7775ddd538833`.
The sample pins the ARM64 image digest and HF model revision, binds the Spark
LAN address on port 30000, and reads an existing API key file. No secrets belong
in this repository. Docker receives credentials from the environment, not the
generated command text. Protect the host Docker socket as privileged access.

## Prepare without disruption

### Claude effort policy

The Anthropic adapter maps effort for served model `qwen3.8-flash-next`:
`low` to `low`, `medium` to `medium`, and `high`/`xhigh`/`max`/`ultracode`
to `xhigh`. Omitted or null effort explicitly sets `enable_thinking=false`
(with `low` as the valid internal template effort value). A model-scoped protocol validator
normalizes `ultracode` before the upstream effort enum validates it.
Other model IDs retain upstream behavior. Explicit low effort is not thinking-off.
`start.sh` regenerates and mounts both patches; changes require a service restart.
Slot launchers default to
`CLAUDE_CODE_EFFORT_LEVEL=low` alongside `--effort low`.

Copy `.env.sample` to `.env` and verify the host address and key-file path.
Run `./start.sh --preflight`. This only validates configuration, the exact cached
checkpoint and a locally cached image. Missing assets cause failure; it never
pulls, builds PLE, starts inference, or stops another container.

`--no-launch` is NOT a read-only dry run: upstream preparation can pull the image
and build PLE. Authorize asset downloads/builds separately before running it.
`./download.sh` fetches the pinned checkpoint once authorized. Budget roughly
99 GB for checkpoint storage plus 27 GB for PLE and additional image storage.
Do not perform memory-heavy preparation alongside Ornith without checking headroom.

Keep the initial defaults: native 262144 context, MTP 3, FP8 KV, 16 GiB KV target,
26 GiB host reserve, four scheduled sequences and 2048-token prefill chunks.
Six client slots may queue; this is not a claim of six full-context capacity.

Recurrent state now defaults to `MAMBA_SSM_CACHE_DTYPE=bfloat16` for finer
prefix-cache blocks. Verify the actual block size in startup logs. This is a
precision trade; keep existing memory reserves and validate real work quality.
Set `MAMBA_SSM_CACHE_DTYPE=float32` to restore the previous state precision.

## Cutover (separate live operation)

### Mamba/MTP cache retention backport

`files/vllm-54713.patch` preserves both upstream commits through
`d3bdc5f7cc6011f0ad30b80b417aa6cdbb4217ab` (vllm-project/vllm#54713).
The generator adapts one context-only alignment-field rename for the pinned
8e685d198 runtime, then applies the commits with strict git context checking.
Both generated core modules are mounted read-only; no image upgrade is needed.
Four upstream regression cases fail before and pass after on this image using
`tests/run_mamba_retention_regression.py`; the adapter only translates retention
configuration to the older environment-variable API. Run it CPU-only, supplying
the pinned upstream test file and generated overlays under `/tmp`.
This addresses one sparse-retention miss, not every cause of prefix-cache misses.
Rollback uses the previous launcher without these two mounts, preserving `.env`.

1. Inform PM first. Snapshot active assignments, pane identities, source container
   and settings. Include Codex Router consumers of port 30000 in the drain.
2. Interrupt all six panes using direct tmux and verify idle. Unload MoP to stop
   watchdog/nudges from racing the switch. Preserve work; use PM re-handoff, not
   continuation files or `--continue` on a model change.
3. Stop (do not remove) `ornith-b12x-serve`; launch the prepared recipe. Keep slots
   paused until authenticated health and exact-model checks pass. Then verify
   `/v1/messages` streaming and tool calls with the actual client authentication.
   The launcher readiness check alone does NOT prove Anthropic API compatibility.
4. Update the slot profile/router model ID only after that proof. Fresh-launch
   slots sequentially without `--continue`; restore MoP and have PM re-handoff
   active assignments one by one, verifying progress before the next admission.
5. On errors, OOM or stalled progress, stop this recipe and `docker start
   ornith-b12x-serve`; verify authenticated serving before resuming clients.

This fork does not modify the live slot wrapper, router, MoP, or source service.
Cold-start duration and client compatibility still require on-Spark validation.
