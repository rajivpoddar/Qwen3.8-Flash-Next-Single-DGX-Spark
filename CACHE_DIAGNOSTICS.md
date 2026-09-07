# Bounded prefix-cache diagnostic mode

Run `MAX_NUM_SEQS=6 VLLM_HIT_DEBUG=1 ./start.sh` during an authorized maintenance
restart. This retains the model, MTP, KV precision and chunk settings. With the
flag absent, no diagnostic overlays are mounted. No cache policy changes here.

The instrumentation adapts the four observation points in
https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound/blob/main/src/patch_hit_debug.py
(Apache-2.0) to this fork's existing retention overlays and adds request-level
prefix fingerprints. No prompt text, raw token IDs or credentials are logged.

Each module expires its logging 30 minutes after its first diagnostic event and
caps each event category at 2,000 events. Expiry stops logs without another
restart; remove the environment flag at the next planned restart to remove all
overlays. These are diagnostic limits, not service limits.

Look for `[hit-debug]` in the container logs:

- `request`: request ID, prompt length, cached length, shared-prefix boundary,
  SHA-256 fingerprints of the first 4K/32K/full prompt token sequence.
- `reconcile`: group hit lengths and the combined cache hit. These are lengths
  after reconciliation, not a claim that the shortest group alone caused a miss.
- `publish`: real/hashed recurrent checkpoints in the current publication range.
- `evict`: cached blocks removed, with their group IDs.
- `chunk`: actual scheduled prompt boundaries.

Compare consecutive requests from the same session, not unrelated fleet-wide
totals. Changed early fingerprints indicate changed input; stable fingerprints
with low reuse require examining checkpoint publication and eviction. No single
log point by itself proves causality. A cold first request is expected to miss.

Validation: `python3 tests/test_hit_debug.py` after source extraction. This checks
anchor drift, gate bounds/expiry, and AST equality of original classes after
removing diagnostic statements. Real API/slot smoke checks are still required.

## First live trace, 2026-09-07

Concurrency restored to 6, long-prefill threshold 1024, same pinned image/model.
Authenticated model listing and Anthropic `OK` inference passed. S1 and S2
returned to productive work before further slot admission.

IMPORTANT correction to the earlier offline applicability assessment: the live
split function logs `block=8`, not 1664. With generic grid 8 and recurrent grid
1664, the #54076 candidate changes the end at start=1024/budget=1024 from 2048
to 1664, and at start=142336 from 143360 to 143104. The prior identical-case test
used generic=1664 and therefore does NOT establish a no-op on this runtime.
This was the pre-fix diagnostic baseline; see the geometry repair below.

First resumed S1 request: 145111 prompt tokens, cold. Follow-up: 147353 prompt,
31616 cached, shared-prefix boundary 44928. First-4K and first-32K fingerprints
were unchanged across these requests. S2 follow-up: 202257 prompt, 59904 cached.
This identifies poor live reuse but does not prove which mechanism caused it.

## Mamba geometry repair

`patch_mamba_geometry.py` narrowly backports the invariants from vLLM
[54076](https://github.com/vllm-project/vllm/pull/54076) and
[53798](https://github.com/vllm-project/vllm/pull/53798):

- Scheduler derives the recurrent block size from actual initialized Mamba
  specs (including wrapped specs), and stops at every crossed state boundary.
- V2 worker binds its cache spec before request admission and seeds resumed
  state indexes using that spec, not the generic minimum block size.
- Existing partial-tail, shared-prefix and deferred-free behavior is retained.
  No cache size, retention, precision, concurrency or MTP tuning is included.

`python3 tests/test_mamba_geometry.py` runs four focused CPU tests. The live
synthetic probe `python3 tests/probe_cache_geometry.py` tests cold, repeat,
tool-result follow-up and shorter-branch output. Its host/key paths are local
test configuration, not a deployment requirement. Cache hit and output safety
must both be checked; high hit rate alone is not sufficient proof.

The geometry overlays are always mounted; diagnostic overlays remain opt-in
and compose on top. Rollback uses the saved pre-geometry launcher and prior
diagnostic generator, without changing model assets or slot transcripts.

Live deployment 2026-09-07: startup logged `generic=8 state=1664`. Four synthetic
cases returned exact `CACHE_OK`, end_turn: cold 13242 tokens/10.160s; repeat
9984 cached + 3258 fresh/2.144s; tool follow-up 9984 cached/2.147s; shorter
branch 3328 cached/2.094s. Seven CPU tests passed (geometry + diagnostics).
This is a within-patched-runtime cold/warm comparison, not an old/new A/B.

Residual: S2's real follow-up reused only 59904 of 209522 tokens; overlapping
long agent prefills still reduce decode throughput. Do not call the full fleet
cache/congestion issue resolved from the synthetic probe. Model settings and
memory budget stayed fixed, but startup's profiled KV allocation changed from
970957 to 927023 tokens, another reason not to treat fleet snapshots as a
controlled performance A/B.
