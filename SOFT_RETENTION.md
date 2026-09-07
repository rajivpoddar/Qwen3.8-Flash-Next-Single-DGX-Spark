# Bounded agent prefix retention

Candidate adaptation of vLLM PR #38514, head
`263525226709bc3fb67afca300512d4eea247ae6`, not a wholesale cherry-pick.

Enable with `VLLM_AGENT_SOFT_RETENTION=1`; default is off. The Anthropic Qwen
Next adapter passes a JSON `heydonna_retention` hint through `vllm_xargs` to
SamplingParams and the core Request. Other models and effort mapping are unchanged.
The first 32K tokens receive priority 90 and later history priority 70, with
300-second TTLs. A hashed metadata user_id is an ownership hint, not auth or
cache isolation. Missing metadata does not globally replace another slot's pins.

Unlike upstream's two-queue implementation, blocks remain in the native free
queue. Allocation evaluates current TTL and picks lowest priority, preserving
LRU order for ties, within at most 4096 free candidates. Larger allocations
fall back to native LRU for the remainder. Everything remains evictable; no
reservation, reference-count change, missing-state bypass or new GPU allocation.
The bounded window can miss a better eviction candidate deeper in the queue.
Equal-priority histories still compete, and expired entries are unprotected.
This is not a guarantee that six 200K histories will remain cached.

Block metadata is cleared on eviction/reset and hash identity is rechecked,
avoiding stale priority when a block is recycled. Existing cached-prefix hints
are refreshed once per request/group, not rescanned on every decode step.
Native hybrid cache matching, geometry, sparse retention and CoW remain intact.

Validation: `python3 -m unittest discover -s tests -p test_soft_retention.py`.
Runtime proof must additionally verify Anthropic conversion -> SamplingParams
-> Request -> block priorities with the pinned image, plus real interleaved
cache reuse and output correctness. CPU simulation alone is not fleet proof.
Logs contain aggregate selection counts, never prompts or metadata user IDs.

Rollback: disable the env flag and restart with the preserved launcher. No model
download, slot clearing, or cache protocol migration is needed.
