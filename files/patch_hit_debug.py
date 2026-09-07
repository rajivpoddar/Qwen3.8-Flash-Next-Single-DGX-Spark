#!/usr/bin/env python3
"""Bounded, opt-in cache diagnostics adapted from Saren-Arterius's Apache-2.0
patch_hit_debug.py. No cache/scheduler decisions are changed. No prompt text or
token IDs are logged. Generated overlays preserve the existing retention patch.
"""
import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATE = '''
import os as _hit_os
import time as _hit_time
import hashlib as _hit_hash
_hit_enabled = _hit_os.environ.get("VLLM_HIT_DEBUG") == "1"
_hit_started = None
_hit_counts = {}
def _hit_allow(kind):
    global _hit_started
    if not _hit_enabled:
        return False
    if _hit_started is None:
        _hit_started = _hit_time.monotonic()
    if _hit_time.monotonic() - _hit_started > 1800:
        return False
    count = _hit_counts.get(kind, 0)
    if count >= 2000:
        return False
    _hit_counts[kind] = count + 1
    return True
'''

def replace(src, old, new):
    assert src.count(old) == 1, f"Diagnostic anchor drift: {old[:100]}"
    return src.replace(old, new)

def build(name, source, old, new, single=False):
    src = (ROOT / source).read_text()
    if single:
        src = replace(src, 'from vllm.v1.request import Request\n',
                      'from vllm.v1.request import Request\nfrom vllm.logger import init_logger\nlogger = init_logger(__name__)\n' + GATE)
    else:
        src = replace(src, '\nlogger = init_logger(__name__)\n',
                      '\nlogger = init_logger(__name__)\n' + GATE)
    src = replace(src, old, new)
    ast.parse(src)
    return ROOT / (name + '_hit_debug.py'), src

def main():
    out = []
    old = '        num_uncached_common_prefix_tokens = longest_hit_length - hit_length\n'
    out.append(build('coordinator', 'kv_cache_coordinator_54713.py', old, old + '''        if _hit_allow("reconcile"):
            logger.info("[hit-debug] reconcile max=%d hit=%d longest=%d groups=%s prefix=%s",
                        max_cache_hit_length, hit_length, longest_hit_length,
                        hit_length_by_group,
                        _hit_hash.sha256(repr(block_hashes[:4]).encode()).hexdigest()[:16])
'''))
    old = '        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)\n'
    out.append(build('manager', 'single_type_kv_cache_manager_54713.py', old, old + '''        if _hit_allow("publish"):
            _hit_blocks = self.req_to_blocks[request.request_id]
            _hit_slice = _hit_blocks[num_cached_blocks_before:num_cached_blocks_after]
            logger.info("[hit-debug] publish req=%s group=%d tokens=%d scan=%d:%d real=%d hashed=%d retention=%s",
                        request.request_id, self.kv_cache_group_id, num_tokens,
                        num_cached_blocks_before, num_cached_blocks_after,
                        sum(not b.is_null for b in _hit_slice),
                        sum(b.block_hash is not None for b in _hit_slice), retention_interval)
''', single=True))
    old = '        evicted_hashes = self._remove_cached_block_hashes(block)\n'
    out.append(build('pool', 'block_pool.orig', old, old + '''        if evicted_hashes and _hit_allow("evict"):
            logger.info("[hit-debug] evict block=%d groups=%s",
                        block.block_id, [get_group_id(h) for h in evicted_hashes])
'''))
    old = '        end = min((s for s in stops if start < s < end), default=end)\n'
    path, src = build('scheduler', os.environ.get('HIT_DEBUG_SCHEDULER', 'scheduler.orig'), old, old + '''        if _hit_allow("chunk"):
            logger.info("[hit-debug] chunk req=%s start=%d end=%d prompt=%d block=%d",
                        request.request_id, start, end, request.num_prompt_tokens, block_size)
''')
    old = '        return blocks, num_local, shared_prefix_boundary, False\n'
    src = replace(src, old, '''        if _hit_allow("request"):
            _hit_tokens = request.prompt_token_ids or []
            logger.info("[hit-debug] request req=%s prompt=%d cached=%d boundary=%d fingerprints=%s",
                        request.request_id, len(_hit_tokens), num_local, shared_prefix_boundary,
                        [(n, _hit_hash.sha256(repr(_hit_tokens[:n]).encode()).hexdigest()[:16])
                         for n in (4096, 32768, len(_hit_tokens)) if n <= len(_hit_tokens)])
''' + old)
    ast.parse(src)
    out.append((path, src))
    # Validate every anchor before writing any output.
    for path, src in out:
        path.write_text(src)
    print('Validated and generated four bounded diagnostic overlays')

if __name__ == '__main__':
    main()
