#!/usr/bin/env python3
"""Compose soft retention with existing geometry, effort and diagnostic overlays."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace(src, old, new):
    assert src.count(old) == 1, f"Soft-retention anchor drift: {old[:100]}"
    return src.replace(old, new)


def build(pool, request, anthropic, manager):
    pool = replace(pool, 'from vllm.v1.request import Request\n',
        'from vllm.v1.request import Request\nfrom vllm.v1.core.retention_policy import SoftRetention\n')
    pool = replace(pool, '        self.enable_kv_cache_events = enable_kv_cache_events\n',
        '        self._soft_retention = SoftRetention()\n        self._retention_log_at = 0.0\n'
        '        self.enable_kv_cache_events = enable_kv_cache_events\n')
    # Refresh at the manager boundary, before its all-hit early return.
    anchor = '        if num_cached_blocks >= num_full_blocks:\n'
    manager = replace(manager, anchor, '''        hints = getattr(request, "soft_retention_hints", None)
        source = getattr(self, "_retention_cow_sources", {}).pop(request.request_id, None)
        if source is not None:
            index, block, old_hash = source
            if block.block_hash == old_hash:
                self.block_pool._soft_retention.mark(block, hints,
                    index * self.block_size, (index + 1) * self.block_size)
        refreshed = getattr(request, "_retention_refreshed_groups", None)
        if refreshed is None:
            refreshed = request._retention_refreshed_groups = set()
        if hints is not None and self.kv_cache_group_id not in refreshed:
            refreshed.add(self.kv_cache_group_id)
            for index, block in enumerate(self.req_to_blocks[request.request_id]):
                self.block_pool._soft_retention.mark(block, hints,
                    index * self.block_size, (index + 1) * self.block_size)
''' + anchor)
    anchor = '        req_blocks[block_idx] = cow_block\n'
    manager = replace(manager, anchor, '''        if not hasattr(self, "_retention_cow_sources"):
            self._retention_cow_sources = {}
        self._retention_cow_sources[request_id] = (block_idx, source_block, source_block.block_hash)
''' + anchor)
    anchor = '        req_blocks = self.req_to_blocks.pop(request_id, [])\n'
    manager = replace(manager, anchor,
        '        getattr(self, "_retention_cow_sources", {}).pop(request_id, None)\n' + anchor)
    anchor = '        if num_cached_blocks >= num_full_blocks:\n'
    pool = replace(pool, anchor, '        hints = getattr(request, "soft_retention_hints", None)\n' + anchor)
    anchor = '            if new_hashes is not None:\n                new_hashes.append(maybe_convert_block_hash(block_hash))\n'
    pool = replace(pool, anchor, '''            self._soft_retention.mark(blk, hints,
                (num_cached_blocks + i) * block_size, num_hash_tokens)
''' + anchor)
    anchor = '        if self.enable_kv_cache_events and not already_cached:\n'
    pool = replace(pool, anchor, '''        self._soft_retention.mark(block, getattr(request, "soft_retention_hints", None),
            num_tokens // block_size * block_size, num_tokens)
''' + anchor)
    pool = replace(pool, '        num_tokens = src_block.block_hash_num_tokens\n',
        '        retention_source_hash = src_block.block_hash\n        num_tokens = src_block.block_hash_num_tokens\n')
    anchor = '            self._insert_block_hash(block_hash, dst_block, num_tokens=num_tokens)\n'
    pool = replace(pool, anchor, anchor + '        self._soft_retention.move(src_block, dst_block, retention_source_hash)\n')
    pool = replace(pool, '        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n',
        '''        ret: list[KVCacheBlock] = self._soft_retention.pop(self.free_block_queue, num_blocks)
        now = self._soft_retention.clock()
        if now - self._retention_log_at >= 60 and self._soft_retention.selections:
            logger.info("[soft-retention] entries=%d selections=%d scanned=%d deferred=%d protected_evicted=%d",
                        len(self._soft_retention.entries), self._soft_retention.selections,
                        self._soft_retention.scanned, self._soft_retention.deferred,
                        self._soft_retention.protected_evicted)
            self._retention_log_at = now
''')
    pool = replace(pool, '        evicted_hashes = self._remove_cached_block_hashes(block)\n',
        '        self._soft_retention.entries.pop(block.block_id, None)\n'
        '        evicted_hashes = self._remove_cached_block_hashes(block)\n')
    pool = replace(pool, '        self.cached_block_hashes_by_block.clear()\n',
        '        self.cached_block_hashes_by_block.clear()\n        self._soft_retention.entries.clear()\n')
    request = replace(request, '        self.kv_transfer_params: dict[str, Any] | None = None\n',
        '''        from vllm.v1.core.retention_policy import parse_hints
        self.soft_retention_hints = parse_hints(sampling_params.extra_args if sampling_params else None)
        self.kv_transfer_params: dict[str, Any] | None = None
''')
    anthropic = replace(anthropic, '        cls._convert_tools(anthropic_request, req)\n        return req\n',
        '''        cls._convert_tools(anthropic_request, req)
        import os
        from vllm.v1.core.retention_policy import anthropic_hints
        hints = anthropic_hints(anthropic_request, os.environ.get("VLLM_AGENT_SOFT_RETENTION") == "1")
        if hints:
            req.vllm_xargs = dict(req.vllm_xargs or {})
            req.vllm_xargs.update(hints)
        return req
''')
    out = dict(pool_soft_retention=pool, request_soft_retention=request,
               anthropic_soft_retention=anthropic, manager_soft_retention=manager)
    for source in out.values():
        ast.parse(source)
    return out


if __name__ == '__main__':
    import os
    out = build((ROOT / os.environ.get('RETENTION_POOL_SOURCE', 'block_pool.orig')).read_text(),
                (ROOT / 'request.orig').read_text(),
                (ROOT / 'anthropic_serving_patched.py').read_text(),
                (ROOT / os.environ.get('RETENTION_MANAGER_SOURCE', 'single_type_kv_cache_manager_54713.py')).read_text())
    for name, source in out.items():
        (ROOT / (name + '.py')).write_text(source)
    print('Validated soft-retention overlays; TTL=300s, scan<=4096, no hard pins')
