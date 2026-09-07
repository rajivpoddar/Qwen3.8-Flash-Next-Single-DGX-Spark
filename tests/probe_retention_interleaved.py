"""CPU-only pressure reproduction using real pinned-runtime lookup and allocation.

No model, GPU, customer prompts, server requests, or serving-process mutations.
Session A publishes aligned FA + four recurrent groups. B then occupies all but
one complete boundary's worth of the same pool. Check A through the actual
hybrid coordinator, not a synthetic min() approximation.
"""
import json
import os
from collections import namedtuple
from types import SimpleNamespace as NS

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec


def run(pressure, eagle=False):
    size, positions, groups = 1664, 6, 5
    pool = BlockPool(num_gpu_blocks=positions * groups + 1,
                     enable_caching=True, hash_block_size=size)
    pool._soft_retention.clock = lambda: 1000
    fa = FullAttentionSpec(block_size=size, num_kv_heads=1, head_size=8,
                           dtype=torch.bfloat16)
    recurrent = MambaSpec(block_size=size, shapes=((8,),),
                           dtypes=(torch.bfloat16,), mamba_cache_mode='align')
    specs = [fa] + [recurrent] * (groups - 1)
    hashes = [f'session-A-boundary-{i}'.encode() for i in range(positions)]
    hints = ('session-A', [dict(start=0, end=None, priority=90, duration=300)])
    allocated = pool.get_new_blocks(positions * groups)
    # Same immediate-free order as the real coordinator: each group tail first.
    for group in range(groups):
        blocks = allocated[group * positions:(group + 1) * positions]
        for index, block in enumerate(blocks):
            pool._insert_block_hash(make_block_hash_with_group_id(hashes[index], group),
                                    block, num_tokens=(index + 1) * size)
            pool._soft_retention.mark(block, hints, index * size, (index + 1) * size)
        pool.free_blocks(reversed(blocks))
    Group = namedtuple('Group', 'spec group_ids manager_cls use_eagle')
    coordinator = NS(
        kv_cache_config=NS(kv_cache_groups=specs), block_pool=pool,
        single_type_managers=[NS(block_size=size) for _ in specs],
        attention_groups=[Group(fa, [0], FullAttentionManager, eagle),
                          Group(recurrent, list(range(1, groups)), MambaManager, False)],
        hash_block_size=size, enable_partial_hash_hits=False,
        _cache_hit_alignment_tokens=size, dcp_world_size=1)
    def lookup():
        return HybridKVCacheCoordinator.find_longest_cache_hit(
            coordinator, hashes, positions * size)[1]
    before = lookup()
    assert before == (positions - int(eagle)) * size
    # Session B's active allocation creates pressure without changing A's tokens.
    b = pool.get_new_blocks(pressure)
    after = lookup()
    result = dict(pressure=pressure, eagle=eagle, before=before, after=after,
                  free_blocks=pool.get_num_free_blocks(),
                  protected_evicted=pool._soft_retention.protected_evicted)
    assert len({x.block_id for x in b}) == pressure
    assert all(x.ref_cnt == 1 for x in b)
    pool.free_blocks(b)
    assert pool.get_num_free_blocks() == positions * groups
    print(json.dumps(result), flush=True)
    return result


results = [run(n) for n in (0, 5, 15, 25, 30)]
if os.environ.get('EXPECT_PREFIX_SURVIVES') == '1':
    assert results[3]['after'] == 1664, results[3]
else:
    assert results[3]['after'] == 0, results[3]
assert results[0]['after'] == 9984
assert results[-1]['after'] == 0  # Exhaustion must still admit B, never hard-pin A.
spec_results = [run(n, eagle=True) for n in (0, 15, 30)]
assert spec_results[1]['after'] == (3328 if os.environ.get('EXPECT_PREFIX_SURVIVES') == '1' else 0)
assert spec_results[-1]['after'] == 0
