"""Run inside the pinned CPU-only image with candidate source mounts."""
import inspect
import os
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.v1.request import Request
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.retention_policy import KEY
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager

os.environ['VLLM_AGENT_SOFT_RETENTION'] = '1'
r = AnthropicMessagesRequest(model='qwen3.8-flash-next', max_tokens=16,
    messages=[{'role':'user', 'content':'test'}], metadata={'user_id':'test-session-6'})
req = AnthropicServingMessages._convert_anthropic_to_openai_request(r)
signature = inspect.signature(req.to_sampling_params)
kwargs = {}
for name, param in signature.parameters.items():
    if param.default is inspect.Parameter.empty:
        if name in ('default_max_tokens', 'max_tokens'): kwargs[name] = 16
        elif name == 'default_sampling_params': kwargs[name] = {}
        else: raise AssertionError(f'Unrecognized sampling argument: {name}')
sampling = req.to_sampling_params(**kwargs)
assert KEY in sampling.extra_args
core = Request('retention-proof', list(range(16)), sampling, None)
assert core.soft_retention_hints[1][0]['priority'] == 90
assert req.chat_template_kwargs['enable_thinking'] is False
pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=8)
blocks = pool.get_new_blocks(20)
for index, block in enumerate(blocks):
    # Actual block type/queue; synthetic opaque hashes need no model weights.
    pool._insert_block_hash(b'proof' + index.to_bytes(4, 'big'), block, num_tokens=8)
    if index < 6: pool._soft_retention.mark(block, core.soft_retention_hints, 0, 8)
pool.free_blocks(blocks)
new = pool.get_new_blocks(25)
assert not set(b.block_id for b in blocks[:6]) & set(b.block_id for b in new)
assert pool.get_num_free_blocks() == 6
# Actual no-new-full-block manager path must refresh the reused checkpoint.
from types import SimpleNamespace as NS
pool._soft_retention.entries.clear()
manager = NS(num_cached_block={core.request_id:1}, block_size=8, kv_cache_group_id=0,
             req_to_blocks={core.request_id:[blocks[0]]}, block_pool=pool)
core._retention_refreshed_groups = set()
SingleTypeKVCacheManager.cache_blocks(manager, core, 8)
assert blocks[0].block_id in pool._soft_retention.entries
# Partial publication and CoW move preserve priority and original expiration.
partial_pool = BlockPool(num_gpu_blocks=12, enable_caching=True, hash_block_size=8)
source, destination = partial_pool.get_new_blocks(2)
core.block_hashes = [b'partial1', b'partial2']
partial_pool.cache_partial_block(core, source, 8, 0, 16)
entry = partial_pool._soft_retention.entries[source.block_id]
partial_pool.move_block_hashes(source, destination)
moved = partial_pool._soft_retention.entries[destination.block_id]
assert entry[1:] == moved[1:]
assert source.block_id not in partial_pool._soft_retention.entries
partial_pool._soft_retention.clock = lambda: entry[2] + 1
assert partial_pool._soft_retention.priority(destination, entry[2] + 1) == 0
# Consumer CoW redirects the block table; refresh must still find cached source.
partial_pool._soft_retention.clock = lambda: 1000
partial_pool._soft_retention.mark(destination, core.soft_retention_hints, 0, 8)
private = partial_pool.get_new_blocks(1)[0]
consumer = NS(req_to_blocks={core.request_id:[destination]}, block_size=16,
              kv_cache_group_id=0, num_cached_block={core.request_id:1},
              block_pool=partial_pool, _pending_cow_copies=[])
SingleTypeKVCacheManager._apply_cow(consumer, core.request_id, 0, destination, private)
partial_pool._soft_retention.clock = lambda: 1100
core._retention_refreshed_groups = set()
SingleTypeKVCacheManager.cache_blocks(consumer, core, 8)
assert partial_pool._soft_retention.entries[destination.block_id][2] == 1400
assert not consumer._retention_cow_sources
print('PASS: Anthropic -> sampling -> core request -> native free-queue protection; omitted thinking unchanged')
print('PASS: actual manager all-hit refresh; partial checkpoint -> CoW -> unchanged expiry -> expiration')
print('PASS: consumer partial-hit CoW source refreshed despite private block-table replacement')
