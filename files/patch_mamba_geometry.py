#!/usr/bin/env python3
"""Backport vLLM #54076 and #53798 to the pinned runtime, preserving local fixes."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def replace(src, old, new):
    assert src.count(old) == 1, f"Mamba geometry anchor drift: {old[:100]}"
    return src.replace(old, new)

def build(scheduler, worker, runner, interface):
    scheduler = replace(scheduler,
        'from vllm.v1.kv_cache_interface import KVCacheConfig\n',
        'from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, UniformTypeKVCacheSpecs\n')
    anchor = '        self.has_mamba_layers = kv_cache_config.has_mamba_layers\n'
    scheduler = replace(scheduler, anchor, anchor + '''        self.mamba_state_block_size = self.cache_config.block_size
        if self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align":
            state_sizes = set()
            for group in kv_cache_config.kv_cache_groups:
                spec = group.kv_cache_spec
                specs = (spec.kv_cache_specs.values()
                         if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,))
                state_sizes.update(s.block_size for s in specs if isinstance(s, MambaSpec))
            assert len(state_sizes) == 1, "Mamba groups must share one state block size"
            self.mamba_state_block_size = state_sizes.pop()
            logger.info("[mamba-geometry] scheduler generic=%d state=%d",
                        self.cache_config.block_size, self.mamba_state_block_size)
''')
    scheduler = replace(scheduler, '        block_size = self.cache_config.block_size\n',
                        '        block_size = self.mamba_state_block_size\n')
    scheduler = replace(scheduler, '            next_block_boundary if start % block_size != 0 else 0,',
                        '            next_block_boundary,')
    anchor = '    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:\n'
    worker = replace(worker, anchor, '''    def set_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        if self._align_mode:
            self._get_mamba_group_info(kv_cache_config)

''' + anchor)
    worker = replace(worker, '            # Seed the running state block from the resumed/prefilled position.\n',
        '            # The block table uses Mamba state blocks, not the generic minimum.\n'
        '            assert self._mamba_spec is not None, "KV cache config not bound"\n')
    worker = replace(worker, '(new_req_data.num_computed_tokens - 1) // self.cache_config.block_size',
                        '(new_req_data.num_computed_tokens - 1) // self._mamba_spec.block_size')
    anchor = '        self.kv_cache_config = kv_cache_config\n'
    runner = replace(runner, anchor, anchor + '        self.model_state.set_kv_cache_config(kv_cache_config)\n')
    anchor = '    def get_additional_cg_support(self)'
    interface = replace(interface, anchor, '''    def set_kv_cache_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Bind final cache geometry before any request is added."""
        return None

''' + anchor)
    result = dict(scheduler_geometry=scheduler, mamba_hybrid_geometry=worker,
                  model_runner_geometry=runner, interface_geometry=interface)
    for src in result.values():
        ast.parse(src)
    return result

def main():
    sources = [(ROOT / (name + '.orig')).read_text()
               for name in ('scheduler', 'mamba_hybrid', 'model_runner', 'interface')]
    for name, src in build(*sources).items():
        (ROOT / (name + '.py')).write_text(src)
    print('Validated and generated scheduler and worker Mamba geometry overlays')

if __name__ == '__main__':
    main()
