import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'files'
spec = importlib.util.spec_from_file_location('patch', ROOT / 'patch_mamba_geometry.py')
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

def method(source, name, scope=None):
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    scope = dict(scope or {})
    exec('from __future__ import annotations\n' + ast.unparse(node), scope)
    return scope[name]

class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = [(ROOT / (n + '.orig')).read_text()
                      for n in ('scheduler', 'mamba_hybrid', 'model_runner', 'interface')]
        cls.out = patch.build(*cls.inputs)

    def test_boundaries_and_negative_control(self):
        old = method(self.inputs[0], '_mamba_block_aligned_split')
        new = method(self.out['scheduler_geometry'], '_mamba_block_aligned_split')
        for size in (880, 1600, 1664):
            s = NS(cache_config=NS(block_size=8), mamba_state_block_size=size,
                   max_num_scheduled_tokens=2048, use_eagle=True,
                   scheduler_config=NS(long_prefill_token_threshold=1024),
                   hash_block_size=8, mamba_partial_cache_hit=False)
            start = 0
            ends = []
            while start < 20000:
                r = NS(num_computed_tokens=start, num_prompt_tokens=20000,
                       num_tokens=20000, shared_prefix_boundary=0)
                count = new(s, r, min(1024, 20000-start))
                self.assertGreater(count, 0)
                self.assertLessEqual(count, min(1024, 20000-start))
                self.assertLessEqual(start+count, (start//size+1)*size)
                start += count
                ends.append(start)
            self.assertTrue(set(range(size, 20000, size)).issubset(ends))
        r.num_computed_tokens = 1024
        self.assertEqual(old(s, r, 1024), 1024)
        self.assertEqual(new(s, r, 1024), 640)

    def test_wrapped_specs_and_disagreement(self):
        class Mamba: 
            def __init__(self, size): self.block_size = size
        class Uniform:
            def __init__(self, *specs): self.kv_cache_specs = dict(enumerate(specs))
        src = self.out['scheduler_geometry']
        part = src.split('        self.mamba_state_block_size =', 1)[1].split('        self.needs_kv_cache_zeroing', 1)[0]
        import textwrap
        code = textwrap.dedent('        self.mamba_state_block_size =' + part)
        scope = dict(self=NS(cache_config=NS(block_size=8, mamba_cache_mode='align'), has_mamba_layers=True),
                     MambaSpec=Mamba, UniformTypeKVCacheSpecs=Uniform,
                     logger=NS(info=lambda *args: None))
        scope['kv_cache_config'] = NS(kv_cache_groups=[NS(kv_cache_spec=Uniform(Mamba(1664), Mamba(1664)))])
        exec(code, scope)
        self.assertEqual(scope['self'].mamba_state_block_size, 1664)
        scope['kv_cache_config'].kv_cache_groups.append(NS(kv_cache_spec=Mamba(880)))
        with self.assertRaises(AssertionError): exec(code, scope)

    def test_worker_uses_bound_geometry(self):
        class Cell:
            def fill_(self, value): self.value = value
        state = NS(_align_mode=True, _mamba_spec=NS(block_size=880),
                   num_accepted_tokens_gpu=[Cell()], _mamba_state_idx_gpu=[Cell()],
                   cache_config=NS(block_size=8))
        add = method(self.out['mamba_hybrid_geometry'], 'add_request',
                     {'super': lambda: NS(add_request=lambda *args: None)})
        for tokens in (0, 880, 107360):
            add(state, 0, NS(num_computed_tokens=tokens))
            self.assertEqual(state._mamba_state_idx_gpu[0].value, (tokens-1)//880)
        self.assertNotEqual((107360-1)//8, state._mamba_state_idx_gpu[0].value)
        state._mamba_spec = None
        with self.assertRaises(AssertionError): add(state, 0, NS(num_computed_tokens=880))
        self.assertIn('self.model_state.set_kv_cache_config(kv_cache_config)', self.out['model_runner_geometry'])

    def test_anchor_drift(self):
        with self.assertRaises(AssertionError): patch.build('', *self.inputs[1:])

if __name__ == '__main__': unittest.main()
