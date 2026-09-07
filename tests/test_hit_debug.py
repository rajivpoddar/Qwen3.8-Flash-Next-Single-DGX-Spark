"""No GPU or vLLM imports: prove generated diagnostics preserve source logic."""
import ast
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'files'
spec = importlib.util.spec_from_file_location('patcher', ROOT / 'patch_hit_debug.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

class StripDiagnostics(ast.NodeTransformer):
    def visit_If(self, node):
        if '_hit_allow' in ast.unparse(node.test):
            return None
        return self.generic_visit(node)

class DiagnosticsTests(unittest.TestCase):
    def test_existing_classes_unchanged(self):
        patcher.main()
        for source, result in [('scheduler.orig', 'scheduler_hit_debug.py'),
                               ('block_pool.orig', 'pool_hit_debug.py'),
                               ('kv_cache_coordinator_54713.py', 'coordinator_hit_debug.py'),
                               ('single_type_kv_cache_manager_54713.py', 'manager_hit_debug.py')]:
            before = ast.parse((ROOT / source).read_text())
            after = StripDiagnostics().visit(ast.parse((ROOT / result).read_text()))
            self.assertEqual([ast.dump(n) for n in before.body if isinstance(n, ast.ClassDef)],
                             [ast.dump(n) for n in after.body if isinstance(n, ast.ClassDef)])

    def test_gate_disabled_and_bounded(self):
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'VLLM_HIT_DEBUG': enabled}):
                scope = {}
                exec(patcher.GATE, scope)
                self.assertEqual(sum(scope['_hit_allow']('test') for _ in range(2100)),
                                 2000 if enabled == '1' else 0)
                if enabled == '1':
                    scope['_hit_started'] -= 1801
                    self.assertFalse(scope['_hit_allow']('other'))

    def test_drift_rejected(self):
        with self.assertRaises(AssertionError):
            patcher.replace('missing', 'anchor', 'new')

if __name__ == '__main__':
    unittest.main()
