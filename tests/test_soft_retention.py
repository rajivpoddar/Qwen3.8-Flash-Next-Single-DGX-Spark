import ast
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1] / 'files'
sys.path.insert(0, str(ROOT))
from retention_policy import SoftRetention, parse_hints, anthropic_hints, KEY
from patch_soft_retention import build


class Queue:
    def __init__(self, blocks):
        self.items = list(blocks)
        self.fake_free_list_head = NS()
        self.fake_free_list_tail = NS()
        self.link()

    def link(self):
        prev = self.fake_free_list_head
        for b in self.items:
            prev.next_free_block = b
            prev = b
        prev.next_free_block = self.fake_free_list_tail

    def remove(self, block):
        self.items = [b for b in self.items if b is not block]
        self.link()

    def popleft_n(self, n):
        assert n <= len(self.items)
        out, self.items = self.items[:n], self.items[n:]
        self.link()
        return out


def blocks(n):
    return [NS(block_id=i, block_hash=('hash', i), is_null=False) for i in range(n)]


def hints(priority=90, duration=10, scope='a'):
    return (scope, [dict(start=0, end=None, priority=priority, duration=duration)])


class RetentionTests(unittest.TestCase):
    def test_native_off_and_empty(self):
        b = blocks(4)
        q = Queue(b)
        p = SoftRetention()
        self.assertEqual(p.pop(q, 0), [])
        self.assertEqual(p.pop(q, 2), b[:2])
        self.assertEqual(p.scanned, 0)

    def test_deep_expired_priority_not_hidden(self):
        now = [0]
        p = SoftRetention(lambda: now[0])
        b = blocks(3)
        p.mark(b[0], hints(50, 100), 0, 8)
        p.mark(b[1], hints(90, 1), 0, 8)
        p.mark(b[2], hints(70, 100), 0, 8)
        now[0] = 2
        self.assertIs(p.pop(Queue(b), 1)[0], b[1])

    def test_all_retained_still_allocates_and_preserves_ties(self):
        b = blocks(9)
        p = SoftRetention(scan_limit=4)
        for x in b: p.mark(x, hints(), 0, 8)
        self.assertEqual(p.pop(Queue(b), 9), b)
        self.assertEqual(p.entries, {})

    def test_six_scopes_survive_unrelated_churn(self):
        b = blocks(30)
        p = SoftRetention()
        for i in range(6): p.mark(b[i], hints(scope=str(i)), 0, 8)
        q = Queue(b)
        self.assertEqual(p.pop(q, 24), b[6:])
        self.assertEqual(q.items, b[:6])
        self.assertEqual(p.protected_evicted, 0)

    def test_recycled_block_and_owner_downgrade(self):
        p = SoftRetention()
        b = blocks(1)[0]
        p.mark(b, hints(), 0, 8)
        p.mark(b, hints(10, scope='b'), 0, 8)
        self.assertEqual(p.priority(b, p.clock()), 90)
        p.mark(b, hints(10), 0, 8)
        self.assertEqual(p.priority(b, p.clock()), 10)
        b.block_hash = 'recycled'
        self.assertEqual(p.priority(b, p.clock()), 0)

    def test_api_scope_validation_and_gating(self):
        r = NS(model='qwen3.8-flash-next', metadata={'user_id':'session-6'})
        self.assertEqual(anthropic_hints(r, False), {})
        data = parse_hints(anthropic_hints(r, True))
        self.assertEqual(len(data[0]), 64)
        self.assertNotIn('session-6', data[0])
        self.assertEqual(data[1][0]['priority'], 90)
        r.model = 'other'
        self.assertEqual(anthropic_hints(r, True), {})
        for bad in [float('nan'), -1, 301, True]:
            d = dict(start=0, end=None, priority=90, duration=bad)
            with self.assertRaises(ValueError): parse_hints({KEY: json.dumps({'directives':[d]})})

    def test_overlay_composes_and_fails_on_drift(self):
        inputs = [(ROOT / f).read_text() for f in
                  ('block_pool.orig', 'request.orig', 'anthropic_serving_patched.py', 'single_type_kv_cache_manager_54713.py')]
        out = build(*inputs)
        for source in out.values(): ast.parse(source)
        self.assertIn('req.vllm_xargs.update(hints)', out['anthropic_soft_retention'])
        self.assertIn('self.soft_retention_hints = parse_hints', out['request_soft_retention'])
        self.assertIn('req.chat_template_kwargs["enable_thinking"] = False', out['anthropic_soft_retention'])
        with self.assertRaises(AssertionError): build('', *inputs[1:])


if __name__ == '__main__': unittest.main()
