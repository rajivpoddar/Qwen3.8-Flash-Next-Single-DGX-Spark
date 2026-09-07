# SPDX-License-Identifier: Apache-2.0
"""Bounded soft retention inspired by vLLM PR #38514 (263525226709).

Keep the native free queue and allocation accounting. Hints only change the
choice among free blocks; they never reserve memory or bypass hybrid lookup.
No lazy priority heap: evaluate TTL at selection time, including deep entries.
"""
import hashlib
import json
import math
import time

KEY = "heydonna_retention"


def anthropic_hints(request, enabled):
    if not enabled or request.model != "qwen3.8-flash-next":
        return {}
    metadata = getattr(request, "metadata", None) or {}
    identity = metadata.get("user_id")
    # Scope is an ownership hint, not authentication or cache isolation.
    scope = hashlib.sha256(str(identity).encode()).hexdigest() if identity else None
    return {KEY: json.dumps({"scope": scope, "directives": [
        {"start": 0, "end": 32768, "priority": 90, "duration": 300},
        {"start": 32768, "end": None, "priority": 70, "duration": 300},
    ]})}


def parse_hints(extra_args):
    raw = (extra_args or {}).get(KEY)
    if raw is None:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) - {"scope", "directives"}:
        raise ValueError("invalid retention hints")
    scope = data.get("scope")
    if scope is not None and (not isinstance(scope, str) or len(scope) > 256):
        raise ValueError("invalid retention scope")
    directives = data.get("directives", [])
    if not isinstance(directives, list) or len(directives) > 8:
        raise ValueError("retention requires at most eight ranges")
    previous_end, previous_priority = 0, 100
    for d in directives:
        if not isinstance(d, dict) or set(d) != {"start", "end", "priority", "duration"}:
            raise ValueError("invalid retention range")
        start, end, priority, duration = (d[k] for k in ("start", "end", "priority", "duration"))
        if (type(start) is not int or start < 0 or previous_end is None
                or start < previous_end or (end is not None and (type(end) is not int or end <= start))
                or type(priority) is not int or not 0 <= priority <= previous_priority
                or type(duration) not in (float, int) or not math.isfinite(duration)
                or not 0 < duration <= 300):
            raise ValueError("invalid retention bounds, priority or TTL")
        previous_end, previous_priority = end, priority
    return scope, directives


class SoftRetention:
    def __init__(self, clock=time.monotonic, scan_limit=4096):
        self.clock = clock
        self.scan_limit = scan_limit
        self.entries = {}
        self.selections = self.scanned = self.deferred = self.protected_evicted = 0

    def mark(self, block, hints, start, end):
        if hints is None or block.is_null or block.block_hash is None:
            return
        scope, directives = hints
        now = self.clock()
        old = self.entries.get(block.block_id)
        if old and (old[0] != block.block_hash or old[2] <= now):
            old = None
            self.entries.pop(block.block_id, None)
        matching = [d for d in directives if d["start"] < end and
                    (d["end"] is None or d["end"] > start)]
        if not matching:
            if old and scope is not None and old[3] == scope:
                self.entries.pop(block.block_id, None)
            return
        chosen = max(matching, key=lambda d: d["priority"])
        if old and chosen["priority"] < old[1] and (scope is None or scope != old[3]):
            return
        self.entries[block.block_id] = (block.block_hash, chosen["priority"],
                                       now + chosen["duration"], scope)

    def priority(self, block, now):
        entry = self.entries.get(block.block_id)
        if entry is None:
            return 0
        if block.block_hash != entry[0] or entry[2] <= now:
            self.entries.pop(block.block_id, None)
            return 0
        return entry[1]

    def move(self, source, destination, source_hash):
        entry = self.entries.pop(source.block_id, None)
        self.entries.pop(destination.block_id, None)
        if entry and entry[0] == source_hash and entry[2] > self.clock():
            self.entries[destination.block_id] = (destination.block_hash, *entry[1:])

    def pop(self, queue, count):
        if not self.entries or count == 0:
            return queue.popleft_n(count)
        # Never scan an unbounded pool in the scheduler hot path. Allocations
        # beyond the bounded candidate window retain the native LRU fallback.
        now = self.clock()
        candidates = []
        block = queue.fake_free_list_head.next_free_block
        while block is not queue.fake_free_list_tail and len(candidates) < self.scan_limit:
            candidates.append((self.priority(block, now), len(candidates), block))
            block = block.next_free_block
        self.selections += 1
        self.scanned += len(candidates)
        chosen = sorted(candidates, key=lambda x: (x[0], x[1]))[:count]
        self.deferred += sum(1 for _, index, _ in chosen if index >= count)
        result = []
        for priority, _, block in chosen:
            queue.remove(block)
            self.protected_evicted += int(priority > 0)
            self.entries.pop(block.block_id, None)
            result.append(block)
        if len(result) < count:
            tail = queue.popleft_n(count - len(result))
            for block in tail:
                self.protected_evicted += int(self.priority(block, now) > 0)
                self.entries.pop(block.block_id, None)
            result.extend(tail)
        return result
