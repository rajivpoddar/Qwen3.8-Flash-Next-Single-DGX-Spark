"""Run upstream #54713 CPU tests in the pinned image; no serving-process changes.

Place upstream test_prefix_caching.py at /tmp/test_prefix_caching_54713.py.
Pass --patched with the two generated overlays in /tmp to test the candidate.
The old runtime reads retention from an env var, not KVCacheConfig.
"""
import importlib
import os
from pathlib import Path
import sys
from math import lcm
from unittest.mock import patch
import pytest

if "--patched" in sys.argv:
    for name in ("single_type_kv_cache_manager", "kv_cache_coordinator"):
        module = importlib.import_module(f"vllm.v1.core.{name}")
        source = Path(f"/tmp/{name}_54713.py").read_text()
        exec(compile(source, f"candidate/{name}.py", "exec"), module.__dict__)

class PinnedRuntimeAdapter:
    def pytest_collection_modifyitems(self, items):
        def make_manager(kv_cache_config, **kwargs):
            kwargs.setdefault("scheduler_block_size", lcm(*(
                g.kv_cache_spec.block_size for g in kv_cache_config.kv_cache_groups)))
            interval = kwargs.pop("retention_interval", None)
            with patch.dict(os.environ):
                if interval is None:
                    os.environ.pop("VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None)
                else:
                    os.environ["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"] = str(interval)
                from vllm.v1.core.kv_cache_manager import KVCacheManager
                return KVCacheManager(kv_cache_config, **kwargs)
        for item in items:
            item.module.make_kv_cache_manager = make_manager

raise SystemExit(pytest.main([
    "-q", "--tb=short", "/tmp/test_prefix_caching_54713.py",
    "-k", "mamba_retention_mtp_boundary or mamba_retention_eagle_backoff",
], plugins=[PinnedRuntimeAdapter()]))
