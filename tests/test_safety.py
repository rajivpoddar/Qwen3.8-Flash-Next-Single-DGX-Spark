import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_api", ROOT / "files/check_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class SafetyTests(unittest.TestCase):
    def test_readiness(self):
        for model, expected in [("wanted", True), ("other", False)]:
            response = MagicMock()
            response.status = 200
            response.read.return_value = json.dumps({"data": [{"id": model}]}).encode()
            response.__enter__.return_value = response
            opener = MagicMock()
            opener.open.return_value = response
            with patch.dict(os.environ, VLLM_API_KEY="test-secret"), patch.object(api.urllib.request, "build_opener", return_value=opener):
                self.assertEqual(api.ready("http://127.0.0.1:1", "wanted"), expected)
                self.assertEqual(opener.open.call_args.args[0].get_header("Authorization"), "Bearer test-secret")
        with patch.dict(os.environ, VLLM_API_KEY=""):
            self.assertFalse(api.ready("http://127.0.0.1:1", "wanted"))
        with patch.dict(os.environ, VLLM_API_KEY="test"), patch.object(api.urllib.request, "build_opener", side_effect=None) as build:
            build.return_value.open.side_effect = OSError("offline")
            self.assertFalse(api.ready("http://127.0.0.1:1", "wanted"))

    def test_preflight_and_cached_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("start.sh", "download.sh"):
                shutil.copy(ROOT / name, root / name)
            revision = "a" * 40
            (root / "key").write_text("test-secret")
            (root / ".env").write_text(f'IMAGE=test@sha256:test\nTP1_MODEL_REVISION={revision}\nAPI_KEY_FILE={root}/key\nBIND_HOST=127.0.0.1\n')
            snapshot = root / "hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}")
            (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "part"}}))
            (snapshot / "part").write_text("fixture")
            (root / "docker").write_text('#!/bin/sh\n[ "$1" = image ] && [ "$2" = inspect ]\n')
            (root / "docker").chmod(0o700)
            env = dict(os.environ, HF_HOME=str(root / "hf"), PATH=str(root) + ":" + os.environ["PATH"])
            for args in (["bash", "start.sh", "--preflight"], ["bash", "download.sh"]):
                result = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("test-secret", result.stdout + result.stderr)
            (snapshot / "part").unlink()
            result = subprocess.run(["bash", "start.sh", "--preflight"], cwd=root, env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
