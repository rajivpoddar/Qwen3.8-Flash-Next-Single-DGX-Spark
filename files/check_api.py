"""Authenticated, exact-model readiness. Never print credentials."""
import json
import os
import sys
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def ready(base, model):
    key = os.environ.get("VLLM_API_KEY", "")
    if not key:
        return False
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        for path in ("/health", "/v1/models"):
            req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + key})
            with opener.open(req, timeout=5) as response:
                if response.status != 200:
                    return False
                if path == "/v1/models":
                    return model in [item["id"] for item in json.load(response)["data"]]
    except Exception:
        return False
    return False


if __name__ == "__main__":
    sys.exit(0 if ready(sys.argv[1], sys.argv[2]) else 1)
