from __future__ import annotations
import json
import urllib.request
import urllib.error


class OllamaHealth:
    def __init__(self, host: str = "http://127.0.0.1:11434"):
        self.host = host.rstrip("/")

    def is_ready(self, timeout_s: float = 0.35) -> bool:
        """
        Lightweight readiness check.
        Hits Ollama's local HTTP API with a short timeout.
        """
        url = f"{self.host}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                if resp.status != 200:
                    return False
                # Optional parse to ensure it's valid JSON
                _ = json.loads(resp.read().decode("utf-8"))
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return False
        except Exception:
            return False