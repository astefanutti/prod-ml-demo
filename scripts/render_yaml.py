#!/usr/bin/env python3
"""Render SparkApplication YAML templates by substituting environment variables.

Usage:
    python3 scripts/render_yaml.py <template.yaml> [VAR=val ...]
    python3 scripts/render_yaml.py <template.yaml> [VAR=val ...] > rendered.yaml
"""
import os
import re
import sys
from pathlib import Path

def _load_env_file(path: Path) -> None:
    """Parse a .env file without requiring python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key not in os.environ:
            os.environ[key] = value

_load_env_file(Path(__file__).parent.parent / ".env")


def render(template_text: str, overrides: dict) -> str:
    env = {**os.environ, **overrides}

    def _sub(m):
        key = m.group(1)
        default = m.group(2)
        return env.get(key, default if default is not None else m.group(0))

    result = re.sub(r'\$\{(\w+)(?::-([^}]*))?\}', _sub, template_text)
    # Only substitute bare $VAR when VAR is ALL_CAPS (avoids eating Grafana's $__file, $__interval, etc.)
    result = re.sub(r'\$([A-Z][A-Z0-9_]*)', lambda m: env.get(m.group(1), m.group(0)), result)
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <template.yaml> [KEY=VALUE ...]", file=sys.stderr)
        sys.exit(1)

    template_path = sys.argv[1]
    overrides = {}
    for arg in sys.argv[2:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k] = v

    template = Path(template_path).read_text()
    print(render(template, overrides))
