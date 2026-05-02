#!/usr/bin/env python3
"""RunPod pod control helper for CARA-FinSent Phase 8 (REST API).

Reads the RunPod API key from the project ``.env`` (variable name: ``POD_API_KEY``)
and the pod ID from ``RUNPOD_POD_ID`` if set, otherwise falls back to the
default Phase 8 pod ``zvqce99epg537l``.

Subcommands
-----------
    status   Print the pod's current state (RUNNING, EXITED, etc.).
    start    Resume a stopped pod (re-billing begins).
    stop     Stop the pod (preserves /workspace; cheaper $0.008/hr storage).
    wait     Block until the pod is RUNNING.

Uses the RunPod REST API at https://rest.runpod.io/v1.
The legacy GraphQL endpoint at api.runpod.io has been retired (HTTP 403).

Phase 8 cost-discipline rule: leaving the pod RUNNING idle is the dominant
budget leak. Always run ``stop`` after a sweep + result transfer is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POD_ID = 'zvqce99epg537l'
RUNPOD_REST = 'https://rest.runpod.io/v1'


def _load_env_var(name: str) -> str | None:
    """Read a variable from the project .env without using python-dotenv."""
    env_path = PROJECT_ROOT / '.env'
    if env_path.is_file():
        try:
            for raw in env_path.read_text(encoding='utf-8').splitlines():
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, _, value = line.partition('=')
                if key.strip() == name:
                    return value.strip().strip('"').strip("'")
        except OSError:
            pass
    return os.environ.get(name)


def _api_key() -> str:
    key = (
        _load_env_var('POD_API_KEY')
        or _load_env_var('API_KEY')
        or _load_env_var('RUNPOD_API_KEY')
    )
    if not key:
        sys.exit('[ERR] RunPod API key not found. Set POD_API_KEY in .env')
    return key


def _pod_id() -> str:
    return _load_env_var('RUNPOD_POD_ID') or DEFAULT_POD_ID


def _request(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    url = f'{RUNPOD_REST}{path}'
    data = None
    headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        sys.exit(f'[ERR] RunPod REST {method} {path} -> HTTP {exc.code}: '
                 f'{exc.read().decode("utf-8", "ignore")}')
    except urllib.error.URLError as exc:
        sys.exit(f'[ERR] RunPod REST network error: {exc.reason}')
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {'_raw': raw}


def _summarize(pod: dict) -> dict:
    keys = ['id', 'name', 'desiredStatus', 'machineId', 'gpuCount', 'costPerHr',
            'uptimeSeconds', 'lastStatusChange']
    return {k: pod.get(k) for k in keys if k in pod}


def cmd_status(_: argparse.Namespace) -> int:
    pod = _request('GET', f'/pods/{_pod_id()}', _api_key())
    print(json.dumps(_summarize(pod), indent=2))
    return 0


def cmd_stop(_: argparse.Namespace) -> int:
    pod = _request('POST', f'/pods/{_pod_id()}/stop', _api_key(), body={})
    print('[OK] stop ->', json.dumps(_summarize(pod), indent=2))
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    body = {'gpuCount': args.gpu_count}
    pod = _request('POST', f'/pods/{_pod_id()}/start', _api_key(), body=body)
    print('[OK] start ->', json.dumps(_summarize(pod), indent=2))
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    api_key = _api_key()
    pod_id = _pod_id()
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        pod = _request('GET', f'/pods/{pod_id}', api_key)
        status = pod.get('desiredStatus')
        uptime = pod.get('uptimeSeconds')
        print(f'[wait] status={status} uptime={uptime}s', flush=True)
        if status == 'RUNNING' and uptime:
            return 0
        time.sleep(args.interval)
    sys.exit(f'[ERR] timed out after {args.timeout}s waiting for pod RUNNING')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status').set_defaults(func=cmd_status)
    sub.add_parser('stop').set_defaults(func=cmd_stop)
    p_start = sub.add_parser('start')
    p_start.add_argument('--gpu_count', type=int, default=1)
    p_start.set_defaults(func=cmd_start)
    p_wait = sub.add_parser('wait')
    p_wait.add_argument('--timeout', type=int, default=300)
    p_wait.add_argument('--interval', type=int, default=10)
    p_wait.set_defaults(func=cmd_wait)
    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
