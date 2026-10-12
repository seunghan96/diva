"""Shared paths + a rate-limited SEC EDGAR client (fair-access: declared User-Agent, <= 8 req/s, backoff on 429/503)."""
import os, time, json, gzip, threading
import requests

ROOT = './envs_real'
RAW = f'{ROOT}/raw'
DATA = f'{ROOT}/data'
VOI = f'{ROOT}/voi'
LOGS = f'{ROOT}/logs'
EQ = '<path>'
UA = 'Academic research (anonymous) contact@example.com'
for d in (RAW, DATA, VOI, LOGS):
    os.makedirs(d, exist_ok=True)


class SEC:
    def __init__(self, rps=8.0):
        self.rps = rps
        self.min_dt = 1.0 / rps
        self.last = 0.0
        self.lock = threading.Lock()
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': UA, 'Accept-Encoding': 'gzip, deflate'})
        self.n = 0
        self.throttled = 0

    def _adapt(self):
        # share EDGAR's 10 req/s budget with other project fetchers on this IP: if another SEC fetcher
        # (the scripts/data/fetch_*.py) is running, drop to 2.5 req/s, else use the configured rate.
        import re as _re
        busy = False
        for lf in ('<path>',
                   '<path>'):
            try:
                if time.time() - os.path.getmtime(lf) < 240:
                    last = open(lf).read().strip().split('\n')[-1]
                    if _re.search(r'\bsec\b|edgar|8-K|ex99|submissions|companyfacts', last, _re.I):
                        busy = True
            except Exception:
                pass
        self.min_dt = 1.0 / (2.5 if busy else self.rps)

    def get(self, url, tries=6):
        if self.n % 50 == 0:
            self._adapt()
        for k in range(tries):
            with self.lock:
                dt = time.time() - self.last
                if dt < self.min_dt:
                    time.sleep(self.min_dt - dt)
                self.last = time.time()
            try:
                r = self.s.get(url, timeout=30)
            except Exception as e:
                time.sleep(2 * (k + 1)); continue
            self.n += 1
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            # 429 / 403 (throttle) / 5xx -> back off; SEC throttles an IP for ~10 min after excess traffic
            self.throttled += 1
            time.sleep(min(600, 30 * 2 ** k))
        raise RuntimeError('SEC unavailable: ' + url)


def write_jsonl_gz(path, rows):
    tmp = path + '.tmp'
    with gzip.open(tmp, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    os.replace(tmp, path)


def read_jsonl_gz(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)
