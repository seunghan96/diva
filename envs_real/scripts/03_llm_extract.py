"""LLM extraction of guidance / segment breadth / management tone from masked 8-K Item 2.02 press releases.

usage: python 03_llm_extract.py TAG MODEL [--once]
  Uses the shared fleet (infoseek/serving FleetClient, load-balanced over all healthy replicas of MODEL).
Streams over raw/ex99/*.jsonl.gz (re-scans until the fetch log says DONE); output data/llm/{TAG}/{cik}.jsonl.gz.
Structured output is enforced with vLLM JSON-schema decoding (temperature 0). The input is the MASKED snippet
(scripts/mask.py), so the extractor never sees the company identity or calendar dates.
"""
import os, sys, json, time, asyncio, argparse, glob, re, random, hashlib
import httpx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, DATA, LOGS, read_jsonl_gz, write_jsonl_gz
from mask import mask_text, select_snippet
import pandas as pd

GUIDE = ['raised', 'maintained', 'lowered', 'initiated', 'mixed', 'withdrawn', 'none']
SEG = ['all_up', 'mostly_up', 'mixed', 'mostly_down', 'all_down', 'not_reported']
TONE = ['positive', 'neutral', 'negative']
SCHEMA = {
    'type': 'object',
    'properties': {
        'guidance': {'type': 'string', 'enum': GUIDE},
        'guidance_metrics': {'type': 'string', 'enum': ['revenue', 'eps', 'revenue_and_eps', 'other', 'none']},
        'guidance_horizon': {'type': 'string', 'enum': ['next_quarter', 'full_year', 'multi_year', 'several', 'none']},
        'guidance_quote': {'type': 'string', 'maxLength': 300},
        'segments': {'type': 'string', 'enum': SEG},
        'tone': {'type': 'string', 'enum': TONE},
    },
    'required': ['guidance', 'guidance_metrics', 'guidance_horizon', 'guidance_quote', 'segments', 'tone'],
}
SYS = 'You are a careful financial analyst who extracts facts from earnings press releases. Answer only with the requested JSON.'
INSTR = """Below are excerpts of a company's quarterly earnings press release (company identity and calendar dates are masked; [Y+0] is the year of the release, [Y+1] the next year, [Y-1] the previous year).

Extract:
1. guidance: the company's forward-looking financial guidance (outlook for revenue, EPS, earnings, margins, cash flow, etc.) as stated in THIS release, relative to its previous guidance:
   - "raised": guidance increased (including raising the low end or the midpoint).
   - "maintained": guidance reaffirmed / reiterated / unchanged.
   - "lowered": guidance reduced (including lowering the high end or the midpoint).
   - "initiated": new guidance for a new period with no previous guidance to compare (e.g., first outlook for the next fiscal year).
   - "mixed": some guided metrics raised and others lowered.
   - "withdrawn": guidance withdrawn or suspended.
   - "none": the release gives no forward-looking financial guidance.
   Use only explicit statements. Do not infer guidance from results or from generic optimism.
2. guidance_metrics, guidance_horizon: which metrics and which period the guidance covers ("none" if no guidance).
3. guidance_quote: the shortest verbatim phrase that supports the guidance label ("" if none).
4. segments: how the company's reported business segments / divisions performed vs. the prior-year period (by revenue or sales): all_up, mostly_up, mixed, mostly_down, all_down, or not_reported (no segment breakdown given, or only one segment).
5. tone: the overall tone of management's commentary: positive, neutral or negative.

Press release excerpts:
<<<
{text}
>>>

Return the JSON object."""


def load_universe():
    U = pd.read_csv(f'{RAW}/universe.csv')
    out = {}
    for _, r in U.iterrows():
        out.setdefault(int(r.cik), {'names': set(), 'tickers': set()})
        out[int(r.cik)]['names'].add(str(r.company_name))
        out[int(r.cik)]['tickers'].add(str(r.ticker).split('~')[0])
    return out


def entity_info(cik, U):
    e = U.get(cik, {'names': set(), 'tickers': set()})
    names, tickers = set(e['names']), set(e['tickers'])
    fp = f'{RAW}/filings/{cik}.json'
    if os.path.exists(fp):
        j = json.load(open(fp))
        if j.get('name'):
            names.add(j['name'])
        for t in j.get('tickers') or []:
            tickers.add(t)
    return list(names), list(tickers)


def prompt_for(row, U):
    names, tickers = entity_info(row['cik'], U)
    y = int(row['filingDate'][:4])
    snip = select_snippet(row['text'] or '')
    return mask_text(snip, names, tickers, y)


SERV = './serving'
sys.path.insert(0, SERV)


async def one(fc, model, text, stats):
    msgs = [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': INSTR.format(text=text)}]
    params = dict(temperature=0.0, max_tokens=300,
                  response_format={'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': SCHEMA}},
                  chat_template_kwargs={'enable_thinking': False})
    try:
        r = await fc.chat(model, msgs, **params)
        if 'error' in r:
            stats['err'] = stats.get('err', 0) + 1
            return None
        c = r['choices'][0]['message']['content']
        stats['ok'] += 1
        return json.loads(c)
    except Exception as e:
        stats['exc'] = stats.get('exc', 0) + 1
        return None


async def run(args):
    from client import FleetClient
    out_dir = f'{DATA}/llm/{args.tag}'
    os.makedirs(out_dir, exist_ok=True)
    U = load_universe()
    stats = {'ok': 0}
    t0 = time.time()
    async with FleetClient(cache_dir=f'{DATA}/llm_cache', per_replica=args.conc) as fc:
        while True:
            done_fetch = os.path.exists(f'{LOGS}/fetchB.log') and 'DONE' in open(f'{LOGS}/fetchB.log').read()
            todo = [f for f in sorted(glob.glob(f'{RAW}/ex99/*.jsonl.gz'))
                    if not os.path.exists(f'{out_dir}/{os.path.basename(f)}')]
            if not todo:
                if done_fetch or args.once:
                    break
                await asyncio.sleep(60); continue
            for i in range(0, len(todo), 8):
                chunk = todo[i:i + 8]
                jobs, meta = [], []
                for f in chunk:
                    for r in read_jsonl_gz(f):
                        if not r.get('text'):
                            continue
                        if args.subset < 1.0 and int(hashlib.md5(r['accn'].encode()).hexdigest()[:8], 16) / 16 ** 8 >= args.subset:
                            continue
                        txt = prompt_for(r, U)
                        jobs.append(one(fc, args.model, txt, stats))
                        meta.append((f, r['accn'], r['cik'], len(txt)))
                res = await asyncio.gather(*jobs)
                by = {}
                for (f, accn, cik, n), x in zip(meta, res):
                    by.setdefault(f, []).append({'accn': accn, 'cik': cik, 'snip_len': n, 'model': args.model, 'out': x})
                for f in chunk:
                    write_jsonl_gz(f'{out_dir}/{os.path.basename(f)}', by.get(f, []))
                fc.flush()
                print(f'{time.strftime("%H:%M:%S")} {args.tag} files+{len(chunk)} {stats} {time.time()-t0:.0f}s', flush=True)
    print('DONE', stats, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('tag'); ap.add_argument('model')
    ap.add_argument('--conc', type=int, default=48)
    ap.add_argument('--subset', type=float, default=1.0, help='deterministic accession-hash subset (validators)')
    ap.add_argument('--once', action='store_true')
    asyncio.run(run(ap.parse_args()))
