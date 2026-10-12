"""Entity + date masking for SEC press-release text, and a guidance/segment-focused snippet selector.

mask_text(text, names, tickers, ref_year):
  - company name variants and tickers -> "the Company" / "[TICKER]"
  - 4-digit years -> relative "[Y+k]" w.r.t. the announcement year (fiscal-year meaning is preserved, the calendar is not)
  - full dates ("July 30, 2026", "7/30/2026", "2026-07-30") -> "[DATE]"; bare "Month dd" -> "[DATE]"
  - URLs, e-mail, phone numbers, street addresses (best effort) -> removed / "[CONTACT]"
"""
import re

MONTHS = 'January|February|March|April|May|June|July|August|September|October|November|December|Jan\\.?|Feb\\.?|Mar\\.?|Apr\\.?|Jun\\.?|Jul\\.?|Aug\\.?|Sept?\\.?|Oct\\.?|Nov\\.?|Dec\\.?'
SUFFIX = r'(,?\s+(Inc\.?|Incorporated|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|plc|PLC|N\.V\.|S\.A\.|LLC|L\.P\.|Holdings?|Group|& Co\.?))*'
GENERIC = {'the', 'inc', 'corp', 'corporation', 'company', 'co', 'group', 'holdings', 'holding', 'international', 'american',
           'national', 'united', 'general', 'first', 'new', 'global', 'technologies', 'systems', 'energy', 'financial',
           'services', 'bank', 'trust', 'resources', 'industries', 'brands', 'and', '&', 'of', 'plc', 'ltd', 'communications',
           'health', 'healthcare', 'capital', 'properties', 'realty', 'pharmaceuticals', 'labs', 'laboratories', 'motors'}


def name_patterns(names, tickers):
    pats = []
    for n in names:
        if not n:
            continue
        n = n.strip(); n = n.title() if n.isupper() else n
        for _ in range(3):
            n = re.sub(r'[,.]?\s*(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|plc|PLC|N\.V\.|LLC|L\.P\.|/[A-Z]{2,3}/?)$', '', n, flags=re.I).strip(' ,.')
        n = re.sub(r'^The\s+', '', n, flags=re.I)
        if len(n) < 2:
            continue
        pats.append(r'\b' + re.escape(n).replace('\\ ', r'\s+') + SUFFIX + r'(?![A-Za-z])')
        toks = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9'&\-]+", n) if t.lower() not in GENERIC and len(t) >= 4]
        if toks:  # distinctive first token alone (e.g. "Apple", "Microsoft"), only if capitalised in text
            pats.append(r'\b' + re.escape(toks[0]) + r"(?:'s)?\b")
    out = []
    for p in sorted(set(pats), key=len, reverse=True):
        out.append(re.compile(p, re.I))
    tk = [re.compile(r'(?<![A-Za-z0-9])' + re.escape(t) + r'(?![A-Za-z0-9])') for t in tickers if t and len(t) >= 2]
    return out, tk


def mask_text(text, names, tickers, ref_year):
    if not text:
        return text
    t = text
    t = re.sub(r'https?://\S+|www\.\S+', '[URL]', t)
    t = re.sub(r'\S+@\S+\.\w+', '[CONTACT]', t)
    t = re.sub(r'\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b', '[CONTACT]', t)
    t = re.sub(r'\((NYSE|NASDAQ|Nasdaq|NYSE American|Nasdaq GS|NasdaqGS)\s*:\s*[A-Z.]+\)', '([EXCHANGE]: [TICKER])', t)
    t = re.sub(r'\b(NYSE|NASDAQ|Nasdaq)\s*:\s*[A-Z.]+\b', '[EXCHANGE]: [TICKER]', t)
    # executives quoted in the release: "said Jane Doe, Chief Executive Officer" / "Jane Doe, chairman and CEO"
    people = set()
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z'\-]+){1,2}),?\s+(?:the\s+Company's\s+|our\s+)?(?:[Cc]hair|[Cc]hief|CEO|CFO|COO|[Pp]resident|[Ee]xecutive [Vv]ice|[Ss]enior [Vv]ice|[Vv]ice [Cc]hair|[Tt]reasurer)", t):
        people.update(w for w in m.group(1).split() if len(w) > 2 and not w.endswith('.'))
    for m in re.finditer(r"\bsaid\s+([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z'\-]+){1,2})", t):
        people.update(w for w in m.group(1).split() if len(w) > 2 and not w.endswith('.'))
    for m in re.finditer(r"\b(?:Officer|CEO|CFO|COO|President|Chairman|Chairwoman|Chair)(?:\s+and\s+[A-Z]{3})?,?\s+([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zA-Z'\-]+){1,2})", t):
        people.update(w for w in m.group(1).split() if len(w) > 1 and not w.endswith('.') and w not in ('The', 'Company', 'Chief', 'Officer'))
    for w in sorted(people, key=len, reverse=True):
        t = re.sub(r'\b' + re.escape(w) + r"\b", '[PERSON]', t)
    t = re.sub(r'\[PERSON\](\s+\[PERSON\])+', '[PERSON]', t)
    pats, tks = name_patterns(names, tickers)
    for p in pats:
        t = p.sub('the Company', t)
    for p in tks:
        t = p.sub('[TICKER]', t)
    # datelines ("DUBLIN--(BUSINESS WIRE)--", "Dallas, Texas, July 30, 2026")
    t = re.sub(r"[A-Z][A-Za-z.' ]{2,30}(,\s*[A-Z][A-Za-z.' ]{2,20})?\s*[-\u2010-\u2014]*\s*\((BUSINESS WIRE|Business Wire|PR ?Newswire|PRNewswire|GLOBE ?NEWSWIRE|Globe ?Newswire|ACCESSWIRE|ACCESS Newswire)\)", '[DATELINE]', t)
    t = re.sub(r"\b[A-Z][a-zA-Z.]+(?: [A-Z][a-zA-Z.]+)?,\s+[A-Z][a-zA-Z.]+(?: [A-Z][a-zA-Z.]+)?,?\s+(?=(" + MONTHS + r")\s+\d)", '[DATELINE], ', t)
    # dates
    t = re.sub(r'\b(' + MONTHS + r')\s+\d{1,2}(st|nd|rd|th)?,?\s+(19|20)\d{2}\b', '[DATE]', t)
    t = re.sub(r'\b\d{1,2}\s+(' + MONTHS + r')\s+(19|20)\d{2}\b', '[DATE]', t)
    t = re.sub(r'\b\d{1,2}/\d{1,2}/(19|20)?\d{2}\b', '[DATE]', t)
    t = re.sub(r'\b(19|20)\d{2}-\d{2}-\d{2}\b', '[DATE]', t)
    t = re.sub(r'\b(' + MONTHS + r')\s+\d{1,2}(st|nd|rd|th)?\b', '[DATE]', t)

    def yr(m):
        y = int(m.group(0))
        k = y - ref_year
        return f'[Y{k:+d}]' if k else '[Y+0]'
    t = re.sub(r"(?<![\d$.,])(19[89]\d|20[0-4]\d)(?![\d%])", yr, t)
    t = re.sub(r"\b(FY|Q[1-4]\s?)'?(\d{2})\b", lambda m: m.group(1) + f"[Y{int(m.group(2)) + 2000 - ref_year:+d}]", t)
    return t


KW = re.compile(r'outlook|guidance|expect|anticipat|forecast|project(s|ed|ing)?\b|full[- ]year|fiscal\s+(year\s+)?\[?Y|'
                r'reaffirm|reiterat|rais(e|es|ed|ing)\b|lower(s|ed|ing)?\b|narrow|updat(e|es|ed)\s+(its|the|our)|range of|'
                r'segment|division|business unit|organic|withdr|suspend', re.I)


def select_snippet(text, head=1500, cap=7000):
    """Headline/summary + keyword lines (narrative only, table rows with many cells skipped) + 1 line of context."""
    if not text:
        return ''
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    out, used, n = [], set(), 0
    acc = ''
    for i, l in enumerate(lines):
        if len(acc) >= head:
            break
        acc += l + '\n'; used.add(i)
    out.append(acc)
    n = len(acc)
    for i, l in enumerate(lines):
        if i in used or l.count('|') >= 3 or len(l) < 25:
            continue
        if KW.search(l):
            for j in (i, i + 1):
                if j < len(lines) and j not in used and lines[j].count('|') < 3:
                    used.add(j)
                    out.append(lines[j] + '\n'); n += len(lines[j]) + 1
            if n >= cap:
                break
    return ''.join(out)[:cap]
