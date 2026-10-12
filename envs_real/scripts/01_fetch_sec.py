"""Phase A: universe (S&P 500 entities that were PIT members at any time 2015-01-01..2026-09-30, from the r1 equity layer),
SEC XBRL companyfacts (whitelisted concepts, every vintage kept with its `filed` date for point-in-time use) and
the submissions index (all 8-K filings with Item 2.02, 2014-07..2026-09).
Phase B: for each Item 2.02 8-K, the filing index page -> EX-99.x press release -> plain text (first 80k chars).
Resumable per CIK. Output: raw/universe.csv, raw/facts/{cik}.jsonl.gz, raw/filings/{cik}.json, raw/ex99/{cik}.jsonl.gz
"""
import os, sys, json, re, time
import pandas as pd
from lxml import html as LH
sys.path.insert(0, os.path.dirname(__file__))
from common import *

CONCEPTS = """Revenues RevenueFromContractWithCustomerExcludingAssessedTax RevenueFromContractWithCustomerIncludingAssessedTax
SalesRevenueNet SalesRevenueGoodsNet SalesRevenueServicesNet RevenuesNetOfInterestExpense InterestAndDividendIncomeOperating
InterestIncomeExpenseNet NoninterestIncome RegulatedAndUnregulatedOperatingRevenue ElectricUtilityRevenue
CostOfRevenue CostOfGoodsAndServicesSold CostOfGoodsSold CostOfServices GrossProfit OperatingIncomeLoss OperatingExpenses
IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest
IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments
NetIncomeLoss ProfitLoss EarningsPerShareDiluted EarningsPerShareBasic EarningsPerShareBasicAndDiluted
WeightedAverageNumberOfDilutedSharesOutstanding WeightedAverageNumberOfSharesOutstandingBasic
NetCashProvidedByUsedInOperatingActivities NetCashProvidedByUsedInOperatingActivitiesContinuingOperations
PaymentsToAcquirePropertyPlantAndEquipment PaymentsToAcquireProductiveAssets PaymentsForCapitalImprovements
PaymentsForRepurchaseOfCommonStock PaymentsForRepurchaseOfEquity PaymentsOfDividends PaymentsOfDividendsCommonStock
LongTermDebt LongTermDebtNoncurrent LongTermDebtCurrent DebtCurrent ShortTermBorrowings CommercialPaper
LongTermDebtAndCapitalLeaseObligations LongTermDebtAndCapitalLeaseObligationsCurrent DebtLongtermAndShorttermCombinedAmount
Assets Liabilities StockholdersEquity StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest
CashAndCashEquivalentsAtCarryingValue CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents
ResearchAndDevelopmentExpense SellingGeneralAndAdministrativeExpense InventoryNet AccountsReceivableNetCurrent
ShareBasedCompensation DepreciationDepletionAndAmortization DepreciationAndAmortization""".split()
CSET = set(CONCEPTS)
FORMS = {'10-Q', '10-K', '10-Q/A', '10-K/A', '10-KT', '20-F', '40-F', '6-K'}


def universe():
    m = pd.read_parquet(f'{EQ}/processed/meta.parquet')
    s = m[(m.last_member_date_pit.isna()) | (m.last_member_date_pit >= '2015-01-01')].copy()
    s = s[s.cik.astype(str).str.fullmatch(r'\d+')]
    s['cik'] = s.cik.astype(str).str.lstrip('0').astype(int)
    s = s[['ticker', 'company_name', 'cik', 'gics_sector', 'gics_sub_industry', 'first_member_date_pit', 'last_member_date_pit']]
    s.to_csv(f'{RAW}/universe.csv', index=False)
    return s


def facts(sec, cik):
    out = f'{RAW}/facts/{cik}.jsonl.gz'
    if os.path.exists(out):
        return
    r = sec.get(f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json')
    rows = []
    if r is not None:
        j = r.json()
        for tax in ('us-gaap', 'dei', 'ifrs-full'):
            for c, d in j.get('facts', {}).get(tax, {}).items():
                if tax == 'us-gaap' and c not in CSET:
                    continue
                if tax == 'dei' and c != 'EntityCommonStockSharesOutstanding':
                    continue
                if tax == 'ifrs-full' and c not in ('Revenue', 'ProfitLoss', 'Assets', 'BasicEarningsLossPerShare', 'DilutedEarningsLossPerShare'):
                    continue
                for unit, L in d.get('units', {}).items():
                    for x in L:
                        if x.get('form') not in FORMS:
                            continue
                        rows.append({'c': c, 'u': unit, 's': x.get('start'), 'e': x['end'], 'v': x['val'], 'fy': x.get('fy'),
                                     'fp': x.get('fp'), 'form': x.get('form'), 'filed': x['filed'], 'accn': x.get('accn'),
                                     'frame': x.get('frame')})
    os.makedirs(f'{RAW}/facts', exist_ok=True)
    write_jsonl_gz(out, rows)


def filings(sec, cik):
    out = f'{RAW}/filings/{cik}.json'
    if os.path.exists(out):
        return json.load(open(out))
    r = sec.get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json')
    if r is None:
        return []
    j = r.json()
    blocks = [j['filings']['recent']]
    for f in j['filings'].get('files', []):
        if f.get('filingTo', '9999') < '2014-07-01':
            continue
        rr = sec.get('https://data.sec.gov/submissions/' + f['name'])
        if rr is not None:
            blocks.append(rr.json())
    rows = []
    for b in blocks:
        n = len(b['accessionNumber'])
        for i in range(n):
            form = b['form'][i]
            if form not in ('8-K', '10-Q', '10-K', '10-K/A', '10-Q/A'):
                continue
            fd = b['filingDate'][i]
            if fd < '2014-07-01':
                continue
            rows.append({'accn': b['accessionNumber'][i], 'form': form, 'filingDate': fd,
                         'acceptance': b['acceptanceDateTime'][i], 'reportDate': b['reportDate'][i],
                         'items': b['items'][i], 'primaryDocument': b['primaryDocument'][i]})
    os.makedirs(f'{RAW}/filings', exist_ok=True)
    json.dump({'name': j.get('name'), 'tickers': j.get('tickers'), 'sic': j.get('sic'), 'fiscalYearEnd': j.get('fiscalYearEnd'),
               'filings': rows}, open(out, 'w'))
    return json.load(open(out))


def html_text(b):
    try:
        doc = LH.fromstring(b)
    except Exception:
        return b.decode('utf-8', 'ignore') if isinstance(b, bytes) else b
    for bad in doc.xpath('//script|//style'):
        bad.drop_tree()
    # keep table rows on their own lines; cells separated by ' | '
    for td in doc.xpath('//td|//th'):
        td.tail = (td.tail or '') + ' | '
    for el in doc.xpath('//tr|//p|//div|//br|//li|//h1|//h2|//h3|//h4'):
        el.tail = (el.tail or '') + '\n'
    t = doc.text_content()
    t = t.replace('\xa0', ' ')
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'( \| )+', ' | ', t)
    t = re.sub(r'\n\s*\n+', '\n', t)
    return t.strip()


def exhibits(sec, cik, fl):
    out = f'{RAW}/ex99/{cik}.jsonl.gz'
    if os.path.exists(out):
        return 0
    rows = []
    for f in fl['filings']:
        if f['form'] != '8-K' or '2.02' not in (f['items'] or '') or f['filingDate'] < '2014-10-01':
            continue
        acc = f['accn']; nod = acc.replace('-', '')
        base = f'https://www.sec.gov/Archives/edgar/data/{cik}/{nod}/'
        r = sec.get(base + f'{acc}-index.htm')
        if r is None:
            rows.append({**f, 'cik': cik, 'ex_doc': None, 'text': None, 'err': 'noindex'}); continue
        doc = LH.fromstring(r.content)
        cands = []
        for tr in doc.xpath('//table[@class="tableFile"]//tr'):
            tds = [td.text_content().strip() for td in tr.xpath('./td')]
            links = tr.xpath('.//a/@href')
            if len(tds) >= 4 and links:
                cands.append((tds[3].upper(), links[0].split('/')[-1].replace('ix?doc=', '')))
        ex = [c for c in cands if c[0].startswith('EX-99')]
        if not ex:  # some issuers put the release in the 8-K body
            ex = [c for c in cands if c[0] == '8-K']
        if not ex:
            rows.append({**f, 'cik': cik, 'ex_doc': None, 'text': None, 'err': 'noex'}); continue
        typ, name = ex[0]
        rr = sec.get(base + name)
        if rr is None:
            rows.append({**f, 'cik': cik, 'ex_doc': name, 'text': None, 'err': 'nodoc'}); continue
        txt = html_text(rr.content) if name.lower().endswith(('htm', 'html')) else rr.content.decode('utf-8', 'ignore')
        rows.append({**f, 'cik': cik, 'ex_type': typ, 'ex_doc': name, 'n_ex99': len(ex), 'text': txt[:80000], 'text_len': len(txt)})
    os.makedirs(f'{RAW}/ex99', exist_ok=True)
    write_jsonl_gz(out, rows)
    return len(rows)


if __name__ == '__main__':
    phase = sys.argv[1]
    sec = SEC(rps=8.0)
    U = universe() if not os.path.exists(f'{RAW}/universe.csv') else pd.read_csv(f'{RAW}/universe.csv')
    ciks = sorted(set(U.cik.astype(int)))
    t0 = time.time()
    for i, cik in enumerate(ciks):
        if phase == 'A' and os.path.exists(f'{RAW}/facts/{cik}.jsonl.gz') and os.path.exists(f'{RAW}/filings/{cik}.json'):
            continue
        if phase == 'A':
            facts(sec, cik); filings(sec, cik)
        else:
            fl = filings(sec, cik)
            n = exhibits(sec, cik, fl) if fl else 0
        if i % 10 == 0:
            print(f'{time.strftime("%H:%M:%S")} {phase} {i}/{len(ciks)} cik={cik} req={sec.n} {time.time()-t0:.0f}s', flush=True)
    print('DONE', phase, len(ciks), sec.n, flush=True)
