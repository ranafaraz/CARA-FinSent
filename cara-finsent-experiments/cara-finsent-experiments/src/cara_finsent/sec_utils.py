from __future__ import annotations

import re
import time
from typing import Dict

import requests
from bs4 import BeautifulSoup

SEC_TICKERS_URL = 'https://www.sec.gov/files/company_tickers.json'
SEC_SUBMISSIONS_URL = 'https://data.sec.gov/submissions/CIK{cik10}.json'
SEC_ARCHIVES_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dashes}/{primary_doc}'


def sec_headers(user_agent: str) -> dict:
    if not user_agent or '@' not in user_agent:
        raise ValueError('SEC requires a descriptive USER_AGENT with contact email, e.g. "Your Name your@email.com"')
    return {'User-Agent': user_agent, 'Accept-Encoding': 'gzip, deflate', 'Host': 'www.sec.gov'}


def load_ticker_cik_map(user_agent: str) -> Dict[str, str]:
    r = requests.get(SEC_TICKERS_URL, headers=sec_headers(user_agent), timeout=30)
    r.raise_for_status()
    data = r.json()
    mapping = {}
    for _, item in data.items():
        ticker = item['ticker'].upper()
        cik = str(item['cik_str']).zfill(10)
        mapping[ticker] = cik
    return mapping


def fetch_submissions(cik10: str, user_agent: str) -> dict:
    url = SEC_SUBMISSIONS_URL.format(cik10=cik10)
    headers = sec_headers(user_agent)
    headers['Host'] = 'data.sec.gov'
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def filing_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'table']):
        tag.decompose()
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fetch_filing_text(cik10: str, accession: str, primary_doc: str, user_agent: str) -> tuple[str, str]:
    cik_no_zeros = str(int(cik10))
    accession_no_dashes = accession.replace('-', '')
    url = SEC_ARCHIVES_URL.format(cik=cik_no_zeros, accession_no_dashes=accession_no_dashes, primary_doc=primary_doc)
    r = requests.get(url, headers=sec_headers(user_agent), timeout=60)
    r.raise_for_status()
    return filing_text_from_html(r.text), url


def rough_extract_item(text: str, item: str = '1A') -> str:
    clean = re.sub(r'\s+', ' ', text)
    item_pattern = item.replace('.', '\\.')
    start_re = re.compile(rf'item\s+{item_pattern}\.?\s+', re.I)
    starts = list(start_re.finditer(clean))
    if not starts:
        return ''
    start = starts[0].start()
    next_item_re = re.compile(r'item\s+(1B|2|3|4|5|6|7|7A|8|9|10|11|12|13|14|15)\.?\s+', re.I)
    nexts = [m.start() for m in next_item_re.finditer(clean, pos=start + 10) if m.start() > start]
    end = min(nexts) if nexts else min(len(clean), start + 200000)
    return clean[start:end].strip()


def sleep_polite(seconds: float):
    if seconds > 0:
        time.sleep(seconds)
