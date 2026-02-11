# analyzer.py
import os
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import OpenDartReader
from bs4 import BeautifulSoup

class Config:
    """설정 클래스"""
    def __init__(self, CLIENT_ID: str, CLIENT_SECRET: str, DART_API_KEY: str, OPENAI_API_KEY: str):
        self.CLIENT_ID = CLIENT_ID
        self.CLIENT_SECRET = CLIENT_SECRET
        self.DART_API_KEY = DART_API_KEY
        self.OPENAI_API_KEY = OPENAI_API_KEY

class RegexCache:
    """정규식 캐시 클래스"""
    def __init__(self, companies: List[str]):
        self.companies = companies
        escaped = [re.escape(c) for c in companies]
        pattern = '|'.join(escaped)
        self.regex = re.compile(pattern)
    
    def search(self, text: str) -> Optional[re.Match]:
        return self.regex.search(text)

class DartProcessor:
    """DART 보고서 처리 클래스"""
    def __init__(self, api_key: str):
        self.dart = OpenDartReader.OpenDartReader(api_key)
        self.cache_dir = "dart_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def process(self, company_name: str, stock_code: str = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """DART 보고서 조회"""
        cache_file = os.path.join(self.cache_dir, f"{company_name.replace(' ', '_')}.txt")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        report_nm = lines[0].strip()
                        dart_text = ''.join(lines[1:])
                        return report_nm, dart_text, None
            except:
                pass
        
        try:
            corp_code = None
            
            if stock_code:
                clean_code = stock_code.replace('A', '').replace('a', '').strip().zfill(6)
                try:
                    corp_list = self.dart.corp_codes
                    matched = corp_list[corp_list['stock_code'] == clean_code]
                    if not matched.empty:
                        corp_code = matched.iloc[0]['corp_code']
                except:
                    pass
            
            if not corp_code:
                try:
                    corp_list = self.dart.corp_codes
                    matched = corp_list[corp_list['corp_name'].str.contains(company_name, na=False)]
                    if matched.empty:
                        return None, None, f"'{company_name}' 기업코드를 찾을 수 없음"
                    corp_code = matched.iloc[0]['corp_code']
                except Exception as e:
                    return None, None, f"기업코드 검색 실패: {str(e)}"
            
            try:
                reports = self.dart.list(corp_code=corp_code, kind='A', kind_detail='A001')
                if reports.empty:
                    return None, None, "DART 사업보고서 없음"
                latest = reports.iloc[0]
                report_nm = latest['report_nm']
                rcept_no = latest['rcept_no']
            except Exception as e:
                return None, None, f"보고서 조회 실패: {str(e)}"
            
            try:
                doc = self.dart.document(rcept_no)
                if not doc or doc.empty:
                    return report_nm, None, "보고서 본문 없음"
                dart_text = '\n'.join(doc['bsns_summ_ctnt'].dropna().astype(str).tolist())
                if not dart_text or len(dart_text) < 100:
                    return report_nm, None, "보고서 내용 부족"
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        f.write(f"{report_nm}\n{dart_text}")
                except:
                    pass
                return report_nm, dart_text, None
            except Exception as e:
                return report_nm, None, f"보고서 본문 조회 실패: {str(e)}"
        except Exception as e:
            return None, None, f"DART 처리 오류: {str(e)}"

async def fetch_naver_news(session: aiohttp.ClientSession, company_name: str, 
                          config: Config, start_date: str, end_date: str) -> List[Dict]:
    """네이버 뉴스 검색"""
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": config.CLIENT_ID,
        "X-Naver-Client-Secret": config.CLIENT_SECRET
    }
    articles = []
    for offset in range(0, 1000, 100):
        params = {"query": company_name, "display": 100, "start": offset + 1, "sort": "date"}
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    break
                data = await response.json()
                items = data.get('items', [])
                if not items:
                    break
                for item in items:
                    title = BeautifulSoup(item['title'], 'html.parser').get_text()
                    description = BeautifulSoup(item['description'], 'html.parser').get_text()
                    try:
                        pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
                        pub_date = pub_date.replace(tzinfo=None)
                    except:
                        continue
                    if pub_date < datetime.strptime(start_date, '%Y%m%d') or pub_date > datetime.strptime(end_date, '%Y%m%d'):
                        continue
                    articles.append({'title': title, 'body': description, 'pub_date': pub_date, 'link': item.get('link', '')})
                if len(items) < 100:
                    break
        except:
            break
    return articles

def filter_articles(articles: List[Dict], company_name: str, regex_cache: RegexCache) -> List[Dict]:
    """기사 필터링"""
    filtered = []
    for article in articles:
        text = article['title'] + ' ' + article['body']
        if company_name in text or (regex_cache and regex_cache.search(text)):
            filtered.append(article)
    return filtered

async def run_news_pipeline(company_name: str, config: Config, regex_cache: RegexCache = None) -> Tuple[List[Dict], int]:
    """뉴스 수집 파이프라인"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    async with aiohttp.ClientSession() as session:
        articles = await fetch_naver_news(session, company_name, config, start_str, end_str)
    filtered = filter_articles(articles, company_name, regex_cache)
    seen_titles = set()
    unique = []
    for article in filtered:
        if article['title'] not in seen_titles:
            seen_titles.add(article['title'])
            unique.append(article)
    unique.sort(key=lambda x: x['pub_date'], reverse=True)
    return unique, len(unique)
