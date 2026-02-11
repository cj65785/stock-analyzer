# analyzer.py
import os
import re
import asyncio
import aiohttp
import OpenDartReader
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
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
        self.dart = OpenDartReader(api_key)
        self.cache_dir = "dart_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def process(self, company_name: str, stock_code: str = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        DART 보고서 조회
        
        Args:
            company_name: 회사명
            stock_code: 종목코드 (예: 'A005930')
        
        Returns:
            (보고서명, 보고서내용, 에러메시지)
        """
        cache_file = os.path.join(self.cache_dir, f"{company_name.replace(' ', '_')}.txt")
        
        # 캐시 확인
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
            # 종목코드가 있으면 코드로 조회 (우선)
            if stock_code:
                # 'A' 제거하여 순수 숫자만 추출
                clean_code = stock_code.replace('A', '').replace('a', '').strip()
                
                # 종목코드로 최근 사업보고서 조회
                try:
                    reports = self.dart.list(corp_code=clean_code, kind='A', kind_detail='A001')
                    if reports.empty:
                        return None, None, f"[종목코드: {clean_code}] DART 사업보고서 없음"
                    
                    # 가장 최근 보고서
                    latest = reports.iloc[0]
                    report_nm = latest['report_nm']
                    rcept_no = latest['rcept_no']
                    
                except Exception as e:
                    # 종목코드 조회 실패 시 종목명으로 재시도
                    return self._search_by_name(company_name, cache_file)
            
            # 종목코드가 없으면 종목명으로 검색 (기존 방식)
            else:
                return self._search_by_name(company_name, cache_file)
            
            # 보고서 본문 가져오기
            doc = self.dart.document(rcept_no)
            
            if not doc or doc.empty:
                return report_nm, None, "보고서 본문 없음"
            
            # 텍스트 추출
            dart_text = '\n'.join(doc['bsns_summ_ctnt'].dropna().astype(str).tolist())
            
            if not dart_text or len(dart_text) < 100:
                return report_nm, None, "보고서 내용 부족"
            
            # 캐시 저장
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(f"{report_nm}\n{dart_text}")
            except:
                pass
            
            return report_nm, dart_text, None
            
        except Exception as e:
            return None, None, f"DART 처리 오류: {str(e)}"
    
    def _search_by_name(self, company_name: str, cache_file: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """종목명으로 DART 검색"""
        try:
            reports = self.dart.list(company=company_name, kind='A', kind_detail='A001')
            if reports.empty:
                return None, None, "DART 사업보고서 없음"
            
            # 종목명으로 필터링
            reports = reports[reports['corp_name'].str.contains(company_name, na=False)]
            if reports.empty:
                return None, None, f"'{company_name}' 관련 보고서 없음"
            
            # 가장 최근 보고서
            latest = reports.iloc[0]
            report_nm = latest['report_nm']
            rcept_no = latest['rcept_no']
            
            # 보고서 본문 가져오기
            doc = self.dart.document(rcept_no)
            
            if not doc or doc.empty:
                return report_nm, None, "보고서 본문 없음"
            
            # 텍스트 추출
            dart_text = '\n'.join(doc['bsns_summ_ctnt'].dropna().astype(str).tolist())
            
            if not dart_text or len(dart_text) < 100:
                return report_nm, None, "보고서 내용 부족"
            
            # 캐시 저장
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(f"{report_nm}\n{dart_text}")
            except:
                pass
            
            return report_nm, dart_text, None
            
        except Exception as e:
            return None, None, f"종목명 검색 실패: {str(e)}"

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
        params = {
            "query": company_name,
            "display": 100,
            "start": offset + 1,
            "sort": "date"
        }
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    break
                
                data = await response.json()
                items = data.get('items', [])
                
                if not items:
                    break
                
                for item in items:
                    # HTML 태그 제거
                    title = BeautifulSoup(item['title'], 'html.parser').get_text()
                    description = BeautifulSoup(item['description'], 'html.parser').get_text()
                    
                    # 날짜 파싱
                    try:
                        pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
                        pub_date = pub_date.replace(tzinfo=None)
                    except:
                        continue
                    
                    # 날짜 필터링
                    if pub_date < datetime.strptime(start_date, '%Y%m%d') or pub_date > datetime.strptime(end_date, '%Y%m%d'):
                        continue
                    
                    articles.append({
                        'title': title,
                        'body': description,
                        'pub_date': pub_date,
                        'link': item.get('link', '')
                    })
                
                # 마지막 페이지 확인
                if len(items) < 100:
                    break
                
        except Exception as e:
            break
    
    return articles

def filter_articles(articles: List[Dict], company_name: str, regex_cache: RegexCache) -> List[Dict]:
    """기사 필터링"""
    filtered = []
    
    for article in articles:
        text = article['title'] + ' ' + article['body']
        
        # 회사명이 포함되어 있는지 확인
        if company_name in text or (regex_cache and regex_cache.search(text)):
            filtered.append(article)
    
    return filtered

async def run_news_pipeline(company_name: str, config: Config, regex_cache: RegexCache = None) -> Tuple[List[Dict], int]:
    """뉴스 수집 파이프라인"""
    # 최근 6개월
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')
    
    async with aiohttp.ClientSession() as session:
        articles = await fetch_naver_news(session, company_name, config, start_str, end_str)
    
    # 필터링
    filtered = filter_articles(articles, company_name, regex_cache)
    
    # 중복 제거 (제목 기준)
    seen_titles = set()
    unique = []
    for article in filtered:
        if article['title'] not in seen_titles:
            seen_titles.add(article['title'])
            unique.append(article)
    
    # 날짜순 정렬
    unique.sort(key=lambda x: x['pub_date'], reverse=True)
    
    return unique, len(unique)

