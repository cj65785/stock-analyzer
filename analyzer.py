# analyzer.py
import asyncio
import aiohttp
import datetime
import re
import requests
import OpenDartReader
import pandas as pd
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
from collections import defaultdict

class Config:
    def __init__(self, CLIENT_ID: str, CLIENT_SECRET: str, DART_API_KEY: str, OPENAI_API_KEY: str):
        self.CLIENT_ID = CLIENT_ID
        self.CLIENT_SECRET = CLIENT_SECRET
        self.DART_API_KEY = DART_API_KEY
        self.OPENAI_API_KEY = OPENAI_API_KEY
        
        # 기본값들
        self.MONTHS_AGO = 6
        self.MAX_CONCURRENT = 10
        self.REQUEST_TIMEOUT = 20
        self.RETRY_COUNT = 3
        self.MIN_BODY_LENGTH = 100
        self.MAX_OTHER_COMPANIES = 5
        self.SIMILARITY_THRESHOLD = 0.6
        self.BODY_HEAD_CHECK = 2000
        
        self.KEYWORDS = [
            "매출", "수출", "계약", "수주", "출시", "허가", "양산", "인수", "진출", "신사업", "투자", "공급"
        ]
        
        self.TITLE_BLACKLIST = [
            "특징주", "목표가", "신고가", "급락", "급등", "상한가", "폭등", "상승폭", "하락폭", "상승률",
            "급등락", "장마감", "시황", "[특징주]", "[속보]","장을 마쳤다", "일 장중", "오늘의 주목주", "전날보다",
            "상승 마감", "하락 마감", "주말뉴스 FULL", "팍스경제TV","동일업종 등락률", "거래일 종가", "투자 알고리즘",
            "브리핑", "바이오스냅", "공시모음", "e공시", "e종목", "더밸류", "데일리인베스트", "IB토마토", "인포스탁",
            "버핏 연구소", "리얼스탁", "한경유레카", "헬로스톡", "로보인베스팅", "골든클럽", "투자원정대", "오늘의 IR", "주요 공시", "IR Page",
            "스포츠", "법률신문", "조세회계", "표창", "훈장","기념식", "후원", "선임", "광고",
            "포럼", "증여", "상속", "수요예측", "문화대상", "브랜드평", "상장폐지", "로펌", "횡령", "VC 하우스", "주식쇼", "데이터랩",
            "오류안내", "후속주", "로또", "평판지수", "브랜드평판", "지금이뉴스", "사외이사", "별세", "저PER", "사람인",
            "사업자등록번호", "3파전", "엔지니어상", "장관 표창", "내달 퇴임", "소집공고", "지분 매각", "주식등의 대량보유자", "who is?",
            "개인정보 항목", "면접 후기", "채용", "부시장", "민원처리반", "임금 체불", "총동문회", "점포거래소", "투자 핫플레이스",
            "[상보]", "유료서비스", "marketin", "프리미엄", "simplywall", "AI리포터", "DealSite", "지속가능경영보고서"
        ]
        
        self.BODY_BLACKLIST = self.TITLE_BLACKLIST.copy()


class RegexCache:
    def __init__(self, companies: List[str]):
        self.companies = companies
        self.patterns = {}
        for company in companies:
            pattern = f"{re.escape(company)}(?=[ 은는이가을를의와과로서에.,\"'\n\r]|$|[^가-힣a-zA-Z0-9])"
            self.patterns[company] = re.compile(pattern)
    
    def count_matches(self, text: str, exclude: str = None) -> int:
        count = 0
        for company, pattern in self.patterns.items():
            if company == exclude:
                continue
            if pattern.search(text):
                count += 1
                if count >= 10:
                    break
        return count
    
    def find_any(self, text: str, exclude: str = None) -> bool:
        for company, pattern in self.patterns.items():
            if company == exclude:
                continue
            if pattern.search(text):
                return True
        return False


def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("&quot;", '"').replace("&lt;", "<")
    text = text.replace("&gt;", ">").replace("&amp;", "&")
    return text.strip()


def parse_date(date_str: str) -> Optional[datetime.datetime]:
    try:
        dt = datetime.datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z")
        return dt.replace(tzinfo=None)
    except:
        return None


def similarity(s1: str, s2: str) -> float:
    return SequenceMatcher(None, s1, s2).ratio()


def clean_body_final(text: str) -> str:
    if not text:
        return ""
    
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_match = re.search(email_pattern, text)
    if email_match:
        text = text[:email_match.start()].strip()
    
    cutoff_patterns = [
        r'관련\s*기사', r'다른\s*기사', r'추천\s*기사', r'인기\s*기사',
        r'더\s*보기', r'See more', r'Tag\s*#', r'#바이오',
        r'저작권자', r'무단\s*전재', r'재배포\s*금지',
        r'Copyright', r'All rights reserved', r'개인정보\s*보호',
        r'구독\s*신청', r'뉴스\s*스탠드', r'좋아요\s*슬퍼요',
        r'기사\s*제보', r'댓글\s*작성', r'많이\s*본\s*뉴스',
        r'지금\s*뜨는', r'공유하기', r'URL\s*복사',
        r'글자\s*크기', r'기사\s*듣기', r'인쇄하기', r'읽기모드',
    ]
    
    for pattern in cutoff_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()].strip()
    
    lines = text.split('\n')
    clean_lines = []
    
    noise_patterns = [
        r'기자\s*=', r'특파원\s*=', r'©|ⓒ',
        r'사진\s*=', r'출처\s*:', r'자료\s*:',
        r'\d{2,4}-\d{2,4}-\d{4}', r'FAX|Fax|fax',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 15:
            continue
        if any(re.search(p, line) for p in noise_patterns):
            continue
        clean_lines.append(line)
    
    text = '\n'.join(clean_lines)
    text = re.sub(r'\n{2,}', '\n', text)
    
    return text.strip()


class HTTPClient:
    def __init__(self, config: Config):
        self.config = config
        self.session = None
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.REQUEST_TIMEOUT)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str) -> Tuple[int, str]:
        for attempt in range(self.config.RETRY_COUNT):
            try:
                async with self.session.get(url) as resp:
                    content = await resp.read()
                    try:
                        text = content.decode('utf-8')
                    except:
                        try:
                            text = content.decode('euc-kr')
                        except:
                            text = content.decode('cp949', errors='ignore')
                    return (resp.status, text)
            except Exception as e:
                if attempt == self.config.RETRY_COUNT - 1:
                    pass
                await asyncio.sleep(0.5 * (attempt + 1))
        return (0, "")


async def extract_body(url: str, client: HTTPClient) -> str:
    status, html = await client.fetch(url)
    if status != 200 or not html:
        return ""
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        selectors = [
            'div#dic_area', 'div#articleBodyContents', 'div.article_body',
            'div#article-view-content-div', 'div.news_cnt_detail_wrap',
            'article.article-body', 'div#newsct_article', 'div.article-body',
            'article', 'div#content',
        ]
        
        body_elem = None
        for sel in selectors:
            body_elem = soup.select_one(sel)
            if body_elem:
                break
        
        if not body_elem:
            body_elem = soup.find('body') or soup
        
        for tag in body_elem.find_all(['script', 'style', 'header', 'footer',
                                      'nav', 'aside', 'form', 'iframe', 'button']):
            tag.decompose()
        
        text = body_elem.get_text(separator='\n')
        body = clean_body_final(text)
        
        return body
        
    except:
        return ""


async def search_naver(target: str, config: Config, regex_cache: RegexCache) -> List[Dict]:
    cutoff = datetime.datetime.now() - datetime.timedelta(days=config.MONTHS_AGO * 30)
    headers = {
        "X-Naver-Client-Id": config.CLIENT_ID,
        "X-Naver-Client-Secret": config.CLIENT_SECRET
    }
    
    collected = []
    seen_urls = set()
    
    async with aiohttp.ClientSession(headers=headers) as session:
        for keyword in config.KEYWORDS:
            query = f'"{target}" "{keyword}"'
            
            for start in range(1, 1001, 100):
                url = f"https://openapi.naver.com/v1/search/news.json?query={quote(query)}&display=100&start={start}&sort=date"
                
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            break
                        data = await resp.json()
                        items = data.get('items', [])
                        if not items:
                            break
                        
                        stop = False
                        for item in items:
                            pub_date = parse_date(item.get('pubDate', ''))
                            if not pub_date or pub_date < cutoff:
                                stop = True
                                break
                            
                            link = item.get('originallink') or item.get('link')
                            if link in seen_urls:
                                continue
                            
                            title = clean_html(item.get('title', ''))
                            
                            bl_found = None
                            for bl in config.TITLE_BLACKLIST:
                                if bl in title:
                                    bl_found = bl
                                    break
                            if bl_found:
                                continue
                            
                            if target not in title:
                                if regex_cache.find_any(title, exclude=target):
                                    continue
                            
                            seen_urls.add(link)
                            collected.append({
                                'title': title,
                                'link': link,
                                'date': item['pubDate'],
                                'pub_date': pub_date
                            })
                        
                        if stop:
                            break
                except Exception as e:
                    break
    
    return collected


def deduplicate(articles: List[Dict], threshold: float) -> List[Dict]:
    seen_urls = set()
    by_date = defaultdict(list)
    unique = []
    
    for art in articles:
        url = art['link']
        if url in seen_urls:
            continue
        
        date_key = art['pub_date'].strftime('%Y-%m-%d')
        
        is_dup = False
        for existing in by_date[date_key]:
            if similarity(art['title'], existing['title']) >= threshold:
                is_dup = True
                break
        
        if is_dup:
            continue
        
        seen_urls.add(url)
        by_date[date_key].append(art)
        unique.append(art)
    
    return unique


class DartProcessor:
    def __init__(self, api_key: str):
        import shutil
        from pathlib import Path
        
        cache_dir = Path.home() / '.OpenDart'
        
        try:
            self.dart = OpenDartReader(api_key)
        except Exception as e:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            self.dart = OpenDartReader(api_key)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            if len(line) < 3: continue
            lines.append(line)
        return '\n'.join(lines).strip()

    def find_listed_corp_code(self, company_name: str, stock_code: str = None) -> Optional[str]:
        """종목코드 또는 종목명으로 corp_code 찾기"""
        try:
            df = self.dart.corp_codes
            
            # 1. 종목코드로 먼저 찾기 (우선)
            if stock_code:
                clean_code = stock_code.replace('A', '').replace('a', '').strip().zfill(6)
                matched = df[df['stock_code'] == clean_code]
                if not matched.empty:
                    return matched.iloc[0]['corp_code']
            
            # 2. 종목명으로 찾기 (fallback)
            candidates = df[df['corp_name'] == company_name]
            if candidates.empty:
                candidates = df[df['corp_name'].str.replace(" ", "") == company_name.replace(" ", "")]
            
            if candidates.empty:
                return None
                
            if 'stock_code' in candidates.columns:
                listed = candidates[candidates['stock_code'].notnull() & (candidates['stock_code'].str.strip() != '')]
                if not listed.empty:
                    return listed.iloc[0]['corp_code']
            
            return candidates.iloc[0]['corp_code']
        except Exception as e:
            return None

def _get_latest_report_code(self, corp_code):
        """
        가장 최신의 정기공시(사업/반기/분기)를 찾아 보고서 번호와 제목을 반환합니다.
        (단순 사업보고서만 찾으면 1년 전 데이터를 볼 위험이 있어 수정함)
        """
        try:
            # 1년치 공시 목록 조회
            end_dt = datetime.datetime.now().strftime('%Y%m%d')
            start_dt = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
            
            # 전체 공시 목록 가져오기
            reports = self.dart.list(corp_code=corp_code, start=start_dt, end=end_dt, final=False)
            
            if reports is None or reports.empty:
                return None, None
            
            # 보고서명에 '사업보고서', '분기보고서', '반기보고서'가 포함된 것만 필터링
            target_reports = reports[reports['report_nm'].str.contains('사업보고서|분기보고서|반기보고서', regex=True)]
            
            if target_reports.empty:
                return None, None
                
            # 접수일자(rcept_dt) 기준 내림차순 정렬하여 가장 최신 것 선택
            latest = target_reports.sort_values(by='rcept_dt', ascending=False).iloc[0]
            
            return latest['rcept_no'], latest['report_nm']
            
        except Exception as e:
            print(f"DART 목록 조회 실패: {e}")
            return None, None

    def _extract_core_content(self, text):
        """
        보고서 전체가 아니라 'II. 사업의 내용' 등 핵심 파트만 추출합니다.
        (GPT 토큰 절약 및 정확도 향상)
        """
        try:
            # 정규식으로 '사업의 내용' 섹션 추출 시도
            # 패턴: "II. 사업의 내용" ~ "III. 재무" 사이
            pattern = r'(II\.?|2\.)\s*사업의\s*내용.*?(III\.?|3\.)\s*재무'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            
            if match:
                # 찾았으면 해당 부분만 반환
                return match.group(0).strip()
            else:
                # 못 찾았으면 앞부분 30,000자만 반환 (너무 길면 잘림)
                return text[:30000]
        except:
            return text[:30000]

    def process(self, company_name: str, stock_code: str = None) -> Tuple[str, str, str]:
        try:
            # 종목코드가 없으면 DART에서 찾기
            if not stock_code:
                code = self.dart.find_corp_code(company_name)
            else:
                code = stock_code
                
            if not code:
                return None, None, "DART 기업코드를 찾을 수 없습니다."

            # [수정] 최신 보고서(분기/반기 포함) 찾기
            rcept_no, report_nm = self._get_latest_report_code(code)
            
            if not rcept_no:
                return None, None, "최근 1년 내 정기공시(사업/반기/분기)가 없습니다."

            # 보고서 원문 다운로드
            xml_text = self.dart.document(rcept_no)
            if not xml_text:
                return report_nm, None, "보고서 원문 데이터가 비어있습니다."

            # [수정] 핵심 내용만 스마트하게 추출
            dart_text = self._extract_core_content(xml_text)
            
            return report_nm, dart_text, ""
            
        except Exception as e:
            return None, None, f"DART 처리 중 오류: {str(e)}"


async def run_news_pipeline(target: str, config: Config, regex_cache: RegexCache) -> Tuple[List[Dict], int]:
    articles = await search_naver(target, config, regex_cache)
    if not articles:
        return [], 0
    
    articles = deduplicate(articles, config.SIMILARITY_THRESHOLD)
    
    semaphore = asyncio.Semaphore(config.MAX_CONCURRENT)
    
    async with HTTPClient(config) as client:
        async def process(art):
            async with semaphore:
                body = await extract_body(art['link'], client)
                
                if not body:
                    return None
                if len(body) < config.MIN_BODY_LENGTH:
                    return None
                if target not in body:
                    return None
                if target not in body[:config.BODY_HEAD_CHECK]:
                    return None
                if regex_cache.count_matches(body[:3000], exclude=target) >= config.MAX_OTHER_COMPANIES:
                    return None
                for bl in config.BODY_BLACKLIST:
                    if bl in body:
                        return None
                
                art['body'] = body
                return art
        
        tasks = [process(art) for art in articles]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if r]
    
    valid.sort(key=lambda x: x['pub_date'], reverse=True)
    return valid, len(valid)

