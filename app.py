# app.py (System Log Mode - Final Clean Version)
import streamlit as st
import asyncio
import pandas as pd
import time
import warnings
import math
from datetime import datetime
from openai import AsyncOpenAI
from io import BytesIO
from database import Database
from analyzer import (
    Config, RegexCache, DartProcessor, 
    run_news_pipeline
)

# 1. 설정 및 초기화
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

st.set_page_config(
    page_title="Server Logs",  # 제목을 서버 로그로 위장
    page_icon="terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS: 불필요한 장식 제거, 가독성 확보 (폰트 크기 정상화)
st.markdown("""
<style>
    /* 상단 헤더 숨김 */
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* 탭 스타일: 심플한 텍스트 */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: bold;
    }
    
    /* Expander(리스트): 깔끔한 로그 스타일 */
    .streamlit-expanderHeader {
        font-family: 'Consolas', 'Courier New', monospace; /* 개발자 폰트 느낌 */
        font-size: 14px;
        color: #333;
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    
    /* 본문 텍스트 박스 스타일 */
    .log-box {
        background-color: #f1f3f5;
        border-left: 4px solid #adb5bd;
        padding: 10px;
        margin-bottom: 10px;
        font-family: 'Malgun Gothic', sans-serif;
        font-size: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .log-label {
        font-size: 12px;
        font-weight: bold;
        color: #555;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 3. 데이터베이스 & 설정 연결
@st.cache_resource
def get_database():
    return Database(st.secrets.get("DATABASE_URL"))

db = get_database()

@st.cache_resource
def get_config():
    return Config(
        CLIENT_ID=st.secrets.get("NAVER_CLIENT_ID"),
        CLIENT_SECRET=st.secrets.get("NAVER_CLIENT_SECRET"),
        DART_API_KEY=st.secrets.get("DART_API_KEY"),
        OPENAI_API_KEY=st.secrets.get("OPENAI_API_KEY")
    )

config = get_config()
openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)

@st.cache_resource
def load_companies():
    try:
        try: df = pd.read_csv('krx_stocks.csv', encoding='cp949')
        except: df = pd.read_csv('krx_stocks.csv', encoding='utf-8')
        code_map = dict(zip(df['종목명'], df['종목코드']))
        companies = df['종목명'].dropna().astype(str).str.strip().tolist()
        return companies, RegexCache(companies), code_map
    except: return [], None, {}

ALL_COMPANIES, REGEX_CACHE, CODE_MAP = load_companies()

# 4. 분석 로직 (간결 요약)
async def analyze_news_with_gpt(company_name: str, articles: list) -> str:
    if not articles: return "No Data."
    articles.sort(key=lambda x: x['pub_date'], reverse=True)
    context = ""
    for i, art in enumerate(articles[:5]): # 뉴스 너무 많으면 토큰 낭비니 상위 5개만
        d = art['pub_date'].strftime('%y-%m-%d')
        context += f"[{d}] {art['title']}\n"
    
    prompt = f"'{company_name}' 뉴스. 주가 재료 위주 3줄 요약. 음슴체.\n{context}"
    try:
        res = await openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e: return f"Err: {e}"

async def analyze_dart_with_gpt(company_name: str, report_nm: str, dart_text: str) -> str:
    if not dart_text or len(dart_text) < 100: return "No Data."
    prompt = f"'{company_name}' 공시({report_nm}). 핵심 호재 3줄 요약. 음슴체.\n{dart_text[:20000]}"
    try:
        res = await openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e: return f"Err: {e}"

async def analyze_company(company_name: str, stock_code: str = None, progress_callback=None):
    if progress_callback: progress_callback(f"Analyzing {company_name}...")
    dart_proc = DartProcessor(config.DART_API_KEY)
    r_nm, d_txt, d_err = dart_proc.process(company_name, stock_code)
    d_res = await analyze_dart_with_gpt(company_name, r_nm, d_txt) if d_txt else "-"
    arts, cnt = await run_news_pipeline(company_name, config, REGEX_CACHE)
    n_res = await analyze_news_with_gpt(company_name, arts)
    
    db.add_result(company_name=company_name, dart_report=r_nm or "-", dart_result=d_res, dart_error=d_err or "", news_count=cnt, news_result=n_res)
    return True

# 5. 메인 UI (탭 구성)
tab1, tab2, tab3 = st.tabs(["EXECUTE", "LOGS", "ARCHIVE"])

# [Tab 1] 실행 (심플한 입력창)
with tab1:
    if 'is_processing' not in st.session_state: st.session_state.is_processing = False
    if 'pending_companies' not in st.session_state: st.session_state.pending_companies = []

    c1, c2 = st.columns([8, 1])
    with c1:
        companies_input = st.text_area("Target Input", value='\n'.join(st.session_state.pending_companies) if st.session_state.pending_companies and not st.session_state.is_processing else "", height=100, label_visibility="collapsed", placeholder="Enter targets...")
    with c2:
        if st.button("RUN", use_container_width=True, disabled=st.session_state.is_processing):
            if companies_input.strip():
                st.session_state.pending_companies = [c.strip() for c in companies_input.split('\n') if c.strip()]
                st.session_state.is_processing = True
                st.rerun()

    # 진행상황 (로그 스타일)
    if st.session_state.is_processing and st.session_state.pending_companies:
        BATCH = 5
        curr = st.session_state.pending_companies[:BATCH]
        st.code(f">> Processing batch... Remaining: {len(st.session_state.pending_companies)}")
        
        for c in curr:
            asyncio.run(analyze_company(c, CODE_MAP.get(c)))
        
        st.session_state.pending_companies = st.session_state.pending_companies[BATCH:]
        
        if st.session_state.pending_companies:
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.is_processing = False
            st.rerun()

# [Tab 2] 결과 목록 (System Log 스타일)
with tab2:
    if 'page' not in st.session_state: st.session_state.page = 1
    all_res = db.get_all_results(limit=10000)
    
    # 검색바
    search = st.text_input("Grep", placeholder="Search keyword...", label_visibility="collapsed")
    targets = [r for r in all_res if search in r['company_name']] if search else all_res
    
    # 페이징
    PER_PAGE = 50
    total_pg = math.ceil(len(targets)/PER_PAGE) if targets else 1
    if st.session_state.page > total_pg: st.session_state.page = 1
    start = (st.session_state.page-1)*PER_PAGE
    view_data = targets[start:start+PER_PAGE]

    # 리스트 출력
    if not view_data:
        st.caption("No logs found.")
    else:
        for i, row in enumerate(view_data):
            # 인덱싱 및 날짜 포맷
            idx_num = len(targets) - (start + i)
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime('%m-%d %H:%M')
            
            # 즐겨찾기 표시 (★)
            star = "★" if row.get('is_bookmarked') else ""
            
            # 제목 생성: [번호] 종목명 (날짜) ★
            # 모바일에서도 잘 보이게 폰트 크기 정상화된 Expander 사용
            label = f"[{idx_num:03d}] {row['company_name']} {star} | {date_str}"
            
            with st.expander(label):
                # 1. 상단 버튼 (가로 배치)
                b1, b2, b3 = st.columns([2, 2, 6])
                with b1:
                    if st.button(f"{'Unsave' if row.get('is_bookmarked') else 'Save'}", key=f"s_{row['id']}"):
                        db.toggle_bookmark(row['id'])
                        st.rerun()
                with b2:
                    if st.button("Delete", key=f"d_{row['id']}"):
                        db.delete_result(row['id'])
                        st.rerun()
                
                # 2. 본문 내용 (박스 형태)
                st.markdown(f"<div class='log-label'>[DART Report: {row['dart_report']}]</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='log-box'>{row['dart_result']}</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='log-label'>[News Summary ({row['news_count']} articles)]</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='log-box'>{row['news_result']}</div>", unsafe_allow_html=True)

    # 페이징 컨트롤
    st.write("---")
    c_p, c_c, c_n = st.columns([1, 2, 1])
    with c_p:
        if st.session_state.page > 1 and st.button("< Prev"):
            st.session_state.page -= 1
            st.rerun()
    with c_c:
        st.markdown(f"<div style='text-align:center;'>Page {st.session_state.page} / {total_pg}</div>", unsafe_allow_html=True)
    with c_n:
        if st.session_state.page < total_pg and st.button("Next >"):
            st.session_state.page += 1
            st.rerun()

# [Tab 3] 아카이브 (동일 스타일)
with tab3:
    bk_list = db.get_bookmarked_results()
    
    # 엑셀 다운로드
    if bk_list:
        df = pd.DataFrame(bk_list)
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        out.seek(0)
        st.download_button("Export Excel", data=out, file_name="archive.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    st.write("") # 간격
    
    if not bk_list:
        st.caption("No archived logs.")
    else:
        for row in bk_list:
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            date_str = dt.strftime('%m-%d')
            
            with st.expander(f"{row['company_name']} | {date_str}"):
                if st.button("Remove", key=f"rm_{row['id']}"):
                    db.toggle_bookmark(row['id'])
                    st.rerun()
                
                st.markdown(f"<div class='log-label'>DART</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='log-box'>{row['dart_result']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='log-label'>NEWS</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='log-box'>{row['news_result']}</div>", unsafe_allow_html=True)
