# app.py (Mobile-Friendly BBS Mode)
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

# [설정] 경고 차단
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# 페이지 설정
st.set_page_config(
    page_title="System Admin", 
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 (모바일 최적화 & 공백 제거)
st.markdown("""
<style>
    /* 1. 전체 여백 제거 (모바일 화면 활용 극대화) */
    header {visibility: hidden;}
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.5rem !important; /* 모바일 좌우 여백 최소화 */
        padding-right: 0.5rem !important;
    }
    
    /* 2. 요소 간격 강제 삭제 */
    .element-container { margin-bottom: 0rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0rem !important; }
    
    /* 3. Expander (게시판 리스트 스타일) */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        border-bottom: 1px solid #e0e0e0 !important;
        margin-bottom: 0px !important;
        border-radius: 0px !important;
    }
    .stExpander > details > summary {
        padding: 8px 2px !important; /* 터치하기 좋게 패딩 살짝 확보 */
        font-size: 14px !important;  /* 모바일 가독성 위해 폰트 14px */
        color: #222 !important;
        min-height: 40px !important;
    }
    .stExpander > details > summary:hover {
        background-color: #f5f5f5;
    }
    .stExpander > details > div {
        padding: 10px !important;
        background-color: #fafafa;
    }

    /* 4. 버튼 스타일 (작고 심플하게) */
    .stButton > button {
        height: 32px; /* 터치 편하게 높이 확보 */
        font-size: 13px;
        padding: 0 12px;
        border: 1px solid #ccc;
        background-color: #fff;
        width: 100%; /* 컬럼 안에서 꽉 차게 */
    }
    .stButton > button:active {
        background-color: #eee;
    }

    /* 5. 본문 텍스트 (줄간격 확보) */
    .report-text {
        font-size: 14px !important;
        line-height: 1.5 !important;
        color: #333;
        white-space: pre-wrap; /* 줄바꿈 보존 */
        margin-bottom: 10px !important;
    }
    
    /* 6. 섹션 헤더 */
    .inner-header {
        font-size: 12px;
        color: #666;
        font-weight: bold;
        border-bottom: 1px solid #ccc;
        margin-bottom: 5px !important;
        padding-bottom: 2px;
        display: block;
    }

    /* 7. 탭 스타일 */
    .stTabs [data-baseweb="tab"] {
        height: 40px; /* 터치용 높이 */
        font-size: 14px;
        padding: 0 15px;
    }
    
    /* 8. 입력창 */
    .stTextArea textarea, .stTextInput input {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 & 설정
@st.cache_resource
def get_database():
    database_url = st.secrets.get("DATABASE_URL")
    return Database(database_url)

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

# --- 분석 함수 ---
async def analyze_news_with_gpt(company_name: str, articles: list) -> str:
    if not articles: return "-"
    articles.sort(key=lambda x: x['pub_date'], reverse=True)
    context = ""
    for i, art in enumerate(articles):
        d = art['pub_date'].strftime('%y.%m.%d')
        context += f"[{d}] {art['title']}\n"
    
    prompt = f"'{company_name}' 뉴스 요약. 호재 위주. 음슴체. 3줄 이내.\n{context}"
    try:
        res = await openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e: return f"Err: {e}"

async def analyze_dart_with_gpt(company_name: str, report_nm: str, dart_text: str) -> str:
    if not dart_text or len(dart_text) < 100: return "-"
    prompt = f"'{company_name}' 공시({report_nm}) 요약. 핵심 모멘텀만. 음슴체. 3줄 이내.\n{dart_text[:30000]}"
    try:
        res = await openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e: return f"Err: {e}"

async def analyze_company(company_name: str, stock_code: str = None, progress_callback=None):
    if progress_callback: progress_callback(f"{company_name}..")
    dart_proc = DartProcessor(config.DART_API_KEY)
    r_nm, d_txt, d_err = dart_proc.process(company_name, stock_code)
    d_res = await analyze_dart_with_gpt(company_name, r_nm, d_txt) if d_txt else "-"
    arts, cnt = await run_news_pipeline(company_name, config, REGEX_CACHE)
    n_res = await analyze_news_with_gpt(company_name, arts)
    
    db.add_result(company_name=company_name, dart_report=r_nm or "-", dart_result=d_res, dart_error=d_err or "", news_count=cnt, news_result=n_res)
    return True

# ==================== UI ====================

tab1, tab2, tab3 = st.tabs(["데이터수집", "분석결과", "보관함"])

# [1] 데이터 수집
with tab1:
    if 'is_processing' not in st.session_state: st.session_state.is_processing = False
    if 'pending_companies' not in st.session_state: st.session_state.pending_companies = []

    c1, c2 = st.columns([8, 2]) # 버튼 크기 확보
    with c1:
        companies_input = st.text_area("Input", value='\n'.join(st.session_state.pending_companies) if st.session_state.pending_companies and not st.session_state.is_processing else "", height=80, label_visibility="collapsed", placeholder="종목명 입력 (엔터 구분)")
    with c2:
        if st.button("실행", use_container_width=True, disabled=st.session_state.is_processing):
            if companies_input.strip():
                st.session_state.pending_companies = [c.strip() for c in companies_input.split('\n') if c.strip()]
                st.session_state.is_processing = True
                st.rerun()

    if st.session_state.is_processing and st.session_state.pending_companies:
        BATCH = 5
        curr = st.session_state.pending_companies[:BATCH]
        st.caption(f"작업중... 남은 건수: {len(st.session_state.pending_companies)}")
        for c in curr:
            asyncio.run(analyze_company(c, CODE_MAP.get(c)))
        st.session_state.pending_companies = st.session_state.pending_companies[BATCH:]
        if st.session_state.pending_companies:
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.is_processing = False
            st.rerun()

# [2] 분석 결과 (모바일 최적화 BBS)
with tab2:
    if 'page' not in st.session_state: st.session_state.page = 1
    all_res = db.get_all_results(limit=10000)
    
    # 상단 컨트롤 (검색)
    c_s, c_cnt = st.columns([7, 3])
    with c_s:
        kw = st.text_input("검색", label_visibility="collapsed", placeholder="종목명 검색")
    with c_cnt:
        st.caption(f"Total: {len(all_res)}")

    targets = [r for r in all_res if kw in r['company_name']] if kw else all_res
    
    # 페이징
    PER_PAGE = 50
    total_pg = math.ceil(len(targets)/PER_PAGE) if targets else 1
    if st.session_state.page > total_pg: st.session_state.page = 1
    start = (st.session_state.page-1)*PER_PAGE
    view_data = targets[start:start+PER_PAGE]

    # 헤더 (모바일에서는 No, 제목만 보이게)
    h = st.columns([1, 6, 3])
    h[0].markdown("<div class='inner-header'>No</div>", unsafe_allow_html=True)
    h[1].markdown("<div class='inner-header'>제목 (터치)</div>", unsafe_allow_html=True)
    h[2].markdown("<div class='inner-header'>날짜</div>", unsafe_allow_html=True)

    if not view_data:
        st.caption("데이터 없음")
    else:
        for i, row in enumerate(view_data):
            num = len(targets) - (start + i)
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            d_str = dt.strftime('%m-%d')
            mark = "★" if row.get('is_bookmarked') else ""
            
            summary = (row['dart_result'][:35] + "..").replace('\n', ' ') if row['dart_result'] else "-"
            
            # Expander: 클릭 시 펼쳐짐
            with st.expander(f"{num} | {row['company_name']} {mark} | {summary}"):
                
                # [수정] 1단: 버튼 영역 (상단 배치, 가로로 나열)
                # 모바일 터치를 위해 버튼 크기 넉넉하게 columns로 분배
                btn_cols = st.columns([3, 3, 4]) 
                with btn_cols[0]:
                    if st.button(f"{'★ 해제' if row.get('is_bookmarked') else '☆ 보관'}", key=f"bk_{row['id']}"):
                        db.toggle_bookmark(row['id'])
                        st.rerun()
                with btn_cols[1]:
                    if st.button("🗑 삭제", key=f"del_{row['id']}"):
                        db.delete_result(row['id'])
                        st.rerun()
                
                st.write("") # 간격
                
                # [수정] 2단: 본문 영역 (통짜로 넓게)
                st.markdown(f"**{row['company_name']}** ({dt.strftime('%Y-%m-%d %H:%M')})")
                
                st.markdown("<div class='inner-header'>DART 공시 분석</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-text'>{row['dart_result']}</div>", unsafe_allow_html=True)
                
                st.write("") # 섹션 간격
                
                st.markdown("<div class='inner-header'>뉴스 모멘텀</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-text'>{row['news_result']}</div>", unsafe_allow_html=True)

    # 페이징
    st.write("")
    cp, cc, cn = st.columns([2, 4, 2])
    with cp:
        if st.session_state.page > 1 and st.button("◀ 이전"):
            st.session_state.page -= 1
            st.rerun()
    with cc:
        st.markdown(f"<div style='text-align:center; padding-top:7px;'>{st.session_state.page} / {total_pg}</div>", unsafe_allow_html=True)
    with cn:
        if st.session_state.page < total_pg and st.button("다음 ▶"):
            st.session_state.page += 1
            st.rerun()

# [3] 보관함 (구조 동일)
with tab3:
    bk_list = db.get_bookmarked_results()
    
    if bk_list:
        df = pd.DataFrame(bk_list)
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        out.seek(0)
        st.download_button("Excel 다운로드", data=out, file_name="saved.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    h = st.columns([8, 2])
    h[0].markdown("<div class='inner-header'>보관된 항목</div>", unsafe_allow_html=True)
    h[1].markdown("<div class='inner-header'>날짜</div>", unsafe_allow_html=True)

    if not bk_list: st.caption("보관된 항목이 없습니다.")
    else:
        for row in bk_list:
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            d_str = dt.strftime('%m-%d')
            summary = (row['dart_result'][:35] + "..").replace('\n', ' ') if row['dart_result'] else "-"
            
            with st.expander(f"{row['company_name']} | {summary}"):
                # 버튼 상단
                btn_cols = st.columns([3, 7])
                with btn_cols[0]:
                    if st.button("💔 보관 해제", key=f"ubk_{row['id']}"):
                        db.toggle_bookmark(row['id'])
                        st.rerun()
                
                st.write("")
                # 본문 하단
                st.markdown("<div class='inner-header'>DART</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-text'>{row['dart_result']}</div>", unsafe_allow_html=True)
                st.markdown("<div class='inner-header'>NEWS</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-text'>{row['news_result']}</div>", unsafe_allow_html=True)
