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

# CSS 스타일 (모바일 최적화 - 이메일 리스트 스타일)
st.markdown("""
<style>
    /* ===== 전체 레이아웃 ===== */
    header {visibility: hidden;}
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }

    /* ===== 요소 간격 완전 제거 ===== */
    .element-container { margin-bottom: 0rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0rem !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] { gap: 0rem !important; }

    /* ===== BBS 리스트 - Expander 스타일 ===== */
    /* 게시판 행: 간격 없이 줄줄이 붙어있는 형태 */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        border-bottom: 1px solid #e0e0e0 !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        margin: 0px !important;
        padding: 0px !important;
        border-radius: 0px !important;
    }
    
    /* 행 제목 (접힌 상태) */
    .stExpander > details > summary {
        padding: 10px 4px !important;
        font-size: 13.5px !important;
        color: #333 !important;
        min-height: 38px !important;
        line-height: 1.3 !important;
        font-weight: 400 !important;
        border: none !important;
        margin: 0 !important;
    }
    .stExpander > details > summary:hover {
        background-color: #f8f8f8 !important;
    }
    
    /* 펼쳐진 내용 영역 */
    .stExpander > details[open] > summary {
        background-color: #f0f2f6 !important;
        font-weight: 600 !important;
    }
    .stExpander > details > div {
        padding: 12px 8px 16px 8px !important;
        background-color: #fafbfc !important;
        border-bottom: 2px solid #d0d0d0 !important;
    }

    /* ===== 버튼 스타일 ===== */
    .stButton > button {
        height: 34px;
        font-size: 12.5px;
        padding: 0 10px;
        border: 1px solid #d0d0d0;
        background-color: #fff;
        border-radius: 4px;
        color: #444;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #f5f5f5;
        border-color: #aaa;
    }
    .stButton > button:active {
        background-color: #e8e8e8;
    }

    /* ===== 본문 텍스트 ===== */
    .report-body {
        font-size: 13.5px !important;
        line-height: 1.65 !important;
        color: #222;
        white-space: pre-wrap;
        padding: 8px 2px;
        word-break: keep-all;
    }

    /* ===== 섹션 라벨 ===== */
    .section-label {
        font-size: 11.5px;
        color: #888;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 4px 0 2px 0;
        margin-top: 8px;
        border-top: 1px solid #eaeaea;
        display: block;
    }
    .section-label:first-of-type {
        border-top: none;
        margin-top: 0;
    }
    
    /* ===== 게시글 내 이전/다음 네비 ===== */
    .post-nav {
        font-size: 12px;
        color: #666;
        padding: 6px 0;
        border-top: 1px solid #e8e8e8;
        margin-top: 10px;
    }
    .post-nav-label {
        color: #999;
        font-size: 11px;
        min-width: 45px;
        display: inline-block;
    }
    .post-nav-title {
        color: #444;
    }

    /* ===== 탭 스타일 ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        font-size: 13.5px;
        padding: 0 16px;
        color: #666;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #111;
        font-weight: 600;
    }

    /* ===== 입력창 ===== */
    .stTextArea textarea, .stTextInput input {
        font-size: 14px;
    }
    
    /* ===== 페이징 텍스트 ===== */
    .page-info {
        text-align: center;
        font-size: 12px;
        color: #999;
        padding-top: 8px;
    }
    
    /* ===== 리스트 헤더 ===== */
    .list-header {
        font-size: 11px;
        color: #999;
        padding: 6px 4px;
        border-bottom: 2px solid #ccc;
        font-weight: 600;
        margin-bottom: 0 !important;
    }
    
    /* ===== 총 건수 뱃지 ===== */
    .total-badge {
        font-size: 11px;
        color: #999;
        text-align: right;
        padding: 6px 4px;
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

tab1, tab2, tab3 = st.tabs(["수집", "결과", "보관"])

# ──────────────────────────────────────
# [1] 데이터 수집
# ──────────────────────────────────────
with tab1:
    if 'is_processing' not in st.session_state: st.session_state.is_processing = False
    if 'pending_companies' not in st.session_state: st.session_state.pending_companies = []

    c1, c2 = st.columns([8, 2])
    with c1:
        companies_input = st.text_area(
            "Input", 
            value='\n'.join(st.session_state.pending_companies) if st.session_state.pending_companies and not st.session_state.is_processing else "", 
            height=80, label_visibility="collapsed", 
            placeholder="종목명 입력 (엔터 구분)"
        )
    with c2:
        if st.button("실행", use_container_width=True, disabled=st.session_state.is_processing):
            if companies_input.strip():
                st.session_state.pending_companies = [c.strip() for c in companies_input.split('\n') if c.strip()]
                st.session_state.is_processing = True
                st.rerun()

    if st.session_state.is_processing and st.session_state.pending_companies:
        BATCH = 5
        curr = st.session_state.pending_companies[:BATCH]
        st.caption(f"⏳ 작업중... 남은 {len(st.session_state.pending_companies)}건")
        for c in curr:
            asyncio.run(analyze_company(c, CODE_MAP.get(c)))
        st.session_state.pending_companies = st.session_state.pending_companies[BATCH:]
        if st.session_state.pending_companies:
            time.sleep(0.5)
            st.rerun()
        else:
            st.session_state.is_processing = False
            st.rerun()

# ──────────────────────────────────────
# [2] 분석 결과 (BBS 스타일)
# ──────────────────────────────────────
with tab2:
    if 'page' not in st.session_state: st.session_state.page = 1
    all_res = db.get_all_results(limit=10000)
    
    # 검색 바
    c_s, c_cnt = st.columns([7, 3])
    with c_s:
        kw = st.text_input("검색", label_visibility="collapsed", placeholder="종목명 검색")
    with c_cnt:
        st.markdown(f"<div class='total-badge'>{len(all_res)}건</div>", unsafe_allow_html=True)

    targets = [r for r in all_res if kw in r['company_name']] if kw else all_res
    
    # 페이징 계산
    PER_PAGE = 50
    total_pg = math.ceil(len(targets) / PER_PAGE) if targets else 1
    if st.session_state.page > total_pg: st.session_state.page = 1
    start = (st.session_state.page - 1) * PER_PAGE
    view_data = targets[start:start + PER_PAGE]

    # 리스트 헤더
    h1, h2 = st.columns([7, 3])
    h1.markdown("<div class='list-header'>종목명</div>", unsafe_allow_html=True)
    h2.markdown("<div class='list-header' style='text-align:right;'>날짜</div>", unsafe_allow_html=True)

    if not view_data:
        st.caption("데이터 없음")
    else:
        for i, row in enumerate(view_data):
            global_idx = start + i  # targets 기준 인덱스
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            d_str = dt.strftime('%m.%d %H:%M')
            mark = " ★" if row.get('is_bookmarked') else ""
            
            # 한 줄 요약 (제목 역할)
            title_line = f"{row['company_name']}{mark}　·　{d_str}"
            
            with st.expander(title_line):
                # ── 상단: 액션 버튼 ──
                bc1, bc2, bc3 = st.columns([3, 3, 4])
                with bc1:
                    bk_label = "★ 해제" if row.get('is_bookmarked') else "☆ 보관"
                    if st.button(bk_label, key=f"bk_{row['id']}"):
                        db.toggle_bookmark(row['id'])
                        st.rerun()
                with bc2:
                    if st.button("삭제", key=f"del_{row['id']}"):
                        db.delete_result(row['id'])
                        st.rerun()
                
                # ── 본문: DART ──
                st.markdown("<span class='section-label'>DART 공시</span>", unsafe_allow_html=True)
                dart_text = row['dart_result'] or "-"
                st.markdown(f"<div class='report-body'>{dart_text}</div>", unsafe_allow_html=True)
                
                # ── 본문: 뉴스 ──
                st.markdown("<span class='section-label'>뉴스 모멘텀</span>", unsafe_allow_html=True)
                news_text = row['news_result'] or "-"
                st.markdown(f"<div class='report-body'>{news_text}</div>", unsafe_allow_html=True)
                
                # ── 하단: 이전글/다음글 네비게이션 ──
                prev_row = targets[global_idx - 1] if global_idx > 0 else None
                next_row = targets[global_idx + 1] if global_idx < len(targets) - 1 else None
                
                nav_html = "<div class='post-nav'>"
                if prev_row:
                    prev_dt = prev_row['created_at']
                    if isinstance(prev_dt, str): prev_dt = datetime.strptime(prev_dt, '%Y-%m-%d %H:%M:%S')
                    nav_html += f"<span class='post-nav-label'>▲ 이전</span> <span class='post-nav-title'>{prev_row['company_name']}　{prev_dt.strftime('%m.%d')}</span><br>"
                if next_row:
                    next_dt = next_row['created_at']
                    if isinstance(next_dt, str): next_dt = datetime.strptime(next_dt, '%Y-%m-%d %H:%M:%S')
                    nav_html += f"<span class='post-nav-label'>▼ 다음</span> <span class='post-nav-title'>{next_row['company_name']}　{next_dt.strftime('%m.%d')}</span>"
                nav_html += "</div>"
                st.markdown(nav_html, unsafe_allow_html=True)

    # 페이징 컨트롤
    if total_pg > 1:
        st.write("")
        cp, cc, cn = st.columns([2, 4, 2])
        with cp:
            if st.session_state.page > 1 and st.button("◀ 이전", key="pg_prev"):
                st.session_state.page -= 1
                st.rerun()
        with cc:
            st.markdown(f"<div class='page-info'>{st.session_state.page} / {total_pg}</div>", unsafe_allow_html=True)
        with cn:
            if st.session_state.page < total_pg and st.button("다음 ▶", key="pg_next"):
                st.session_state.page += 1
                st.rerun()

# ──────────────────────────────────────
# [3] 보관함
# ──────────────────────────────────────
with tab3:
    bk_list = db.get_bookmarked_results()
    
    if bk_list:
        df = pd.DataFrame(bk_list)
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        out.seek(0)
        st.download_button("Excel 다운로드", data=out, file_name="saved.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # 헤더
    h1, h2 = st.columns([7, 3])
    h1.markdown("<div class='list-header'>종목명</div>", unsafe_allow_html=True)
    h2.markdown("<div class='list-header' style='text-align:right;'>날짜</div>", unsafe_allow_html=True)

    if not bk_list: 
        st.caption("보관된 항목이 없습니다.")
    else:
        for row in bk_list:
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            d_str = dt.strftime('%m.%d %H:%M')
            
            with st.expander(f"★ {row['company_name']}　·　{d_str}"):
                bc1, bc2 = st.columns([3, 7])
                with bc1:
                    if st.button("보관 해제", key=f"ubk_{row['id']}"):
                        db.toggle_bookmark(row['id'])
                        st.rerun()
                
                st.markdown("<span class='section-label'>DART 공시</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-body'>{row['dart_result'] or '-'}</div>", unsafe_allow_html=True)
                
                st.markdown("<span class='section-label'>뉴스 모멘텀</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-body'>{row['news_result'] or '-'}</div>", unsafe_allow_html=True)
