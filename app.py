# app.py (인트라넷 게시판 모드)
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

# --------------------------------------------------------------------------
# [설정] 경고 차단
# --------------------------------------------------------------------------
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# 페이지 설정 (제목을 '업무일지' 등으로 위장)
st.set_page_config(
    page_title="업무 관리 시스템", 
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 (인트라넷 게시판 스타일 - 투박함의 미학)
st.markdown("""
<style>
    /* 1. 상단 헤더 숨김 및 여백 제거 */
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }
    
    /* 2. 탭 스타일 (폴더 탭 느낌) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 1px solid #ddd;
    }
    .stTabs [data-baseweb="tab"] {
        height: 35px;
        font-size: 14px;
        color: #555;
        border: 1px solid transparent;
        border-radius: 5px 5px 0 0;
        padding: 0 15px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #fff;
        border: 1px solid #ddd;
        border-bottom: 1px solid #fff;
        color: #000;
        font-weight: bold;
    }
    
    /* 3. 게시판 리스트 스타일 (표 처럼 보이게) */
    .board-header {
        font-weight: bold;
        background-color: #f5f5f5;
        padding: 8px 5px;
        border-top: 2px solid #555;
        border-bottom: 1px solid #ddd;
        font-size: 13px;
        text-align: center;
        margin-bottom: 0px;
    }
    .board-row {
        padding: 0px;
        border-bottom: 1px solid #eee;
        font-size: 13px;
    }
    .board-row:hover {
        background-color: #f9f9f9;
    }
    
    /* Expander 커스텀 (게시글 제목 역할) */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }
    .stExpander > details > summary {
        padding: 8px 5px !important;
        border-bottom: 1px solid #eee;
        font-size: 13px !important;
        color: #333 !important;
    }
    .stExpander > details > summary:hover {
        background-color: #f9f9f9;
        color: #000 !important;
    }
    .stExpander > details > div {
        padding: 15px;
        background-color: #fafafa;
        border-bottom: 1px solid #ddd;
    }

    /* 4. 버튼 및 입력창 (심플 그레이) */
    .stButton > button {
        border: 1px solid #ccc;
        background-color: #f8f8f8;
        color: #333;
        font-size: 12px;
        height: 28px;
        padding: 0 10px;
    }
    .stButton > button:hover {
        border-color: #999;
        color: #000;
    }
    /* 중요 버튼만 약간 진하게 */
    .primary-btn > button {
        background-color: #555 !important;
        color: white !important;
    }

    /* 5. 본문 텍스트 (문서 느낌) */
    .report-text {
        font-family: 'Malgun Gothic', sans-serif;
        font-size: 13px;
        line-height: 1.6;
        color: #444;
        white-space: pre-wrap;
    }
    .section-title {
        font-weight: bold;
        color: #000;
        margin-top: 10px;
        margin-bottom: 5px;
        font-size: 14px;
        border-left: 3px solid #555;
        padding-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 데이터베이스 초기화
@st.cache_resource
def get_database():
    database_url = st.secrets.get("DATABASE_URL")
    return Database(database_url)

db = get_database()

# Config 초기화
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

# 상장사 목록 로드
@st.cache_resource
def load_companies():
    try:
        try:
            df = pd.read_csv('krx_stocks.csv', encoding='cp949')
        except:
            df = pd.read_csv('krx_stocks.csv', encoding='utf-8')
        
        code_map = dict(zip(df['종목명'], df['종목코드']))
        companies = df['종목명'].dropna().astype(str).str.strip().tolist()
        return companies, RegexCache(companies), code_map
    except Exception as e:
        return [], None, {}

ALL_COMPANIES, REGEX_CACHE, CODE_MAP = load_companies()

# GPT 분석 함수 (뉴스 - 간결체)
async def analyze_news_with_gpt(company_name: str, articles: list) -> str:
    if not articles:
        return "데이터 없음"
    
    articles.sort(key=lambda x: x['pub_date'], reverse=True)
    context = ""
    for i, art in enumerate(articles):
        d = art['pub_date'].strftime('%y-%m-%d')
        context += f"[{d}] {art['title']}\n"

    system_prompt = f"""
"{company_name}" 뉴스 요약. 
주가 상승 모멘텀(수주,계약,실적 등) 위주로 작성.
음슴체 사용. 서론/결론 생략.
형식:
- [날짜] 내용 요약
"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Err: {e}"

# GPT 분석 함수 (DART - 간결체)
async def analyze_dart_with_gpt(company_name: str, report_nm: str, dart_text: str) -> str:
    if not dart_text or len(dart_text) < 100:
        return "내용 없음"
    
    dart_context = dart_text[:40000]

    system_prompt = f"""
"{company_name}" 공시({report_nm}) 요약.
기업 가치 관련 핵심 내용만 추출.
음슴체 사용. 잡담 금지.
형식:
- 핵심 내용 1
- 핵심 내용 2
"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{dart_context}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Err: {e}"

# 통합 분석 함수
async def analyze_company(company_name: str, stock_code: str = None, progress_callback=None):
    if progress_callback: progress_callback(f"분석중.. {company_name}")
    
    dart_processor = DartProcessor(config.DART_API_KEY)
    report_nm, dart_text, dart_error = dart_processor.process(company_name, stock_code)
    
    dart_result = await analyze_dart_with_gpt(company_name, report_nm, dart_text) if dart_text else "보고서 없음"
    
    articles, news_count = await run_news_pipeline(company_name, config, REGEX_CACHE)
    news_result = await analyze_news_with_gpt(company_name, articles)
    
    db.add_result(
        company_name=company_name,
        dart_report=report_nm or "-",
        dart_result=dart_result,
        dart_error=dart_error or "",
        news_count=news_count,
        news_result=news_result
    )
    return True

# ==================== UI (BBS Mode) ====================

# 탭 구성 (직관적인 한글)
tab1, tab2, tab3 = st.tabs(["데이터 수집", "분석 결과", "관심 종목"])

# ===== 탭 1: 데이터 수집 (입력폼) =====
with tab1:
    col_input, col_btn = st.columns([8, 1])
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'pending_companies' not in st.session_state:
        st.session_state.pending_companies = []

    with col_input:
        companies_input = st.text_area(
            "종목 리스트",
            value='\n'.join(st.session_state.pending_companies) if st.session_state.pending_companies and not st.session_state.is_processing else "",
            placeholder="분석할 종목명을 줄바꿈으로 입력하세요",
            height=100,
            label_visibility="collapsed"
        )
    
    with col_btn:
        st.write("") # 줄맞춤
        analyze_button = st.button("실행", use_container_width=True, disabled=st.session_state.is_processing)
    
    if analyze_button:
        if not companies_input.strip():
            st.error("종목명을 입력하세요.")
        else:
            companies_list = [c.strip() for c in companies_input.split('\n') if c.strip()]
            st.session_state.pending_companies = companies_list
            st.session_state.is_processing = True
            st.rerun()

    # 진행 상황 (텍스트로만 심플하게)
    if st.session_state.is_processing and st.session_state.pending_companies:
        BATCH_SIZE = 5
        total = len(st.session_state.pending_companies)
        current_batch = st.session_state.pending_companies[:BATCH_SIZE]
        
        status_box = st.empty()
        status_box.text(f"▷ 작업 진행중... 잔여: {total}건")
        
        for idx, company in enumerate(current_batch):
            stock_code = CODE_MAP.get(company)
            asyncio.run(analyze_company(company, stock_code))
        
        st.session_state.pending_companies = st.session_state.pending_companies[BATCH_SIZE:]
        
        if st.session_state.pending_companies:
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.is_processing = False
            status_box.text("▶ 작업 완료.")
            st.rerun()

# ===== 탭 2: 분석 결과 (게시판 형태 + 페이지네이션) =====
with tab2:
    # 1. 데이터 조회 (전체 가져오기 - 페이지네이션 위해)
    # limit를 아주 크게 잡아서 사실상 다 가져옴
    if 'page' not in st.session_state:
        st.session_state.page = 1
        
    all_results = db.get_all_results(limit=10000) 
    
    # 검색 필터
    col_search, col_dummy = st.columns([3, 7])
    with col_search:
        search_kw = st.text_input("검색", placeholder="종목명 검색", label_visibility="collapsed")
    
    if search_kw:
        # 검색 시에는 전체 필터링
        filtered_results = [r for r in all_results if search_kw in r['company_name']]
        st.session_state.page = 1 # 검색하면 1페이지로
    else:
        filtered_results = all_results

    # 2. 페이지네이션 계산
    ITEMS_PER_PAGE = 50
    total_items = len(filtered_results)
    total_pages = math.ceil(total_items / ITEMS_PER_PAGE) if total_items > 0 else 1
    
    # 페이지 범위 체크
    if st.session_state.page > total_pages: st.session_state.page = total_pages
    if st.session_state.page < 1: st.session_state.page = 1
    
    start_idx = (st.session_state.page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    
    # 현재 페이지 데이터 슬라이싱
    page_data = filtered_results[start_idx:end_idx]

    # 3. 게시판 헤더 출력
    # (삭제 기능은 게시글 내부로 이동하여 리스트를 깔끔하게 유지)
    header_cols = st.columns([0.5, 1, 6, 2, 0.5])
    header_cols[0].markdown("<div class='board-header'>No</div>", unsafe_allow_html=True)
    header_cols[1].markdown("<div class='board-header'>구분</div>", unsafe_allow_html=True)
    header_cols[2].markdown("<div class='board-header'>제목</div>", unsafe_allow_html=True)
    header_cols[3].markdown("<div class='board-header'>작성일</div>", unsafe_allow_html=True)
    header_cols[4].markdown("<div class='board-header'>-</div>", unsafe_allow_html=True)

    # 4. 리스트 출력 loop
    if not page_data:
        st.markdown("<div style='text-align:center; padding:20px; color:#999;'>데이터가 없습니다.</div>", unsafe_allow_html=True)
    else:
        for idx, row in enumerate(page_data):
            # 순번 계산 (전체 기준 내림차순 or 그냥 페이지 내 순번)
            # 여기선 DB ID 사용하거나 역순 번호
            display_num = total_items - (start_idx + idx)
            
            created_at = row['created_at']
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%Y-%m-%d')
            
            is_bm = "★" if row.get('is_bookmarked') else ""
            
            # 한 행(Row)의 레이아웃
            # Streamlit Expander를 사용하되, 라벨을 게시글 제목처럼 꾸밈
            # 제목 포맷: [종목명] 분석 결과 요약 ...
            
            summary_text = f"[{row['company_name']}] 기업분석 보고서 {is_bm}"
            
            # Expander 시작
            with st.expander(summary_text):
                # 게시글 내부 (상세 내용)
                c_head, c_body = st.columns([2, 8])
                
                with c_head:
                    st.markdown(f"**{row['company_name']}**")
                    st.caption(f"분석일시: {created_at.strftime('%Y-%m-%d %H:%M')}")
                    
                    # 기능 버튼들 (작게)
                    if st.button("관심종목 등록/해제", key=f"bk_{row['id']}"):
                        db.toggle_bookmark(row['id'])
                        st.rerun()
                    
                    if st.button("데이터 삭제", key=f"del_{row['id']}"):
                        db.delete_result(row['id'])
                        st.rerun()

                with c_body:
                    st.markdown("<div class='section-title'>DART 공시 분석</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='report-text'>{row['dart_result']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-title'>뉴스 모멘텀</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='report-text'>{row['news_result']}</div>", unsafe_allow_html=True)

    # 5. 페이지네이션 컨트롤 (하단 중앙)
    st.write("") # 간격
    col_prev, col_page, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.session_state.page > 1:
            if st.button("◀ 이전", use_container_width=True):
                st.session_state.page -= 1
                st.rerun()
                
    with col_page:
        st.markdown(f"<div style='text-align:center; padding-top:5px; font-size:13px;'>Page {st.session_state.page} / {total_pages}</div>", unsafe_allow_html=True)
        
    with col_next:
        if st.session_state.page < total_pages:
            if st.button("다음 ▶", use_container_width=True):
                st.session_state.page += 1
                st.rerun()

# ===== 탭 3: 관심 종목 (동일한 게시판 스타일) =====
with tab3:
    bookmarked = db.get_bookmarked_results()
    
    header_cols = st.columns([0.5, 8, 2])
    header_cols[0].markdown("<div class='board-header'>No</div>", unsafe_allow_html=True)
    header_cols[1].markdown("<div class='board-header'>제목</div>", unsafe_allow_html=True)
    header_cols[2].markdown("<div class='board-header'>작성일</div>", unsafe_allow_html=True)
    
    if not bookmarked:
        st.markdown("<div style='text-align:center; padding:20px; color:#999;'>보관된 문서가 없습니다.</div>", unsafe_allow_html=True)
    else:
        for idx, row in enumerate(bookmarked):
            created_at = row['created_at']
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%Y-%m-%d')
            
            with st.expander(f"[{row['company_name']}] 주요 모멘텀 요약본"):
                st.markdown(f"<div class='section-title'>DART: {row['dart_report']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-text'>{row['dart_result']}</div>", unsafe_allow_html=True)
                st.markdown("<div class='section-title'>NEWS</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-text'>{row['news_result']}</div>", unsafe_allow_html=True)
                
                if st.button("보관 해제", key=f"bm_del_{row['id']}"):
                    db.toggle_bookmark(row['id'])
                    st.rerun()

    # 엑셀 다운로드 (우측 하단 작게)
    st.write("")
    if bookmarked:
        df_bm = pd.DataFrame(bookmarked)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_bm.to_excel(writer, index=False, sheet_name='Saved')
        output.seek(0)
        
        c1, c2 = st.columns([8, 2])
        with c2:
            st.download_button("Excel 다운로드", data=output, file_name="saved_list.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
