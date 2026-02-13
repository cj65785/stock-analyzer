# app.py (Stealth Mode)
import streamlit as st
import asyncio
import pandas as pd
import time
import warnings
from datetime import datetime
from openai import AsyncOpenAI
from io import BytesIO
from database import Database
from analyzer import (
    Config, RegexCache, DartProcessor, 
    run_news_pipeline
)

# --------------------------------------------------------------------------
# [설정] 경고 메시지 차단
# --------------------------------------------------------------------------
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

# 페이지 설정 (제목도 '시스템 관리' 같은 걸로 위장 가능)
st.set_page_config(
    page_title="System Admin", 
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS 스타일 (극한의 공백 제거 및 은밀 모드)
st.markdown("""
<style>
    /* 1. 상단 헤더 및 여백 제거 */
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    
    /* 2. 탭 스타일 (작고 심플하게) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 30px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 2px;
        padding-bottom: 2px;
        font-size: 14px;
    }
    
    /* 3. Expander (게시판처럼 밀착) */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
        margin-bottom: 0px !important; /* 간격 제거 */
        border-bottom: 1px solid #f0f0f0 !important; /* 게시판 구분선 느낌 */
    }
    .stExpander > details > summary {
        padding-top: 5px !important;
        padding-bottom: 5px !important;
        font-size: 14px !important; /* 글자 크기 축소 */
    }
    .stExpander > details > div {
        padding-bottom: 5px !important;
    }

    /* 4. 버튼 및 입력창 크기 축소 */
    .stButton > button {
        height: 30px;
        padding-top: 0px;
        padding-bottom: 0px;
        font-size: 13px;
    }
    .stTextArea > label, .stTextInput > label {
        font-size: 13px;
    }
    
    /* 5. 체크박스 정렬 및 크기 */
    div[data-testid="stCheckbox"] {
        min-height: 20px;
        margin-top: 5px;
    }
    div[data-testid="stCheckbox"] label span {
        padding-left: 0px;
    }
    
    /* 6. 섹션 헤더 (은밀하게) */
    .section-header {
        background-color: #f8f9fa; 
        padding: 5px; 
        border-radius: 3px; 
        margin-top: 10px; 
        font-size: 13px; 
        font-weight: bold;
        color: #555;
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

# GPT 분석 함수 (뉴스)
async def analyze_news_with_gpt(company_name: str, articles: list) -> str:
    if not articles:
        return "데이터 없음"
    
    articles.sort(key=lambda x: x['pub_date'], reverse=True)
    context = ""
    for i, art in enumerate(articles):
        d = art['pub_date'].strftime('%y-%m-%d')
        context += f"[{d}] {art['title']}\n{art['body'][:3000]}...\n"

    system_prompt = f"""
        당신은 주식 시장의 '모멘텀 전문 분석가'입니다. 

        [작성 규칙]
        1. "{company_name}"의 기업 가치(Valuation) 리레이팅을 유발할 수 있는 모든 모멘텀을 적을 것
        ※ 모멘텀 :  '매출', '수출', '수주', '계약', '신제품', "양산", '캐파', 'M&A'
        2. 반드시 "{company_name}" 회사와 직접 관련된 내용만 작성하며, 창작이 아닌 기사 속 내용만으로 작성할 것
        3. 중복된 기사는 하나로 합치고, 구체적인 "숫자"나 "시기", "국가", "계약 상대방" 등이 언급된 경우 반드시 넣어주기 바랍니다.
        4. 산업 전반의 동향, 다른 회사의 사례, 일반적인 시장 분석은 절대 포함하지 마십시오.
        5. 문체: 개조식, 명사형 종결(~음, ~임, ~함), 인사말 및 미사여구 없는 핵심 내용만 작성할 것
        
        [출력 포맷]
        1️⃣ 모멘텀 제목 (yyyy.mm.dd.)
        - {company_name}의 모멘텀 관련 핵심 내용 요약
        
        2️⃣ 모멘텀 제목 (yyyy.mm.dd.)
        - {company_name}의 모멘텀 관련 핵심 내용 요약
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

# GPT 분석 함수 (DART)
async def analyze_dart_with_gpt(company_name: str, report_nm: str, dart_text: str) -> str:
    if not dart_text or len(dart_text) < 100:
        return "내용 없음"
    
    dart_context = dart_text[:40000]

    system_prompt = f"""
        당신은 주식 시장의 '모멘텀 전문 분석가'입니다.
        
        [작성 규칙]
        1. "{company_name}"의 기업 가치(Valuation) 리레이팅을 유발할 수 있는 모든 모멘텀을 적을 것
        2. 신사업 진출, 신규 고객 확보, 증설, M&A, 퀄테스트 통과, 벤더 등록, 수출 지역 다변화 등 구체적인 근거를 포함할 것
        3. 현황을 적는 것이 아닌, 기업 가치를 레벨업 시키는 핵심 성과 및 미래 기대감을 적을 것
        4. 반드시 주어진 자료 내의 내용만으로 작성하며, 외부 지식을 가져오거나 없는 내용을 추론하지 말 것
        5. 문체: 개조식, 명사형 종결(~음, ~임, ~함), 인사말 및 미사여구 없는 핵심 내용만 작성할 것
        
        [출력 포맷]
        - 모멘텀 내용 1
        
        - 모멘텀 내용 2
        
        - 모멘텀 내용 3
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
    if progress_callback: progress_callback(f"{company_name} DART..")
    
    dart_processor = DartProcessor(config.DART_API_KEY)
    report_nm, dart_text, dart_error = dart_processor.process(company_name, stock_code)
    
    if dart_text:
        if progress_callback: progress_callback(f"{company_name} DART AI..")
        dart_result = await analyze_dart_with_gpt(company_name, report_nm, dart_text)
    else:
        dart_result = "보고서 없음"
    
    if progress_callback: progress_callback(f"{company_name} News..")
    
    articles, news_count = await run_news_pipeline(company_name, config, REGEX_CACHE)
    
    if progress_callback: progress_callback(f"{company_name} News AI..")
    
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

# ==================== UI (Stealth) ====================

# 탭 이름도 심플하게 변경
tab1, tab2, tab3 = st.tabs(["분석", "목록", "보관함"])

# ===== 탭 1: 분석 =====
with tab1:
    st.caption("Batch Analysis System") # 아주 작게 표시
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'pending_companies' not in st.session_state:
        st.session_state.pending_companies = []
    
    companies_input = st.text_area(
        "Target List", # 영어로 써두면 더 업무 같음
        value='\n'.join(st.session_state.pending_companies) if st.session_state.pending_companies and not st.session_state.is_processing else "",
        placeholder="종목명 입력",
        height=100, # 높이 줄임
        key="companies_input",
        disabled=st.session_state.is_processing,
        label_visibility="collapsed" # 라벨 숨김
    )
    
    col1, col2 = st.columns([1, 6])
    with col1:
        analyze_button = st.button("실행", type="primary", use_container_width=True, disabled=st.session_state.is_processing)
    
    if analyze_button:
        if not companies_input.strip():
            st.warning("입력 필요")
        else:
            companies_list = [c.strip() for c in companies_input.split('\n') if c.strip()]
            st.session_state.pending_companies = companies_list
            st.session_state.is_processing = True
            st.rerun()

    if st.session_state.is_processing and st.session_state.pending_companies:
        BATCH_SIZE = 5
        total = len(st.session_state.pending_companies)
        current_batch = st.session_state.pending_companies[:BATCH_SIZE]
        
        status_text = st.empty()
        status_text.caption(f"Processing.. {total} left")
        
        progress_bar = st.progress(0)
        
        processed_count = 0
        
        for idx, company in enumerate(current_batch):
            stock_code = CODE_MAP.get(company)
            def update_status(msg):
                status_text.caption(f"[{idx+1}/{len(current_batch)}] {msg}")
            
            try:
                asyncio.run(analyze_company(company, stock_code, update_status))
                processed_count += 1
            except Exception as e:
                st.error(f"{company} Err")
            
            progress_bar.progress((idx + 1) / len(current_batch))
        
        st.session_state.pending_companies = st.session_state.pending_companies[BATCH_SIZE:]
        
        if st.session_state.pending_companies:
            time.sleep(1) 
            st.rerun() 
        else:
            st.session_state.is_processing = False
            status_text.text("Done.")
            st.rerun()

# ===== 탭 2: 목록 (게시판 스타일) =====
with tab2:
    col_search, col_action, col_cnt = st.columns([3, 1, 1])
    
    with col_search:
        search_keyword = st.text_input("Search", placeholder="Search", key="search_all", label_visibility="collapsed")
    
    with col_cnt:
        total_count = db.get_count()
        st.caption(f"Total: {total_count}")

    if search_keyword:
        results = db.search_results(search_keyword)
    else:
        results = db.get_all_results(limit=50) # 로딩 속도 위해 50개로
    
    with col_action:
        if st.button("삭제", type="secondary", use_container_width=True):
            deleted_count = 0
            for result in results:
                if st.session_state.get(f"del_{result['id']}"):
                    db.delete_result(result['id'])
                    deleted_count += 1
            if deleted_count > 0:
                st.rerun()

    # 리스트 출력 (헤더 없음, 밀착형)
    if not results:
        st.caption("No Data")
    else:
        # 게시판 헤더 느낌 (옵션)
        # st.markdown("| 선택 | 종목 | 시간 |")
        
        for result in results:
            created_at = result['created_at']
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%m-%d %H:%M')
            
            # 북마크 아이콘 심플하게
            mark = "★" if result.get('is_bookmarked') else "☆"
            
            # 레이아웃: [체크] [내용(Expander)]
            c_chk, c_body = st.columns([0.5, 12])
            
            with c_chk:
                st.checkbox("del", key=f"del_{result['id']}", label_visibility="collapsed")
            
            with c_body:
                # Expander 제목을 한줄로 심플하게: "종목명 (시간) ★"
                with st.expander(f"{result['company_name']} ({date_str}) {mark}"):
                    # 내부 내용
                    c_btn, _ = st.columns([1, 5])
                    with c_btn:
                        if st.button("북마크", key=f"bk_{result['id']}"):
                            db.toggle_bookmark(result['id'])
                            st.rerun()
                    
                    st.markdown('<div class="section-header">DART</div>', unsafe_allow_html=True)
                    st.text(result['dart_result']) # write 대신 text로 줄간격 축소
                    
                    st.markdown('<div class="section-header">NEWS</div>', unsafe_allow_html=True)
                    st.text(result['news_result'])
                    
    # 엑셀 다운로드 (작게)
    if results:
        df = db.to_dataframe()
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        output.seek(0)
        st.download_button("Excel", data=output, file_name="data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ===== 탭 3: 보관함 =====
with tab3:
    bookmarked = db.get_bookmarked_results()
    
    if not bookmarked:
        st.caption("Empty")
    else:
        st.caption(f"Saved: {len(bookmarked)}")
        for result in bookmarked:
            created_at = result['created_at']
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%m-%d %H:%M')
            
            with st.expander(f"{result['company_name']} ({date_str})"):
                if st.button("해제", key=f"unbk_{result['id']}"):
                    db.toggle_bookmark(result['id'])
                    st.rerun()
                
                st.markdown('<div class="section-header">DART</div>', unsafe_allow_html=True)
                st.text(result['dart_result'])
                
                st.markdown('<div class="section-header">NEWS</div>', unsafe_allow_html=True)
                st.text(result['news_result'])
        
        # 엑셀 (보관함)
        df_bm = pd.DataFrame(bookmarked)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_bm.to_excel(writer, index=False, sheet_name='Saved')
        output.seek(0)
        st.download_button("Save Excel", data=output, file_name="saved.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

