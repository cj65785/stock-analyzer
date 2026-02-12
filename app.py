# app.py
import streamlit as st
import asyncio
import pandas as pd
import time
from datetime import datetime
from openai import AsyncOpenAI
from io import BytesIO
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
from database import Database
from analyzer import (
    Config, RegexCache, DartProcessor, 
    run_news_pipeline
)

# 페이지 설정
st.set_page_config(
    page_title="📊 종목 분석 게시판",
    page_icon="📊",
    layout="wide"
)

# CSS 스타일
st.markdown("""
<style>
    .main {max-width: 1200px; margin: 0 auto;}
    .stExpander {border: 1px solid #e0e0e0; border-radius: 5px; margin-bottom: 10px;}
    .company-title {font-size: 20px; font-weight: bold; color: #1f77b4;}
    .date-text {color: #666; font-size: 14px;}
    .section-header {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-top: 20px;}
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

# 상장사 목록 로드 (종목코드 포함)
@st.cache_resource
def load_companies():
    try:
        # cp949 인코딩 시도
        try:
            df = pd.read_csv('krx_stocks.csv', encoding='cp949')
        except:
            df = pd.read_csv('krx_stocks.csv', encoding='utf-8')
        
        # 종목코드 매핑 생성 (종목명 -> 종목코드)
        code_map = dict(zip(df['종목명'], df['종목코드']))
        companies = df['종목명'].dropna().astype(str).str.strip().tolist()
        return companies, RegexCache(companies), code_map
    except Exception as e:
        st.error(f"CSV 로드 오류: {e}")
        return [], None, {}

ALL_COMPANIES, REGEX_CACHE, CODE_MAP = load_companies()

# GPT 분석 함수
async def analyze_news_with_gpt(company_name: str, articles: list) -> str:
    if not articles:
        return "분석할 뉴스 기사가 없습니다."
    
    articles.sort(key=lambda x: x['pub_date'], reverse=True)
    context = ""
    for i, art in enumerate(articles):
        d = art['pub_date'].strftime('%Y-%m-%d')
        context += f"[[기사 {i+1}]] {d} / {art['title']}\n{art['body'][:5000]}...\n\n"

    system_prompt = f"""
당신은 주식 시장의 '모멘텀 전문 분석가'입니다. 
제공된 뉴스 기사들을 정밀 분석하여, 이 회사의 미래 기업 가치 상승에 기여할 수 있는 '핵심 모멘텀'만 추출하세요.

[대원칙]
⚠️ 반드시 "{company_name}" 회사와 직접 관련된 내용만 작성하십시오.
- 산업 전반의 동향, 다른 회사의 사례, 일반적인 시장 분석은 절대 포함하지 마십시오.
- "{company_name}"이 주어(主語)가 되는 문장만 작성하십시오.

[작성 규칙]
1. 단순히 실적을 나열하거나 이미 반영된 뉴스는 제외하십시오.
2. '매출', '수출', '수주', '계약', '신제품', "양산", '캐파', 'M&A' 등 미래 주가를 끌어올릴 강력한 재료 위주로 요약하십시오.
3. 중복된 내용은 하나로 합치고, 구체적인 숫자나 시기 등이 언급된 경우 반드시 넣어주기 바랍니다.
4. 반드시 아래 포맷을 엄격하게 지키십시오. 서론이나 결론(인사말 등)은 절대 쓰지 마십시오.
5. 창작이 아닌 기사의 내용을 근거로 요약해야합니다.
6. 투자와 관련없는 내용은 배제하되, 가능한 많은 모멘텀을 작성합니다.

[서식 규칙]
- **볼드체**, 헤더(##) 등 마크다운 문법을 절대 사용하지 마십시오. 순수 텍스트로만 작성하십시오.
- 문체: 개조식, 명사형 종결 (~음, ~임, ~함)
- 아이콘 활용: 💊(임상/신약), 🤝(계약/파트너십), 🌍(해외진출), 🏭(생산능력), 💡(신규사업) 등

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
                {"role": "user", "content": f"[기사 목록]\n{context}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 오류: {e}"

async def analyze_dart_with_gpt(company_name: str, report_nm: str, dart_text: str) -> str:
    if not dart_text or len(dart_text) < 100:
        return "DART 보고서를 찾을 수 없거나 내용이 부족합니다."
    
    dart_context = dart_text[:50000]

    system_prompt = f"""
당신은 주식 시장의 '모멘텀 전문 분석가'입니다.
제공된 DART 사업보고서를 분석하여, "{company_name}"의 기업 가치 상승에 기여할 수 있는 '핵심 모멘텀'만 추출하세요.

[작성 규칙]
1. 기업 가치(Valuation) 리레이팅을 유발할 수 있는 모든 재료를 상세히 적으십시오.
2. 신사업 진출, 신규 고객 확보, 증설, M&A, 퀄테스트 통과, 벤더 등록, 수출 지역 다변화 등 구체적인 근거를 포함하여 상세하게 작성하십시오.
3. 현황을 적는 것이 아닌, 기업 가치를 레벨업 시키는 핵심 성과 및 미래 기대감을 적습니다.
4. 반드시 주어진 자료 내의 내용만으로 작성하며, 외부 지식을 가져오거나 없는 내용을 추론하지 마십시오.

[서식 규칙]
- **볼드체**, 헤더(##) 등 마크다운 문법을 절대 사용하지 마십시오. 순수 텍스트로만 작성하십시오.
- 문체: 개조식, 명사형 종결 (~음, ~임, ~함), 구구절절 쓰지말고 압축적으로 쓸 것

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
                {"role": "user", "content": f"기업명: {company_name}\n보고서: {report_nm}\n\n[DART 사업보고서 내용]\n{dart_context}"}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 오류: {e}"

# 단일 종목 분석 함수 (종목코드 추가)
async def analyze_company(company_name: str, stock_code: str = None, progress_callback=None):
    # 1. DART 분석 (종목코드 사용)
    if progress_callback:
        progress_callback(f"📊 {company_name} DART 분석 중...")
    
    dart_processor = DartProcessor(config.DART_API_KEY)
    report_nm, dart_text, dart_error = dart_processor.process(company_name, stock_code)
    
    if progress_callback:
        progress_callback(f"🤖 {company_name} DART GPT 분석 중...")
    
    dart_result = await analyze_dart_with_gpt(company_name, report_nm, dart_text)
    
    # 2. 뉴스 분석
    if progress_callback:
        progress_callback(f"📰 {company_name} 뉴스 수집 중...")
    
    articles, news_count = await run_news_pipeline(company_name, config, REGEX_CACHE)
    
    if progress_callback:
        progress_callback(f"🤖 {company_name} 뉴스 GPT 분석 중...")
    
    news_result = await analyze_news_with_gpt(company_name, articles)
    
    # 3. DB 저장
    db.add_result(
        company_name=company_name,
        dart_report=report_nm or "없음",
        dart_result=dart_result,
        dart_error=dart_error or "",
        news_count=news_count,
        news_result=news_result
    )
    
    return {
        'company': company_name,
        'dart_report': report_nm,
        'dart_result': dart_result,
        'dart_error': dart_error,
        'news_count': news_count,
        'news_result': news_result
    }

# ==================== UI ====================

st.title("📊 종목 분석 게시판")
st.markdown("---")

# 탭 생성 (3개)
tab1, tab2, tab3 = st.tabs(["🚀 새 분석", "📋 전체 결과", "⭐ 즐겨찾기"])

# ===== 탭 1: 새 분석 =====
with tab1:
    st.header("🚀 새 분석 시작 (자동 이어하기 모드)")
    
    # [수정 1] 처리 상태를 관리할 플래그 초기화
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    # [기존 유지] 입력창 상태 관리
    if 'pending_companies' not in st.session_state:
        st.session_state.pending_companies = []
    
    # [기존 유지] 입력 UI
    companies_input = st.text_area(
        "종목명 입력 (줄바꿈으로 구분)",
        value='\n'.join(st.session_state.pending_companies) if st.session_state.pending_companies and not st.session_state.is_processing else "",
        placeholder="삼성전자\nSK하이닉스\n케어젠",
        height=150,
        key="companies_input",
        disabled=st.session_state.is_processing # 처리 중엔 입력 막기
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        # [수정 2] 버튼 로직 변경
        # 버튼을 누르면 '처리 중' 상태로 바꾸고 즉시 리로드합니다.
        analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True, disabled=st.session_state.is_processing)
    
    # 버튼 클릭 시 초기 세팅
    if analyze_button:
        if not companies_input.strip():
            st.warning("⚠️ 종목명을 입력해주세요.")
        else:
            # 입력된 목록을 리스트로 변환하여 저장
            companies_list = [c.strip() for c in companies_input.split('\n') if c.strip()]
            st.session_state.pending_companies = companies_list
            st.session_state.is_processing = True # 처리 시작 플래그 ON
            st.rerun() # 로직 시작을 위해 리로드

    # [수정 3] 자동 배치 처리 로직 (리로드 될 때마다 실행됨)
    if st.session_state.is_processing and st.session_state.pending_companies:
        
        # 1. 배치 설정 (한 번에 5개씩)
        BATCH_SIZE = 5
        total_remaining = len(st.session_state.pending_companies)
        
        # 남은 것 중 앞에서 5개만 가져옴
        current_batch = st.session_state.pending_companies[:BATCH_SIZE]
        
        st.info(f"🔄 자동 처리 중... (남은 종목: {total_remaining}개 / 이번 배치: {len(current_batch)}개)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_count = 0
        
        # 2. 배치 루프 (5개만 실행)
        for idx, company in enumerate(current_batch):
            status_text.markdown(f"**[{idx+1}/{len(current_batch)}] 🔍 {company} 분석 중...**")
            
            # 종목코드 매핑
            stock_code = CODE_MAP.get(company)
            if not stock_code:
                st.warning(f"⚠️ {company}: 종목코드를 찾을 수 없습니다. 종목명으로 시도합니다.")
            
            # 콜백 함수 정의
            def update_status(msg):
                status_text.text(f"[{idx+1}/{len(current_batch)}] {msg}")
            
            try:
                # [핵심] 기존의 비동기 분석 함수 호출
                # 마스터님 코드의 analyze_company 함수를 그대로 사용
                asyncio.run(analyze_company(company, stock_code, update_status))
                processed_count += 1
                
            except Exception as e:
                st.error(f"❌ {company} 오류: {e}")
                # 실패해도 다음 루프로 진행
            
            progress_bar.progress((idx + 1) / len(current_batch))
        
        # 3. 처리 완료된 목록 제거 (Queue Pop)
        # 방금 처리한 개수만큼 리스트 앞에서 잘라냄
        st.session_state.pending_companies = st.session_state.pending_companies[BATCH_SIZE:]
        
        # 4. 다음 작업 결정
        if st.session_state.pending_companies:
            # 아직 남았으면 -> 잠시 대기 후 리로드 (메모리 초기화)
            status_text.text(f"✅ {processed_count}개 완료! 메모리 정리를 위해 1초 뒤 이어합니다...")
            time.sleep(1) 
            st.rerun() 
        else:
            # 다 끝났으면 -> 종료 처리
            st.session_state.is_processing = False
            st.session_state.pending_companies = [] # 목록 비우기
            
            status_text.text("✨ 모든 분석 완료!")
            progress_bar.progress(1.0)
            st.balloons()
            st.success("✅ 모든 작업이 끝났습니다! '전체 결과' 탭을 확인하세요.")
            
            # 완료 후 입력창 초기화를 위한 리로드 버튼 (선택사항)
            if st.button("새로 시작하기"):
                st.rerun()

# ===== 탭 2: 전체 결과 =====
with tab2:
    st.header("📋 전체 결과")
    
    # 검색 & 통계
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_keyword = st.text_input("🔍 검색", placeholder="종목명 입력", key="search_all")
    with col2:
        st.write("")  # 간격
    with col3:
        total_count = db.get_count()
        st.metric("총 분석 수", f"{total_count}개")
    
    # 결과 조회
    if search_keyword:
        results = db.search_results(search_keyword)
    else:
        results = db.get_all_results(limit=100)
    
    if not results:
        st.info("📝 분석 결과가 없습니다. '새 분석' 탭에서 종목을 분석해보세요!")
    else:
        # 결과 표시
        for result in results:
            created_at = result['created_at']
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%Y-%m-%d %H:%M')
            
            # 북마크 상태
            bookmark_icon = "⭐" if result.get('is_bookmarked') else "☆"
            
            with st.expander(f"📌 {result['company_name']} - {date_str}"):
                # 북마크 버튼
                col_bookmark, col_space = st.columns([1, 5])
                with col_bookmark:
                    if st.button(f"{bookmark_icon} 즐겨찾기", key=f"bookmark_{result['id']}"):
                        db.toggle_bookmark(result['id'])
                        st.rerun()
                
                # DART 결과
                st.markdown('<div class="section-header">📊 DART 보고서 모멘텀</div>', unsafe_allow_html=True)
                if result['dart_error']:
                    st.warning(f"⚠️ {result['dart_error']}")
                else:
                    st.write(f"**보고서:** {result['dart_report']}")
                    st.text(result['dart_result'])
                
                st.markdown("---")
                
                # 뉴스 결과
                st.markdown('<div class="section-header">📰 뉴스 모멘텀 (최근 6개월)</div>', unsafe_allow_html=True)
                st.write(f"**수집 기사:** {result['news_count']}건")
                st.text(result['news_result'])
    
    # 엑셀 다운로드
    if results:
        st.markdown("---")
        df = db.to_dataframe()
        
        # Excel 파일로 변환
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='분석결과')
        output.seek(0)
        
        st.download_button(
            label="📥 전체 결과 엑셀 다운로드",
            data=output,
            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ===== 탭 3: 즐겨찾기 =====
with tab3:
    st.header("⭐ 즐겨찾기")
    
    # 즐겨찾기 결과 조회
    bookmarked_results = db.get_bookmarked_results()
    
    if not bookmarked_results:
        st.info("⭐ 즐겨찾기한 종목이 없습니다. '전체 결과' 탭에서 ☆ 버튼을 클릭하세요!")
    else:
        st.success(f"📌 즐겨찾기: {len(bookmarked_results)}개")
        
        # 결과 표시
        for result in bookmarked_results:
            created_at = result['created_at']
            if isinstance(created_at, str):
                created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%Y-%m-%d %H:%M')
            
            with st.expander(f"⭐ {result['company_name']} - {date_str}"):
                # 북마크 해제 버튼
                if st.button("☆ 즐겨찾기 해제", key=f"unbookmark_{result['id']}"):
                    db.toggle_bookmark(result['id'])
                    st.rerun()
                
                # DART 결과
                st.markdown('<div class="section-header">📊 DART 보고서 모멘텀</div>', unsafe_allow_html=True)
                if result['dart_error']:
                    st.warning(f"⚠️ {result['dart_error']}")
                else:
                    st.write(f"**보고서:** {result['dart_report']}")
                    st.text(result['dart_result'])
                
                st.markdown("---")
                
                # 뉴스 결과
                st.markdown('<div class="section-header">📰 뉴스 모멘텀 (최근 6개월)</div>', unsafe_allow_html=True)
                st.write(f"**수집 기사:** {result['news_count']}건")
                st.text(result['news_result'])
        
        # 엑셀 다운로드
        st.markdown("---")
        df_bookmarked = pd.DataFrame(bookmarked_results)
        
        # Excel 파일로 변환
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_bookmarked.to_excel(writer, index=False, sheet_name='즐겨찾기')
        output.seek(0)
        
        st.download_button(
            label="📥 즐겨찾기 엑셀 다운로드",
            data=output,
            file_name=f"bookmarked_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



