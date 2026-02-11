# app.py
import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime
from openai import AsyncOpenAI
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
        CLIENT_ID=st.secrets["NAVER_CLIENT_ID"],
        CLIENT_SECRET=st.secrets["NAVER_CLIENT_SECRET"],
        DART_API_KEY=st.secrets["DART_API_KEY"],
        OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
    )

try:
    config = get_config()
    openai_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
except:
    st.error("⚠️ API 키를 설정해주세요. `.streamlit/secrets.toml` 파일을 확인하세요.")
    st.stop()

# 상장사 목록 로드
@st.cache_resource
def load_companies():
    try:
        df = pd.read_csv('krx_stocks.csv', encoding='utf-8')
        companies = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        return companies, RegexCache(companies)
    except:
        return [], None

ALL_COMPANIES, REGEX_CACHE = load_companies()

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

# 단일 종목 분석 함수
async def analyze_company(company_name: str, progress_callback=None):
    # 1. DART 분석
    if progress_callback:
        progress_callback(f"📊 {company_name} DART 분석 중...")
    
    dart_processor = DartProcessor(config.DART_API_KEY)
    report_nm, dart_text, dart_error = dart_processor.process(company_name)
    
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

# 탭 생성
tab1, tab2 = st.tabs(["🚀 새 분석", "📋 전체 결과"])

# ===== 탭 1: 새 분석 =====
with tab1:
    st.header("🚀 새 분석 시작")
    
    companies_input = st.text_area(
        "종목명 입력 (줄바꿈으로 구분)",
        placeholder="삼성전자\nSK하이닉스\n케어젠",
        height=150
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 분석 시작", type="primary", use_container_width=True)
    
    if analyze_button:
        if not companies_input.strip():
            st.warning("⚠️ 종목명을 입력해주세요.")
        else:
            companies_list = [c.strip() for c in companies_input.split('\n') if c.strip()]
            
            # 종목명 검증
            if ALL_COMPANIES:
                invalid = []
                for company in companies_list:
                    if company not in ALL_COMPANIES and company.replace(" ", "") not in ALL_COMPANIES:
                        invalid.append(company)
                
                if invalid:
                    st.error(f"⚠️ 다음 종목을 찾을 수 없습니다: {', '.join(invalid)}")
                    st.stop()
            
            st.success(f"✅ 총 {len(companies_list)}개 종목 분석 시작")
            
            # 프로그레스 바
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 비동기 실행
            for idx, company in enumerate(companies_list):
                status_text.text(f"[{idx+1}/{len(companies_list)}] {company} 분석 중...")
                
                def update_status(msg):
                    status_text.text(f"[{idx+1}/{len(companies_list)}] {msg}")
                
                try:
                    result = asyncio.run(analyze_company(company, update_status))
                    st.success(f"✅ {company} 완료")
                except Exception as e:
                    st.error(f"❌ {company} 오류: {e}")
                
                progress_bar.progress((idx + 1) / len(companies_list))
            
            status_text.text("✨ 모든 분석 완료!")
            st.balloons()
            
            # 자동으로 전체 결과 탭으로 이동 안내
            st.info("👉 '전체 결과' 탭에서 결과를 확인하세요!")

# ===== 탭 2: 전체 결과 =====
with tab2:
    st.header("📋 전체 결과")
    
    # 검색 & 정렬
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_keyword = st.text_input("🔍 검색", placeholder="종목명 입력")
    with col2:
        st.write("")  # 간격
    with col3:
        total_count = db.get_count()
        st.metric("총 분석 수", f"{total_count}개")
    
    # 결과 조회
    if search_keyword:
        results = db.search_results(search_keyword)
    else:
        results = db.get_all_results(limit=50)
    
    if not results:
        st.info("📝 분석 결과가 없습니다. '새 분석' 탭에서 종목을 분석해보세요!")
    else:
        # 결과 표시
        for result in results:
            created_at = datetime.strptime(result['created_at'], '%Y-%m-%d %H:%M:%S')
            date_str = created_at.strftime('%Y-%m-%d %H:%M')
            
            with st.expander(f"📌 {result['company_name']} - {date_str}"):
                # 삭제 버튼
                col_del1, col_del2 = st.columns([5, 1])
                # 삭제 버튼 부분 전체를 이렇게 교체:
                with col_del2:
                    delete_key = f"delete_confirm_{result['id']}"
                    if delete_key not in st.session_state:
                        st.session_state[delete_key] = False
                    
                    if not st.session_state[delete_key]:
                        if st.button("🗑️ 삭제", key=f"del_{result['id']}"):
                            st.session_state[delete_key] = True
                    else:
                        col_confirm1, col_confirm2 = st.columns(2)
                        with col_confirm1:
                            if st.button("✅ 확인", key=f"confirm_{result['id']}"):
                                db.delete_result(result['id'])
                                del st.session_state[delete_key]
                                st.success("삭제됨")
                        with col_confirm2:
                            if st.button("❌ 취소", key=f"cancel_{result['id']}"):
                                st.session_state[delete_key] = False
                
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
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 전체 결과 CSV 다운로드",
            data=csv,
            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# 사이드바
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    1. **🚀 새 분석 탭**
       - 종목명을 줄바꿈으로 입력
       - 분석 시작 버튼 클릭
       - 진행 상황 확인
    
    2. **📋 전체 결과 탭**
       - 과거 분석 결과 조회
       - 검색 기능 사용
       - 결과 삭제 가능
       - CSV 다운로드
    
    3. **💡 팁**
       - 어디서든 접속 가능
       - 결과는 영구 저장
       - 한 번에 여러 종목 분석
    """)
    
    st.markdown("---")

    st.caption("Made with ❤️ by Streamlit")

