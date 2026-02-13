# app.py (Mobile-Friendly BBS Mode v2)
import streamlit as st
import asyncio
import pandas as pd
import time
import warnings
import math
import html as html_lib
from datetime import datetime
from openai import AsyncOpenAI
from io import BytesIO
from database import Database
from analyzer import (
    Config, RegexCache, DartProcessor, 
    run_news_pipeline
)

warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

st.set_page_config(
    page_title="System Admin", 
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════
# CSS: 모바일 BBS 스타일 (공백 극한 제거)
# ═══════════════════════════════════════════
st.markdown("""
<style>
    /* ── 글로벌 리셋 ── */
    header, footer, #MainMenu {visibility: hidden !important; height: 0 !important;}
    .block-container {
        padding: 0.4rem 0.6rem 2rem 0.6rem !important;
        max-width: 100% !important;
    }
    
    /* Streamlit 기본 간격 전부 제거 */
    .element-container { margin: 0 !important; padding: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        gap: 0 !important; padding: 0 !important; margin: 0 !important;
    }
    .stMarkdown { min-height: 0 !important; }
    div[data-testid="stHorizontalBlock"] { gap: 0.3rem !important; }

    /* ── 탭 ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 2px solid #ddd; padding: 0 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px; font-size: 13px; padding: 0 18px;
        color: #888; border-bottom: 2px solid transparent; margin-bottom: -2px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #d32f2f; font-weight: 700; border-bottom: 2px solid #d32f2f;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 0.3rem !important; }

    /* ── BBS Expander 행 ── */
    .stExpander {
        border: none !important; box-shadow: none !important;
        background: transparent !important;
        border-bottom: 1px solid #e4e4e4 !important;
        margin: 0 !important; padding: 0 !important; border-radius: 0 !important;
    }
    .stExpander > details > summary {
        padding: 9px 4px !important; font-size: 13px !important;
        color: #333 !important; min-height: 0 !important;
        line-height: 1.35 !important; font-weight: 400 !important;
    }
    .stExpander > details > summary:hover { background-color: #f9f9f9 !important; }
    .stExpander > details > summary p { margin: 0 !important; padding: 0 !important; }
    .stExpander > details[open] > summary {
        background-color: #f5f5f5 !important;
        border-bottom: 1px solid #ddd !important;
        font-weight: 600 !important;
    }
    .stExpander > details > div[data-testid="stExpanderDetails"] {
        padding: 0 !important; background-color: #fff !important;
    }

    /* ── 버튼 ── */
    .stButton > button {
        height: 28px; font-size: 11.5px; padding: 0 12px;
        border: 1px solid #ccc; background: #fafafa;
        border-radius: 3px; color: #555; width: auto; min-width: 55px;
    }
    .stButton > button:hover { background: #f0f0f0; border-color: #aaa; }

    /* ── 입력 ── */
    .stTextArea textarea, .stTextInput input { font-size: 13px; }
    .stDownloadButton > button { height: 28px; font-size: 11.5px; padding: 0 12px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# 데이터베이스 & 설정
# ═══════════════════════════════════════════
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

# ═══════════════════════════════════════════
# GPT 분석
# ═══════════════════════════════════════════
async def analyze_news_with_gpt(company_name: str, articles: list) -> str:
    if not articles: return "-"
    articles.sort(key=lambda x: x['pub_date'], reverse=True)
    context = "".join(f"[{a['pub_date'].strftime('%Y.%m.%d')}] {a['title']}\n" for a in articles)
    prompt = f"""당신은 주식 시장의 '모멘텀 전문 분석가'입니다. 
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

{context}"""
    try:
        res = await openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1)
        return res.choices[0].message.content
    except Exception as e: return f"Err: {e}"

async def analyze_dart_with_gpt(company_name: str, report_nm: str, dart_text: str) -> str:
    if not dart_text or len(dart_text) < 100: return "-"
    prompt = f"""당신은 주식 시장의 '모멘텀 전문 분석가'입니다.
        
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

{dart_text[:30000]}"""
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

# ═══════════════════════════════════════════
# 본문 HTML 렌더 (Streamlit 여백 간섭 완전 회피)
# ═══════════════════════════════════════════
def render_post(row, prev_row=None, next_row=None):
    """게시글 전체를 단일 HTML로 렌더링 — Streamlit 마크다운 여백 문제 원천 차단"""
    dart = html_lib.escape(row.get('dart_result') or '-').replace('\n', '<br>')
    news = html_lib.escape(row.get('news_result') or '-').replace('\n', '<br>')

    # 이전/다음
    nav_items = []
    if prev_row:
        p_dt = prev_row['created_at']
        if isinstance(p_dt, str): p_dt = datetime.strptime(p_dt, '%Y-%m-%d %H:%M:%S')
        nav_items.append(f'<div style="padding:4px 0;"><span style="color:#bbb;font-size:11px;display:inline-block;width:40px;">▲이전</span>'
                         f'<span style="color:#555;font-size:12px;">{html_lib.escape(prev_row["company_name"])}&nbsp;{p_dt.strftime("%m.%d")}</span></div>')
    if next_row:
        n_dt = next_row['created_at']
        if isinstance(n_dt, str): n_dt = datetime.strptime(n_dt, '%Y-%m-%d %H:%M:%S')
        nav_items.append(f'<div style="padding:4px 0;"><span style="color:#bbb;font-size:11px;display:inline-block;width:40px;">▼다음</span>'
                         f'<span style="color:#555;font-size:12px;">{html_lib.escape(next_row["company_name"])}&nbsp;{n_dt.strftime("%m.%d")}</span></div>')
    nav_html = f'<div style="margin-top:8px;padding-top:6px;border-top:1px solid #e8e8e8;">{"".join(nav_items)}</div>' if nav_items else ""

    return f"""<div style="padding:10px 8px 12px 8px;font-family:-apple-system,'Malgun Gothic',sans-serif;">
<div style="font-size:11px;color:#aaa;letter-spacing:0.3px;font-weight:600;">DART 공시</div>
<div style="font-size:13px;line-height:1.7;color:#222;padding:4px 0 10px 0;">{dart}</div>
<div style="border-top:1px solid #f0f0f0;padding-top:8px;font-size:11px;color:#aaa;letter-spacing:0.3px;font-weight:600;">뉴스 모멘텀</div>
<div style="font-size:13px;line-height:1.7;color:#222;padding:4px 0 2px 0;">{news}</div>
{nav_html}
</div>"""


# ==================== UI ====================

tab1, tab2, tab3 = st.tabs(["수집", "결과", "보관"])

# ──── [1] 수집 ────
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
            time.sleep(0.5); st.rerun()
        else:
            st.session_state.is_processing = False; st.rerun()

# ──── [2] 결과 ────
with tab2:
    if 'page' not in st.session_state: st.session_state.page = 1
    all_res = db.get_all_results(limit=10000)

    c_s, c_cnt = st.columns([8, 2])
    with c_s:
        kw = st.text_input("검색", label_visibility="collapsed", placeholder="종목명 검색")
    with c_cnt:
        st.markdown(f"<div style='text-align:right;font-size:11px;color:#aaa;padding:8px 2px 0 0;'>{len(all_res)}건</div>", unsafe_allow_html=True)

    targets = [r for r in all_res if kw in r['company_name']] if kw else all_res

    PER_PAGE = 50
    total_pg = math.ceil(len(targets) / PER_PAGE) if targets else 1
    if st.session_state.page > total_pg: st.session_state.page = 1
    start = (st.session_state.page - 1) * PER_PAGE
    view_data = targets[start:start + PER_PAGE]

    # 헤더
    st.markdown('<div style="display:flex;justify-content:space-between;padding:4px;border-bottom:2px solid #bbb;">'
                '<span style="font-size:11px;color:#999;font-weight:600;">종목명</span>'
                '<span style="font-size:11px;color:#999;font-weight:600;">날짜</span></div>', unsafe_allow_html=True)

    if not view_data:
        st.caption("데이터 없음")
    else:
        for i, row in enumerate(view_data):
            gidx = start + i
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            mark = " ★" if row.get('is_bookmarked') else ""

            with st.expander(f"**{row['company_name']}**{mark}　·　{dt.strftime('%m.%d %H:%M')}"):
                # 버튼 (왼쪽 정렬, 나머지 공간은 빈칸)
                b1, b2, _ = st.columns([1.5, 1.5, 9])
                with b1:
                    lbl = "★ 해제" if row.get('is_bookmarked') else "☆ 보관"
                    if st.button(lbl, key=f"bk_{row['id']}"): db.toggle_bookmark(row['id']); st.rerun()
                with b2:
                    if st.button("삭제", key=f"del_{row['id']}"): db.delete_result(row['id']); st.rerun()

                prev_r = targets[gidx - 1] if gidx > 0 else None
                next_r = targets[gidx + 1] if gidx < len(targets) - 1 else None
                st.markdown(render_post(row, prev_r, next_r), unsafe_allow_html=True)

    if total_pg > 1:
        cp, cc, cn = st.columns([2, 4, 2])
        with cp:
            if st.session_state.page > 1 and st.button("◀ 이전", key="pg_prev"):
                st.session_state.page -= 1; st.rerun()
        with cc:
            st.markdown(f"<div style='text-align:center;font-size:12px;color:#aaa;padding-top:8px;'>{st.session_state.page}/{total_pg}</div>", unsafe_allow_html=True)
        with cn:
            if st.session_state.page < total_pg and st.button("다음 ▶", key="pg_next"):
                st.session_state.page += 1; st.rerun()

# ──── [3] 보관 ────
with tab3:
    bk_list = db.get_bookmarked_results()

    if bk_list:
        df = pd.DataFrame(bk_list)
        out = BytesIO()
        with pd.ExcelWriter(out, engine='openpyxl') as writer: df.to_excel(writer, index=False)
        out.seek(0)
        st.download_button("Excel", data=out, file_name="saved.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown('<div style="display:flex;justify-content:space-between;padding:4px;border-bottom:2px solid #bbb;">'
                '<span style="font-size:11px;color:#999;font-weight:600;">종목명</span>'
                '<span style="font-size:11px;color:#999;font-weight:600;">날짜</span></div>', unsafe_allow_html=True)

    if not bk_list:
        st.caption("보관된 항목이 없습니다.")
    else:
        for row in bk_list:
            dt = row['created_at']
            if isinstance(dt, str): dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            with st.expander(f"★ **{row['company_name']}**　·　{dt.strftime('%m.%d %H:%M')}"):
                b1, _ = st.columns([1.5, 10])
                with b1:
                    if st.button("해제", key=f"ubk_{row['id']}"): db.toggle_bookmark(row['id']); st.rerun()
                st.markdown(render_post(row), unsafe_allow_html=True)
