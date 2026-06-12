import streamlit as st
from pathlib import Path
import textwrap
import html

try:
    import plotly.io as pio
    import plotly.graph_objects as go
    pio.templates["aqua_vivid"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, Segoe UI, sans-serif", color="#0B1D33", size=13),
            paper_bgcolor="rgba(255,255,255,0.96)",
            plot_bgcolor="rgba(247,253,252,0.96)",
            colorway=["#0B1D33", "#1FA7B6", "#81D8D0", "#FF6B6B", "#7FD3C6", "#C7D2DC"],
            margin=dict(l=45, r=25, t=58, b=45),
            xaxis=dict(gridcolor="rgba(11,29,51,0.08)", zerolinecolor="rgba(11,29,51,0.12)", linecolor="rgba(11,29,51,0.16)", tickfont=dict(color="#42606F"), titlefont=dict(color="#0B1D33")),
            yaxis=dict(gridcolor="rgba(11,29,51,0.08)", zerolinecolor="rgba(11,29,51,0.12)", linecolor="rgba(11,29,51,0.16)", tickfont=dict(color="#42606F"), titlefont=dict(color="#0B1D33")),
            legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color="#0B1D33"))
        )
    )
    pio.templates.default = "aqua_vivid"
except Exception:
    pass


# ======================================================
# PROJECT PATHS
# ======================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT_DIR / "pages"

PAGE_FILES = {
    "Bài 1 - Cobb-Douglas": "1_Bai_1_Cobb_Douglas.py",
    "Bài 2 - LP ngân sách": "2_Bai_2_LP_Ngan_Sach.py",
    "Bài 3 - Priority Index": "3_Bai_3_Priority_Index.py",
    "Bài 4 - Regional LP": "4_Bai_4_Regional_LP.py",
    "Bài 5 - MIP dự án": "5_Bai_5_MIP_Du_An.py",
    "Bài 6 - TOPSIS vùng": "6_Bai_6_TOPSIS_Vung.py",
    "Bài 7 - NSGA-II Pareto": "7_Bai_7_NSGA_II_Pareto.py",
    "Bài 8 - Tối ưu động": "8_Bai_8_Dynamic_Optimization.py",
    "Bài 9 - AI và lao động": "9_Bai_9_AI_Labor_Market.py",
    "Bài 10 - Stochastic Programming": "10_Bai_10_Stochastic_Programming.py",
    "Bài 11 - Q-learning": "11_Bai_11_Q_Learning_Policy.py",
    "Bài 12 - AIDEOM-VN tích hợp": "12_Bai_12_AIDEOM_Integrated.py",
}

MENU_OPTIONS = ["Trang chủ"] + list(PAGE_FILES.keys())


# ======================================================
# PAGE CONFIG + COMMON CSS
# ======================================================

def setup_page(page_title="AIDEOM-VN", page_icon="🧠"):
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide"
    )

    st.markdown(

        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');
        :root{--navy:#07182C;--navy-2:#0B1D33;--navy-3:#102A45;--teal:#1FA7B6;--aqua:#81D8D0;--mint:#DDF8F5;--mint-2:#ECFBF9;--white:#FFFFFF;--cream:#FFF9F2;--coral:#FF6B6B;--blue-soft:#DFF4FF;--line:#CFE8EA;--text:#0B1D33;--soft-text:#42606F;--shadow:rgba(11,29,51,.10)}
        html,body,[class*="css"]{font-family:'Inter','Segoe UI',sans-serif}.stApp{background:radial-gradient(circle at 8% 8%,rgba(129,216,208,.62) 0,rgba(129,216,208,.30) 18%,transparent 36%),radial-gradient(circle at 82% 4%,rgba(31,167,182,.34) 0,rgba(31,167,182,.16) 18%,transparent 34%),radial-gradient(circle at 92% 88%,rgba(255,107,107,.15) 0,transparent 30%),linear-gradient(135deg,#F3FFFD 0%,#E5F8F6 34%,#FFFFFF 72%,#F4FBFF 100%);color:var(--text)}
        [data-testid="stAppViewContainer"]{background:linear-gradient(120deg,rgba(255,255,255,.50),rgba(255,255,255,.10)),radial-gradient(circle at 16% 26%,rgba(129,216,208,.25),transparent 26%),radial-gradient(circle at 64% 18%,rgba(31,167,182,.14),transparent 24%)}.block-container{padding-top:2.2rem;padding-bottom:3rem}
        [data-testid="stSidebar"]{background:radial-gradient(circle at 50% 0%,rgba(129,216,208,.20),transparent 34%),linear-gradient(180deg,#06182B 0%,#0B1D33 56%,#0E2C48 100%);border-right:1px solid rgba(129,216,208,.18);box-shadow:12px 0 34px rgba(11,29,51,.18)}[data-testid="stSidebarNav"]{display:none}[data-testid="stSidebar"] *{color:#EAFDFC!important}[data-testid="stSidebar"] h2{font-family:'Manrope','Inter',sans-serif;letter-spacing:.2px;color:#fff!important}[data-testid="stSidebar"] .stCaption{color:rgba(234,253,252,.72)!important}[data-testid="stSidebar"] .stRadio>div{gap:.38rem}[data-testid="stSidebar"] .stRadio label{background:rgba(255,255,255,.065);border:1px solid rgba(129,216,208,.16);border-radius:14px;padding:10px 13px;margin-bottom:7px;transition:all .18s ease}[data-testid="stSidebar"] .stRadio label:hover{background:rgba(129,216,208,.18);border-color:rgba(129,216,208,.60);transform:translateX(2px)}
        .main-title{font-family:'Manrope','Inter',sans-serif;font-size:46px;font-weight:800;color:var(--navy-2);margin-bottom:4px;letter-spacing:-.6px}.sub-title{font-size:20px;font-weight:800;color:#0E9BA8;margin-bottom:18px}.page-title{font-family:'Manrope','Inter',sans-serif;font-size:37px;font-weight:800;color:var(--navy-2);margin-bottom:8px;line-height:1.24;letter-spacing:-.4px}.page-subtitle{font-size:16px;color:var(--soft-text);line-height:1.72;margin-bottom:18px}
        .intro-box{background:linear-gradient(135deg,rgba(255,255,255,.92) 0%,rgba(221,248,245,.96) 100%),radial-gradient(circle at top right,rgba(129,216,208,.34),transparent 40%);border:1px solid rgba(129,216,208,.52);border-left:8px solid var(--teal);border-radius:22px;padding:19px 23px;margin-top:14px;margin-bottom:22px;color:var(--text);line-height:1.72;box-shadow:0 16px 36px rgba(31,167,182,.13)}.note-box{background:linear-gradient(180deg,#fff 0%,#EAFBF9 100%);border-left:5px solid var(--aqua);border-radius:16px;padding:14px 18px;margin-top:12px;margin-bottom:20px;color:#154153;line-height:1.65;box-shadow:0 10px 24px rgba(31,167,182,.07)}.soft-card{background:linear-gradient(180deg,rgba(255,255,255,.98) 0%,rgba(247,253,252,.98) 100%);border:1px solid rgba(129,216,208,.48);border-radius:21px;padding:18px;box-shadow:0 15px 30px rgba(11,29,51,.075);min-height:142px;margin-bottom:18px}
        .metric-title{font-size:15px;font-weight:800;color:#0E8794;margin-bottom:10px;line-height:1.35}.metric-value{font-size:31px;font-weight:850;color:var(--navy-2);margin-bottom:10px;letter-spacing:-.3px}.metric-note{font-size:13px;color:var(--soft-text);line-height:1.48}.source-text{font-size:13px;color:#5A7683;font-style:italic;margin-top:4px;margin-bottom:12px;line-height:1.55}.section-caption{font-size:15px;color:var(--soft-text);line-height:1.7;margin-bottom:14px}h1,h2,h3,h4{color:var(--navy-2)!important;font-family:'Manrope','Inter',sans-serif}
        div[data-testid="stMetric"]{background:linear-gradient(180deg,#fff 0%,#F4FFFD 100%);border:1px solid rgba(129,216,208,.48);border-radius:18px;padding:14px 16px;box-shadow:0 12px 26px rgba(11,29,51,.065)}div[data-testid="stDataFrame"],.stDataFrame{border-radius:16px;overflow:hidden;border:1px solid rgba(129,216,208,.42);box-shadow:0 12px 26px rgba(11,29,51,.055)}
        .stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,#0E9BA8 0%,#81D8D0 100%);color:#07182C!important;border:none;border-radius:13px;padding:.58rem 1.08rem;font-weight:850;box-shadow:0 12px 26px rgba(31,167,182,.24)}.stButton>button:hover,.stDownloadButton>button:hover{background:linear-gradient(135deg,#0B8794 0%,#67CEC5 100%);transform:translateY(-1px)}
        div[data-testid="stSlider"] label,div[data-testid="stSelectbox"] label,div[data-testid="stNumberInput"] label,div[data-testid="stTextInput"] label,div[data-testid="stMultiselect"] label,div[data-testid="stCheckbox"] label{color:#0B1D33!important;font-weight:800!important}div[data-testid="stSlider"] div[data-baseweb="slider"]>div{background:#CFF1EE!important}div[data-testid="stSlider"] div[role="slider"]{background-color:#0E9BA8!important;border-color:#0E9BA8!important;box-shadow:0 0 0 4px rgba(31,167,182,.17)!important}
        .stTabs [data-baseweb="tab-list"]{gap:8px}.stTabs [data-baseweb="tab"]{border-radius:999px;background:rgba(255,255,255,.88);border:1px solid rgba(129,216,208,.42);color:#164657;padding:.38rem .95rem;font-weight:800}.stTabs [aria-selected="true"]{background:#81D8D0!important;color:#07182C!important;border-color:#81D8D0!important;box-shadow:0 7px 16px rgba(31,167,182,.18)}div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,.stTextInput input,.stNumberInput input{border-radius:12px!important;border:1px solid rgba(129,216,208,.44)!important;background:rgba(255,255,255,.98)!important}.stAlert{border-radius:15px;border:1px solid rgba(129,216,208,.44)}
        </style>
        """
,
        unsafe_allow_html=True
    )


# ======================================================
# NAVIGATION
# ======================================================

def go_to(page_name):
    if page_name == "Trang chủ":
        st.session_state["selected_page"] = "Trang chủ"
        st.switch_page("app.py")

    file_name = PAGE_FILES.get(page_name)

    if file_name is None:
        st.error("Không tìm thấy trang được chọn.")
        return

    page_path = PAGES_DIR / file_name

    if not page_path.exists():
        st.error(f"Không tìm thấy file: {file_name}")
        return

    st.session_state["selected_page"] = page_name
    st.switch_page(str(page_path))


def render_sidebar(current_page="Trang chủ"):
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = current_page

    st.session_state["selected_page"] = current_page

    with st.sidebar:
        st.markdown("## AIDEOM-VN")
        st.caption("Dashboard mô phỏng và phân tích chính sách kinh tế")

        selected_page = st.radio(
            "Chọn bài",
            MENU_OPTIONS,
            index=MENU_OPTIONS.index(current_page),
        )

        if selected_page != current_page:
            go_to(selected_page)

        st.markdown("---")
        st.caption("Theo dõi dữ liệu, mô phỏng kịch bản và kết quả tối ưu hóa.")


# ======================================================
# REUSABLE COMPONENTS
# ======================================================

def page_header(title, subtitle=None):
    st.markdown(
        f"""
        <div class="page-title">{title}</div>
        """,
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f"""
            <div class="page-subtitle">{subtitle}</div>
            """,
            unsafe_allow_html=True
        )


def clean_html_text(text):
    clean_text = textwrap.dedent(str(text)).strip()
    clean_text = html.escape(clean_text)
    clean_text = clean_text.replace("\n", "<br>")
    return clean_text


def info_box(text):
    clean_text = clean_html_text(text)

    st.markdown(
        f'<div class="intro-box">{clean_text}</div>',
        unsafe_allow_html=True
    )


def note_box(text):
    clean_text = clean_html_text(text)

    st.markdown(
        f'<div class="note-box">{clean_text}</div>',
        unsafe_allow_html=True
    )


def source_note(text):
    clean_text = clean_html_text(text)

    st.markdown(
        f'<div class="source-text">Nguồn: {clean_text}</div>',
        unsafe_allow_html=True
    )


def section_caption(text):
    clean_text = clean_html_text(text)

    st.markdown(
        f'<div class="section-caption">{clean_text}</div>',
        unsafe_allow_html=True
    )


def kpi_card(title, value, note):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )