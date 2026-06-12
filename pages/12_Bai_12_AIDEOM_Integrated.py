
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from utils.aideom_ui import (
    setup_page,
    render_sidebar,
    source_note,
    section_caption,
    kpi_card,
)

# ======================================================
# PAGE SETUP
# ======================================================

setup_page("Bài 12 - AIDEOM-VN tích hợp")

def safe_render_bai12_sidebar():
    """
    Dùng sidebar chung nếu tìm thấy đúng nhãn Bài 12 trong MENU_OPTIONS.
    Cách này tránh lỗi list.index nếu nhãn trong utils/aideom_ui.py khác dấu ':' hoặc '-'.
    """
    try:
        import utils.aideom_ui as ui
        menu_options = getattr(ui, "MENU_OPTIONS", [])

        candidates = [
            "Bài 12: AIDEOM-VN tích hợp",
            "Bài 12 - AIDEOM-VN tích hợp",
            "Bài 12 AIDEOM Integrated",
            "Bài 12 - AIDEOM Integrated",
            "Bài 12 AIDEOM-VN tích hợp",
        ]

        for name in candidates:
            if name in menu_options:
                render_sidebar(name)
                return

        for name in menu_options:
            normalized = str(name).lower()
            if "bài 12" in normalized or "bai 12" in normalized or "aideom" in normalized:
                render_sidebar(name)
                return

        render_sidebar("Bài 12: AIDEOM-VN tích hợp")

    except Exception:
        st.sidebar.markdown("### AIDEOM-VN")
        st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
        st.sidebar.markdown("---")
        st.sidebar.info(
            "Đang ở Bài 12. Nếu muốn hiện menu chung, kiểm tra lại tên Bài 12 trong MENU_OPTIONS của utils/aideom_ui.py."
        )

safe_render_bai12_sidebar()

PLOT_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "doubleClick": False,
    "responsive": True,
}

BLUE = "#1FA7B6"
SKY = "#1FA7B6"
SKY_LIGHT = "#81D8D0"
SKY_PALE = "#F8FCFC"
GREEN = "#E6F7F5"
PINK = "#FAD7D7"
LAVENDER = "#F1FBFA"

def show_plot(fig):
    fig.update_layout(
        dragmode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1f2937"),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

st.title("Bài 12. Đồ án tích hợp AIDEOM-VN")

st.write(
    """
    Trang này tích hợp các kết quả chính của hệ thống AIDEOM-VN thành một dashboard ra quyết định.
    Mục tiêu là so sánh 5 kịch bản chính sách theo các chỉ tiêu kinh tế, chuyển đổi số, AI, lao động,
    rủi ro môi trường và năng lực chống chịu.
    """
)

source_note(
    """
    Dashboard sử dụng các tệp dữ liệu trong thư mục data gồm vietnam_macro_2020_2025.csv,
    vietnam_sectors_2024.csv và vietnam_regions_2024.csv. Các tham số và khuyến nghị là mô phỏng phục vụ phân tích,
    không phải khuyến nghị ngân sách hay chính sách chính thức.
    """
)

# ======================================================
# LOAD DATA
# ======================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MACRO_FILE = DATA_DIR / "vietnam_macro_2020_2025.csv"
SECTOR_FILE = DATA_DIR / "vietnam_sectors_2024.csv"
REGION_FILE = DATA_DIR / "vietnam_regions_2024.csv"

def safe_read_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

macro_df = safe_read_csv(MACRO_FILE)
sector_df = safe_read_csv(SECTOR_FILE)
region_df = safe_read_csv(REGION_FILE)

base_values = {
    "gdp_2025": 12847.6,
    "digital_2025": 19.5,
    "ai_firms_2025": 80.1,
    "trained_labor_2025": 29.2,
    "fdi_2025": 27.6,
    "export_2025": 475.0,
}

def get_macro_value(possible_cols, fallback):
    if macro_df.empty:
        return fallback

    for col in possible_cols:
        if col in macro_df.columns:
            try:
                return float(macro_df[col].iloc[-1])
            except Exception:
                pass

    return fallback

gdp_base = get_macro_value(
    ["GDP_trillion_VND", "gdp", "GDP", "GDP_nghin_ty_VND", "Y_GDP"],
    base_values["gdp_2025"]
)

digital_base = get_macro_value(
    ["digital_economy_pct", "digital_index", "D", "digital_gdp_pct"],
    base_values["digital_2025"]
)

ai_base = get_macro_value(
    ["AI_firms_thousand", "ai_firms", "AI", "ai_enterprises"],
    base_values["ai_firms_2025"]
)

human_base = get_macro_value(
    ["trained_labor_pct", "H", "human_capital", "labor_trained_pct"],
    base_values["trained_labor_2025"]
)

fdi_base = get_macro_value(
    ["FDI", "fdi", "FDI_billion_USD"],
    base_values["fdi_2025"]
)

export_base = get_macro_value(
    ["export", "exports", "Export_billion_USD"],
    base_values["export_2025"]
)

num_regions = len(region_df) if not region_df.empty else 6
num_sectors = len(sector_df) if not sector_df.empty else 8

# ======================================================
# SCENARIOS
# ======================================================

scenario_table = pd.DataFrame({
    "Kịch bản": [
        "S1. Truyền thống",
        "S2. Số hóa nhanh",
        "S3. AI dẫn dắt",
        "S4. Bao trùm số",
        "S5. Tối ưu cân bằng",
    ],
    "K": [0.70, 0.25, 0.20, 0.30, 0.35],
    "D": [0.10, 0.45, 0.20, 0.20, 0.25],
    "AI": [0.10, 0.15, 0.45, 0.10, 0.20],
    "H": [0.10, 0.15, 0.15, 0.40, 0.20],
    "Diễn giải": [
        "Ưu tiên vốn vật chất, hạ tầng truyền thống, xuất khẩu và FDI.",
        "Ưu tiên chính phủ số, doanh nghiệp số, dữ liệu và nền tảng số.",
        "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu và tự động hóa.",
        "Ưu tiên nhân lực, SME, vùng yếu, giáo dục số và nông nghiệp số.",
        "Kịch bản cân bằng giữa tăng trưởng, số hóa, AI, nhân lực và rủi ro.",
    ]
})

# ======================================================
# SECTION 1: SETTINGS
# ======================================================

st.markdown("---")
st.header("1. Thiết lập tổng hợp hệ thống")

section_caption(
    """
    Phần này cho phép điều chỉnh ngân sách, giả định tăng trưởng nền, mức ưu tiên rủi ro và mức ưu tiên bao trùm.
    Cơ cấu K-D-AI-H của từng kịch bản có thể chỉnh trực tiếp; hệ thống sẽ tự chuẩn hóa tổng tỷ trọng của mỗi kịch bản về 1.
    """
)

setting_col, chart_col = st.columns([1.05, 1])

with setting_col:
    setting_box = st.container(border=True)

    with setting_box:
        st.markdown("#### Tham số mô phỏng 2026–2030")

        c1, c2 = st.columns(2)

        with c1:
            annual_budget = st.number_input(
                "Ngân sách chính sách mỗi năm",
                min_value=20000,
                max_value=150000,
                value=65000,
                step=5000,
            )

            base_growth = st.number_input(
                "Tăng trưởng nền nếu không can thiệp",
                min_value=0.00,
                max_value=0.10,
                value=0.055,
                step=0.005,
                format="%.3f",
            )

        with c2:
            risk_aversion = st.number_input(
                "Mức ưu tiên giảm rủi ro",
                min_value=0.00,
                max_value=1.00,
                value=0.35,
                step=0.05,
                format="%.2f",
            )

            inclusion_weight = st.number_input(
                "Mức ưu tiên bao trùm",
                min_value=0.00,
                max_value=1.00,
                value=0.30,
                step=0.05,
                format="%.2f",
            )

        st.markdown("#### Cơ cấu 5 kịch bản")

        scenario_input = st.data_editor(
            scenario_table,
            hide_index=True,
            use_container_width=True,
            disabled=["Kịch bản", "Diễn giải"],
            column_config={
                "K": st.column_config.NumberColumn("K", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
                "D": st.column_config.NumberColumn("D", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
                "AI": st.column_config.NumberColumn("AI", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
                "H": st.column_config.NumberColumn("H", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
            },
        )

        for idx in scenario_input.index:
            total = scenario_input.loc[idx, ["K", "D", "AI", "H"]].sum()
            if total > 0:
                scenario_input.loc[idx, ["K", "D", "AI", "H"]] = scenario_input.loc[idx, ["K", "D", "AI", "H"]] / total

with chart_col:
    st.markdown("#### Cơ cấu phân bổ theo kịch bản")

    fig = go.Figure()

    colors = {
        "K": SKY_LIGHT,
        "D": SKY,
        "AI": "#FF6B6B",
        "H": GREEN,
    }

    names = {
        "K": "Vốn vật chất K",
        "D": "Số hóa D",
        "AI": "AI",
        "H": "Nhân lực H",
    }

    for col in ["K", "D", "AI", "H"]:
        fig.add_trace(
            go.Bar(
                x=scenario_input["Kịch bản"],
                y=scenario_input[col],
                name=names[col],
                marker_color=colors[col],
                hovertemplate=(
                    "Kịch bản: %{x}<br>"
                    f"Hạng mục: {names[col]}<br>"
                    "Tỷ trọng: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        height=340,
        title="Cơ cấu K-D-AI-H",
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Tỷ trọng",
        legend=dict(orientation="h", y=-0.22),
    )

    show_plot(fig)

# ======================================================
# CORE SIMULATION
# ======================================================

def simulate_scenario(row):
    K = row["K"]
    D = row["D"]
    AI = row["AI"]
    H = row["H"]

    growth_bonus = (
        0.015 * K
        + 0.030 * D
        + 0.040 * AI
        + 0.025 * H
    )

    risk_penalty = (
        risk_aversion * (
            0.018 * AI
            + 0.010 * K
            - 0.012 * H
            - 0.006 * D
        )
    )

    avg_growth = base_growth + growth_bonus - risk_penalty
    gdp_2030 = gdp_base * ((1 + avg_growth) ** 5)

    digital_2030 = digital_base + 8.0 * D + 3.0 * AI + 2.5 * H
    ai_readiness_2030 = ai_base + 22.0 * AI + 7.0 * D + 5.0 * H
    human_2030 = human_base + 10.0 * H + 2.0 * D

    new_jobs = annual_budget * (0.00035 * D + 0.00042 * AI + 0.00028 * H)
    displaced_jobs = annual_budget * (0.00030 * AI + 0.00012 * D - 0.00015 * H)
    displaced_jobs = max(displaced_jobs, 0)
    upgraded_jobs = annual_budget * (0.00045 * H + 0.00010 * D)
    net_jobs = new_jobs + upgraded_jobs - displaced_jobs

    cyber_risk = 100 * (0.35 * AI + 0.18 * D - 0.22 * H + 0.08 * K)
    env_risk = 100 * (0.38 * K + 0.20 * D + 0.26 * AI - 0.12 * H)
    dependency_risk = 100 * (0.25 * AI + 0.18 * D - 0.10 * H)

    cyber_risk = float(np.clip(cyber_risk, 0, 100))
    env_risk = float(np.clip(env_risk, 0, 100))
    dependency_risk = float(np.clip(dependency_risk, 0, 100))

    resilience = 100 - (0.35 * cyber_risk + 0.35 * env_risk + 0.30 * dependency_risk)
    resilience = float(np.clip(resilience, 0, 100))

    inclusion_score = 100 * (0.55 * H + 0.25 * D + 0.10 * AI + 0.10 * (1 - K))
    innovation_score = 100 * (0.45 * AI + 0.30 * D + 0.15 * H + 0.10 * K)

    normalized_growth = min(max(avg_growth / 0.10, 0), 1)
    normalized_digital = min(max(digital_2030 / 40, 0), 1)
    normalized_innovation = min(max(innovation_score / 100, 0), 1)
    normalized_inclusion = min(max(inclusion_score / 100, 0), 1)
    normalized_resilience = min(max(resilience / 100, 0), 1)

    composite = (
        0.30 * normalized_growth
        + 0.20 * normalized_digital
        + 0.20 * normalized_innovation
        + inclusion_weight * 0.20 * normalized_inclusion
        + 0.10 * normalized_resilience
    )

    return pd.Series({
        "GDP 2030": gdp_2030,
        "Tăng trưởng TB": avg_growth,
        "Digital 2030": digital_2030,
        "AI readiness 2030": ai_readiness_2030,
        "Nhân lực số 2030": human_2030,
        "NewJob": new_jobs,
        "UpgradeJob": upgraded_jobs,
        "DisplacedJob": displaced_jobs,
        "NetJob": net_jobs,
        "Cyber risk": cyber_risk,
        "Environmental risk": env_risk,
        "Dependency risk": dependency_risk,
        "Resilience": resilience,
        "Inclusion score": inclusion_score,
        "Innovation score": innovation_score,
        "AIDEOM score": composite,
        "Growth component": 0.30 * normalized_growth,
        "Digital component": 0.20 * normalized_digital,
        "Innovation component": 0.20 * normalized_innovation,
        "Inclusion component": inclusion_weight * 0.20 * normalized_inclusion,
        "Resilience component": 0.10 * normalized_resilience,
    })

kpi_df = scenario_input.copy()
kpi_values = kpi_df.apply(simulate_scenario, axis=1)
kpi_df = pd.concat([kpi_df, kpi_values], axis=1)

best_idx = kpi_df["AIDEOM score"].idxmax()
best_scenario = kpi_df.loc[best_idx, "Kịch bản"]

# ======================================================
# SECTION 2: SYSTEM OVERVIEW
# ======================================================

st.markdown("---")
st.header("2. Bức tranh tích hợp AIDEOM-VN")

section_caption(
    """
    Bài 12 không chỉ trình bày lại kết quả riêng lẻ của các bài trước, mà kết nối chúng thành một logic ra quyết định:
    dữ liệu vĩ mô, ngành và vùng được đưa vào các mô hình dự báo, tối ưu, mô phỏng rủi ro và cuối cùng chuyển thành so sánh kịch bản.
    """
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    kpi_card("Kịch bản tốt nhất", best_scenario, "Theo AIDEOM score hiện tại.")

with m2:
    kpi_card("GDP 2030 cao nhất", f"{kpi_df['GDP 2030'].max():,.0f}", "Nghìn tỷ VND theo mô phỏng.")

with m3:
    kpi_card("NetJob cao nhất", f"{kpi_df['NetJob'].max():,.0f}", "Việc làm ròng cao nhất.")

with m4:
    avg_risk_min = kpi_df[["Cyber risk", "Environmental risk", "Dependency risk"]].mean(axis=1).min()
    kpi_card("Rủi ro thấp nhất", f"{avg_risk_min:.1f}", "Trung bình ba nhóm rủi ro.")

module_df = pd.DataFrame({
    "Nhóm mô hình": [
        "Dự báo vĩ mô",
        "Ưu tiên ngành/vùng",
        "Tối ưu phân bổ",
        "Lao động và an sinh",
        "Bất định và thích nghi",
        "Dashboard tích hợp",
    ],
    "Bài liên quan": [
        "Bài 1, Bài 8",
        "Bài 3, Bài 6",
        "Bài 2, Bài 4, Bài 5, Bài 7",
        "Bài 9",
        "Bài 10, Bài 11",
        "Bài 12",
    ],
    "Vai trò trong AIDEOM-VN": [
        "Tạo nền tham chiếu GDP, số hóa, AI và nhân lực đến 2030.",
        "Xác định ngành/vùng có năng lực hấp thụ công nghệ và nhu cầu ưu tiên.",
        "Chuyển mục tiêu chính sách thành phân bổ nguồn lực có ràng buộc.",
        "Kiểm tra việc làm mới, việc làm nâng cấp và việc làm bị thay thế.",
        "Đánh giá chính sách trong điều kiện rủi ro, kịch bản và trạng thái thay đổi.",
        "Tổng hợp thành so sánh kịch bản và khuyến nghị có cảnh báo.",
    ],
})

overview_col, flow_col = st.columns([1.05, 0.95])

with overview_col:
    st.markdown("#### Cấu trúc tích hợp")

    st.dataframe(module_df, use_container_width=True)

with flow_col:
    st.markdown("#### Luồng ra quyết định")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=16,
                    line=dict(color="#94a3b8", width=0.5),
                    label=[
                        "Dữ liệu vĩ mô",
                        "Dữ liệu ngành",
                        "Dữ liệu vùng",
                        "Dự báo & tối ưu",
                        "Lao động & rủi ro",
                        "AIDEOM score",
                        "Khuyến nghị chính sách",
                    ],
                    color=[SKY_LIGHT, SKY_LIGHT, SKY_LIGHT, SKY, GREEN, "#FF6B6B", BLUE],
                ),
                link=dict(
                    source=[0, 1, 2, 3, 4, 5],
                    target=[3, 3, 3, 5, 5, 6],
                    value=[2, 2, 2, 3, 2, 4],
                    color=["rgba(183,215,239,0.45)"] * 6,
                ),
            )
        ]
    )

    fig.update_layout(
        title="Từ dữ liệu đến khuyến nghị",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
    )

    show_plot(fig)

# ======================================================
# SECTION 3: KPI COMPARISON
# ======================================================

st.markdown("---")
st.header("3. So sánh 5 kịch bản chính sách")

section_caption(
    """
    Bảng và biểu đồ dưới đây cho thấy mỗi kịch bản có ưu thế khác nhau.
    Kịch bản có GDP cao nhất chưa chắc có NetJob cao nhất; kịch bản có đổi mới mạnh cũng có thể đi kèm rủi ro cao hơn.
    """
)

kpi_display = kpi_df[
    [
        "Kịch bản",
        "GDP 2030",
        "Tăng trưởng TB",
        "Digital 2030",
        "AI readiness 2030",
        "Nhân lực số 2030",
        "NetJob",
        "Cyber risk",
        "Environmental risk",
        "Dependency risk",
        "Resilience",
        "AIDEOM score",
    ]
].copy()

st.dataframe(
    kpi_display.style.format({
        "GDP 2030": "{:,.0f}",
        "Tăng trưởng TB": "{:.2%}",
        "Digital 2030": "{:.2f}",
        "AI readiness 2030": "{:.2f}",
        "Nhân lực số 2030": "{:.2f}",
        "NetJob": "{:,.0f}",
        "Cyber risk": "{:.1f}",
        "Environmental risk": "{:.1f}",
        "Dependency risk": "{:.1f}",
        "Resilience": "{:.1f}",
        "AIDEOM score": "{:.4f}",
    }),
    use_container_width=True,
)

chart_a, chart_b = st.columns([1, 1])

with chart_a:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=kpi_df["Kịch bản"],
            y=kpi_df["GDP 2030"],
            name="GDP 2030",
            marker_color=SKY,
            marker_line=dict(color=BLUE, width=1),
            hovertemplate="Kịch bản: %{x}<br>GDP 2030: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="GDP dự báo năm 2030",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Nghìn tỷ VND",
    )

    show_plot(fig)

with chart_b:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=kpi_df["Digital 2030"],
            y=kpi_df["AI readiness 2030"],
            mode="markers+text",
            text=kpi_df["Kịch bản"],
            textposition="top center",
            marker=dict(
                size=np.maximum(kpi_df["NetJob"] / max(kpi_df["NetJob"].max(), 1) * 45, 14),
                color=kpi_df["Resilience"],
                colorscale=[
                    [0, SKY_LIGHT],
                    [0.5, SKY],
                    [1, BLUE],
                ],
                showscale=True,
                colorbar=dict(title="Resilience"),
                line=dict(color="#334155", width=0.6),
            ),
            hovertemplate=(
                "Kịch bản: %{text}<br>"
                "Digital 2030: %{x:.2f}<br>"
                "AI readiness 2030: %{y:.2f}<br>"
                "Resilience: %{marker.color:.1f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Bản đồ Digital - AI - Resilience",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Digital 2030",
        yaxis_title="AI readiness 2030",
    )

    show_plot(fig)

chart_c, chart_d = st.columns([1, 1])

with chart_c:
    risk_long = kpi_df.melt(
        id_vars="Kịch bản",
        value_vars=["Cyber risk", "Environmental risk", "Dependency risk"],
        var_name="Loại rủi ro",
        value_name="Điểm rủi ro",
    )

    risk_colors = {
        "Cyber risk": SKY_LIGHT,
        "Environmental risk": SKY,
        "Dependency risk": "#FF6B6B",
    }

    fig = go.Figure()

    for risk in risk_long["Loại rủi ro"].unique():
        temp = risk_long[risk_long["Loại rủi ro"] == risk]
        fig.add_trace(
            go.Bar(
                x=temp["Kịch bản"],
                y=temp["Điểm rủi ro"],
                name=risk,
                marker_color=risk_colors.get(risk, SKY),
                hovertemplate="Kịch bản: %{x}<br>Rủi ro: %{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        title="So sánh rủi ro theo kịch bản",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Điểm rủi ro",
        legend=dict(orientation="h", y=-0.22),
    )

    show_plot(fig)

with chart_d:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=kpi_df["Kịch bản"],
            y=kpi_df["NetJob"],
            name="NetJob",
            marker_color=GREEN,
            marker_line=dict(color=BLUE, width=1),
            hovertemplate="Kịch bản: %{x}<br>NetJob: %{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=kpi_df["Kịch bản"],
            y=kpi_df["DisplacedJob"],
            name="DisplacedJob",
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=8, color=SKY),
            hovertemplate="Kịch bản: %{x}<br>DisplacedJob: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Việc làm ròng và việc làm bị thay thế",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Số việc làm",
        legend=dict(orientation="h", y=-0.22),
    )

    show_plot(fig)

# ======================================================
# SECTION 4: SCORE DECOMPOSITION
# ======================================================

st.markdown("---")
st.header("4. Phân rã AIDEOM score")

section_caption(
    """
    Điểm AIDEOM được tổng hợp từ tăng trưởng, số hóa, đổi mới, bao trùm và năng lực chống chịu.
    Phân rã điểm giúp giải thích vì sao một kịch bản được xếp cao hơn, thay vì chỉ nhìn vào một con số tổng hợp.
    """
)

score_cols = [
    "Growth component",
    "Digital component",
    "Innovation component",
    "Inclusion component",
    "Resilience component",
]

score_names = {
    "Growth component": "Tăng trưởng",
    "Digital component": "Số hóa",
    "Innovation component": "Đổi mới",
    "Inclusion component": "Bao trùm",
    "Resilience component": "Chống chịu",
}

score_color = {
    "Growth component": SKY_LIGHT,
    "Digital component": SKY,
    "Innovation component": "#FF6B6B",
    "Inclusion component": GREEN,
    "Resilience component": LAVENDER,
}

score_col, score_chart_col = st.columns([1, 1])

with score_col:
    score_display = kpi_df[["Kịch bản"] + score_cols + ["AIDEOM score"]].rename(columns=score_names)

    st.dataframe(
        score_display.style.format({
            "Tăng trưởng": "{:.4f}",
            "Số hóa": "{:.4f}",
            "Đổi mới": "{:.4f}",
            "Bao trùm": "{:.4f}",
            "Chống chịu": "{:.4f}",
            "AIDEOM score": "{:.4f}",
        }),
        use_container_width=True,
    )

with score_chart_col:
    fig = go.Figure()

    for col in score_cols:
        fig.add_trace(
            go.Bar(
                x=kpi_df["Kịch bản"],
                y=kpi_df[col],
                name=score_names[col],
                marker_color=score_color[col],
                hovertemplate=(
                    "Kịch bản: %{x}<br>"
                    f"Thành phần: {score_names[col]}<br>"
                    "Điểm: %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Phân rã điểm AIDEOM",
        height=350,
        margin=dict(l=10, r=10, t=45, b=20),
        yaxis_title="Điểm thành phần",
        legend=dict(orientation="h", y=-0.22),
    )

    show_plot(fig)

# ======================================================
# SECTION 5: ALLOCATION AND WARNINGS
# ======================================================

st.markdown("---")
st.header("5. Phân bổ và cảnh báo rủi ro")

selected_scenario = st.selectbox(
    "Chọn kịch bản để xem chi tiết",
    kpi_df["Kịch bản"].tolist(),
    index=int(best_idx),
)

selected_row = kpi_df[kpi_df["Kịch bản"] == selected_scenario].iloc[0]

detail_col, radar_col = st.columns([1, 1])

with detail_col:
    st.markdown("#### Tóm tắt kịch bản")

    st.write(selected_row["Diễn giải"])

    detail_table = pd.DataFrame({
        "Chỉ tiêu": [
            "K",
            "D",
            "AI",
            "H",
            "GDP 2030",
            "Digital 2030",
            "AI readiness 2030",
            "Nhân lực số 2030",
            "NetJob",
            "Cyber risk",
            "Environmental risk",
            "Dependency risk",
            "Resilience",
        ],
        "Giá trị": [
            selected_row["K"],
            selected_row["D"],
            selected_row["AI"],
            selected_row["H"],
            selected_row["GDP 2030"],
            selected_row["Digital 2030"],
            selected_row["AI readiness 2030"],
            selected_row["Nhân lực số 2030"],
            selected_row["NetJob"],
            selected_row["Cyber risk"],
            selected_row["Environmental risk"],
            selected_row["Dependency risk"],
            selected_row["Resilience"],
        ],
    })

    st.dataframe(
        detail_table.style.format({"Giá trị": "{:,.3f}"}),
        use_container_width=True,
    )

with radar_col:
    st.markdown("#### Hồ sơ KPI của kịch bản")

    radar_labels = [
        "Tăng trưởng",
        "Số hóa",
        "AI",
        "Nhân lực",
        "Bao trùm",
        "Chống chịu",
    ]

    radar_values = [
        min(selected_row["Tăng trưởng TB"] / 0.10, 1),
        min(selected_row["Digital 2030"] / 40, 1),
        min(selected_row["AI readiness 2030"] / 110, 1),
        min(selected_row["Nhân lực số 2030"] / 45, 1),
        min(selected_row["Inclusion score"] / 100, 1),
        min(selected_row["Resilience"] / 100, 1),
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=selected_scenario,
            line=dict(color=BLUE),
            fillcolor="rgba(183,215,239,0.45)",
            hovertemplate="Chỉ tiêu: %{theta}<br>Điểm chuẩn hóa: %{r:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Radar KPI chuẩn hóa",
    )

    show_plot(fig)

warnings = []

if selected_row["Cyber risk"] >= 45:
    warnings.append("Rủi ro an ninh mạng cao do tỷ trọng AI/chuyển đổi số lớn nhưng nhân lực chưa đủ bù đắp.")

if selected_row["Environmental risk"] >= 40:
    warnings.append("Rủi ro môi trường đáng chú ý, đặc biệt nếu đầu tư hạ tầng và trung tâm dữ liệu tăng nhanh.")

if selected_row["DisplacedJob"] > selected_row["NewJob"]:
    warnings.append("Việc làm bị thay thế lớn hơn việc làm mới, cần tăng đào tạo lại hoặc giảm tốc độ tự động hóa.")

if selected_row["H"] < 0.18 and selected_row["AI"] > 0.30:
    warnings.append("Tỷ trọng AI cao nhưng nhân lực thấp, có nguy cơ thiếu năng lực hấp thụ công nghệ.")

if selected_row["D"] < 0.18 and selected_row["AI"] > 0.30:
    warnings.append("AI tăng nhanh khi nền tảng số chưa đủ mạnh, có nguy cơ triển khai phân mảnh.")

if not warnings:
    warnings.append("Không có cảnh báo lớn theo ngưỡng mô hình hiện tại.")

st.markdown("#### Cảnh báo chính sách")

for w in warnings:
    st.warning(w)

# ======================================================
# SECTION 6: SENSITIVITY
# ======================================================

st.markdown("---")
st.header("6. Kiểm tra độ nhạy của khuyến nghị")

section_caption(
    """
    Phần này kiểm tra khuyến nghị có ổn định khi thay đổi mức ưu tiên bao trùm và mức ưu tiên giảm rủi ro hay không.
    Nếu kịch bản tốt nhất thay đổi mạnh, cần trình bày kết quả như một lựa chọn chính sách có điều kiện thay vì một kết luận tuyệt đối.
    """
)

sens_rows = []
risk_grid = [max(0, risk_aversion - 0.20), risk_aversion, min(1, risk_aversion + 0.20)]
incl_grid = [max(0, inclusion_weight - 0.20), inclusion_weight, min(1, inclusion_weight + 0.20)]

for r in risk_grid:
    for inc in incl_grid:
        old_risk = risk_aversion
        old_inc = inclusion_weight

        # Local recompute with alternative weights.
        temp_scores = []
        for _, row in scenario_input.iterrows():
            K = row["K"]
            D = row["D"]
            AI = row["AI"]
            H = row["H"]

            growth_bonus = 0.015 * K + 0.030 * D + 0.040 * AI + 0.025 * H
            risk_penalty = r * (0.018 * AI + 0.010 * K - 0.012 * H - 0.006 * D)
            avg_growth = base_growth + growth_bonus - risk_penalty

            digital_2030 = digital_base + 8.0 * D + 3.0 * AI + 2.5 * H
            innovation_score = 100 * (0.45 * AI + 0.30 * D + 0.15 * H + 0.10 * K)

            cyber_risk = 100 * (0.35 * AI + 0.18 * D - 0.22 * H + 0.08 * K)
            env_risk = 100 * (0.38 * K + 0.20 * D + 0.26 * AI - 0.12 * H)
            dependency_risk = 100 * (0.25 * AI + 0.18 * D - 0.10 * H)

            cyber_risk = float(np.clip(cyber_risk, 0, 100))
            env_risk = float(np.clip(env_risk, 0, 100))
            dependency_risk = float(np.clip(dependency_risk, 0, 100))
            resilience = 100 - (0.35 * cyber_risk + 0.35 * env_risk + 0.30 * dependency_risk)
            resilience = float(np.clip(resilience, 0, 100))

            inclusion_score = 100 * (0.55 * H + 0.25 * D + 0.10 * AI + 0.10 * (1 - K))

            score = (
                0.30 * min(max(avg_growth / 0.10, 0), 1)
                + 0.20 * min(max(digital_2030 / 40, 0), 1)
                + 0.20 * min(max(innovation_score / 100, 0), 1)
                + inc * 0.20 * min(max(inclusion_score / 100, 0), 1)
                + 0.10 * min(max(resilience / 100, 0), 1)
            )

            temp_scores.append(score)

        temp_best_idx = int(np.argmax(temp_scores))
        sens_rows.append({
            "Ưu tiên giảm rủi ro": r,
            "Ưu tiên bao trùm": inc,
            "Kịch bản tốt nhất": scenario_input.iloc[temp_best_idx]["Kịch bản"],
            "AIDEOM score": temp_scores[temp_best_idx],
        })

sens_df = pd.DataFrame(sens_rows)

sens_col, sens_chart_col = st.columns([1, 1])

with sens_col:
    st.dataframe(
        sens_df.style.format({
            "Ưu tiên giảm rủi ro": "{:.2f}",
            "Ưu tiên bao trùm": "{:.2f}",
            "AIDEOM score": "{:.4f}",
        }),
        use_container_width=True,
    )

with sens_chart_col:
    best_counts = sens_df["Kịch bản tốt nhất"].value_counts().reset_index()
    best_counts.columns = ["Kịch bản", "Số lần đứng đầu"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=best_counts["Kịch bản"],
            y=best_counts["Số lần đứng đầu"],
            marker_color=SKY,
            marker_line=dict(color=BLUE, width=1),
            hovertemplate="Kịch bản: %{x}<br>Số lần đứng đầu: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Độ ổn định của kịch bản tốt nhất",
        height=320,
        margin=dict(l=10, r=10, t=45, b=20),
        yaxis_title="Số lần",
        xaxis=dict(tickangle=0),
    )

    show_plot(fig)

# ======================================================
# SECTION 7: POLICY INTERPRETATION
# ======================================================

st.markdown("---")
st.header("7. Diễn giải chính sách tích hợp")

top_gdp = kpi_df.sort_values("GDP 2030", ascending=False).iloc[0]["Kịch bản"]
top_netjob = kpi_df.sort_values("NetJob", ascending=False).iloc[0]["Kịch bản"]
top_resilience = kpi_df.sort_values("Resilience", ascending=False).iloc[0]["Kịch bản"]
lowest_risk = kpi_df.assign(
    avg_risk=kpi_df[["Cyber risk", "Environmental risk", "Dependency risk"]].mean(axis=1)
).sort_values("avg_risk").iloc[0]["Kịch bản"]

st.markdown("#### Kết quả tổng hợp")

st.write(
    f"""
    Kết quả mô phỏng cho thấy kịch bản có GDP 2030 cao nhất là {top_gdp}, trong khi kịch bản có NetJob cao nhất là {top_netjob}.
    Kịch bản có năng lực chống chịu cao nhất là {top_resilience}, còn kịch bản có mức rủi ro tổng hợp thấp nhất là {lowest_risk}.
    Điều này cho thấy không có một kịch bản duy nhất tối ưu tuyệt đối trên mọi mục tiêu.
    Chính sách số và AI luôn tồn tại đánh đổi giữa tăng trưởng, tốc độ đổi mới, việc làm, môi trường và an ninh dữ liệu.
    """
)

st.markdown("#### Hàm ý theo 5 kịch bản")

st.write(
    """
    Kịch bản truyền thống có ưu điểm là ít gây xáo trộn lao động và phù hợp với logic tăng trưởng dựa trên vốn vật chất, FDI và xuất khẩu.
    Tuy nhiên, nếu duy trì quá lâu, kịch bản này có thể làm Việt Nam chậm đạt các mục tiêu về kinh tế số, dữ liệu và năng lực AI.

    Kịch bản số hóa nhanh phù hợp khi mục tiêu trước mắt là nâng nền tảng chuyển đổi số quốc gia.
    Đây là kịch bản có ý nghĩa nền móng, nhất là với dịch vụ công trực tuyến, doanh nghiệp số, thanh toán số và dữ liệu dùng chung.

    Kịch bản AI dẫn dắt tạo tác động đổi mới và tăng trưởng nhanh, nhưng đi kèm rủi ro về an ninh dữ liệu, phụ thuộc công nghệ,
    phát thải từ hạ tầng tính toán và dịch chuyển lao động. Vì vậy, nếu chọn kịch bản này, chính sách phải kèm đầu tư nhân lực,
    tiêu chuẩn dữ liệu, quản trị rủi ro AI và an toàn mạng.

    Kịch bản bao trùm số phù hợp khi ưu tiên giảm khoảng cách số, bảo vệ lao động, hỗ trợ SME, nông nghiệp số và vùng yếu.
    Kịch bản này có thể không tạo GDP cao nhất trong ngắn hạn, nhưng giúp tăng tính bền vững xã hội của chuyển đổi số.

    Kịch bản tối ưu cân bằng là hướng dung hòa. Nó không nhất thiết đứng đầu ở từng chỉ tiêu riêng lẻ, nhưng thường có tính ổn định hơn
    vì kết hợp tăng trưởng, số hóa, AI và nhân lực. Đây là hướng phù hợp nếu Chính phủ muốn tránh cả hai cực đoan:
    đầu tư quá chậm làm lỡ cơ hội công nghệ, hoặc đầu tư quá nhanh làm tăng rủi ro xã hội.
    """
)

st.markdown("#### Liên hệ chính sách Việt Nam")

st.write(
    """
    Kết quả Bài 12 phù hợp với cách tiếp cận chính sách hiện nay của Việt Nam: chuyển đổi số, AI và đổi mới sáng tạo không nên được xem là các mục tiêu tách rời.
    Quyết định 749/QĐ-TTg nhấn mạnh chuyển đổi số quốc gia như nền tảng thay đổi phương thức quản trị và hoạt động kinh tế - xã hội.
    Quyết định 411/QĐ-TTg đặt trọng tâm vào phát triển kinh tế số và xã hội số.
    Quyết định 127/QĐ-TTg định hướng phát triển nghiên cứu, phát triển và ứng dụng AI đến năm 2030.
    Nghị quyết 57-NQ/TW tiếp tục nhấn mạnh khoa học công nghệ, đổi mới sáng tạo và chuyển đổi số như động lực phát triển.

    Từ góc nhìn AIDEOM-VN, điểm quan trọng là các chính sách này cần được phối hợp theo chuỗi:
    hạ tầng số và dữ liệu tạo nền tảng, AI tạo năng suất và đổi mới, nhân lực số tạo năng lực hấp thụ,
    còn mô hình rủi ro giúp kiểm soát tác động phụ về lao động, môi trường và an ninh dữ liệu.
    Nếu thiếu một trong các mắt xích này, hiệu quả của chính sách AI và chuyển đổi số sẽ bị hạn chế.
    """
)

st.markdown("#### Tính phù hợp với yêu cầu đồ án tích hợp")

st.write(
    f"""
    Về mặt kỹ thuật, dashboard đã tích hợp dữ liệu vĩ mô, dữ liệu ngành và dữ liệu vùng; có các mô hình mô phỏng tăng trưởng,
    chuyển đổi số, AI readiness, lao động, rủi ro và năng lực chống chịu. Về mặt ra quyết định, dashboard không chỉ đưa ra một kịch bản tốt nhất là {best_scenario},
    mà còn cho thấy vì sao kịch bản đó được chọn thông qua phân rã AIDEOM score, cảnh báo rủi ro và kiểm tra độ nhạy.
    """
)

st.write(
    """
    Điểm cần nhấn mạnh khi bảo vệ là AIDEOM-VN không thay thế quyết định chính sách. Hệ thống đóng vai trò như một decision support system:
    giúp minh bạch hóa giả định, lượng hóa đánh đổi, kiểm tra độ nhạy và tạo cơ sở thảo luận giữa mục tiêu tăng trưởng, số hóa,
    đổi mới sáng tạo, bao trùm xã hội và an toàn hệ thống.
    """
)

st.markdown("#### Khuyến nghị cuối cùng")

if best_scenario == "S3. AI dẫn dắt":
    recommendation = (
        "Mô hình đang nghiêng về AI dẫn dắt. Khuyến nghị là triển khai AI theo cụm ngành/vùng có năng lực hấp thụ tốt, "
        "nhưng phải tăng tỷ trọng nhân lực số, an ninh dữ liệu và tiêu chuẩn quản trị AI để giảm rủi ro."
    )
elif best_scenario == "S4. Bao trùm số":
    recommendation = (
        "Mô hình đang nghiêng về bao trùm số. Khuyến nghị là ưu tiên đào tạo lại, vùng yếu, SME và nông nghiệp số, "
        "đồng thời chọn một số dự án AI vừa sức để tránh chậm đổi mới."
    )
elif best_scenario == "S2. Số hóa nhanh":
    recommendation = (
        "Mô hình đang nghiêng về số hóa nhanh. Khuyến nghị là tập trung dữ liệu, dịch vụ công số, hạ tầng số và doanh nghiệp số "
        "trước khi mở rộng các ứng dụng AI phức tạp."
    )
elif best_scenario == "S1. Truyền thống":
    recommendation = (
        "Mô hình đang nghiêng về truyền thống. Khuyến nghị là cần thận trọng vì cách tiếp cận này ổn định ngắn hạn nhưng có thể làm chậm mục tiêu kinh tế số và AI."
    )
else:
    recommendation = (
        "Mô hình đang nghiêng về tối ưu cân bằng. Khuyến nghị là chọn lộ trình kết hợp: đầu tư đủ cho số hóa và AI, "
        "nhưng không giảm vai trò của nhân lực số và quản trị rủi ro."
    )

st.success(recommendation)

st.markdown(
    """
    <p style="
        font-size: 14px;
        color: #64748b;
        font-style: italic;
        line-height: 1.6;
        margin-top: 10px;
        margin-bottom: 8px;
    ">
    Lưu ý: Tất cả tham số, điểm số và khuyến nghị trong dashboard là kết quả mô phỏng phục vụ phân tích.
    Các kết quả này chỉ nên được dùng như cơ sở thảo luận, không phải kết luận thống kê chính thức hay khuyến nghị chính sách cuối cùng.
    </p>
    """,
    unsafe_allow_html=True,
)
