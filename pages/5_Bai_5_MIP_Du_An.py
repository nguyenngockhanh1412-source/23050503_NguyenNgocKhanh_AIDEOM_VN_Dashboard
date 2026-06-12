import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

setup_page("Bài 5 - MIP dự án")
render_sidebar("Bài 5 - MIP dự án")

st.title("Bài 5. Quy hoạch nguyên hỗn hợp lựa chọn dự án chuyển đổi số")

st.write(
    """
    Bài 5 sử dụng mô hình quy hoạch nguyên hỗn hợp để lựa chọn danh mục dự án chuyển đổi số.
    Điểm khác biệt của bài toán là mỗi dự án chỉ có hai trạng thái: được chọn hoặc không được chọn.
    Vì vậy, mô hình không chỉ xếp hạng từng dự án riêng lẻ, mà tìm tổ hợp dự án tối ưu trong điều kiện ngân sách,
    tiến độ giải ngân, ràng buộc tiên quyết và yêu cầu chiến lược.
    """
)

# ======================================================
# IMPORT SOLVER
# ======================================================

try:
    import pulp
except ImportError:
    st.error("Bạn chưa cài PuLP. Hãy chạy: .\\.venv\\Scripts\\python.exe -m pip install pulp")
    st.stop()

# ======================================================
# DATA
# ======================================================

project_data = pd.DataFrame({
    "Mã": [f"P{i}" for i in range(1, 16)],
    "Tên dự án": [
        "Trung tâm dữ liệu Hòa Lạc",
        "Trung tâm dữ liệu TP.HCM",
        "Cáp quang 5G vùng sâu",
        "Hệ thống định danh điện tử VNeID 2.0",
        "Cổng dịch vụ công quốc gia v3",
        "Y tế số quốc gia",
        "Giáo dục số K-12 toàn quốc",
        "Trung tâm AI quốc gia + supercomputing",
        "Sandbox tài chính số",
        "Logistics thông minh + cảng biển số",
        "Nông nghiệp số ĐBSCL",
        "Đào tạo 50.000 kỹ sư AI/bán dẫn",
        "Khu công nghiệp bán dẫn Bắc Ninh - Bắc Giang",
        "An ninh mạng quốc gia",
        "Open Data + dữ liệu mở quốc gia",
    ],
    "Lĩnh vực": [
        "Hạ tầng",
        "Hạ tầng",
        "Hạ tầng",
        "Chính phủ số",
        "Chính phủ số",
        "Y tế số",
        "Giáo dục",
        "AI",
        "Tài chính số",
        "Logistics",
        "Nông nghiệp",
        "Nhân lực",
        "Bán dẫn",
        "An ninh",
        "Dữ liệu",
    ],
    "Chi phí": [
        12000, 11500, 18000, 4500, 3200,
        5800, 6500, 15000, 2500, 7200,
        4800, 8500, 20000, 3800, 1500
    ],
    "Lợi ích NPV": [
        21500, 20800, 32500, 9200, 6800,
        11400, 12200, 28500, 5800, 13800,
        8500, 16200, 35000, 7500, 3800
    ],
    "Năm 1-2": [
        8500, 7500, 12000, 3500, 2500,
        4000, 4500, 9000, 1800, 5000,
        3500, 5500, 13000, 2800, 1200
    ],
})

project_data["Năm 3-5"] = project_data["Chi phí"] - project_data["Năm 1-2"]

def completion_probability(field):
    if field == "Hạ tầng":
        return 0.85
    if field == "Chính phủ số":
        return 0.75
    if field in ["AI", "Bán dẫn"]:
        return 0.65
    return 0.80

project_data["Xác suất đúng tiến độ"] = project_data["Lĩnh vực"].apply(completion_probability)
project_data["Lợi ích kỳ vọng"] = project_data["Lợi ích NPV"] * project_data["Xác suất đúng tiến độ"]
project_data["NPV/Chi phí"] = project_data["Lợi ích NPV"] / project_data["Chi phí"]
project_data["Chiến lược"] = project_data["Mã"].isin(["P8", "P12", "P13", "P14"]).astype(int)

field_colors = {
    "Hạ tầng": "#0B1D33",
    "Chính phủ số": "#1FA7B6",
    "Y tế số": "#81D8D0",
    "Giáo dục": "#E6F7F5",
    "AI": "#FF6B6B",
    "Tài chính số": "#F1FBFA",
    "Logistics": "#7FD3C6",
    "Nông nghiệp": "#DDF8F5",
    "Nhân lực": "#C7D2DC",
    "Bán dẫn": "#4FC7BE",
    "An ninh": "#BFEFEB",
    "Dữ liệu": "#F8FCFC",
}

# ======================================================
# SOLVER
# ======================================================

def solve_mip(
    df,
    total_budget=80000,
    early_budget=40000,
    min_projects=7,
    max_projects=11,
    force_p14=True,
    keep_center_exclusion=True,
    require_p1_p2=False,
    objective_mode="NPV gốc",
):
    model = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)

    codes = df["Mã"].tolist()
    y = pulp.LpVariable.dicts("y", codes, lowBound=0, upBound=1, cat="Binary")

    if objective_mode == "Lợi ích kỳ vọng có rủi ro":
        objective_col = "Lợi ích kỳ vọng"
    else:
        objective_col = "Lợi ích NPV"

    model += pulp.lpSum(
        df.loc[i, objective_col] * y[df.loc[i, "Mã"]]
        for i in df.index
    )

    model += pulp.lpSum(
        df.loc[i, "Chi phí"] * y[df.loc[i, "Mã"]]
        for i in df.index
    ) <= total_budget

    model += pulp.lpSum(
        df.loc[i, "Năm 1-2"] * y[df.loc[i, "Mã"]]
        for i in df.index
    ) <= early_budget

    if keep_center_exclusion:
        model += y["P1"] + y["P2"] <= 1

    model += y["P8"] <= y["P12"]
    model += y["P13"] <= y["P12"]
    model += y["P4"] + y["P5"] >= 1

    if force_p14:
        model += y["P14"] >= 1

    model += pulp.lpSum(y[p] for p in codes) >= min_projects
    model += pulp.lpSum(y[p] for p in codes) <= max_projects

    if require_p1_p2:
        model += y["P1"] == 1
        model += y["P2"] == 1

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return None, status

    result = df.copy()
    result["Chọn"] = result["Mã"].apply(lambda p: int(round(y[p].value())))
    result["Trạng thái"] = result["Chọn"].map({1: "Được chọn", 0: "Không chọn"})

    selected = result[result["Chọn"] == 1].copy()
    rejected = result[result["Chọn"] == 0].copy()

    total_cost = selected["Chi phí"].sum()
    total_early = selected["Năm 1-2"].sum()
    total_benefit = selected["Lợi ích NPV"].sum()
    total_expected = selected["Lợi ích kỳ vọng"].sum()

    summary = {
        "status": status,
        "objective_value": pulp.value(model.objective),
        "objective_mode": objective_mode,
        "total_cost": total_cost,
        "total_early": total_early,
        "total_late": selected["Năm 3-5"].sum(),
        "total_benefit": total_benefit,
        "total_expected": total_expected,
        "npv_ratio": total_benefit / total_cost if total_cost > 0 else np.nan,
        "expected_ratio": total_expected / total_cost if total_cost > 0 else np.nan,
        "num_projects": len(selected),
        "strategic_count": int(selected["Chiến lược"].sum()),
        "selected_fields": selected["Lĩnh vực"].nunique(),
        "p15_selected": int("P15" in selected["Mã"].tolist()),
        "p14_selected": int("P14" in selected["Mã"].tolist()),
        "p8_selected": int("P8" in selected["Mã"].tolist()),
        "p13_selected": int("P13" in selected["Mã"].tolist()),
    }

    return {
        "result": result,
        "selected": selected,
        "rejected": rejected,
        "summary": summary,
    }, status


def selected_codes(solution):
    if solution is None:
        return []
    return solution["selected"]["Mã"].tolist()


def compare_sets(base_codes, other_codes):
    added = [p for p in other_codes if p not in base_codes]
    removed = [p for p in base_codes if p not in other_codes]
    return added, removed


# ======================================================
# SECTION 1: MODEL
# ======================================================

st.markdown("---")
st.header("1. Mô hình toán học")

st.latex(r"""
\max Z = \sum_i B_i y_i,\quad y_i \in \{0,1\}
""")

st.write(
    """
    Trong mô hình, yᵢ là biến nhị phân: yᵢ = 1 nếu dự án i được chọn và yᵢ = 0 nếu dự án không được chọn.
    Bᵢ là lợi ích NPV kỳ vọng của dự án. Bài toán tối đa hóa tổng lợi ích của danh mục, nhưng phải tuân thủ ngân sách 5 năm,
    ngân sách giai đoạn 1–2, ràng buộc loại trừ, ràng buộc tiên quyết và yêu cầu số lượng dự án.
    """
)

st.write(
    """
    Đây là dạng knapsack tổng quát hóa. Điểm quan trọng của MIP là một dự án có tỷ suất lợi ích/chi phí cao chưa chắc được chọn,
    vì mô hình lựa chọn cả danh mục dưới nhiều ràng buộc đồng thời.
    """
)

source_note(
    """
    Dữ liệu 15 dự án, ràng buộc ngân sách và ràng buộc logic được lấy theo yêu cầu Bài 5 trong bộ đề.
    Các kết quả là mô phỏng phục vụ phân tích, không thay thế quyết định đầu tư chính thức.
    """
)

# ======================================================
# SECTION 2: SETTINGS
# ======================================================

st.markdown("---")
st.header("2. Thiết lập kịch bản và ràng buộc")

scenario = st.selectbox(
    "Chọn kịch bản phân tích",
    [
        "Theo đề bài",
        "Nới ngân sách lên 100.000 tỷ",
        "Kiểm tra yêu cầu có cả P1 và P2",
        "Bỏ bắt buộc P14",
        "Tối đa hóa lợi ích kỳ vọng có rủi ro",
    ],
)

if scenario == "Theo đề bài":
    default_budget = 80000
    default_early = 40000
    default_min = 7
    default_max = 11
    default_force_p14 = True
    default_exclusion = True
    default_p1p2 = False
    default_objective = "NPV gốc"

elif scenario == "Nới ngân sách lên 100.000 tỷ":
    default_budget = 100000
    default_early = 40000
    default_min = 7
    default_max = 11
    default_force_p14 = True
    default_exclusion = True
    default_p1p2 = False
    default_objective = "NPV gốc"

elif scenario == "Kiểm tra yêu cầu có cả P1 và P2":
    default_budget = 80000
    default_early = 40000
    default_min = 7
    default_max = 11
    default_force_p14 = True
    default_exclusion = True
    default_p1p2 = True
    default_objective = "NPV gốc"

elif scenario == "Bỏ bắt buộc P14":
    default_budget = 80000
    default_early = 40000
    default_min = 7
    default_max = 11
    default_force_p14 = False
    default_exclusion = True
    default_p1p2 = False
    default_objective = "NPV gốc"

else:
    default_budget = 80000
    default_early = 40000
    default_min = 7
    default_max = 11
    default_force_p14 = True
    default_exclusion = True
    default_p1p2 = False
    default_objective = "Lợi ích kỳ vọng có rủi ro"

control_col, rule_col = st.columns([1.2, 0.8])

with control_col:
    c1, c2 = st.columns(2)

    with c1:
        total_budget = st.number_input(
            "Ngân sách tổng 5 năm",
            min_value=30000,
            max_value=120000,
            value=default_budget,
            step=5000,
        )

        early_budget = st.number_input(
            "Ngân sách năm 1–2",
            min_value=10000,
            max_value=80000,
            value=default_early,
            step=2500,
        )

    with c2:
        min_projects = st.number_input(
            "Số dự án tối thiểu",
            min_value=1,
            max_value=15,
            value=default_min,
            step=1,
        )

        max_projects = st.number_input(
            "Số dự án tối đa",
            min_value=1,
            max_value=15,
            value=default_max,
            step=1,
        )

    objective_mode = st.selectbox(
        "Hàm mục tiêu",
        ["NPV gốc", "Lợi ích kỳ vọng có rủi ro"],
        index=0 if default_objective == "NPV gốc" else 1,
    )

with rule_col:
    st.markdown("#### Ràng buộc logic")

    keep_center_exclusion = st.checkbox(
        "Loại trừ P1 và P2: chỉ chọn một trung tâm dữ liệu",
        value=default_exclusion,
    )

    require_p1_p2 = st.checkbox(
        "Yêu cầu có cả P1 và P2 để kiểm tra redundancy",
        value=default_p1p2,
    )

    force_p14 = st.checkbox(
        "Bắt buộc P14 an ninh mạng",
        value=default_force_p14,
    )

    st.write("P8 và P13 chỉ được chọn nếu P12 được chọn.")
    st.write("P4 hoặc P5 phải có ít nhất một dự án chính phủ số.")

budget_pressure = total_budget / 80000
early_pressure = early_budget / 40000

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card("Ngân sách tổng", f"{total_budget:,.0f}", "Tỷ VND cho toàn bộ giai đoạn 2026–2030.")

with k2:
    kpi_card("Ngân sách năm 1–2", f"{early_budget:,.0f}", "Tỷ VND giới hạn cho giai đoạn đầu.")

with k3:
    kpi_card("Số dự án yêu cầu", f"{min_projects}–{max_projects}", "Khoảng số lượng dự án được phép chọn.")

with k4:
    kpi_card("Hàm mục tiêu", objective_mode, "NPV gốc hoặc lợi ích kỳ vọng sau rủi ro.")

if require_p1_p2 and keep_center_exclusion:
    st.warning(
        """
        Bạn đang bật đồng thời yêu cầu có cả P1 và P2, trong khi vẫn giữ ràng buộc loại trừ P1 + P2 ≤ 1.
        Trường hợp này dùng để kiểm tra câu hỏi 5.4.3 và thường sẽ không khả thi.
        """
    )

# ======================================================
# SECTION 3: INPUT DATA
# ======================================================

st.markdown("---")
st.header("3. Dữ liệu 15 dự án")

section_caption(
    """
    Bảng dữ liệu gồm chi phí toàn kỳ, lợi ích NPV, ngân sách năm 1–2, ngân sách năm 3–5, xác suất hoàn thành đúng tiến độ
    và lợi ích kỳ vọng sau điều chỉnh rủi ro.
    """
)

data_col, eff_col = st.columns([1.15, 0.85])

with data_col:
    st.dataframe(
        project_data.style.format({
            "Chi phí": "{:,.0f}",
            "Lợi ích NPV": "{:,.0f}",
            "Năm 1-2": "{:,.0f}",
            "Năm 3-5": "{:,.0f}",
            "Xác suất đúng tiến độ": "{:.2f}",
            "Lợi ích kỳ vọng": "{:,.0f}",
            "NPV/Chi phí": "{:.2f}",
        }),
        use_container_width=True,
    )

with eff_col:
    eff_plot = project_data.sort_values("NPV/Chi phí", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=eff_plot["Mã"],
            x=eff_plot["NPV/Chi phí"],
            orientation="h",
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate=(
                "Dự án: %{y}<br>"
                "NPV/Chi phí: %{x:.2f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Tỷ suất NPV/chi phí",
        height=430,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="NPV/Chi phí",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 4: SOLVE CURRENT SCENARIO
# ======================================================

st.markdown("---")
st.header("4. Kết quả tối ưu của kịch bản đang chọn")

solution, status = solve_mip(
    project_data,
    total_budget=total_budget,
    early_budget=early_budget,
    min_projects=min_projects,
    max_projects=max_projects,
    force_p14=force_p14,
    keep_center_exclusion=keep_center_exclusion,
    require_p1_p2=require_p1_p2,
    objective_mode=objective_mode,
)

if solution is None:
    st.error(f"Mô hình không tìm được nghiệm tối ưu. Trạng thái: {status}")

    st.write(
        """
        Trường hợp không khả thi thường xảy ra khi các ràng buộc logic mâu thuẫn với nhau.
        Ví dụ, nếu vừa bắt buộc chọn cả P1 và P2, vừa giữ ràng buộc loại trừ P1 + P2 ≤ 1,
        mô hình không thể thỏa mãn đồng thời hai điều kiện này.
        """
    )

else:
    result = solution["result"]
    selected = solution["selected"]
    rejected = solution["rejected"]
    summary = solution["summary"]

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        kpi_card("Số dự án được chọn", f"{summary['num_projects']}", "Số dự án trong danh mục tối ưu.")

    with m2:
        kpi_card("Tổng chi phí", f"{summary['total_cost']:,.0f}", "Tỷ VND dùng trong toàn kỳ.")

    with m3:
        kpi_card("Tổng lợi ích NPV", f"{summary['total_benefit']:,.0f}", "Tỷ VND lợi ích của danh mục.")

    with m4:
        kpi_card("NPV biên", f"{summary['npv_ratio']:.2f}", "Tổng NPV chia cho tổng chi phí.")

    table_col, chart_col = st.columns([1.05, 0.95])

    with table_col:
        st.markdown("#### Danh mục dự án được chọn")

        selected_display = selected[[
            "Mã",
            "Tên dự án",
            "Lĩnh vực",
            "Chi phí",
            "Lợi ích NPV",
            "Năm 1-2",
            "Năm 3-5",
            "NPV/Chi phí",
        ]].copy()

        st.dataframe(
            selected_display.style.format({
                "Chi phí": "{:,.0f}",
                "Lợi ích NPV": "{:,.0f}",
                "Năm 1-2": "{:,.0f}",
                "Năm 3-5": "{:,.0f}",
                "NPV/Chi phí": "{:.2f}",
            }),
            use_container_width=True,
        )

    with chart_col:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=selected["Mã"],
                y=selected["Lợi ích NPV"],
                name="Lợi ích NPV",
                marker_color="#1FA7B6",
                marker_line=dict(color="#1FA7B6", width=1),
                hovertemplate=(
                    "Dự án: %{x}<br>"
                    "Lợi ích NPV: %{y:,.0f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Lợi ích NPV của các dự án được chọn",
            height=360,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Dự án",
            yaxis_title="Tỷ VND",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Kiểm tra sử dụng nguồn lực")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        resource_df = pd.DataFrame({
            "Ràng buộc": [
                "Ngân sách tổng 5 năm",
                "Ngân sách năm 1–2",
                "Số dự án tối thiểu",
                "Số dự án tối đa",
            ],
            "Giá trị sử dụng": [
                summary["total_cost"],
                summary["total_early"],
                summary["num_projects"],
                summary["num_projects"],
            ],
            "Ngưỡng": [
                total_budget,
                early_budget,
                min_projects,
                max_projects,
            ],
        })

        resource_df["Tỷ lệ sử dụng (%)"] = resource_df["Giá trị sử dụng"] / resource_df["Ngưỡng"] * 100

        st.dataframe(
            resource_df.style.format({
                "Giá trị sử dụng": "{:,.0f}",
                "Ngưỡng": "{:,.0f}",
                "Tỷ lệ sử dụng (%)": "{:.1f}",
            }),
            use_container_width=True,
        )

    with res_col2:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Tổng 5 năm", "Năm 1–2"],
                y=[summary["total_cost"], summary["total_early"]],
                name="Đã dùng",
                marker_color="#1FA7B6",
                hovertemplate="Nguồn lực: %{x}<br>Đã dùng: %{y:,.0f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Bar(
                x=["Tổng 5 năm", "Năm 1–2"],
                y=[total_budget - summary["total_cost"], early_budget - summary["total_early"]],
                name="Còn dư",
                marker_color="#E6F7F5",
                hovertemplate="Nguồn lực: %{x}<br>Còn dư: %{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Cơ cấu sử dụng ngân sách",
            height=320,
            margin=dict(l=10, r=10, t=45, b=20),
            barmode="stack",
            yaxis_title="Tỷ VND",
            legend=dict(orientation="h", y=-0.20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Trực quan logic chọn/không chọn")

    fig = go.Figure()

    for state, color in [("Không chọn", "#81D8D0"), ("Được chọn", "#1FA7B6")]:
        temp = result[result["Trạng thái"] == state]
        fig.add_trace(
            go.Scatter(
                x=temp["Chi phí"],
                y=temp["Lợi ích NPV"],
                mode="markers+text",
                text=temp["Mã"],
                textposition="top center",
                name=state,
                marker=dict(
                    size=temp["Năm 1-2"] / 300,
                    color=color,
                    line=dict(color="#334155", width=0.5),
                    opacity=0.85,
                ),
                hovertemplate=(
                    "Dự án: %{text}<br>"
                    "Chi phí: %{x:,.0f}<br>"
                    "Lợi ích NPV: %{y:,.0f}<br>"
                    "Quy mô điểm: ngân sách năm 1–2<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Ma trận chi phí - lợi ích",
        height=390,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Chi phí",
        yaxis_title="Lợi ích NPV",
        legend=dict(orientation="h", y=-0.20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Tất cả dự án và trạng thái lựa chọn")

    compact_table = result[[
        "Mã",
        "Tên dự án",
        "Lĩnh vực",
        "Chi phí",
        "Lợi ích NPV",
        "Năm 1-2",
        "Xác suất đúng tiến độ",
        "Lợi ích kỳ vọng",
        "Trạng thái",
    ]].copy()

    st.dataframe(
        compact_table.style.format({
            "Chi phí": "{:,.0f}",
            "Lợi ích NPV": "{:,.0f}",
            "Năm 1-2": "{:,.0f}",
            "Xác suất đúng tiến độ": "{:.2f}",
            "Lợi ích kỳ vọng": "{:,.0f}",
        }),
        use_container_width=True,
    )

# ======================================================
# SECTION 5: SCENARIO COMPARISON
# ======================================================

st.markdown("---")
st.header("5. So sánh các kịch bản bắt buộc trong đề")

baseline, baseline_status = solve_mip(project_data)
budget_100, budget_100_status = solve_mip(project_data, total_budget=100000)
both_with_exclusion, both_with_exclusion_status = solve_mip(
    project_data,
    require_p1_p2=True,
    keep_center_exclusion=True,
)
both_without_exclusion, both_without_exclusion_status = solve_mip(
    project_data,
    require_p1_p2=True,
    keep_center_exclusion=False,
)
without_p14, without_p14_status = solve_mip(
    project_data,
    force_p14=False,
)
risk_solution, risk_status = solve_mip(
    project_data,
    objective_mode="Lợi ích kỳ vọng có rủi ro",
)

scenario_results = []

scenario_objects = {
    "Theo đề bài": (baseline, baseline_status),
    "Ngân sách 100.000 tỷ": (budget_100, budget_100_status),
    "Bắt buộc P1 và P2, vẫn giữ loại trừ": (both_with_exclusion, both_with_exclusion_status),
    "Bắt buộc P1 và P2, bỏ loại trừ": (both_without_exclusion, both_without_exclusion_status),
    "Bỏ bắt buộc P14": (without_p14, without_p14_status),
    "Tối đa hóa lợi ích kỳ vọng": (risk_solution, risk_status),
}

base_codes = selected_codes(baseline)

for name, (sol, stat) in scenario_objects.items():
    if sol is None:
        scenario_results.append({
            "Kịch bản": name,
            "Trạng thái": stat,
            "Z*": np.nan,
            "Tổng chi phí": np.nan,
            "Số dự án": np.nan,
            "Dự án được chọn": "Không khả thi",
            "Thay đổi so với gốc": "Không so sánh",
        })
    else:
        codes = selected_codes(sol)
        added, removed = compare_sets(base_codes, codes)

        if name == "Theo đề bài":
            change_text = "Kịch bản gốc"
        else:
            add_text = ", ".join(added) if len(added) > 0 else "không thêm"
            remove_text = ", ".join(removed) if len(removed) > 0 else "không bớt"
            change_text = f"Thêm: {add_text}; Bớt: {remove_text}"

        scenario_results.append({
            "Kịch bản": name,
            "Trạng thái": stat,
            "Z*": sol["summary"]["objective_value"],
            "Tổng chi phí": sol["summary"]["total_cost"],
            "Số dự án": sol["summary"]["num_projects"],
            "Dự án được chọn": ", ".join(codes),
            "Thay đổi so với gốc": change_text,
        })

scenario_df = pd.DataFrame(scenario_results)

comp_col, comp_chart_col = st.columns([1.15, 0.85])

with comp_col:
    st.dataframe(
        scenario_df.style.format({
            "Z*": "{:,.0f}",
            "Tổng chi phí": "{:,.0f}",
            "Số dự án": "{:.0f}",
        }),
        use_container_width=True,
    )

with comp_chart_col:
    plot_df = scenario_df.dropna(subset=["Z*"])

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df["Kịch bản"],
            y=plot_df["Z*"],
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Kịch bản: %{x}<br>Z*: %{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="So sánh Z* giữa các kịch bản",
        height=390,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=20),
        yaxis_title="Z*",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 6: POLICY DISCUSSION
# ======================================================

st.markdown("---")
st.header("6. Diễn giải và thảo luận chính sách")

if baseline is not None:
    base_summary = baseline["summary"]
    base_selected = baseline["selected"]
    base_rejected = baseline["rejected"]

    p15_in_base = "P15" in base_selected["Mã"].tolist()

    if p15_in_base:
        p15_comment = (
            "Trong nghiệm theo bộ tham số hiện tại, P15 được chọn. Điều này cho thấy dự án Open Data có tỷ suất NPV/chi phí cao và chi phí nhỏ, "
            "nên có thể đi vào danh mục khi các ràng buộc còn đủ không gian. Tuy nhiên, câu hỏi của đề vẫn có ý nghĩa: nếu P15 bị loại trong một kịch bản khác, "
            "nguyên nhân không phải vì dự án kém hiệu quả đơn lẻ, mà vì MIP tối ưu toàn bộ danh mục dưới ràng buộc ngân sách, ngân sách năm 1–2, số dự án và các điều kiện tiên quyết."
        )
    else:
        p15_comment = (
            "P15 có tỷ suất NPV/chi phí cao nhưng không được chọn trong nghiệm hiện tại. Lý do là MIP không tối đa hóa tỷ suất đơn lẻ, "
            "mà tối đa hóa tổng lợi ích của cả danh mục. Một dự án nhỏ, hiệu quả/chi phí cao vẫn có thể bị loại nếu nó không giúp tăng Z* bằng các tổ hợp dự án khác, "
            "hoặc nếu danh mục đã bị giới hạn bởi số lượng dự án, ngân sách năm 1–2 hoặc các ràng buộc chiến lược."
        )

    st.markdown("#### Vì sao P15 có thể bị bỏ qua dù tỷ suất lợi ích/chi phí cao?")

    st.write(p15_comment)

    st.write(
        """
        Về mặt chính sách, đây không nhất thiết là kết quả xấu. Open Data là dự án nền tảng, có chi phí thấp và hiệu quả lan tỏa dài hạn,
        nhưng mô hình MIP trong đề đang đo lợi ích bằng NPV trực tiếp. Nếu nhà hoạch định chính sách muốn bảo vệ các dự án nền tảng dữ liệu mở,
        có thể thêm ràng buộc bắt buộc chọn P15, hoặc bổ sung hệ số cộng hưởng cho những dự án phụ thuộc vào dữ liệu mở.
        """
    )

    st.markdown("#### Ràng buộc bắt buộc P14 có làm giảm Z* không?")

    if without_p14 is not None:
        p14_cost = without_p14["summary"]["objective_value"] - baseline["summary"]["objective_value"]

        if p14_cost > 0:
            p14_text = (
                f"Khi bỏ ràng buộc bắt buộc P14, Z* tăng thêm khoảng {p14_cost:,.0f}. "
                "Điều này cho thấy yêu cầu bắt buộc an ninh mạng có chi phí cơ hội về mặt tối đa hóa NPV."
            )
        elif abs(p14_cost) < 1e-6:
            p14_text = (
                "Khi bỏ ràng buộc bắt buộc P14, Z* gần như không đổi. Điều này cho thấy P14 vẫn được mô hình lựa chọn hoặc không tạo ra chi phí cơ hội đáng kể trong bộ tham số hiện tại."
            )
        else:
            p14_text = (
                "Kết quả so sánh cho thấy nghiệm không bắt buộc P14 thấp hơn nghiệm gốc, điều này có thể do ràng buộc khác khiến cấu trúc danh mục thay đổi."
            )

        st.write(p14_text)

    st.write(
        """
        Về mặt chính sách, bắt buộc P14 là hợp lý dù có thể làm giảm Z*. An ninh mạng là điều kiện nền cho chuyển đổi số quốc gia.
        Nếu thiếu năng lực bảo vệ hệ thống, các dự án dữ liệu, AI, y tế số, chính phủ số hoặc hạ tầng số đều có thể phát sinh rủi ro an toàn thông tin.
        Vì vậy, P14 nên được hiểu như một ràng buộc an toàn hệ thống, không chỉ là một dự án có NPV riêng.
        """
    )

    st.markdown("#### Mô hình hóa hiệu ứng cộng hưởng giữa P8 và P13 như thế nào?")

    st.write(
        """
        Mô hình hiện tại giả định lợi ích của các dự án là độc lập cộng gộp. Trong thực tế, P8 về trung tâm AI quốc gia và P13 về khu công nghiệp bán dẫn
        có thể tạo lợi ích cộng hưởng: hạ tầng tính toán AI làm tăng nhu cầu chip, còn năng lực bán dẫn giúp chủ động hơn về phần cứng cho AI.
        """
    )

    st.write(
        """
        Có thể mô hình hóa cộng hưởng bằng cách thêm một biến nhị phân mới, ví dụ z_8_13, với các ràng buộc z_8_13 ≤ y_8, z_8_13 ≤ y_13,
        và z_8_13 ≥ y_8 + y_13 − 1. Sau đó, hàm mục tiêu được cộng thêm một khoản synergy · z_8_13.
        Khi cả P8 và P13 cùng được chọn, z_8_13 = 1 và mô hình ghi nhận phần lợi ích cộng hưởng; nếu chỉ một trong hai dự án được chọn,
        khoản cộng hưởng bằng 0.
        """
    )

    st.markdown("#### Nới ngân sách lên 100.000 tỷ làm thay đổi gì?")

    if budget_100 is not None:
        added_100, removed_100 = compare_sets(base_codes, selected_codes(budget_100))
        add_text = ", ".join(added_100) if added_100 else "không thêm dự án nào"
        remove_text = ", ".join(removed_100) if removed_100 else "không loại dự án nào khỏi danh mục gốc"

        st.write(
            f"""
            Khi ngân sách tăng lên 100.000 tỷ, danh mục thay đổi như sau: thêm {add_text}; bớt {remove_text}.
            Nếu Z* không tăng nhiều, điều đó có nghĩa ràng buộc thực sự không nằm ở ngân sách tổng 5 năm,
            mà có thể nằm ở ngân sách năm 1–2, ràng buộc tiên quyết hoặc giới hạn số lượng dự án.
            """
        )

    st.markdown("#### Kiểm tra yêu cầu có cả P1 và P2")

    if both_with_exclusion is None:
        st.write(
            """
            Khi Quốc hội yêu cầu có cả P1 và P2 nhưng mô hình vẫn giữ ràng buộc loại trừ P1 + P2 ≤ 1, bài toán không khả thi.
            Đây là kết quả hợp lý về mặt logic: không thể vừa yêu cầu chọn cả hai trung tâm dữ liệu, vừa quy định chỉ được chọn một trong hai.
            """
        )

    if both_without_exclusion is not None:
        diff_both = both_without_exclusion["summary"]["objective_value"] - baseline["summary"]["objective_value"]
        st.write(
            f"""
            Nếu bỏ ràng buộc loại trừ để cho phép redundancy, mô hình có thể khả thi và Z* thay đổi khoảng {diff_both:,.0f}
            so với kịch bản gốc. Điều này minh họa rằng redundancy có thể làm tăng khả năng chống chịu hạ tầng số,
            nhưng cũng làm thay đổi cấu trúc danh mục và tạo chi phí cơ hội so với mục tiêu tối đa hóa NPV.
            """
        )

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
    Lưu ý: Kết quả của Bài 5 là mô phỏng phục vụ phân tích. Mô hình MIP giúp lượng hóa đánh đổi giữa lợi ích, chi phí,
    tiến độ ngân sách, điều kiện tiên quyết và yêu cầu chiến lược, nhưng không thay thế quy trình thẩm định đầu tư công thực tế.
    </p>
    """,
    unsafe_allow_html=True,
)