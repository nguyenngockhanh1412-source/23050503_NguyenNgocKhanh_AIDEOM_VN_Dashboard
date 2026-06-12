
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

try:
    import pulp
except ImportError:
    st.error("Bạn chưa cài PuLP. Hãy chạy trong Terminal: python -m pip install pulp")
    st.stop()


# ======================================================
# PAGE SETUP
# ======================================================

setup_page("Bài 9 - AI và lao động")
render_sidebar("Bài 9 - AI và lao động")

st.title("Bài 9. Tác động AI tới thị trường lao động Việt Nam")

st.write(
    """
    Dashboard mô phỏng tác động của AI và tự động hóa tới việc làm theo ngành.
    Mục tiêu là phân bổ ngân sách giữa đầu tư AI và đào tạo lại lao động sao cho NetJob toàn nền kinh tế dương,
    đồng thời kiểm tra khả năng bảo vệ các nhóm ngành dễ bị tổn thương.
    """
)

source_note(
    """
    Dữ liệu 8 ngành được lấy từ vietnam_sectors_2024.csv theo yêu cầu Bài 9. Các tham số a1, a2, b1, c1, d1
    được dùng để mô phỏng việc làm mới, việc làm nâng cấp, việc làm bị thay thế và năng lực đào tạo lại.
    Kết quả là mô phỏng phục vụ phân tích, không thay thế dự báo chính thức về thị trường lao động Việt Nam.
    """
)

# ======================================================
# LOAD DATA
# ======================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "vietnam_sectors_2024.csv"

if not DATA_FILE.exists():
    st.error(f"Không tìm thấy file dữ liệu: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)
if "include_labor_model" in df.columns:
    df = df[df["include_labor_model"] == 1].copy()

required_cols = [
    "sector", "labor", "risk", "a1", "a2", "b1", "c1", "d1", "vulnerable"
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error("File vietnam_sectors_2024.csv đang thiếu các cột sau:")
    st.write(missing_cols)
    st.stop()

sectors = df["sector"].tolist()

# ======================================================
# FUNCTIONS
# ======================================================

def solve_labor_model(
    data,
    budget=30000,
    min_total_netjob=0,
    training_capacity=18000,
    vulnerable_min_training=0.25,
    max_displacement_share=None
):
    model = pulp.LpProblem("AI_Labor_Market_Optimization", pulp.LpMaximize)

    x_ai = pulp.LpVariable.dicts(
        "x_AI",
        sectors,
        lowBound=0,
        cat="Continuous"
    )

    x_h = pulp.LpVariable.dicts(
        "x_H",
        sectors,
        lowBound=0,
        cat="Continuous"
    )

    new_job = {}
    upgrade_job = {}
    displaced_job = {}
    net_job = {}

    for _, row in data.iterrows():
        s = row["sector"]

        new_job[s] = row["a1"] * x_ai[s] + row["a2"] * x_h[s]
        upgrade_job[s] = row["b1"] * x_h[s]
        displaced_job[s] = row["c1"] * x_ai[s] * row["risk"] / 100
        net_job[s] = new_job[s] + upgrade_job[s] - displaced_job[s]

    model += pulp.lpSum(net_job[s] for s in sectors)

    model += pulp.lpSum(x_ai[s] + x_h[s] for s in sectors) <= budget
    model += pulp.lpSum(x_h[s] for s in sectors) <= training_capacity

    for s in sectors:
        model += net_job[s] >= 0

    model += pulp.lpSum(net_job[s] for s in sectors) >= min_total_netjob

    for _, row in data.iterrows():
        s = row["sector"]
        model += displaced_job[s] <= row["d1"] * x_h[s]

    vulnerable_sectors = data[data["vulnerable"] == 1]["sector"].tolist()

    if vulnerable_sectors:
        model += (
            pulp.lpSum(x_h[s] for s in vulnerable_sectors)
            >= vulnerable_min_training * pulp.lpSum(x_h[s] for s in sectors)
        )

    if max_displacement_share is not None:
        for _, row in data.iterrows():
            s = row["sector"]
            model += displaced_job[s] <= max_displacement_share * row["labor"] * 1_000_000

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return None, status

    rows = []

    for _, row in data.iterrows():
        s = row["sector"]
        ai_val = x_ai[s].value()
        h_val = x_h[s].value()

        new_val = pulp.value(new_job[s])
        up_val = pulp.value(upgrade_job[s])
        disp_val = pulp.value(displaced_job[s])
        net_val = pulp.value(net_job[s])

        rows.append({
            "sector": s,
            "labor": row["labor"],
            "risk": row["risk"],
            "x_AI": ai_val,
            "x_H": h_val,
            "NewJob": new_val,
            "UpgradeJob": up_val,
            "DisplacedJob": disp_val,
            "NetJob": net_val,
            "displaced_share_labor": disp_val / (row["labor"] * 1_000_000),
            "vulnerable": row["vulnerable"],
        })

    result = pd.DataFrame(rows)

    summary = {
        "total_budget_used": result["x_AI"].sum() + result["x_H"].sum(),
        "total_ai": result["x_AI"].sum(),
        "total_training": result["x_H"].sum(),
        "total_new": result["NewJob"].sum(),
        "total_upgrade": result["UpgradeJob"].sum(),
        "total_displaced": result["DisplacedJob"].sum(),
        "total_net": result["NetJob"].sum(),
        "training_capacity_used": result["x_H"].sum() / training_capacity if training_capacity > 0 else np.nan,
    }

    return {
        "result": result,
        "summary": summary
    }, status


def find_min_training_for_sector_2(data, budget=30000):
    sector_name = "CN chế biến chế tạo"

    values = np.arange(0, 12001, 250)
    feasible_rows = []

    for min_h in values:
        model = pulp.LpProblem("Min_Training_Sector_2", pulp.LpMaximize)

        x_ai = pulp.LpVariable.dicts("x_AI", sectors, lowBound=0, cat="Continuous")
        x_h = pulp.LpVariable.dicts("x_H", sectors, lowBound=0, cat="Continuous")

        new_job = {}
        upgrade_job = {}
        displaced_job = {}
        net_job = {}

        for _, row in data.iterrows():
            s = row["sector"]

            new_job[s] = row["a1"] * x_ai[s] + row["a2"] * x_h[s]
            upgrade_job[s] = row["b1"] * x_h[s]
            displaced_job[s] = row["c1"] * x_ai[s] * row["risk"] / 100
            net_job[s] = new_job[s] + upgrade_job[s] - displaced_job[s]

        model += pulp.lpSum(net_job[s] for s in sectors)
        model += pulp.lpSum(x_ai[s] + x_h[s] for s in sectors) <= budget
        model += x_h[sector_name] >= min_h
        model += net_job[sector_name] >= 0

        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        status = pulp.LpStatus[model.status]

        if status == "Optimal":
            feasible_rows.append({
                "min_x_H_sector_2": min_h,
                "NetJob_sector_2": pulp.value(net_job[sector_name]),
                "x_AI_sector_2": x_ai[sector_name].value(),
                "x_H_sector_2": x_h[sector_name].value(),
            })

    if not feasible_rows:
        return None, None

    feasible_df = pd.DataFrame(feasible_rows)
    feasible_df = feasible_df.sort_values("min_x_H_sector_2")

    return feasible_df.iloc[0], feasible_df


def list_text(items):
    return ", ".join(items) if len(items) > 0 else "không có ngành nào theo ngưỡng hiện tại"


def format_money(x):
    return f"{x:,.0f}"


# ======================================================
# MODEL SETUP PANEL
# ======================================================

st.markdown("---")
st.header("Thiết lập mô hình")

section_caption(
    """
    Phần thiết lập được đặt ở đầu bài để người dùng thay đổi tham số rồi quan sát ngay các tab kết quả bên dưới.
    Các tham số gồm ngân sách AI - đào tạo, năng lực đào tạo lại, yêu cầu NetJob tối thiểu và ràng buộc an sinh xã hội.
    """
)

setting_box = st.container(border=True)

with setting_box:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Nguồn lực")
        budget = st.number_input(
            "Tổng ngân sách",
            min_value=10000,
            max_value=60000,
            value=30000,
            step=1000
        )

        training_capacity = st.number_input(
            "Năng lực đào tạo lại tối đa",
            min_value=5000,
            max_value=30000,
            value=18000,
            step=1000
        )

    with col2:
        st.markdown("#### Ràng buộc lao động")
        vulnerable_min_training = st.number_input(
            "Tỷ trọng đào tạo tối thiểu cho nhóm dễ tổn thương",
            min_value=0.00,
            max_value=0.80,
            value=0.25,
            step=0.05,
            format="%.2f"
        )

        min_total_netjob = st.number_input(
            "NetJob tối thiểu toàn nền kinh tế",
            min_value=0,
            max_value=500000,
            value=0,
            step=10000
        )

    with col3:
        st.markdown("#### An sinh xã hội")
        use_safety_constraint = st.checkbox(
            "Bật ràng buộc không ngành nào mất quá 5% lao động",
            value=False
        )

        max_displacement_share = 0.05 if use_safety_constraint else None

        st.write(
            """
            Ràng buộc này giúp kiểm tra tốc độ tự động hóa có vượt quá ngưỡng an toàn lao động hay không.
            """
        )

# ======================================================
# SOLVE
# ======================================================

solution, status = solve_labor_model(
    df,
    budget=budget,
    min_total_netjob=min_total_netjob,
    training_capacity=training_capacity,
    vulnerable_min_training=vulnerable_min_training,
    max_displacement_share=max_displacement_share
)

if solution is None:
    st.error(f"Mô hình không tìm được nghiệm tối ưu. Trạng thái: {status}")
    st.info("Gợi ý: nới ngân sách, tăng năng lực đào tạo lại, giảm NetJob tối thiểu hoặc tắt ràng buộc an sinh 5%.")
    st.stop()

result_df = solution["result"]
summary = solution["summary"]

safety_solution, safety_status = solve_labor_model(
    df,
    budget=budget,
    min_total_netjob=min_total_netjob,
    training_capacity=training_capacity,
    vulnerable_min_training=vulnerable_min_training,
    max_displacement_share=0.05
)

min_train_row, min_train_grid = find_min_training_for_sector_2(df, budget=budget)

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card("Ngân sách đã dùng", f"{summary['total_budget_used']:,.0f}", "Tổng đầu tư AI và đào tạo lại.")

with k2:
    kpi_card("Tổng đầu tư AI", f"{summary['total_ai']:,.0f}", "Ngân sách phân bổ cho tự động hóa và AI.")

with k3:
    kpi_card("Tổng đào tạo lại", f"{summary['total_training']:,.0f}", "Ngân sách cho đào tạo lại lao động.")

with k4:
    kpi_card("NetJob toàn nền kinh tế", f"{summary['total_net']:,.0f}", "Việc làm ròng sau khi trừ thay thế.")


# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Tổng quan", "Phân bổ tối ưu", "Mô phỏng dễ tổn thương", "Thảo luận chính sách"]
)

# ======================================================
# TAB 1
# ======================================================

with tab1:
    st.header("1. Tổng quan mô hình")

    st.write(
        """
        Mô hình xem mỗi ngành như một đơn vị nhận đầu tư AI và đầu tư đào tạo lại.
        Đầu tư AI có thể tạo việc làm mới nhưng cũng làm tăng việc làm bị thay thế.
        Đầu tư đào tạo lại giúp nâng cấp việc làm và giới hạn tốc độ dịch chuyển lao động.
        """
    )

    st.markdown("#### Công thức mô hình")

    formula_col, explain_col = st.columns([0.95, 1.05])

    with formula_col:
        st.latex(r"""
        NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i
        """)

        st.latex(r"""
        DisplacedJob_i \leq d_i x^H_i
        """)

    with explain_col:
        st.write(
            """
            Trong đó, x_AI là đầu tư AI, x_H là đầu tư đào tạo lại. NewJob phản ánh việc làm mới do AI và đào tạo tạo ra;
            UpgradeJob là việc làm được nâng cấp nhờ đào tạo lại; DisplacedJob là việc làm bị thay thế bởi tự động hóa.
            Ràng buộc DisplacedJob ≤ dᵢx_H thể hiện nguyên tắc tốc độ tự động hóa không được vượt quá năng lực đào tạo lại.
            """
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Số ngành", f"{len(df)}", "Số ngành trong mô hình.")

    with c2:
        kpi_card("Việc làm mới", f"{summary['total_new']:,.0f}", "NewJob toàn nền kinh tế.")

    with c3:
        kpi_card("Việc làm nâng cấp", f"{summary['total_upgrade']:,.0f}", "UpgradeJob toàn nền kinh tế.")

    with c4:
        kpi_card("Việc làm bị thay thế", f"{summary['total_displaced']:,.0f}", "DisplacedJob toàn nền kinh tế.")

    st.markdown("#### Dữ liệu 8 ngành")

    display_df = df.rename(columns={
        "sector": "Ngành",
        "labor": "Lao động",
        "risk": "Risk",
        "a1": "a1",
        "a2": "a2",
        "b1": "b1",
        "c1": "c1",
        "d1": "d1",
        "vulnerable": "Dễ tổn thương"
    })

    data_col, risk_col = st.columns([1.15, 0.85])

    with data_col:
        st.dataframe(
            display_df.style.format({
                "Lao động": "{:.2f}",
                "Risk": "{:.0f}",
                "a1": "{:.1f}",
                "a2": "{:.1f}",
                "b1": "{:.1f}",
                "c1": "{:.1f}",
                "d1": "{:.1f}",
                "Dễ tổn thương": "{:.0f}",
            }),
            use_container_width=True
        )

    with risk_col:
        plot_df = df.sort_values("risk", ascending=True)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=plot_df["sector"],
                x=plot_df["risk"],
                orientation="h",
                marker_color="#1FA7B6",
                marker_line=dict(color="#1FA7B6", width=1),
                hovertemplate="Ngành: %{y}<br>Risk: %{x:.0f}%<extra></extra>",
            )
        )

        fig.update_layout(
            title="Rủi ro tự động hóa theo ngành",
            height=390,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Risk (%)",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 2
# ======================================================

with tab2:
    st.header("2. Phân bổ tối ưu")

    st.write(
        """
        Phần này thể hiện cách mô hình phân bổ ngân sách giữa đầu tư AI và đào tạo lại theo từng ngành.
        Kết quả cần được đọc cùng NetJob và DisplacedJob, không chỉ đọc riêng ngân sách AI.
        """
    )

    display_result = result_df.rename(columns={
        "sector": "Ngành",
        "x_AI": "Đầu tư AI",
        "x_H": "Đào tạo lại",
        "NewJob": "NewJob",
        "UpgradeJob": "UpgradeJob",
        "DisplacedJob": "DisplacedJob",
        "NetJob": "NetJob",
        "displaced_share_labor": "Tỷ lệ thay thế/LĐ",
    })

    alloc_col, alloc_chart_col = st.columns([1.05, 0.95])

    with alloc_col:
        st.dataframe(
            display_result[
                [
                    "Ngành",
                    "Đầu tư AI",
                    "Đào tạo lại",
                    "NewJob",
                    "UpgradeJob",
                    "DisplacedJob",
                    "NetJob",
                    "Tỷ lệ thay thế/LĐ",
                ]
            ].style.format({
                "Đầu tư AI": "{:,.0f}",
                "Đào tạo lại": "{:,.0f}",
                "NewJob": "{:,.0f}",
                "UpgradeJob": "{:,.0f}",
                "DisplacedJob": "{:,.0f}",
                "NetJob": "{:,.0f}",
                "Tỷ lệ thay thế/LĐ": "{:.2%}",
            }),
            use_container_width=True
        )

    with alloc_chart_col:
        plot_df = result_df.sort_values("x_H", ascending=True)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=plot_df["sector"],
                x=plot_df["x_AI"],
                name="Đầu tư AI",
                orientation="h",
                marker_color="#1FA7B6",
                hovertemplate="Ngành: %{y}<br>Đầu tư AI: %{x:,.0f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Bar(
                y=plot_df["sector"],
                x=plot_df["x_H"],
                name="Đào tạo lại",
                orientation="h",
                marker_color="#E6F7F5",
                hovertemplate="Ngành: %{y}<br>Đào tạo lại: %{x:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Cơ cấu đầu tư AI và đào tạo lại",
            height=420,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Ngân sách",
            yaxis_title="",
            barmode="stack",
            legend=dict(orientation="h", y=-0.17),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cấu trúc việc làm ròng")

    sectors_order = result_df.sort_values("NetJob")["sector"].tolist()
    plot_df = result_df.set_index("sector").loc[sectors_order].reset_index()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=plot_df["sector"],
            x=plot_df["NewJob"],
            orientation="h",
            name="NewJob",
            marker_color="#1FA7B6",
            hovertemplate="Ngành: %{y}<br>NewJob: %{x:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=plot_df["sector"],
            x=plot_df["UpgradeJob"],
            orientation="h",
            name="UpgradeJob",
            marker_color="#E6F7F5",
            hovertemplate="Ngành: %{y}<br>UpgradeJob: %{x:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            y=plot_df["sector"],
            x=-plot_df["DisplacedJob"],
            orientation="h",
            name="DisplacedJob",
            marker_color="#FAD7D7",
            hovertemplate="Ngành: %{y}<br>DisplacedJob: %{customdata:,.0f}<extra></extra>",
            customdata=plot_df["DisplacedJob"],
        )
    )

    fig.add_vline(x=0, line_width=1, line_color="#334155")

    fig.update_layout(
        title="Luồng tạo - nâng cấp - thay thế việc làm",
        height=430,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Số việc làm",
        yaxis_title="",
        barmode="relative",
        legend=dict(orientation="h", y=-0.16),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 3
# ======================================================

with tab3:
    st.header("3. Mô phỏng nhóm dễ tổn thương")

    st.write(
        """
        Module này kiểm tra tác động của chính sách tới các ngành có nhiều lao động phổ thông hoặc dễ bị dịch chuyển.
        Đây là phần mở rộng để mô hình không chỉ tối đa hóa NetJob, mà còn đánh giá rủi ro an sinh xã hội.
        """
    )

    vulnerable_df = result_df[result_df["vulnerable"] == 1].copy()
    vulnerable_names = vulnerable_df["sector"].tolist()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Ngành dễ tổn thương", f"{len(vulnerable_df)}", "Số ngành được gắn cờ dễ tổn thương.")

    with c2:
        kpi_card("Đào tạo lại nhóm này", f"{vulnerable_df['x_H'].sum():,.0f}", "Tổng x_H cho nhóm dễ tổn thương.")

    with c3:
        kpi_card("Việc làm bị thay thế", f"{vulnerable_df['DisplacedJob'].sum():,.0f}", "DisplacedJob của nhóm dễ tổn thương.")

    with c4:
        kpi_card("NetJob nhóm này", f"{vulnerable_df['NetJob'].sum():,.0f}", "NetJob của nhóm dễ tổn thương.")

    vuln_col, vuln_chart_col = st.columns([1.05, 0.95])

    with vuln_col:
        st.markdown("#### Kết quả nhóm dễ tổn thương")

        st.dataframe(
            vulnerable_df[
                [
                    "sector",
                    "labor",
                    "risk",
                    "x_AI",
                    "x_H",
                    "DisplacedJob",
                    "NetJob",
                    "displaced_share_labor"
                ]
            ].rename(columns={
                "sector": "Ngành",
                "labor": "Lao động",
                "risk": "Risk",
                "x_AI": "Đầu tư AI",
                "x_H": "Đào tạo lại",
                "displaced_share_labor": "Tỷ lệ thay thế/LĐ"
            }).style.format({
                "Lao động": "{:.2f}",
                "Risk": "{:.0f}",
                "Đầu tư AI": "{:,.0f}",
                "Đào tạo lại": "{:,.0f}",
                "DisplacedJob": "{:,.0f}",
                "NetJob": "{:,.0f}",
                "Tỷ lệ thay thế/LĐ": "{:.2%}",
            }),
            use_container_width=True
        )

    with vuln_chart_col:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=vulnerable_df["sector"],
                y=vulnerable_df["DisplacedJob"],
                name="DisplacedJob",
                marker_color="#FAD7D7",
                hovertemplate="Ngành: %{x}<br>DisplacedJob: %{y:,.0f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Bar(
                x=vulnerable_df["sector"],
                y=vulnerable_df["NetJob"],
                name="NetJob",
                marker_color="#1FA7B6",
                hovertemplate="Ngành: %{x}<br>NetJob: %{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="DisplacedJob và NetJob của nhóm dễ tổn thương",
            height=360,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis=dict(tickangle=12),
            yaxis_title="Số việc làm",
            barmode="group",
            legend=dict(orientation="h", y=-0.20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Stress test ràng buộc an sinh 5%")

    if safety_solution is None:
        st.error("Khi thêm ràng buộc không ngành nào mất quá 5% lao động, mô hình không còn khả thi.")
    else:
        safety_summary = safety_solution["summary"]

        compare_safety = pd.DataFrame({
            "Chỉ tiêu": [
                "NetJob toàn nền kinh tế",
                "Tổng đầu tư AI",
                "Tổng đào tạo lại",
                "Việc làm bị thay thế",
            ],
            "Mô hình gốc": [
                summary["total_net"],
                summary["total_ai"],
                summary["total_training"],
                summary["total_displaced"],
            ],
            "Có ràng buộc an sinh 5%": [
                safety_summary["total_net"],
                safety_summary["total_ai"],
                safety_summary["total_training"],
                safety_summary["total_displaced"],
            ],
        })

        st.dataframe(
            compare_safety.style.format({
                "Mô hình gốc": "{:,.0f}",
                "Có ràng buộc an sinh 5%": "{:,.0f}",
            }),
            use_container_width=True
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=compare_safety["Chỉ tiêu"],
                y=compare_safety["Mô hình gốc"],
                name="Mô hình gốc",
                marker_color="#1FA7B6",
                hovertemplate="Chỉ tiêu: %{x}<br>Mô hình gốc: %{y:,.0f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Bar(
                x=compare_safety["Chỉ tiêu"],
                y=compare_safety["Có ràng buộc an sinh 5%"],
                name="Có ràng buộc an sinh 5%",
                marker_color="#E6F7F5",
                hovertemplate="Chỉ tiêu: %{x}<br>Có ràng buộc: %{y:,.0f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="So sánh mô hình gốc và mô hình có ràng buộc an sinh",
            height=340,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis=dict(tickangle=8),
            yaxis_title="Giá trị",
            barmode="group",
            legend=dict(orientation="h", y=-0.20),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Ngưỡng đào tạo tối thiểu cho ngành CN chế biến chế tạo")

    if min_train_row is not None:
        st.write(
            f"""
            Theo mô phỏng lưới, mức x_H tối thiểu cho ngành CN chế biến chế tạo để NetJob của ngành này không âm là khoảng
            {min_train_row["min_x_H_sector_2"]:,.0f}. Tại ngưỡng này, NetJob của ngành đạt khoảng {min_train_row["NetJob_sector_2"]:,.0f}.
            """
        )

        grid_col, grid_chart_col = st.columns([1, 1])

        with grid_col:
            st.dataframe(
                min_train_grid.head(12).style.format({
                    "min_x_H_sector_2": "{:,.0f}",
                    "NetJob_sector_2": "{:,.0f}",
                    "x_AI_sector_2": "{:,.0f}",
                    "x_H_sector_2": "{:,.0f}",
                }),
                use_container_width=True
            )

        with grid_chart_col:
            plot_grid = min_train_grid.head(25)

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=plot_grid["min_x_H_sector_2"],
                    y=plot_grid["NetJob_sector_2"],
                    mode="lines+markers",
                    line=dict(color="#1FA7B6", width=2.5),
                    marker=dict(size=7, color="#1FA7B6"),
                    hovertemplate=(
                        "x_H tối thiểu: %{x:,.0f}<br>"
                        "NetJob CN chế biến: %{y:,.0f}<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title="Ngưỡng đào tạo và NetJob ngành chế biến chế tạo",
                height=320,
                margin=dict(l=10, r=10, t=45, b=20),
                xaxis_title="x_H tối thiểu",
                yaxis_title="NetJob",
                plot_bgcolor="white",
                paper_bgcolor="white",
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Không tìm được ngưỡng đào tạo khả thi cho ngành CN chế biến chế tạo trong khoảng kiểm tra.")

# ======================================================
# TAB 4
# ======================================================

with tab4:
    st.header("4. Thảo luận chính sách")

    top_training_sector = result_df.sort_values("x_H", ascending=False).iloc[0]
    finance_row = result_df[result_df["sector"] == "Tài chính-Ngân hàng"].iloc[0]
    agri_row = result_df[result_df["sector"] == "Nông-Lâm-Thủy sản"].iloc[0]
    manufacturing_row = result_df[result_df["sector"] == "CN chế biến chế tạo"].iloc[0]

    low_ai_high_displaced = result_df[
        (result_df["x_AI"] <= result_df["x_AI"].median()) &
        (result_df["DisplacedJob"] >= result_df["DisplacedJob"].median())
    ]["sector"].tolist()

    st.markdown("#### Ngành cần đào tạo lại nhiều nhất")

    st.write(
        f"""
        Kết quả tối ưu cho thấy ngành cần đầu tư đào tạo lại nhiều nhất là {top_training_sector["sector"]},
        với mức x_H khoảng {top_training_sector["x_H"]:,.0f}. Điều này cần được hiểu theo logic của mô hình:
        ngành nào có quy mô lao động lớn, rủi ro tự động hóa đáng kể hoặc hệ số nâng cấp việc làm từ đào tạo cao thì thường cần ngân sách đào tạo lại lớn hơn.
        """
    )

    st.write(
        """
        Cách diễn giải này khá phù hợp với thực tiễn Việt Nam. Các ngành có nhiều lao động như nông nghiệp, chế biến chế tạo, xây dựng và bán buôn - bán lẻ
        thường chịu tác động mạnh khi tự động hóa và AI thay đổi quy trình sản xuất, phân phối và dịch vụ. Vì vậy, chính sách lao động không nên
        chỉ tập trung vào tạo việc làm công nghệ cao mới, mà phải đi kèm đào tạo lại quy mô lớn cho nhóm lao động đang làm việc trong các ngành truyền thống.
        """
    )

    st.markdown("#### Ngành Tài chính - Ngân hàng")

    st.write(
        f"""
        Ngành Tài chính - Ngân hàng có mức rủi ro thay thế cao trong dữ liệu đầu vào, nhưng đồng thời cũng có hệ số tạo việc làm mới từ AI rất lớn.
        Trong nghiệm hiện tại, mô hình phân bổ x_AI khoảng {finance_row["x_AI"]:,.0f} và x_H khoảng {finance_row["x_H"]:,.0f} cho ngành này,
        tạo NetJob khoảng {finance_row["NetJob"]:,.0f}.
        """
    )

    st.write(
        """
        Hàm ý chính sách là không nên nhìn Tài chính - Ngân hàng chỉ như một ngành bị AI thay thế.
        Đây là ngành có khả năng chuyển đổi việc làm theo hướng công nghệ cao: phân tích dữ liệu, quản trị rủi ro, an ninh mạng,
        cá nhân hóa dịch vụ tài chính, tự động hóa quy trình và kiểm soát tuân thủ. Tuy nhiên, vì rủi ro thay thế cũng cao,
        chiến lược phù hợp không phải là giảm đầu tư AI, mà là gắn đầu tư AI với tái đào tạo nội bộ, chuyển đổi kỹ năng
        và bảo vệ nhóm lao động làm công việc lặp lại.
        """
    )

    st.markdown("#### Có nên đầu tư AI vào Nông-Lâm-Thủy sản không?")

    st.write(
        f"""
        Ngành Nông-Lâm-Thủy sản có hệ số tạo việc làm AI thấp hơn nhiều ngành khác, nhưng quy mô lao động rất lớn.
        Trong nghiệm hiện tại, ngành này nhận x_AI khoảng {agri_row["x_AI"]:,.0f} và x_H khoảng {agri_row["x_H"]:,.0f}.
        Điểm quan trọng là mô hình không khuyến nghị đầu tư AI vào nông nghiệp chỉ vì tạo nhiều việc làm công nghệ cao trực tiếp.
        Nếu đầu tư AI vào ngành này, mục tiêu hợp lý hơn là nâng năng suất, giảm rủi ro lao động dịch chuyển và hỗ trợ chuyển đổi kỹ năng.
        """
    )

    st.write(
        """
        Vì vậy, câu trả lời không phải là “không đầu tư AI vào nông nghiệp”, mà là đầu tư AI có chọn lọc.
        AI có thể hỗ trợ dự báo mùa vụ, quản lý chuỗi cung ứng, truy xuất nguồn gốc, tối ưu tưới tiêu, cảnh báo dịch bệnh
        và thương mại số cho nông sản. Nhưng nếu chỉ đầu tư AI mà không đi kèm đào tạo lại, khuyến nông số và hỗ trợ doanh nghiệp nhỏ,
        lợi ích có thể không lan tỏa đến nhóm lao động lớn nhất. Mô hình vì vậy nhấn mạnh vai trò của x_H đối với ngành này.
        """
    )

    st.markdown("#### Ràng buộc tốc độ tự động hóa và an sinh xã hội")

    st.write(
        """
        Phát biểu “tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại” được biểu diễn trực tiếp bằng ràng buộc:
        DisplacedJob_i <= d_i * x_H_i. Ràng buộc này buộc lượng việc làm bị thay thế trong mỗi ngành không được vượt quá khả năng hấp thụ
        của chương trình đào tạo lại trong ngành đó.
        """
    )

    st.write(
        """
        Đây là điểm khác biệt quan trọng giữa mô hình lao động và mô hình tối đa hóa công nghệ thuần túy.
        Nếu không có ràng buộc này, nghiệm tối ưu có thể dồn quá nhiều đầu tư vào AI ở các ngành có hiệu suất cao,
        nhưng gây dịch chuyển lao động lớn hơn năng lực đào tạo lại. Khi đó NetJob toàn nền kinh tế có thể dương trên giấy,
        nhưng một số nhóm lao động vẫn bị tổn thương nặng.
        """
    )

    st.markdown("#### Đề xuất ràng buộc bổ sung")

    st.write(
        f"""
        Ngoài ràng buộc DisplacedJob_i <= d_i * x_H_i, có thể bổ sung ràng buộc an sinh xã hội:
        DisplacedJob_i <= 0.05 * L_i. Ràng buộc này giới hạn số việc làm bị thay thế trong mỗi ngành không vượt quá 5% lao động của ngành.
        Trong dashboard, ràng buộc này được kiểm tra tại module stress test. Nếu bài toán vẫn khả thi, chính sách có thể vừa duy trì NetJob dương
        vừa kiểm soát mức độ tổn thương ngành. Nếu bài toán không khả thi, điều đó cho thấy tốc độ triển khai AI theo ngân sách hiện tại quá nhanh
        so với năng lực đào tạo lại và cần giảm nhịp tự động hóa hoặc tăng ngân sách đào tạo.
        """
    )

    st.write(
        """
        Có thể bổ sung thêm một ràng buộc cho nhóm dễ tổn thương:
        tổng đào tạo lại cho các ngành nông nghiệp, xây dựng và bán buôn - bán lẻ phải chiếm ít nhất một tỷ lệ nhất định trong tổng x_H.
        Ràng buộc này giúp tránh tình huống ngân sách đào tạo bị hút quá mạnh vào các ngành có năng suất cao nhưng ít lao động,
        trong khi nhóm lao động lớn lại không được hỗ trợ đủ. Đây là cách đưa mục tiêu an sinh xã hội vào mô hình một cách định lượng.
        """
    )

    st.markdown("#### Liên hệ chính sách Việt Nam")

    st.write(
        """
        Bài toán Bài 9 cho thấy chính sách AI không thể tách khỏi chính sách lao động.
        Nghị quyết 57-NQ/TW đặt khoa học công nghệ, đổi mới sáng tạo và chuyển đổi số là đột phá phát triển;
        Quyết định 127/QĐ-TTg đặt mục tiêu phát triển và ứng dụng AI đến năm 2030;
        Quyết định 749/QĐ-TTg thúc đẩy chuyển đổi số quốc gia; và Quyết định 411/QĐ-TTg nhấn mạnh phát triển kinh tế số, xã hội số.
        Các định hướng này đều ủng hộ việc ứng dụng công nghệ để tăng năng suất, nhưng nếu thiếu đào tạo lại,
        thị trường lao động có thể bị phân hóa.
        """
    )

    st.write(
        """
        Vì vậy, chính sách phù hợp với Việt Nam nên có ba lớp. Lớp thứ nhất là đầu tư AI vào các ngành có khả năng tạo việc làm mới
        và năng suất cao. Lớp thứ hai là đào tạo lại bắt buộc ở các ngành có nhiều lao động và rủi ro dịch chuyển lớn.
        Lớp thứ ba là cơ chế an sinh chuyển tiếp, gồm hỗ trợ thu nhập ngắn hạn, tư vấn nghề nghiệp, đào tạo kỹ năng số cơ bản
        và kết nối việc làm sau đào tạo. Cách tiếp cận này giúp AI trở thành động lực nâng năng suất thay vì trở thành cú sốc
        làm gia tăng bất bình đẳng lao động.
        """
    )

    st.success(
        """
        Kết luận: Mô hình cho thấy đầu tư AI chỉ tạo tác động tích cực bền vững khi đi cùng đào tạo lại và ràng buộc an sinh.
        NetJob dương toàn nền kinh tế là chưa đủ; chính sách còn phải kiểm soát ngành nào bị thay thế, nhóm lao động nào được đào tạo lại
        và tốc độ tự động hóa có vượt quá năng lực hấp thụ của thị trường lao động hay không.
        """
    )
