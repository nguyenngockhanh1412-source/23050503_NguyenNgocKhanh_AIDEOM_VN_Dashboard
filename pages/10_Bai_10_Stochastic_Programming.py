import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pyomo.environ as pyo

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


# ======================================================
# PAGE SETUP
# ======================================================

setup_page("Bài 10 - Stochastic Programming")

def safe_render_bai10_sidebar():
    """
    Gọi sidebar chung đúng nhãn trong MENU_OPTIONS.
    Nếu tên bài trong utils/aideom_ui.py có khác dấu ':' hoặc '-', hàm sẽ tự dò mục chứa Bài 10.
    """
    try:
        import utils.aideom_ui as ui

        menu_options = getattr(ui, "MENU_OPTIONS", [])
        candidates = [
            "Bài 10 Stochastic Programming",
            "Bài 10 - Stochastic Programming",
            "Bài 10: Stochastic Programming",
            "Bài 10 Quy hoạch ngẫu nhiên",
            "Bài 10 - Quy hoạch ngẫu nhiên",
        ]

        for name in candidates:
            if name in menu_options:
                render_sidebar(name)
                return

        for name in menu_options:
            normalized = str(name).lower()
            if "bài 10" in normalized or "bai 10" in normalized or "stochastic" in normalized:
                render_sidebar(name)
                return

        # Nếu không có trong MENU_OPTIONS, vẫn cố gọi nhãn phổ biến nhất.
        render_sidebar("Bài 10 Stochastic Programming")

    except Exception:
        # Không gọi lặp sidebar để tránh lỗi lặp AIDEOM-VN.
        st.sidebar.markdown("### AIDEOM-VN")
        st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
        st.sidebar.markdown("---")
        st.sidebar.info("Đang ở Bài 10. Cần kiểm tra lại tên Bài 10 trong MENU_OPTIONS của utils/aideom_ui.py để hiện menu chung.")

safe_render_bai10_sidebar()

st.title("Bài 10. Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định")

st.write(
    """
    Dashboard mô phỏng bài toán ra quyết định ngân sách trong điều kiện bất định. 
    Chính phủ phải phân bổ ngân sách ban đầu khi chưa biết kịch bản kinh tế nào sẽ xảy ra, 
    sau đó được điều chỉnh bổ sung khi kịch bản thực tế đã được quan sát.
    """
)

source_note(
    """
    Dữ liệu kịch bản, hệ số beta và ràng buộc ngân sách được sử dụng theo yêu cầu Bài 10 trong bộ đề.
    Kết quả là mô phỏng phục vụ phân tích, không thay thế quy trình thẩm định ngân sách chính thức.
    """
)

# ======================================================
# DATA
# ======================================================

items = ["Hạ tầng số", "Chuyển đổi số", "Trí tuệ nhân tạo", "Nhân lực số"]

default_scenarios = pd.DataFrame({
    "Kịch bản": ["Lạc quan", "Cơ sở", "Bi quan", "Khủng hoảng"],
    "Tăng trưởng TG (%)": [3.5, 2.8, 1.5, 0.2],
    "FDI VN (tỷ USD/năm)": [32.0, 27.0, 20.0, 12.0],
    "Xuất khẩu VN tăng (%)": [12.0, 8.0, 3.0, -5.0],
    "Xác suất": [0.30, 0.45, 0.20, 0.05],
})

default_beta = pd.DataFrame({
    "Hạng mục": items,
    "Cơ bản": [1.00, 1.10, 1.25, 0.95],
    "Lạc quan": [1.25, 1.35, 1.55, 1.05],
    "Cơ sở": [1.00, 1.10, 1.25, 0.95],
    "Bi quan": [0.75, 0.85, 0.90, 1.00],
    "Khủng hoảng": [0.40, 0.50, 0.55, 1.10],
})


# ======================================================
# FUNCTIONS
# ======================================================

def normalize_probabilities(scenarios_df):
    out = scenarios_df.copy()
    total = out["Xác suất"].sum()
    if total <= 0:
        out["Xác suất"] = 1 / len(out)
    else:
        out["Xác suất"] = out["Xác suất"] / total
    return out


def get_solver():
    solver = pyo.SolverFactory("appsi_highs")
    if not solver.available(exception_flag=False):
        return None
    return solver


def prepare_beta(beta_df, scenario_names):
    beta_base = dict(zip(beta_df["Hạng mục"], beta_df["Cơ bản"]))
    beta_s = {}
    for s in scenario_names:
        beta_s[s] = dict(zip(beta_df["Hạng mục"], beta_df[s]))
    return beta_base, beta_s


def solve_stochastic_model(
    scenarios_df,
    beta_df,
    first_budget=65000,
    recourse_budget=15000,
    penalty=0.15,
    fixed_x=None,
):
    scenarios_df = normalize_probabilities(scenarios_df)
    scenario_names = scenarios_df["Kịch bản"].tolist()
    prob = dict(zip(scenarios_df["Kịch bản"], scenarios_df["Xác suất"]))
    beta_base, beta_s = prepare_beta(beta_df, scenario_names)

    model = pyo.ConcreteModel()
    model.J = pyo.Set(initialize=items)
    model.S = pyo.Set(initialize=scenario_names)

    model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.S, model.J, domain=pyo.NonNegativeReals)
    model.overuse = pyo.Var(model.S, domain=pyo.NonNegativeReals)

    model.first_budget = pyo.Constraint(
        expr=sum(model.x[j] for j in model.J) <= first_budget
    )

    if fixed_x is not None:
        def fixed_rule(m, j):
            return m.x[j] == float(fixed_x[j])
        model.fixed_x_con = pyo.Constraint(model.J, rule=fixed_rule)

    def recourse_budget_rule(m, s):
        return sum(m.y[s, j] for j in m.J) <= recourse_budget
    model.recourse_budget = pyo.Constraint(model.S, rule=recourse_budget_rule)

    def ai_capacity_rule(m, s):
        return m.y[s, "Trí tuệ nhân tạo"] <= 0.5 * m.x["Nhân lực số"]
    model.ai_capacity = pyo.Constraint(model.S, rule=ai_capacity_rule)

    def overuse_rule(m, s):
        reserve = first_budget - sum(m.x[j] for j in m.J)
        return m.overuse[s] >= sum(m.y[s, j] for j in m.J) - reserve
    model.overuse_con = pyo.Constraint(model.S, rule=overuse_rule)

    model.obj = pyo.Objective(
        expr=
        sum(beta_base[j] * model.x[j] for j in model.J)
        +
        sum(
            prob[s] * (
                sum(beta_s[s][j] * model.y[s, j] for j in model.J)
                - penalty * model.overuse[s]
            )
            for s in model.S
        ),
        sense=pyo.maximize
    )

    solver = get_solver()
    if solver is None:
        return None, "No solver"

    result = solver.solve(model)

    x = {j: pyo.value(model.x[j]) for j in items}

    y_rows = []
    scenario_values = {}

    base_part = sum(beta_base[j] * x[j] for j in items)

    for s in scenario_names:
        recourse_part = sum(beta_s[s][j] * pyo.value(model.y[s, j]) for j in items)
        penalty_part = penalty * pyo.value(model.overuse[s])
        scenario_value = base_part + recourse_part - penalty_part
        scenario_values[s] = scenario_value

        for j in items:
            y_rows.append({
                "Kịch bản": s,
                "Hạng mục": j,
                "Điều chỉnh": pyo.value(model.y[s, j]),
            })

    x_df = pd.DataFrame({
        "Hạng mục": items,
        "Ngân sách ban đầu": [x[j] for j in items],
    })

    y_df = pd.DataFrame(y_rows)

    expected_value = sum(prob[s] * scenario_values[s] for s in scenario_names)

    return {
        "x": x,
        "x_df": x_df,
        "y_df": y_df,
        "expected_value": expected_value,
        "scenario_values": scenario_values,
        "base_part": base_part,
        "prob": prob,
        "status": str(result.solver.termination_condition),
    }, "Optimal"


def solve_expected_value_model(
    scenarios_df,
    beta_df,
    first_budget=65000,
    recourse_budget=15000,
    penalty=0.15,
):
    scenarios_df = normalize_probabilities(scenarios_df)
    scenario_names = scenarios_df["Kịch bản"].tolist()
    prob = dict(zip(scenarios_df["Kịch bản"], scenarios_df["Xác suất"]))
    beta_base, beta_s = prepare_beta(beta_df, scenario_names)

    expected_beta = {
        j: sum(prob[s] * beta_s[s][j] for s in scenario_names)
        for j in items
    }

    model = pyo.ConcreteModel()
    model.J = pyo.Set(initialize=items)

    model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.overuse = pyo.Var(domain=pyo.NonNegativeReals)

    model.first_budget = pyo.Constraint(
        expr=sum(model.x[j] for j in model.J) <= first_budget
    )

    model.recourse_budget = pyo.Constraint(
        expr=sum(model.y[j] for j in model.J) <= recourse_budget
    )

    model.ai_capacity = pyo.Constraint(
        expr=model.y["Trí tuệ nhân tạo"] <= 0.5 * model.x["Nhân lực số"]
    )

    model.overuse_con = pyo.Constraint(
        expr=model.overuse >= sum(model.y[j] for j in model.J) - (first_budget - sum(model.x[j] for j in model.J))
    )

    model.obj = pyo.Objective(
        expr=
        sum(beta_base[j] * model.x[j] for j in model.J)
        +
        sum(expected_beta[j] * model.y[j] for j in model.J)
        - penalty * model.overuse,
        sense=pyo.maximize
    )

    solver = get_solver()
    if solver is None:
        return None, "No solver"

    result = solver.solve(model)

    x = {j: pyo.value(model.x[j]) for j in items}

    x_df = pd.DataFrame({
        "Hạng mục": items,
        "Ngân sách ban đầu": [x[j] for j in items],
    })

    value = pyo.value(model.obj)

    return {
        "x": x,
        "x_df": x_df,
        "value": value,
        "status": str(result.solver.termination_condition),
    }, "Optimal"


def solve_wait_and_see(
    scenarios_df,
    beta_df,
    first_budget=65000,
    recourse_budget=15000,
    penalty=0.15,
):
    scenarios_df = normalize_probabilities(scenarios_df)
    scenario_names = scenarios_df["Kịch bản"].tolist()
    prob = dict(zip(scenarios_df["Kịch bản"], scenarios_df["Xác suất"]))
    beta_base, beta_s = prepare_beta(beta_df, scenario_names)

    rows = []
    values = {}

    for s in scenario_names:
        model = pyo.ConcreteModel()
        model.J = pyo.Set(initialize=items)

        model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
        model.y = pyo.Var(model.J, domain=pyo.NonNegativeReals)
        model.overuse = pyo.Var(domain=pyo.NonNegativeReals)

        model.first_budget = pyo.Constraint(
            expr=sum(model.x[j] for j in model.J) <= first_budget
        )

        model.recourse_budget = pyo.Constraint(
            expr=sum(model.y[j] for j in model.J) <= recourse_budget
        )

        model.ai_capacity = pyo.Constraint(
            expr=model.y["Trí tuệ nhân tạo"] <= 0.5 * model.x["Nhân lực số"]
        )

        model.overuse_con = pyo.Constraint(
            expr=model.overuse >= sum(model.y[j] for j in model.J) - (first_budget - sum(model.x[j] for j in model.J))
        )

        model.obj = pyo.Objective(
            expr=
            sum(beta_base[j] * model.x[j] for j in model.J)
            +
            sum(beta_s[s][j] * model.y[j] for j in model.J)
            - penalty * model.overuse,
            sense=pyo.maximize
        )

        solver = get_solver()
        if solver is None:
            return None, "No solver"

        result = solver.solve(model)
        values[s] = pyo.value(model.obj)

        for j in items:
            rows.append({
                "Kịch bản": s,
                "Hạng mục": j,
                "Ngân sách ban đầu nếu biết trước": pyo.value(model.x[j]),
                "Điều chỉnh nếu biết trước": pyo.value(model.y[j]),
            })

    ws_expected = sum(prob[s] * values[s] for s in scenario_names)

    return {
        "ws_expected_value": ws_expected,
        "scenario_values": values,
        "rows": pd.DataFrame(rows),
    }, "Optimal"


def solve_robust_regret(
    scenarios_df,
    beta_df,
    ws_values,
    first_budget=65000,
    recourse_budget=15000,
    penalty=0.15,
):
    scenarios_df = normalize_probabilities(scenarios_df)
    scenario_names = scenarios_df["Kịch bản"].tolist()
    prob = dict(zip(scenarios_df["Kịch bản"], scenarios_df["Xác suất"]))
    beta_base, beta_s = prepare_beta(beta_df, scenario_names)

    model = pyo.ConcreteModel()
    model.J = pyo.Set(initialize=items)
    model.S = pyo.Set(initialize=scenario_names)

    model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)
    model.y = pyo.Var(model.S, model.J, domain=pyo.NonNegativeReals)
    model.overuse = pyo.Var(model.S, domain=pyo.NonNegativeReals)
    model.R = pyo.Var(domain=pyo.NonNegativeReals)

    model.first_budget = pyo.Constraint(
        expr=sum(model.x[j] for j in model.J) <= first_budget
    )

    def recourse_budget_rule(m, s):
        return sum(m.y[s, j] for j in m.J) <= recourse_budget
    model.recourse_budget = pyo.Constraint(model.S, rule=recourse_budget_rule)

    def ai_capacity_rule(m, s):
        return m.y[s, "Trí tuệ nhân tạo"] <= 0.5 * m.x["Nhân lực số"]
    model.ai_capacity = pyo.Constraint(model.S, rule=ai_capacity_rule)

    def overuse_rule(m, s):
        reserve = first_budget - sum(m.x[j] for j in m.J)
        return m.overuse[s] >= sum(m.y[s, j] for j in m.J) - reserve
    model.overuse_con = pyo.Constraint(model.S, rule=overuse_rule)

    def regret_rule(m, s):
        scenario_value = (
            sum(beta_base[j] * m.x[j] for j in m.J)
            +
            sum(beta_s[s][j] * m.y[s, j] for j in m.J)
            - penalty * m.overuse[s]
        )
        return m.R >= ws_values[s] - scenario_value
    model.regret_con = pyo.Constraint(model.S, rule=regret_rule)

    model.obj = pyo.Objective(expr=model.R, sense=pyo.minimize)

    solver = get_solver()
    if solver is None:
        return None, "No solver"

    result = solver.solve(model)

    x = {j: pyo.value(model.x[j]) for j in items}

    y_rows = []
    scenario_values = {}
    regrets = {}

    for s in scenario_names:
        scenario_value = (
            sum(beta_base[j] * pyo.value(model.x[j]) for j in items)
            +
            sum(beta_s[s][j] * pyo.value(model.y[s, j]) for j in items)
            - penalty * pyo.value(model.overuse[s])
        )

        scenario_values[s] = scenario_value
        regrets[s] = ws_values[s] - scenario_value

        for j in items:
            y_rows.append({
                "Kịch bản": s,
                "Hạng mục": j,
                "Điều chỉnh": pyo.value(model.y[s, j]),
            })

    expected_value = sum(prob[s] * scenario_values[s] for s in scenario_names)

    return {
        "x": x,
        "x_df": pd.DataFrame({
            "Hạng mục": items,
            "Ngân sách ban đầu": [x[j] for j in items],
        }),
        "y_df": pd.DataFrame(y_rows),
        "expected_value": expected_value,
        "scenario_values": scenario_values,
        "regrets": regrets,
        "worst_regret": pyo.value(model.R),
        "status": str(result.solver.termination_condition),
    }, "Optimal"


def to_wide_y(y_df):
    return y_df.pivot(index="Kịch bản", columns="Hạng mục", values="Điều chỉnh").fillna(0)


def share_df(x_df):
    out = x_df.copy()
    total = out["Ngân sách ban đầu"].sum()
    if total > 0:
        out["Tỷ trọng"] = out["Ngân sách ban đầu"] / total
    else:
        out["Tỷ trọng"] = 0
    return out


def list_high_items(x_df, threshold=0.20):
    temp = share_df(x_df)
    selected = temp[temp["Tỷ trọng"] >= threshold]["Hạng mục"].tolist()
    if not selected:
        return "không có hạng mục nào vượt ngưỡng"
    return ", ".join(selected)


# ======================================================
# 0. ABBREVIATION GLOSSARY
# ======================================================

st.markdown("---")
st.header("0. Danh mục từ viết tắt")

section_caption("Các khái niệm dưới đây giúp đọc đúng SP, EV, WS, VSS, EVPI và robust regret trong bài toán quy hoạch ngẫu nhiên hai giai đoạn.")

abbr_df = pd.DataFrame({
    "Từ viết tắt": [
        "SP",
        "EV",
        "WS",
        "VSS",
        "EVPI",
        "Robust regret",
        "Stochastic programming",
        "First-stage",
        "Second-stage recourse",
    ],
    "Viết đầy đủ / nghĩa": [
        "Stochastic Programming - quy hoạch ngẫu nhiên",
        "Expected Value - lời giải theo kịch bản trung bình",
        "Wait-and-See - lời giải khi biết trước kịch bản tương lai",
        "Value of Stochastic Solution - giá trị của lời giải ngẫu nhiên",
        "Expected Value of Perfect Information - giá trị kỳ vọng của thông tin hoàn hảo",
        "Hối tiếc lớn nhất trong các kịch bản, dùng để đo mức độ phòng thủ của quyết định",
        "Phương pháp tối ưu hóa khi có nhiều kịch bản bất định và xác suất xảy ra",
        "Quyết định giai đoạn 1, đưa ra trước khi biết kịch bản",
        "Quyết định điều chỉnh giai đoạn 2, đưa ra sau khi kịch bản xảy ra",
    ],
    "Cách hiểu trong bài 10": [
        "Mô hình chính, xét đủ 4 kịch bản và xác suất.",
        "Giải bài toán bằng kịch bản trung bình rồi đem đánh giá lại trong bất định.",
        "Mức kết quả lý tưởng nếu biết trước tương lai, thường là trần tham chiếu.",
        "Nếu VSS dương, dùng SP tốt hơn cách quyết định theo kịch bản trung bình.",
        "Nếu EVPI lớn, hệ thống dự báo và dữ liệu sớm có giá trị chính sách cao.",
        "Dùng để chọn quyết định ít hối tiếc nhất nếu rơi vào kịch bản bất lợi.",
        "Phù hợp khi chính sách phải quyết định trước nhưng tương lai chưa chắc chắn.",
        "Phân bổ ngân sách ban đầu cho I, D, AI, H.",
        "Điều chỉnh thêm ngân sách cho từng hạng mục trong từng kịch bản.",
    ],
})

st.dataframe(abbr_df, use_container_width=True)


# ======================================================
# 1. SCENARIO TREE
# ======================================================

st.markdown("---")
st.header("1. Cây kịch bản và tham số mô hình")

st.write(
    """
    Người dùng có thể điều chỉnh xác suất kịch bản, mức tăng trưởng, FDI, xuất khẩu và hệ số hiệu quả đầu tư.
    Sau khi điều chỉnh, kết quả SP, EV, WS, VSS, EVPI và robust regret sẽ tự cập nhật.
    """
)

st.markdown("#### Thiết lập ngân sách và chi phí điều chỉnh")

b1, b2, b3 = st.columns(3)

with b1:
    first_budget = st.number_input(
        "Ngân sách giai đoạn 1",
        min_value=30000,
        max_value=100000,
        value=65000,
        step=5000,
    )

with b2:
    recourse_budget = st.number_input(
        "Ngân sách điều chỉnh mỗi kịch bản",
        min_value=5000,
        max_value=40000,
        value=15000,
        step=2500,
    )

with b3:
    penalty = st.number_input(
        "Hệ số phạt vượt dự phòng",
        min_value=0.00,
        max_value=1.00,
        value=0.15,
        step=0.05,
        format="%.2f",
    )

scenario_col, scenario_chart_col = st.columns([1.2, 0.9])

with scenario_col:
    st.markdown("#### Điều chỉnh kịch bản")

    scenario_input = st.data_editor(
        default_scenarios,
        hide_index=True,
        use_container_width=True,
        disabled=["Kịch bản"],
        column_config={
            "Tăng trưởng TG (%)": st.column_config.NumberColumn("Tăng trưởng TG (%)", step=0.1, format="%.1f"),
            "FDI VN (tỷ USD/năm)": st.column_config.NumberColumn("FDI VN (tỷ USD/năm)", step=1.0, format="%.1f"),
            "Xuất khẩu VN tăng (%)": st.column_config.NumberColumn("Xuất khẩu VN tăng (%)", step=1.0, format="%.1f"),
            "Xác suất": st.column_config.NumberColumn("Xác suất", min_value=0.0, max_value=1.0, step=0.01, format="%.2f"),
        }
    )

    scenario_input = normalize_probabilities(scenario_input)
    st.caption(f"Tổng xác suất sau chuẩn hóa: {scenario_input['Xác suất'].sum():.2f}")

with scenario_chart_col:
    st.markdown("#### Biểu đồ kịch bản")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=scenario_input["Kịch bản"],
            y=scenario_input["Xác suất"],
            name="Xác suất",
            marker_color="#1FA7B6",
            hovertemplate="Kịch bản: %{x}<br>Xác suất: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scenario_input["Kịch bản"],
            y=scenario_input["Tăng trưởng TG (%)"],
            name="Tăng trưởng TG",
            mode="lines+markers",
            marker=dict(size=8, color="#1FA7B6"),
            line=dict(color="#1FA7B6", width=2),
            yaxis="y2",
            hovertemplate="Kịch bản: %{x}<br>Tăng trưởng TG: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Xác suất và tăng trưởng thế giới",
        height=290,
        margin=dict(l=10, r=10, t=45, b=20),
        legend=dict(orientation="h", y=-0.25),
        xaxis=dict(tickangle=0),
        yaxis=dict(title="Xác suất", range=[0, max(0.6, scenario_input["Xác suất"].max() + 0.1)]),
        yaxis2=dict(title="Tăng trưởng TG", overlaying="y", side="right"),
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Hệ số hiệu quả đầu tư theo kịch bản")

beta_col, beta_chart_col = st.columns([1.2, 0.9])

with beta_col:
    beta_input = st.data_editor(
        default_beta,
        hide_index=True,
        use_container_width=True,
        disabled=["Hạng mục"],
        column_config={
            "Cơ bản": st.column_config.NumberColumn("Cơ bản", min_value=0.0, max_value=3.0, step=0.05, format="%.2f"),
            "Lạc quan": st.column_config.NumberColumn("Lạc quan", min_value=0.0, max_value=3.0, step=0.05, format="%.2f"),
            "Cơ sở": st.column_config.NumberColumn("Cơ sở", min_value=0.0, max_value=3.0, step=0.05, format="%.2f"),
            "Bi quan": st.column_config.NumberColumn("Bi quan", min_value=0.0, max_value=3.0, step=0.05, format="%.2f"),
            "Khủng hoảng": st.column_config.NumberColumn("Khủng hoảng", min_value=0.0, max_value=3.0, step=0.05, format="%.2f"),
        }
    )

with beta_chart_col:
    heat_data = beta_input.set_index("Hạng mục")[scenario_input["Kịch bản"].tolist()]

    fig = go.Figure(
        data=go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns,
            y=heat_data.index,
            colorscale=[[0, "#F8FBFF"], [0.35, "#81D8D0"], [0.7, "#FF6B6B"], [1, "#1FA7B6"]],
            text=np.round(heat_data.values, 2),
            texttemplate="%{text}",
            hovertemplate="Hạng mục: %{y}<br>Kịch bản: %{x}<br>Hệ số: %{z:.2f}<extra></extra>",
            showscale=True,
        )
    )

    fig.update_layout(
        title="Hiệu quả đầu tư theo kịch bản",
        height=290,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# SOLVE MODELS
# ======================================================

if get_solver() is None:
    st.error("Không tìm thấy solver HiGHS. Hãy kiểm tra đã cài highspy chưa: python -m pip install highspy")
    st.stop()

sp_solution, _ = solve_stochastic_model(
    scenario_input,
    beta_input,
    first_budget=first_budget,
    recourse_budget=recourse_budget,
    penalty=penalty,
)

ev_solution, _ = solve_expected_value_model(
    scenario_input,
    beta_input,
    first_budget=first_budget,
    recourse_budget=recourse_budget,
    penalty=penalty,
)

eev_solution, _ = solve_stochastic_model(
    scenario_input,
    beta_input,
    first_budget=first_budget,
    recourse_budget=recourse_budget,
    penalty=penalty,
    fixed_x=ev_solution["x"],
)

ws_solution, _ = solve_wait_and_see(
    scenario_input,
    beta_input,
    first_budget=first_budget,
    recourse_budget=recourse_budget,
    penalty=penalty,
)

robust_solution, _ = solve_robust_regret(
    scenario_input,
    beta_input,
    ws_solution["scenario_values"],
    first_budget=first_budget,
    recourse_budget=recourse_budget,
    penalty=penalty,
)

sp_value = sp_solution["expected_value"]
eev_value = eev_solution["expected_value"]
ws_value = ws_solution["ws_expected_value"]
robust_value = robust_solution["expected_value"]

vss = sp_value - eev_value
evpi = ws_value - sp_value


# ======================================================
# 2. SP SOLUTION
# ======================================================

st.markdown("---")
st.header("2. Lời giải stochastic programming")

st.write(
    """
    Lời giải SP đưa ra quyết định giai đoạn 1 trước khi biết kịch bản, đồng thời cho phép điều chỉnh giai đoạn 2 theo từng kịch bản.
    """
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    kpi_card("Giá trị kỳ vọng SP", f"{sp_value:,.2f}", "Giá trị kỳ vọng của lời giải ngẫu nhiên.")

with m2:
    kpi_card("Ngân sách ban đầu", f"{sp_solution['x_df']['Ngân sách ban đầu'].sum():,.0f}", "Tổng ngân sách giai đoạn 1.")

with m3:
    kpi_card("VSS", f"{vss:,.2f}", "Giá trị của lời giải ngẫu nhiên.")

with m4:
    kpi_card("EVPI", f"{evpi:,.2f}", "Giá trị thông tin hoàn hảo.")

sp_col1, sp_col2 = st.columns([1, 1])

with sp_col1:
    st.markdown("#### Quyết định giai đoạn 1")

    st.dataframe(
        sp_solution["x_df"].style.format({"Ngân sách ban đầu": "{:,.0f}"}),
        use_container_width=True
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=sp_solution["x_df"]["Hạng mục"],
                y=sp_solution["x_df"]["Ngân sách ban đầu"],
                marker_color=["#1FA7B6", "#0B1D33", "#81D8D0", "#E6F7F5"],
                hovertemplate="Hạng mục: %{x}<br>Ngân sách: %{y:,.0f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Phân bổ ngân sách ban đầu",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Ngân sách",
    )

    st.plotly_chart(fig, use_container_width=True)

with sp_col2:
    y_wide = to_wide_y(sp_solution["y_df"])

    st.markdown("#### Điều chỉnh giai đoạn 2")

    st.dataframe(
        y_wide.style.format("{:,.0f}"),
        use_container_width=True
    )

    fig = go.Figure()

    colors = ["#1FA7B6", "#0B1D33", "#81D8D0", "#E6F7F5"]

    for i, col in enumerate(y_wide.columns):
        fig.add_trace(
            go.Bar(
                x=y_wide.index,
                y=y_wide[col],
                name=col,
                marker_color=colors[i],
                hovertemplate="Kịch bản: %{x}<br>Hạng mục: " + col + "<br>Điều chỉnh: %{y:,.0f}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Cơ cấu điều chỉnh theo kịch bản",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Ngân sách điều chỉnh",
        legend=dict(orientation="h", y=-0.28),
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# 3. COMPARISON
# ======================================================

st.markdown("---")
st.header("3. So sánh EV, SP, WS và robust regret")

comparison_df = pd.DataFrame({
    "Phương pháp": [
        "EV đánh giá trong bất định",
        "SP",
        "Wait-and-see",
        "Robust regret",
    ],
    "Giá trị kỳ vọng": [
        eev_value,
        sp_value,
        ws_value,
        robust_value,
    ],
    "Diễn giải": [
        "Dùng quyết định từ mô hình trung bình rồi đánh giá lại trong 4 kịch bản.",
        "Ra quyết định trước bất định và tối ưu điều chỉnh theo từng kịch bản.",
        "Giả định biết trước kịch bản, là mức giá trị trần của thông tin hoàn hảo.",
        "Giảm hối tiếc trong kịch bản bất lợi thay vì chỉ tối đa hóa kỳ vọng.",
    ]
})

st.dataframe(
    comparison_df.style.format({"Giá trị kỳ vọng": "{:,.2f}"}),
    use_container_width=True
)

comp_col1, comp_col2 = st.columns([1, 1])

with comp_col1:
    fig = go.Figure(
        data=[
            go.Bar(
                y=comparison_df["Phương pháp"],
                x=comparison_df["Giá trị kỳ vọng"],
                orientation="h",
                marker_color=["#F8FCFC", "#1FA7B6", "#E6F7F5", "#0B1D33"],
                hovertemplate="Phương pháp: %{y}<br>Giá trị: %{x:,.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Giá trị kỳ vọng theo phương pháp",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Giá trị",
    )

    st.plotly_chart(fig, use_container_width=True)

with comp_col2:
    method_x = pd.DataFrame({
        "Hạng mục": items,
        "EV": [ev_solution["x"][j] for j in items],
        "SP": [sp_solution["x"][j] for j in items],
        "Robust": [robust_solution["x"][j] for j in items],
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(x=method_x["Hạng mục"], y=method_x["EV"], name="EV", marker_color="#F8FCFC"))
    fig.add_trace(go.Bar(x=method_x["Hạng mục"], y=method_x["SP"], name="SP", marker_color="#1FA7B6"))
    fig.add_trace(go.Bar(x=method_x["Hạng mục"], y=method_x["Robust"], name="Robust", marker_color="#0B1D33"))

    fig.update_traces(hovertemplate="Hạng mục: %{x}<br>Ngân sách: %{y:,.0f}<extra></extra>")

    fig.update_layout(
        barmode="group",
        title="So sánh ngân sách ban đầu",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Ngân sách",
        legend=dict(orientation="h", y=-0.25),
    )

    st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)

with c1:
    kpi_card("VSS", f"{vss:,.2f}", "SP so với lời giải EV đánh giá lại.")

with c2:
    kpi_card("EVPI", f"{evpi:,.2f}", "Giá trị kỳ vọng của thông tin hoàn hảo.")

with c3:
    kpi_card("Worst regret robust", f"{robust_solution['worst_regret']:,.2f}", "Mức hối tiếc lớn nhất của nghiệm robust.")

regret_df = pd.DataFrame({
    "Kịch bản": list(robust_solution["regrets"].keys()),
    "Regret": list(robust_solution["regrets"].values()),
})

reg_col1, reg_col2 = st.columns([1, 1])

with reg_col1:
    st.markdown("#### Regret của nghiệm robust")

    st.dataframe(
        regret_df.style.format({"Regret": "{:,.2f}"}),
        use_container_width=True
    )

with reg_col2:
    fig = go.Figure(
        data=[
            go.Scatter(
                x=regret_df["Kịch bản"],
                y=regret_df["Regret"],
                mode="lines+markers",
                line=dict(color="#1FA7B6", width=2),
                marker=dict(size=8),
                hovertemplate="Kịch bản: %{x}<br>Regret: %{y:,.2f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Mức hối tiếc theo kịch bản",
        height=290,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Regret",
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# 4. POLICY INSURANCE
# ======================================================

st.markdown("---")
st.header("4. Policy Insurance Simulator")

scenario_value_df = pd.DataFrame({
    "Kịch bản": list(sp_solution["scenario_values"].keys()),
    "SP": list(sp_solution["scenario_values"].values()),
    "Robust": list(robust_solution["scenario_values"].values()),
    "Wait-and-see": [ws_solution["scenario_values"][s] for s in sp_solution["scenario_values"].keys()],
})

val_col, chart_col = st.columns([1, 1])

with val_col:
    st.dataframe(
        scenario_value_df.style.format({
            "SP": "{:,.2f}",
            "Robust": "{:,.2f}",
            "Wait-and-see": "{:,.2f}",
        }),
        use_container_width=True
    )

with chart_col:
    fig = go.Figure()

    fig.add_trace(go.Bar(x=scenario_value_df["Kịch bản"], y=scenario_value_df["SP"], name="SP", marker_color="#1FA7B6"))
    fig.add_trace(go.Bar(x=scenario_value_df["Kịch bản"], y=scenario_value_df["Robust"], name="Robust", marker_color="#0B1D33"))
    fig.add_trace(go.Bar(x=scenario_value_df["Kịch bản"], y=scenario_value_df["Wait-and-see"], name="Wait-and-see", marker_color="#E6F7F5"))

    fig.update_traces(hovertemplate="Kịch bản: %{x}<br>Giá trị: %{y:,.2f}<extra></extra>")

    fig.update_layout(
        barmode="group",
        title="Giá trị theo từng kịch bản",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Giá trị",
        legend=dict(orientation="h", y=-0.25),
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================================
# 5. POLICY DISCUSSION
# ======================================================

st.markdown("---")
st.header("5. Diễn giải chính sách")

sp_share = share_df(sp_solution["x_df"])
ev_share = share_df(ev_solution["x_df"])
robust_share = share_df(robust_solution["x_df"])

sp_h = sp_solution["x"]["Nhân lực số"]
ev_h = ev_solution["x"]["Nhân lực số"]
robust_h = robust_solution["x"]["Nhân lực số"]

sp_h_share = sp_share[sp_share["Hạng mục"] == "Nhân lực số"]["Tỷ trọng"].iloc[0]
ev_h_share = ev_share[ev_share["Hạng mục"] == "Nhân lực số"]["Tỷ trọng"].iloc[0]
robust_h_share = robust_share[robust_share["Hạng mục"] == "Nhân lực số"]["Tỷ trọng"].iloc[0]

crisis_name = "Khủng hoảng"
sp_crisis = sp_solution["scenario_values"][crisis_name]
robust_crisis = robust_solution["scenario_values"][crisis_name]
ws_crisis = ws_solution["scenario_values"][crisis_name]

most_sp_item = sp_share.sort_values("Tỷ trọng", ascending=False).iloc[0]["Hạng mục"]
most_robust_item = robust_share.sort_values("Tỷ trọng", ascending=False).iloc[0]["Hạng mục"]

st.markdown("#### Lời giải ngẫu nhiên và vai trò của nhân lực số")

st.write(
    f"""
    So với lời giải EV, lời giải SP phân bổ ngân sách ban đầu cho nhân lực số ở mức {sp_h:,.0f}, 
    trong khi EV phân bổ {ev_h:,.0f}. Tỷ trọng nhân lực số trong SP là khoảng {sp_h_share:.1%}, còn trong EV là khoảng {ev_h_share:.1%}. 
    Nếu SP phân bổ nhiều hơn cho nhân lực số, điều này cho thấy mô hình không chỉ chạy theo kịch bản trung bình, 
    mà còn xét khả năng nền kinh tế rơi vào trạng thái bất lợi.

    Nhân lực số có vai trò như một loại bảo hiểm chính sách. Trong kịch bản thuận lợi, AI và chuyển đổi số có thể tạo tăng trưởng nhanh. 
    Nhưng khi kịch bản xấu xảy ra, năng lực con người giúp doanh nghiệp và khu vực công chuyển đổi việc làm, vận hành nền tảng số,
    khai thác dữ liệu và duy trì dịch vụ thiết yếu. Vì vậy, đầu tư nhân lực không chỉ là chi phí đào tạo, mà là năng lực chống chịu của nền kinh tế.
    """
)

st.markdown("#### VSS, EVPI và giá trị của thông tin bất định")

st.write(
    f"""
    VSS hiện bằng {vss:,.2f}. Nếu VSS dương, việc xét đủ các kịch bản bất định tạo ra giá trị cao hơn so với cách ra quyết định dựa trên kịch bản trung bình. 
    Với Việt Nam, điều này có ý nghĩa vì nền kinh tế có độ mở lớn, phụ thuộc vào xuất khẩu, FDI, chuỗi cung ứng và biến động địa chính trị. 
    Một kế hoạch ngân sách chỉ dựa trên kịch bản cơ sở có thể thiếu linh hoạt nếu tăng trưởng thế giới giảm, xuất khẩu suy yếu hoặc dòng FDI thay đổi.

    EVPI hiện bằng {evpi:,.2f}. Đây là giá trị kỳ vọng của việc biết trước hoàn hảo tương lai. 
    Nếu EVPI lớn, đầu tư vào hệ thống dự báo, dữ liệu thời gian thực, cảnh báo rủi ro, thống kê số và năng lực phân tích chính sách có giá trị cao.
    Nếu EVPI nhỏ, điều đó cho thấy quyết định SP hiện tại đã có mức linh hoạt nhất định để giảm thiệt hại do bất định.
    """
)

st.markdown("#### Robust regret và logic phòng thủ")

st.write(
    f"""
    Lời giải robust regret có tỷ trọng nhân lực số khoảng {robust_h_share:.1%}, với ngân sách nhân lực số là {robust_h:,.0f}.
    Hạng mục chiếm tỷ trọng lớn nhất trong SP là {most_sp_item}, còn trong robust là {most_robust_item}.
    Nếu robust nghiêng nhiều hơn về nhân lực số hoặc chuyển đổi số, điều đó phản ánh logic phòng thủ: 
    thay vì chỉ tối đa hóa giá trị kỳ vọng, mô hình muốn giảm mức hối tiếc trong kịch bản xấu nhất.

    Ở kịch bản khủng hoảng, giá trị SP đạt {sp_crisis:,.2f}, robust đạt {robust_crisis:,.2f}, 
    còn mức tốt nhất nếu biết trước kịch bản là {ws_crisis:,.2f}. 
    Khoảng cách này phản ánh chi phí của việc không biết trước tương lai và phải ra quyết định trong bất định.
    """
)

st.markdown("#### Liên hệ thực tiễn Việt Nam")

st.write(
    """
    COVID-19, các đứt gãy chuỗi cung ứng và các cú sốc thiên tai như bão Yagi cho thấy năng lực chống chịu của nền kinh tế không chỉ phụ thuộc vào vốn vật chất.
    Khi có cú sốc, khả năng duy trì giáo dục trực tuyến, thương mại điện tử, dịch vụ công trực tuyến, logistics số và làm việc từ xa phụ thuộc lớn vào hạ tầng số,
    dữ liệu và nhân lực số.

    Điều này phù hợp với các định hướng chính sách của Việt Nam: Quyết định 749/QĐ-TTg về Chương trình chuyển đổi số quốc gia,
    Quyết định 411/QĐ-TTg về phát triển kinh tế số và xã hội số, Quyết định 127/QĐ-TTg về Chiến lược AI quốc gia,
    và Nghị quyết 57-NQ/TW về khoa học công nghệ, đổi mới sáng tạo và chuyển đổi số. 
    Các văn kiện này đều nhấn mạnh công nghệ, dữ liệu và nhân lực như nền tảng phát triển dài hạn. Bài toán SP làm rõ thêm rằng
    trong điều kiện bất định, các yếu tố này còn có vai trò như lớp bảo hiểm giúp chính sách không bị quá cứng trước cú sốc.
    """
)

st.markdown("#### Hàm ý điều hành ngân sách")

st.write(
    f"""
    Với dữ liệu hiện tại, SP ưu tiên tương đối cao cho {list_high_items(sp_solution["x_df"], threshold=0.20)}.
    Khi người dùng tăng xác suất khủng hoảng hoặc giảm hệ số hiệu quả của AI trong kịch bản xấu, mô hình thường có xu hướng chuyển ngân sách
    sang các hạng mục có tính nền tảng và chống chịu hơn. Ngược lại, khi xác suất lạc quan tăng và hiệu quả AI cao hơn,
    ngân sách có thể nghiêng mạnh hơn về AI.

    Do đó, khuyến nghị không phải là luôn tăng AI hay luôn tăng nhân lực số, mà là thiết kế ngân sách theo trạng thái bất định.
    Khi môi trường quốc tế thuận lợi, AI là công cụ tăng tốc năng suất. Khi rủi ro toàn cầu tăng, nhân lực số, dữ liệu và chuyển đổi số cơ bản
    trở thành lớp bảo hiểm giúp nền kinh tế thích ứng.
    """
)

st.success(
    """
    Kết luận: Bài 10 cho thấy ra quyết định trong bất định không nên dựa vào một kịch bản trung bình duy nhất.
    Stochastic programming giúp lượng hóa giá trị của sự linh hoạt, còn robust regret giúp kiểm tra khả năng chịu đựng của chính sách trong kịch bản xấu.
    """
)