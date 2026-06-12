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

setup_page("Bài 4 - Regional LP")
render_sidebar("Bài 4 - Regional LP")

st.title("Bài 4. Quy hoạch tuyến tính phân bổ ngân sách số theo vùng")

st.write(
    """
    Bài 4 mở rộng bài toán phân bổ ngân sách từ cấp quốc gia sang cấp vùng.
    Mô hình xem xét 6 vùng kinh tế - xã hội và 4 hạng mục đầu tư: hạ tầng số,
    chuyển đổi số doanh nghiệp, năng lực AI và nhân lực số. Mục tiêu là tối đa hóa GDP gain kỳ vọng,
    đồng thời bảo đảm các ràng buộc về ngân sách, sàn/trần vùng, nhân lực số và công bằng số vùng miền.
    """
)

# ======================================================
# IMPORT SOLVERS
# ======================================================

try:
    import pulp
except ImportError:
    st.error(
        "Bạn chưa cài PuLP. Hãy chạy trong Terminal: "
        ".\\.venv\\Scripts\\python.exe -m pip install pulp"
    )
    st.stop()

try:
    import cvxpy as cp
except ImportError:
    st.error(
        "Bạn chưa cài CVXPY. Hãy chạy trong Terminal: "
        ".\\.venv\\Scripts\\python.exe -m pip install cvxpy"
    )
    st.stop()

# ======================================================
# DATA
# ======================================================

regions = [
    "Trung du miền núi phía Bắc",
    "Đồng bằng sông Hồng",
    "Bắc Trung Bộ + DH Trung Bộ",
    "Tây Nguyên",
    "Đông Nam Bộ",
    "Đồng bằng sông Cửu Long",
]

region_short = ["TDMNPB", "ĐBSH", "BTB-DHMT", "Tây Nguyên", "ĐNB", "ĐBSCL"]

items = ["I", "D", "AI", "H"]

item_names = {
    "I": "Hạ tầng số",
    "D": "CĐS doanh nghiệp",
    "AI": "Năng lực AI",
    "H": "Nhân lực số",
}

item_colors = {
    "I": "#0B1D33",
    "D": "#1FA7B6",
    "AI": "#FF6B6B",
    "H": "#E6F7F5",
}

beta_matrix = pd.DataFrame(
    [
        [1.15, 0.85, 0.55, 1.30],
        [0.95, 1.25, 1.40, 1.05],
        [1.05, 0.95, 0.85, 1.15],
        [1.20, 0.75, 0.45, 1.35],
        [0.90, 1.30, 1.55, 1.00],
        [1.10, 0.85, 0.65, 1.25],
    ],
    index=regions,
    columns=items,
)

D0 = {
    "Trung du miền núi phía Bắc": 38,
    "Đồng bằng sông Hồng": 78,
    "Bắc Trung Bộ + DH Trung Bộ": 55,
    "Tây Nguyên": 32,
    "Đông Nam Bộ": 82,
    "Đồng bằng sông Cửu Long": 48,
}

# ======================================================
# SOLVER FUNCTIONS
# ======================================================

def solve_with_pulp(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
    use_fairness=True,
    use_region_cap=True,
):
    model = pulp.LpProblem("VN_Digital_Budget_Regional", pulp.LpMaximize)

    x = pulp.LpVariable.dicts(
        "x",
        (regions, items),
        lowBound=0,
        cat="Continuous",
    )

    model += pulp.lpSum(
        beta_matrix.loc[r, j] * x[r][j]
        for r in regions
        for j in items
    )

    model += pulp.lpSum(x[r][j] for r in regions for j in items) <= budget_total

    for r in regions:
        model += pulp.lpSum(x[r][j] for j in items) >= region_floor
        if use_region_cap:
            model += pulp.lpSum(x[r][j] for j in items) <= region_cap

    model += pulp.lpSum(x[r]["H"] for r in regions) >= human_floor

    if use_fairness:
        M = pulp.LpVariable("Dmax", lowBound=0, cat="Continuous")

        for r in regions:
            model += D0[r] + gamma_fair * x[r]["D"] <= M

        for r in regions:
            model += D0[r] + gamma_fair * x[r]["D"] >= lambda_fair * M

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    status = pulp.LpStatus[model.status]

    if status != "Optimal":
        return None, status

    allocation = pd.DataFrame(index=regions, columns=items)

    for r in regions:
        for j in items:
            allocation.loc[r, j] = x[r][j].value()

    allocation = allocation.astype(float)
    z_value = pulp.value(model.objective)

    return {
        "allocation": allocation,
        "Z": z_value,
        "status": status,
    }, status


def solve_with_cvxpy(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
    use_fairness=True,
    use_region_cap=True,
):
    beta_np = beta_matrix.values.astype(float)
    D0_np = np.array([D0[r] for r in regions], dtype=float)

    x = cp.Variable((len(regions), len(items)), nonneg=True)

    objective = cp.Maximize(cp.sum(cp.multiply(beta_np, x)))
    constraints = [
        cp.sum(x) <= budget_total,
        cp.sum(x, axis=1) >= region_floor,
        cp.sum(x[:, 3]) >= human_floor,
    ]

    if use_region_cap:
        constraints.append(cp.sum(x, axis=1) <= region_cap)

    if use_fairness:
        M = cp.Variable(nonneg=True)
        post_digital = D0_np + gamma_fair * x[:, 1]
        constraints.append(post_digital <= M)
        constraints.append(post_digital >= lambda_fair * M)

    problem = cp.Problem(objective, constraints)

    solved = False
    for solver_name in ["CLARABEL", "SCS"]:
        try:
            problem.solve(solver=solver_name)
            solved = True
            break
        except Exception:
            continue

    if not solved:
        return None, "Solver failed"

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return None, problem.status

    allocation = pd.DataFrame(
        x.value,
        index=regions,
        columns=items,
    )

    return {
        "allocation": allocation,
        "Z": problem.value,
        "status": problem.status,
    }, problem.status


def feasibility_notes(budget_total, region_floor, region_cap, human_floor, gamma_fair, lambda_fair):
    notes = []

    if region_floor * len(regions) > budget_total:
        notes.append(
            f"Tổng sàn 6 vùng là {region_floor * len(regions):,.0f}, lớn hơn tổng ngân sách {budget_total:,.0f}."
        )

    if region_cap * len(regions) < budget_total:
        notes.append(
            f"Tổng trần 6 vùng là {region_cap * len(regions):,.0f}, nhỏ hơn tổng ngân sách {budget_total:,.0f}."
        )

    if region_floor > region_cap:
        notes.append("Sàn mỗi vùng đang lớn hơn trần mỗi vùng.")

    if human_floor > budget_total:
        notes.append("Sàn nhân lực số đang lớn hơn tổng ngân sách.")

    max_initial_d = max(D0.values())

    lambda_max_structural = min(
        (D0[r] + gamma_fair * region_cap) / max_initial_d
        for r in regions
    )

    if lambda_fair > lambda_max_structural:
        notes.append(
            f"Với gamma = {gamma_fair:.4f} và trần vùng = {region_cap:,.0f}, "
            f"lambda tối đa gần đúng chỉ khoảng {lambda_max_structural:.3f}. "
            f"Lambda hiện tại = {lambda_fair:.3f}, nên ràng buộc công bằng có nguy cơ không khả thi."
        )

    return notes, lambda_max_structural


def make_allocation_display(allocation):
    display_df = allocation.rename(columns=item_names).copy()
    display_df["Tổng vùng"] = display_df.sum(axis=1)
    return display_df


def make_digital_after(allocation, gamma_fair, lambda_fair):
    digital_df = pd.DataFrame({
        "Vùng": regions,
        "Tên ngắn": region_short,
        "D0": [D0[r] for r in regions],
        "Đầu tư D": allocation["D"].values,
    })

    digital_df["D sau đầu tư"] = digital_df["D0"] + gamma_fair * digital_df["Đầu tư D"]
    digital_df["Ngưỡng công bằng"] = lambda_fair * digital_df["D sau đầu tư"].max()
    digital_df["Tỷ lệ so với vùng cao nhất (%)"] = (
        digital_df["D sau đầu tư"] / digital_df["D sau đầu tư"].max() * 100
    )

    return digital_df


# ======================================================
# SECTION 1: MODEL
# ======================================================

st.markdown("---")
st.header("1. Mô hình toán học")

st.latex(r"""
\max Z = \sum_r \sum_j \beta_{j,r}x_{j,r}
""")

st.write(
    """
    Trong mô hình, x(j,r) là mức ngân sách phân bổ cho hạng mục j tại vùng r, còn β(j,r)
    là hệ số tác động biên của từng hạng mục tại từng vùng. Mô hình không chỉ tối đa hóa GDP gain kỳ vọng,
    mà còn kiểm soát phân bổ tối thiểu, trần ngân sách vùng, đầu tư nhân lực số và ràng buộc công bằng số giữa các vùng.
    """
)

source_note(
    """
    Dữ liệu hệ số β và chỉ số D0 được dùng để minh họa bài toán LP phân bổ ngân sách số theo vùng trong bộ đề.
    Kết quả là mô phỏng phục vụ phân tích, không thay thế quy trình thẩm định ngân sách chính thức.
    """
)

# ======================================================
# SECTION 2: POLICY PARAMETERS
# ======================================================

st.markdown("---")
st.header("2. Thiết lập tham số chính sách")

section_caption(
    """
    Phần này giữ cách trình bày dạng bảng điều khiển chính sách. Người dùng chọn kịch bản, sau đó có thể điều chỉnh ngân sách,
    sàn vùng, trần vùng, sàn nhân lực số và ràng buộc công bằng số. Các chỉ tiêu kiểm tra nhanh được đặt trong card để dễ quan sát.
    """
)

scenario = st.selectbox(
    "Chọn kịch bản",
    [
        "Khả thi gần đề bài",
        "Strict theo đề bài",
        "Ưu tiên công bằng mạnh",
        "Nới ngân sách đầu tư số",
        "Ưu tiên nhân lực số",
    ],
)

if scenario == "Strict theo đề bài":
    default_budget = 50000.0
    default_floor = 5000.0
    default_cap = 12000.0
    default_human = 12000.0
    default_lambda = 0.70
    default_gamma = 0.002

elif scenario == "Khả thi gần đề bài":
    default_budget = 50000.0
    default_floor = 5000.0
    default_cap = 12000.0
    default_human = 12000.0
    default_lambda = 0.68
    default_gamma = 0.002

elif scenario == "Ưu tiên công bằng mạnh":
    default_budget = 55000.0
    default_floor = 6000.0
    default_cap = 14000.0
    default_human = 13000.0
    default_lambda = 0.75
    default_gamma = 0.0025

elif scenario == "Nới ngân sách đầu tư số":
    default_budget = 60000.0
    default_floor = 5000.0
    default_cap = 15000.0
    default_human = 13000.0
    default_lambda = 0.70
    default_gamma = 0.002

else:
    default_budget = 50000.0
    default_floor = 5000.0
    default_cap = 12000.0
    default_human = 16000.0
    default_lambda = 0.68
    default_gamma = 0.002

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown("#### Quy mô ngân sách")
    budget_total = st.number_input(
        "Tổng ngân sách",
        min_value=10000.0,
        max_value=100000.0,
        value=default_budget,
        step=1000.0,
        help="Tổng ngân sách phân bổ cho 6 vùng và 4 hạng mục, đơn vị tỷ VND.",
    )

    human_floor = st.number_input(
        "Sàn nhân lực số",
        min_value=0.0,
        max_value=100000.0,
        value=default_human,
        step=1000.0,
        help="Tổng đầu tư tối thiểu cho nhân lực số trên toàn quốc.",
    )

with p2:
    st.markdown("#### Ràng buộc vùng")
    region_floor = st.number_input(
        "Sàn mỗi vùng",
        min_value=0.0,
        max_value=50000.0,
        value=default_floor,
        step=500.0,
        help="Mỗi vùng phải nhận ít nhất mức ngân sách này.",
    )

    region_cap = st.number_input(
        "Trần mỗi vùng",
        min_value=0.0,
        max_value=50000.0,
        value=default_cap,
        step=500.0,
        help="Mỗi vùng không được vượt quá mức ngân sách này.",
    )

with p3:
    st.markdown("#### Công bằng số")
    lambda_fair = st.number_input(
        "Lambda công bằng",
        min_value=0.10,
        max_value=1.00,
        value=default_lambda,
        step=0.01,
        format="%.2f",
        help="Vùng yếu nhất sau đầu tư phải đạt ít nhất lambda so với vùng mạnh nhất.",
    )

    gamma_fair = st.number_input(
        "Gamma tác động D",
        min_value=0.0001,
        max_value=0.0100,
        value=default_gamma,
        step=0.0001,
        format="%.4f",
        help="Mức tăng chỉ số số hóa khi đầu tư thêm vào chuyển đổi số doanh nghiệp.",
    )

notes, lambda_max_structural = feasibility_notes(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
)

k1, k2, k3, k4 = st.columns(4)

with k1:
    kpi_card(
        "Tổng sàn 6 vùng",
        f"{region_floor * 6:,.0f}",
        "Tổng ngân sách tối thiểu nếu mỗi vùng đều nhận mức sàn."
    )

with k2:
    kpi_card(
        "Tổng trần 6 vùng",
        f"{region_cap * 6:,.0f}",
        "Tổng ngân sách tối đa nếu mỗi vùng đều chạm trần."
    )

with k3:
    kpi_card(
        "Lambda tối đa gần đúng",
        f"{lambda_max_structural:.3f}",
        "Ngưỡng công bằng gần đúng có thể đạt với trần vùng hiện tại."
    )

with k4:
    kpi_card(
        "Lambda đang chọn",
        f"{lambda_fair:.3f}",
        "Ngưỡng công bằng người dùng đang áp dụng."
    )

if len(notes) == 0:
    st.success("Bộ tham số hiện tại không có dấu hiệu mâu thuẫn cơ bản. Có thể giải mô hình.")
else:
    st.warning("Hệ thống phát hiện một số điểm có thể làm mô hình không khả thi:")
    for note in notes:
        st.write(f"- {note}")

if scenario == "Strict theo đề bài":
    st.info(
        """
        Kịch bản strict theo đề bài giữ lambda = 0.70 và gamma = 0.002.
        Với trần vùng 12.000, mô hình có thể không khả thi vì vùng Tây Nguyên có D0 ban đầu thấp.
        Đây là một kết quả đáng phân tích, không phải lỗi phần mềm.
        """
    )

# ======================================================
# SECTION 3: BETA MATRIX
# ======================================================

st.markdown("---")
st.header("3. Hệ số tác động biên β(j,r)")

section_caption(
    """
    Bảng hệ số β cho biết hiệu quả biên của từng hạng mục đầu tư ở từng vùng.
    Hệ số càng cao thì cùng một đơn vị ngân sách tạo ra GDP gain kỳ vọng càng lớn.
    """
)

beta_display = beta_matrix.rename(columns=item_names)

beta_col, beta_chart_col = st.columns([1, 1])

with beta_col:
    st.dataframe(
        beta_display.style.format("{:.2f}"),
        use_container_width=True,
    )

with beta_chart_col:
    fig = go.Figure()

    for item in items:
        fig.add_trace(
            go.Bar(
                x=region_short,
                y=beta_matrix[item].values,
                name=item_names[item],
                marker_color=item_colors[item],
                hovertemplate=(
                    "Vùng: %{x}<br>"
                    f"Hạng mục: {item_names[item]}<br>"
                    "β: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Hệ số β theo vùng và hạng mục",
        height=340,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Hệ số β",
        barmode="group",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 4: FAIR MODEL
# ======================================================

st.markdown("---")
st.header("4. Kết quả tối ưu có ràng buộc công bằng")

pulp_fair, status_pulp_fair = solve_with_pulp(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
    use_fairness=True,
    use_region_cap=True,
)

cvxpy_fair, status_cvxpy_fair = solve_with_cvxpy(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
    use_fairness=True,
    use_region_cap=True,
)

if pulp_fair is None:
    st.error(f"PuLP không tìm được nghiệm tối ưu. Trạng thái: {status_pulp_fair}")

    st.write(
        """
        Mô hình không khả thi khi các ràng buộc chính sách không thể đồng thời được thỏa mãn.
        Với bộ tham số strict theo đề bài, nguyên nhân thường nằm ở ràng buộc công bằng số:
        các vùng có D0 ban đầu thấp cần quá nhiều đầu tư D để đạt tỷ lệ lambda so với vùng mạnh nhất,
        trong khi lại bị giới hạn bởi trần ngân sách vùng.
        """
    )

    st.info(
        "Gợi ý: chọn kịch bản Khả thi gần đề bài, giảm lambda công bằng, tăng gamma tác động D, hoặc nới trần mỗi vùng."
    )

    # Không dừng app để người dùng vẫn đọc được phân tích dữ liệu và chỉnh tham số.
    st.stop()

alloc_fair = pulp_fair["allocation"]
Z_fair = pulp_fair["Z"]

alloc_fair_display = make_allocation_display(alloc_fair)

c1, c2, c3, c4 = st.columns(4)

with c1:
    kpi_card("Trạng thái PuLP", status_pulp_fair, "Trạng thái nghiệm của mô hình chính.")

with c2:
    kpi_card("Z* PuLP", f"{Z_fair:,.2f}", "GDP gain kỳ vọng theo nghiệm PuLP.")

with c3:
    if cvxpy_fair is not None:
        kpi_card("Z* CVXPY", f"{cvxpy_fair['Z']:,.2f}", "Kết quả đối chiếu bằng CVXPY.")
    else:
        kpi_card("Z* CVXPY", "Không khả thi", f"Trạng thái: {status_cvxpy_fair}")

with c4:
    if cvxpy_fair is not None:
        kpi_card(
            "Chênh lệch solver",
            f"{abs(Z_fair - cvxpy_fair['Z']):,.4f}",
            "Chênh lệch giữa PuLP và CVXPY."
        )
    else:
        kpi_card("Chênh lệch solver", "N/A", "Không có kết quả CVXPY để so sánh.")

alloc_col, stack_col = st.columns([1.05, 0.95])

with alloc_col:
    st.markdown("#### Ma trận phân bổ tối ưu 6 × 4")

    st.dataframe(
        alloc_fair_display.style.format("{:,.1f}"),
        use_container_width=True,
    )

with stack_col:
    fig = go.Figure()

    for item in items:
        fig.add_trace(
            go.Bar(
                y=region_short,
                x=alloc_fair[item].values,
                name=item_names[item],
                orientation="h",
                marker_color=item_colors[item],
                hovertemplate=(
                    "Vùng: %{y}<br>"
                    f"Hạng mục: {item_names[item]}<br>"
                    "Ngân sách: %{x:,.1f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Cơ cấu ngân sách theo vùng",
        height=390,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Tỷ VND",
        yaxis_title="",
        barmode="stack",
        legend=dict(orientation="h", y=-0.18),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 5: DIGITAL GAP
# ======================================================

st.markdown("---")
st.header("5. Kiểm tra công bằng số sau đầu tư")

digital_after = make_digital_after(alloc_fair, gamma_fair, lambda_fair)

digital_col, digital_chart_col = st.columns([1, 1])

with digital_col:
    st.dataframe(
        digital_after[[
            "Vùng",
            "D0",
            "Đầu tư D",
            "D sau đầu tư",
            "Tỷ lệ so với vùng cao nhất (%)",
        ]].style.format({
            "D0": "{:.1f}",
            "Đầu tư D": "{:,.1f}",
            "D sau đầu tư": "{:.2f}",
            "Tỷ lệ so với vùng cao nhất (%)": "{:.1f}",
        }),
        use_container_width=True,
    )

with digital_chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=digital_after["Tên ngắn"],
            y=digital_after["D0"],
            mode="lines+markers",
            name="D0 ban đầu",
            line=dict(color="#1FA7B6", width=2.5),
            marker=dict(size=8),
            hovertemplate="Vùng: %{x}<br>D0: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=digital_after["Tên ngắn"],
            y=digital_after["D sau đầu tư"],
            mode="lines+markers",
            name="D sau đầu tư",
            line=dict(color="#1FA7B6", width=2.5),
            marker=dict(size=8),
            hovertemplate="Vùng: %{x}<br>D sau đầu tư: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=digital_after["Tên ngắn"],
            y=digital_after["Ngưỡng công bằng"],
            mode="lines",
            name="Ngưỡng công bằng",
            line=dict(color="#5FA8D3", width=2, dash="dash"),
            hovertemplate="Ngưỡng công bằng: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Khoảng cách số trước và sau đầu tư",
        height=340,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Vùng",
        yaxis_title="Chỉ số số hóa",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 6: SCENARIO COMPARISON
# ======================================================

st.markdown("---")
st.header("6. So sánh kịch bản chính sách")

pulp_no_fair, status_pulp_no_fair = solve_with_pulp(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
    use_fairness=False,
    use_region_cap=True,
)

pulp_no_cap, status_pulp_no_cap = solve_with_pulp(
    budget_total,
    region_floor,
    region_cap,
    human_floor,
    gamma_fair,
    lambda_fair,
    use_fairness=True,
    use_region_cap=False,
)

if pulp_no_fair is None:
    st.error(f"Mô hình không công bằng không có nghiệm tối ưu. Trạng thái: {status_pulp_no_fair}")
    st.stop()

alloc_no_fair = pulp_no_fair["allocation"]
Z_no_fair = pulp_no_fair["Z"]

if pulp_no_cap is not None:
    alloc_no_cap = pulp_no_cap["allocation"]
    Z_no_cap = pulp_no_cap["Z"]
else:
    alloc_no_cap = None
    Z_no_cap = np.nan

fairness_cost = Z_no_fair - Z_fair
fairness_cost_pct = fairness_cost / Z_no_fair * 100 if Z_no_fair != 0 else 0

scenario_rows = [
    {
        "Kịch bản": "Có công bằng + có trần vùng",
        "Z*": Z_fair,
        "Ghi chú": "Mô hình chính theo ràng buộc công bằng và trần vùng.",
    },
    {
        "Kịch bản": "Không công bằng + có trần vùng",
        "Z*": Z_no_fair,
        "Ghi chú": "Đường chuẩn hiệu quả khi bỏ ràng buộc công bằng.",
    },
]

if pulp_no_cap is not None:
    scenario_rows.append({
        "Kịch bản": "Có công bằng + bỏ trần vùng",
        "Z*": Z_no_cap,
        "Ghi chú": "Kiểm tra vai trò của trần ngân sách vùng.",
    })

scenario_compare = pd.DataFrame(scenario_rows)

s1, s2, s3, s4 = st.columns(4)

with s1:
    kpi_card("Z* có công bằng", f"{Z_fair:,.2f}", "Kịch bản chính.")

with s2:
    kpi_card("Z* không công bằng", f"{Z_no_fair:,.2f}", "Kịch bản bỏ ràng buộc công bằng.")

with s3:
    kpi_card("Chi phí công bằng", f"{fairness_cost:,.2f}", "Chênh lệch Z* do thêm công bằng.")

with s4:
    kpi_card("Tỷ lệ đánh đổi", f"{fairness_cost_pct:.2f}%", "Chi phí công bằng so với kịch bản không công bằng.")

compare_col, compare_chart_col = st.columns([1, 1])

with compare_col:
    st.dataframe(
        scenario_compare.style.format({"Z*": "{:,.2f}"}),
        use_container_width=True,
    )

with compare_chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=scenario_compare["Kịch bản"],
            y=scenario_compare["Z*"],
            marker_color=["#1FA7B6", "#0B1D33", "#E6F7F5"][:len(scenario_compare)],
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Kịch bản: %{x}<br>Z*: %{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="So sánh giá trị mục tiêu theo kịch bản",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=10),
        yaxis_title="Z*",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

compare_alloc = pd.DataFrame({
    "Vùng": regions,
    "Tên ngắn": region_short,
    "Có công bằng": alloc_fair.sum(axis=1).values,
    "Không công bằng": alloc_no_fair.sum(axis=1).values,
})

compare_alloc["Chênh lệch"] = compare_alloc["Có công bằng"] - compare_alloc["Không công bằng"]

st.markdown("#### Dịch chuyển ngân sách khi thêm ràng buộc công bằng")

shift_col, shift_chart_col = st.columns([1, 1])

with shift_col:
    st.dataframe(
        compare_alloc[["Vùng", "Có công bằng", "Không công bằng", "Chênh lệch"]].style.format({
            "Có công bằng": "{:,.1f}",
            "Không công bằng": "{:,.1f}",
            "Chênh lệch": "{:,.1f}",
        }),
        use_container_width=True,
    )

with shift_chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=compare_alloc["Tên ngắn"],
            x=compare_alloc["Chênh lệch"],
            orientation="h",
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Vùng: %{y}<br>Chênh lệch: %{x:,.1f}<extra></extra>",
        )
    )

    fig.add_vline(x=0, line_width=1, line_color="#334155")

    fig.update_layout(
        title="Có công bằng - Không công bằng",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Tỷ VND",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Ma trận phân bổ không có ràng buộc công bằng")

alloc_no_fair_display = make_allocation_display(alloc_no_fair)

st.dataframe(
    alloc_no_fair_display.style.format("{:,.1f}"),
    use_container_width=True,
)

# ======================================================
# SECTION 7: POLICY INTERPRETATION
# ======================================================

st.markdown("---")
st.header("7. Diễn giải và thảo luận chính sách")

weakest_digital_region = min(D0, key=D0.get)
strongest_digital_region = max(D0, key=D0.get)

top_region_fair = alloc_fair.sum(axis=1).idxmax()
top_region_no_fair = alloc_no_fair.sum(axis=1).idxmax()

item_total_fair = alloc_fair.sum(axis=0)
item_total_no_fair = alloc_no_fair.sum(axis=0)

top_item_fair = item_total_fair.idxmax()
top_item_no_fair = item_total_no_fair.idxmax()

increase_regions = compare_alloc[compare_alloc["Chênh lệch"] > 1e-6]["Vùng"].tolist()
decrease_regions = compare_alloc[compare_alloc["Chênh lệch"] < -1e-6]["Vùng"].tolist()

increase_text = ", ".join(increase_regions) if len(increase_regions) > 0 else "không có vùng nào tăng rõ rệt"
decrease_text = ", ".join(decrease_regions) if len(decrease_regions) > 0 else "không có vùng nào giảm rõ rệt"

if fairness_cost_pct <= 2:
    fairness_eval = (
        f"Chi phí công bằng chỉ khoảng {fairness_cost_pct:.2f}% so với mô hình không công bằng. "
        "Đây là mức đánh đổi thấp, cho thấy có thể tăng tính bao trùm vùng miền mà không làm giảm nhiều GDP gain kỳ vọng."
    )
elif fairness_cost_pct <= 8:
    fairness_eval = (
        f"Chi phí công bằng khoảng {fairness_cost_pct:.2f}%. Đây là mức đánh đổi trung bình: "
        "nền kinh tế phải hy sinh một phần hiệu quả ngắn hạn để đổi lấy phân bổ cân bằng hơn."
    )
else:
    fairness_eval = (
        f"Chi phí công bằng lên tới {fairness_cost_pct:.2f}%. Đây là mức đánh đổi cao, cho thấy ràng buộc công bằng đang khá chặt."
    )

if top_item_fair == "H":
    item_comment = (
        "Nhân lực số trở thành hạng mục nổi bật trong mô hình có công bằng. Điều này hợp lý vì các vùng yếu về số hóa thường cần đầu tư vào con người trước khi hấp thụ AI."
    )
elif top_item_fair == "I":
    item_comment = (
        "Hạ tầng số là hạng mục nổi bật trong mô hình có công bằng. Điều này cho thấy hạ tầng vẫn là điều kiện nền tảng để thu hẹp khoảng cách số."
    )
elif top_item_fair == "D":
    item_comment = (
        "Chuyển đổi số doanh nghiệp được ưu tiên mạnh trong mô hình có công bằng. Đây là hạng mục trực tiếp giúp cải thiện chỉ số số hóa sau đầu tư."
    )
else:
    item_comment = (
        "AI vẫn là hạng mục nổi bật trong mô hình có công bằng. Điều này cho thấy ràng buộc công bằng không loại bỏ đầu tư AI, nhưng buộc AI phải đặt trong cấu trúc phân bổ cân bằng hơn."
    )

st.markdown("#### Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về đâu?")

st.write(
    f"""
    Khi bỏ ràng buộc công bằng, mô hình đạt Z* = {Z_no_fair:,.2f}, cao hơn kịch bản có công bằng là {Z_fair:,.2f}.
    Vùng nhận tổng ngân sách lớn nhất trong mô hình có công bằng là {top_region_fair}, còn trong mô hình không công bằng là {top_region_no_fair}.
    Điều này cho thấy nếu chỉ tối đa hóa hiệu quả biên, vốn có xu hướng chảy về những vùng có hệ số tác động cao hơn,
    như Đồng bằng sông Hồng hoặc Đông Nam Bộ, tùy cấu trúc β và các ràng buộc đi kèm.
    """
)

st.write(
    f"""
    Hậu quả chính sách là khoảng cách số vùng miền có thể bị nới rộng. Vùng có chỉ số số hóa ban đầu thấp nhất là {weakest_digital_region}
    với D0 = {D0[weakest_digital_region]}, trong khi vùng cao nhất là {strongest_digital_region} với D0 = {D0[strongest_digital_region]}.
    Nếu không có cơ chế điều tiết, các vùng mạnh có thể tiếp tục thu hút nhiều vốn số hơn, trong khi vùng yếu thiếu hạ tầng,
    dữ liệu và nhân lực để bắt kịp.
    """
)

st.markdown("#### Trần ngân sách vùng có ý nghĩa gì?")

if pulp_no_cap is not None:
    cap_cost = Z_no_cap - Z_fair
    st.write(
        f"""
        Khi bỏ trần ngân sách vùng nhưng vẫn giữ ràng buộc công bằng, mô hình đạt Z* = {Z_no_cap:,.2f}.
        Chênh lệch so với mô hình có trần là {cap_cost:,.2f}. Điều này cho thấy trần ngân sách vùng có thể làm giảm một phần hiệu quả tối đa,
        nhưng đổi lại giúp tránh tập trung vốn quá mức vào một số vùng có hiệu quả biên cao.
        """
    )
else:
    st.write(
        """
        Trong bộ tham số hiện tại, kịch bản bỏ trần vùng không tìm được nghiệm tối ưu ổn định. Tuy nhiên về mặt chính sách,
        trần ngân sách vùng vẫn có thể hiểu như một cơ chế phân quyền và phân tán nguồn lực, giúp tránh tình trạng một vài vùng hấp thụ phần lớn ngân sách.
        """
    )

st.write(
    """
    Vì vậy, trần ngân sách vùng không chỉ là một giới hạn kỹ thuật trong mô hình. Nó phản ánh lựa chọn chính sách:
    Nhà nước có thể chấp nhận giảm một phần Z* để đổi lấy phân bổ cân bằng hơn, tăng tính chính danh của đầu tư công
    và bảo đảm các vùng yếu vẫn có cơ hội nhận nguồn lực.
    """
)

st.markdown("#### Nên đầu tư gì ở Tây Nguyên?")

st.write(
    """
    Với Tây Nguyên, hệ số AI trong ma trận β tương đối thấp, trong khi D0 ban đầu cũng thấp. Do đó, không nên hiểu kết quả theo hướng
    ưu tiên triển khai AI quy mô lớn ngay từ đầu. Hợp lý hơn là đầu tư vào hạ tầng số và nhân lực số trước, tức tăng nền tảng kết nối,
    kỹ năng số, năng lực vận hành dữ liệu và khả năng hấp thụ công nghệ.
    """
)

st.write(
    """
    Khi hạ tầng và nhân lực đã cải thiện, các khoản đầu tư AI sau đó mới có khả năng tạo tác động thực chất hơn.
    Cách tiếp cận này phù hợp với logic chính sách vùng: vùng yếu không nhất thiết cần công nghệ phức tạp nhất ngay lập tức,
    mà cần điều kiện nền tảng để không bị bỏ lại phía sau trong quá trình chuyển đổi số.
    """
)

st.markdown("#### Đánh đổi hiệu quả và công bằng")

st.write(
    f"""
    Phần chênh lệch {fairness_cost:,.2f} giữa mô hình không công bằng và mô hình có công bằng có thể hiểu là chi phí kinh tế của công bằng vùng miền.
    {fairness_eval} Khi thêm ràng buộc công bằng, các vùng được tăng phân bổ tương đối gồm: {increase_text}.
    Các vùng bị giảm phân bổ tương đối gồm: {decrease_text}.
    """
)

st.write(
    f"""
    Trong mô hình có công bằng, hạng mục nhận tổng ngân sách lớn nhất là {item_names[top_item_fair]}. {item_comment}
    Trong mô hình không công bằng, hạng mục nổi bật nhất là {item_names[top_item_no_fair]}.
    Sự khác biệt này cho thấy ràng buộc công bằng không chỉ thay đổi phân bổ theo vùng, mà còn làm thay đổi cả cấu trúc ưu tiên giữa hạ tầng,
    chuyển đổi số doanh nghiệp, AI và nhân lực số.
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
    Lưu ý: Kết quả của Bài 4 là mô phỏng phục vụ phân tích. Mô hình LP giúp lượng hóa đánh đổi giữa hiệu quả và công bằng vùng miền,
    nhưng không thay thế quyết định ngân sách thực tế. Khi ứng dụng thực tế, cần thay thế bằng số liệu chính thức, đánh giá năng lực hấp thụ của từng vùng
    và tham vấn cơ quan quản lý địa phương.
    </p>
    """,
    unsafe_allow_html=True,
)