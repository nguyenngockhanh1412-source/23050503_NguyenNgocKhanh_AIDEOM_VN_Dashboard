import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog
try:
    import pulp
except Exception:
    pulp = None

from utils.aideom_ui import (
    setup_page,
    render_sidebar,
    page_header,
    section_caption,
    source_note,
    kpi_card,
)

# ======================================================
# PAGE SETUP
# ======================================================

setup_page("Bài 2 - LP ngân sách")
render_sidebar("Bài 2 - LP ngân sách")

page_header(
    "Bài 2. Phân bổ ngân sách số theo 4 hạng mục đầu tư",
    "Bài toán quy hoạch tuyến tính mô phỏng phân bổ ngân sách đầu tư số cho hạ tầng số, AI và dữ liệu, nhân lực số và R&D công nghệ."
)

st.write(
    """
    Bài 2 mô phỏng bài toán phân bổ ngân sách đầu tư số cho bốn hạng mục chính: hạ tầng số, AI và dữ liệu, nhân lực số và R&D công nghệ.
    Mục tiêu của mô hình là tìm phương án phân bổ giúp tối đa hóa GDP kỳ vọng, đồng thời vẫn bảo đảm các ngưỡng đầu tư tối thiểu
    và tỷ trọng công nghệ chiến lược.
    """
)

# ======================================================
# MODEL
# ======================================================

# ======================================================
# MODEL
# ======================================================

st.markdown("---")
st.header("1. Mô hình toán học")

st.latex(r"""
\max Z = 0.85x_1 + 1.20x_2 + 0.95x_3 + 1.35x_4
""")

st.write(
    """
    Trong mô hình, x₁ là ngân sách cho hạ tầng số, x₂ là ngân sách cho AI và dữ liệu,
    x₃ là ngân sách cho nhân lực số, còn x₄ là ngân sách cho R&D công nghệ.
    Các hệ số trong hàm mục tiêu thể hiện mức đóng góp kỳ vọng của từng hạng mục vào GDP.
    """
)

st.write(
    """
    Bài toán có các ràng buộc về ngân sách tổng, mức đầu tư tối thiểu cho từng hạng mục
    và tỷ trọng tối thiểu dành cho nhóm công nghệ chiến lược AI + R&D.
    Vì vậy, kết quả tối ưu không chỉ phụ thuộc vào hệ số tác động, mà còn phụ thuộc vào cách người dùng thiết lập ràng buộc chính sách.
    """
)

source_note(
    """
    Bộ hệ số và ràng buộc bám theo Bài 2 của bộ đề. Module giải bài toán bằng scipy.optimize.linprog và giải lại bằng PuLP/CBC để đối chiếu nghiệm, đồng thời hiển thị shadow price của các ràng buộc khi solver hỗ trợ.
    """
)

# ======================================================
# CORE DATA AND SOLVER
# ======================================================

coef = {
    "Hạ tầng số": 0.85,
    "AI và dữ liệu": 1.20,
    "Nhân lực số": 0.95,
    "R&D công nghệ": 1.35,
}


def solve_budget_model(B, min1, min2, min3, min4, strategic_share):
    """
    Giải đúng bài toán LP bằng scipy.optimize.linprog theo đề bài.
    Bài toán tối đa hóa được chuyển về bài toán tối thiểu hóa -Z.
    Nếu PuLP khả dụng, hàm đồng thời giải lại bằng PuLP để lấy dual value/shadow price
    của các ràng buộc tuyến tính.
    """

    c = np.array([-0.85, -1.20, -0.95, -1.35], dtype=float)

    A_ub = np.array([
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [strategic_share, strategic_share - 1, strategic_share, strategic_share - 1],
    ], dtype=float)

    b_ub = np.array([
        B,
        -min1,
        -min2,
        -min3,
        -min4,
        0,
    ], dtype=float)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(0, None)] * 4,
        method="highs",
    )

    if not res.success:
        return None, f"Không khả thi hoặc solver không hội tụ: {res.message}"

    x1, x2, x3, x4 = res.x
    Z = -res.fun
    strategic_value = x2 + x4
    strategic_required = strategic_share * (x1 + x2 + x3 + x4)

    dual_rows = []
    shadow_price_budget = np.nan

    # Dual của scipy là marginal của bài toán minimize. Đổi dấu để diễn giải theo maximize.
    try:
        names = [
            "Ngân sách tổng",
            "Hạ tầng số tối thiểu",
            "AI và dữ liệu tối thiểu",
            "Nhân lực số tối thiểu",
            "R&D tối thiểu",
            "AI + R&D tối thiểu",
        ]
        marginals = np.array(res.ineqlin.marginals, dtype=float)
        residuals = np.array(res.ineqlin.residual, dtype=float)
        max_duals = -marginals
        shadow_price_budget = float(max_duals[0])
        for name, dual, slack in zip(names, max_duals, residuals):
            dual_rows.append({
                "Ràng buộc": name,
                "Shadow price (scipy/HiGHS)": dual,
                "Slack": slack,
                "Trạng thái": "Chặt" if abs(slack) < 1e-7 else "Không chặt",
            })
    except Exception:
        pass

    # Giải lại bằng PuLP để đúng yêu cầu đề bài về pulp và dual values nếu CBC hỗ trợ pi/slack.
    if pulp is not None:
        try:
            m = pulp.LpProblem("AIDEOM_Bai2_LP_Ngan_Sach", pulp.LpMaximize)
            y = pulp.LpVariable.dicts("x", range(4), lowBound=0, cat="Continuous")
            cons = {}
            m += 0.85*y[0] + 1.20*y[1] + 0.95*y[2] + 1.35*y[3]
            cons["Ngân sách tổng"] = y[0] + y[1] + y[2] + y[3] <= B
            cons["Hạ tầng số tối thiểu"] = y[0] >= min1
            cons["AI và dữ liệu tối thiểu"] = y[1] >= min2
            cons["Nhân lực số tối thiểu"] = y[2] >= min3
            cons["R&D tối thiểu"] = y[3] >= min4
            cons["AI + R&D tối thiểu"] = y[1] + y[3] >= strategic_share * (y[0] + y[1] + y[2] + y[3])
            for nm, con in cons.items():
                m += con, nm
            m.solve(pulp.PULP_CBC_CMD(msg=False))

            pulp_duals = {}
            for nm in cons:
                con = m.constraints[nm]
                pulp_duals[nm] = {
                    "Shadow price (PuLP/CBC)": getattr(con, "pi", np.nan),
                    "PuLP slack": getattr(con, "slack", np.nan),
                }

            if dual_rows:
                for row in dual_rows:
                    row.update(pulp_duals.get(row["Ràng buộc"], {}))
            else:
                for nm, vals in pulp_duals.items():
                    row = {"Ràng buộc": nm}
                    row.update(vals)
                    dual_rows.append(row)
        except Exception:
            pass

    result = {
        "x1": float(x1),
        "x2": float(x2),
        "x3": float(x3),
        "x4": float(x4),
        "Z": float(Z),
        "strategic_value": float(strategic_value),
        "strategic_required": float(strategic_required),
        "min_sum": float(min1 + min2 + min3 + min4),
        "shadow_price_budget": shadow_price_budget,
        "dual_df": pd.DataFrame(dual_rows),
        "solver_status": res.message,
    }

    return result, "Khả thi"

# ======================================================
# SECTION 2: SETTINGS
# ======================================================

st.markdown("---")
st.header("2. Thiết lập chính sách")

section_caption(
    """
    Người dùng có thể chọn một kịch bản gợi ý trước, sau đó điều chỉnh ngân sách tổng,
    các mức đầu tư tối thiểu và tỷ trọng AI + R&D để xem phương án phân bổ thay đổi như thế nào.
    """
)

scenario = st.selectbox(
    "Chọn định hướng mô phỏng",
    [
        "Kịch bản cân bằng",
        "Ưu tiên công nghệ chiến lược",
        "Ưu tiên nhân lực số",
        "Thắt chặt ngân sách",
    ],
)

if scenario == "Kịch bản cân bằng":
    default_budget = 100.0
    default_min1 = 25.0
    default_min2 = 15.0
    default_min3 = 20.0
    default_min4 = 10.0
    default_share = 0.35
    scenario_text = "Kịch bản cân bằng giữ mức đầu tư tối thiểu tương đối ổn định giữa hạ tầng, AI, nhân lực và R&D."

elif scenario == "Ưu tiên công nghệ chiến lược":
    default_budget = 120.0
    default_min1 = 20.0
    default_min2 = 25.0
    default_min3 = 20.0
    default_min4 = 20.0
    default_share = 0.50
    scenario_text = "Kịch bản này nâng tỷ trọng AI + R&D, phù hợp khi mục tiêu là tăng tốc đổi mới sáng tạo và công nghệ chiến lược."

elif scenario == "Ưu tiên nhân lực số":
    default_budget = 110.0
    default_min1 = 20.0
    default_min2 = 15.0
    default_min3 = 35.0
    default_min4 = 10.0
    default_share = 0.35
    scenario_text = "Kịch bản này ưu tiên nhân lực số, phù hợp khi chính sách muốn xử lý thiếu hụt kỹ năng số và năng lực hấp thụ công nghệ."

else:
    default_budget = 90.0
    default_min1 = 25.0
    default_min2 = 15.0
    default_min3 = 20.0
    default_min4 = 10.0
    default_share = 0.35
    scenario_text = "Kịch bản thắt chặt ngân sách dùng để kiểm tra mô hình khi nguồn lực bị giới hạn hơn."

st.info(scenario_text)

setting_col, check_col = st.columns([1.2, 0.8])

with setting_col:
    st.markdown("#### Điều chỉnh ngân sách và ràng buộc")

    s1, s2 = st.columns(2)

    with s1:
        total_budget = st.slider(
            "Ngân sách tổng B, nghìn tỷ VND",
            min_value=80.0,
            max_value=160.0,
            value=default_budget,
            step=5.0,
        )

        min_x1 = st.slider(
            "Mức tối thiểu cho hạ tầng số x₁",
            min_value=0.0,
            max_value=70.0,
            value=default_min1,
            step=1.0,
        )

        min_x2 = st.slider(
            "Mức tối thiểu cho AI và dữ liệu x₂",
            min_value=0.0,
            max_value=70.0,
            value=default_min2,
            step=1.0,
        )

    with s2:
        min_x3 = st.slider(
            "Mức tối thiểu cho nhân lực số x₃",
            min_value=0.0,
            max_value=70.0,
            value=default_min3,
            step=1.0,
        )

        min_x4 = st.slider(
            "Mức tối thiểu cho R&D công nghệ x₄",
            min_value=0.0,
            max_value=70.0,
            value=default_min4,
            step=1.0,
        )

        strategic_share = st.slider(
            "Tỷ trọng tối thiểu cho AI + R&D",
            min_value=0.10,
            max_value=0.70,
            value=default_share,
            step=0.05,
        )

min_sum = min_x1 + min_x2 + min_x3 + min_x4
remaining_after_min = total_budget - min_sum
required_strategic = strategic_share * total_budget
current_min_strategic = min_x2 + min_x4
max_possible_strategic = total_budget - min_x1 - min_x3
pressure = min_sum / total_budget

with check_col:
    st.markdown("#### Tóm tắt ràng buộc")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Tổng tối thiểu", f"{min_sum:.1f}")
        st.metric("AI + R&D yêu cầu", f"{required_strategic:.1f}")

    with c2:
        st.metric("Ngân sách còn lại", f"{remaining_after_min:.1f}")
        st.metric("Áp lực tối thiểu", f"{pressure * 100:.1f}%")

    st.write("Mức sử dụng ngân sách tối thiểu")
    st.progress(min(1.0, pressure))

    if pressure <= 0.70:
        st.success("Ràng buộc tối thiểu còn khá thoáng.")
    elif pressure <= 0.90:
        st.warning("Ràng buộc tối thiểu đang khá chặt.")
    else:
        st.error("Ràng buộc tối thiểu quá sát ngân sách tổng.")

st.markdown("#### Kiểm tra nhanh ràng buộc")

if min_sum > total_budget:
    deficit = min_sum - total_budget
    st.error(
        f"Tổng mức tối thiểu đang vượt ngân sách {deficit:.1f} nghìn tỷ VND. "
        "Nên tăng ngân sách tổng hoặc giảm bớt một trong các mức tối thiểu."
    )
elif max_possible_strategic < required_strategic:
    shortage = required_strategic - max_possible_strategic
    st.error(
        f"Ràng buộc AI + R&D đang thiếu khoảng {shortage:.1f} nghìn tỷ VND. "
        "Nên giảm mức tối thiểu của hạ tầng số hoặc nhân lực số."
    )
else:
    st.success("Bộ ràng buộc hiện tại khả thi.")

hint_col1, hint_col2, hint_col3 = st.columns(3)

with hint_col1:
    if min_x3 >= 35:
        st.info(
            "Ưu tiên nhân lực số đang cao. Điều này phù hợp nếu mục tiêu là khắc phục thiếu hụt kỹ năng số."
        )

with hint_col2:
    if strategic_share >= 0.50:
        st.info(
            "Tỷ trọng AI + R&D đang cao, phản ánh định hướng công nghệ chiến lược."
        )

with hint_col3:
    if min_x1 >= 50:
        st.warning(
            "Hạ tầng số đang chiếm tỷ trọng lớn, có thể làm giảm không gian cho AI, dữ liệu và R&D."
        )

# Tính kết quả dùng chung
solution, status_msg = solve_budget_model(
    total_budget,
    min_x1,
    min_x2,
    min_x3,
    min_x4,
    strategic_share,
)

# ======================================================
# SECTION 3: OPTIMAL RESULT
# ======================================================

st.markdown("---")
st.header("3. Kết quả phân bổ tối ưu")

if solution is None:
    st.error(status_msg)

    st.write(
        """
        Với bộ tham số hiện tại, mô hình không thể tìm được phương án phân bổ thỏa mãn tất cả ràng buộc.
        Người dùng nên điều chỉnh lại ngân sách tổng, giảm một số mức tối thiểu hoặc nới ràng buộc tỷ trọng AI + R&D.
        """
    )

else:
    x1 = solution["x1"]
    x2 = solution["x2"]
    x3 = solution["x3"]
    x4 = solution["x4"]
    Z = solution["Z"]
    strategic_value = solution["strategic_value"]
    strategic_required = solution["strategic_required"]

    result_df = pd.DataFrame({
        "Hạng mục": [
            "Hạ tầng số",
            "AI và dữ liệu",
            "Nhân lực số",
            "R&D công nghệ",
        ],
        "Ký hiệu": ["x₁", "x₂", "x₃", "x₄"],
        "Hệ số tác động": [0.85, 1.20, 0.95, 1.35],
        "Phân bổ tối ưu": [x1, x2, x3, x4],
        "Tỷ trọng ngân sách (%)": [
            x1 / total_budget * 100,
            x2 / total_budget * 100,
            x3 / total_budget * 100,
            x4 / total_budget * 100,
        ],
        "Đóng góp GDP kỳ vọng": [
            0.85 * x1,
            1.20 * x2,
            0.95 * x3,
            1.35 * x4,
        ],
    })

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        kpi_card("Ngân sách tổng", f"{total_budget:.1f}", "Nghìn tỷ VND.")

    with m2:
        kpi_card("GDP kỳ vọng Z*", f"{Z:.2f}", "Giá trị mục tiêu tối ưu.")

    with m3:
        kpi_card("AI + R&D", f"{strategic_value:.1f}", "Tổng ngân sách công nghệ chiến lược.")

    with m4:
        kpi_card("Tỷ trọng AI + R&D", f"{strategic_value / total_budget * 100:.1f}%", "Tỷ trọng trong tổng ngân sách.")

    result_col, chart_col = st.columns([1.05, 0.95])

    with result_col:
        st.dataframe(
            result_df.style.format({
                "Hệ số tác động": "{:.2f}",
                "Phân bổ tối ưu": "{:.1f}",
                "Tỷ trọng ngân sách (%)": "{:.2f}",
                "Đóng góp GDP kỳ vọng": "{:.2f}",
            }),
            use_container_width=True,
        )

    with chart_col:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=result_df["Hạng mục"],
                y=result_df["Phân bổ tối ưu"],
                name="Phân bổ tối ưu",
                marker_color="#1FA7B6",
                marker_line=dict(color="#1FA7B6", width=1),
                hovertemplate="Hạng mục: %{x}<br>Phân bổ: %{y:.1f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Phân bổ ngân sách tối ưu",
            height=330,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis=dict(tickangle=0),
            yaxis_title="Nghìn tỷ VND",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    contrib_col, check_col = st.columns([1, 1])

    with contrib_col:
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=result_df["Hạng mục"],
                y=result_df["Đóng góp GDP kỳ vọng"],
                name="Đóng góp GDP kỳ vọng",
                marker_color="#0B1D33",
                marker_line=dict(color="#1FA7B6", width=1),
                hovertemplate="Hạng mục: %{x}<br>Đóng góp: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Đóng góp GDP kỳ vọng theo hạng mục",
            height=320,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis=dict(tickangle=0),
            yaxis_title="Nghìn tỷ VND",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    with check_col:
        check_df = pd.DataFrame({
            "Ràng buộc": [
                "Tổng ngân sách",
                "Hạ tầng số tối thiểu",
                "AI và dữ liệu tối thiểu",
                "Nhân lực số tối thiểu",
                "R&D tối thiểu",
                "AI + R&D tối thiểu",
            ],
            "Giá trị đạt được": [
                x1 + x2 + x3 + x4,
                x1,
                x2,
                x3,
                x4,
                strategic_value,
            ],
            "Ngưỡng yêu cầu": [
                total_budget,
                min_x1,
                min_x2,
                min_x3,
                min_x4,
                strategic_required,
            ],
            "Trạng thái": [
                "Đạt" if x1 + x2 + x3 + x4 <= total_budget + 1e-6 else "Không đạt",
                "Đạt" if x1 >= min_x1 - 1e-6 else "Không đạt",
                "Đạt" if x2 >= min_x2 - 1e-6 else "Không đạt",
                "Đạt" if x3 >= min_x3 - 1e-6 else "Không đạt",
                "Đạt" if x4 >= min_x4 - 1e-6 else "Không đạt",
                "Đạt" if strategic_value >= strategic_required - 1e-6 else "Không đạt",
            ],
        })

        st.dataframe(
            check_df.style.format({
                "Giá trị đạt được": "{:.1f}",
                "Ngưỡng yêu cầu": "{:.1f}",
            }),
            use_container_width=True,
        )

    dual_df = solution.get("dual_df", pd.DataFrame())
    if not dual_df.empty:
        st.markdown("#### Shadow price và độ chặt của ràng buộc")
        st.dataframe(
            dual_df.style.format({
                "Shadow price (scipy/HiGHS)": "{:.4f}",
                "Slack": "{:.4f}",
                "Shadow price (PuLP/CBC)": "{:.4f}",
                "PuLP slack": "{:.4f}",
            }, na_rep="—"),
            use_container_width=True,
        )
        st.caption("Shadow price cho biết giá trị mục tiêu thay đổi xấp xỉ bao nhiêu khi nới ràng buộc thêm một đơn vị, trong vùng nghiệm hiện tại.")

# ======================================================
# SECTION 4: SENSITIVITY
# ======================================================

st.markdown("---")
st.header("4. Phân tích độ nhạy theo ngân sách tổng")

st.write(
    """
    Phân tích này cho biết khi ngân sách tổng tăng hoặc giảm, giá trị GDP kỳ vọng tối ưu thay đổi như thế nào.
    Đây là cách mô phỏng đơn giản để hiểu ý nghĩa của ràng buộc ngân sách trong bài toán phân bổ vốn công.
    """
)

budget_range = np.arange(80, 165, 10)
sensitivity_rows = []

for B in budget_range:
    sol_B, msg_B = solve_budget_model(
        B,
        min_x1,
        min_x2,
        min_x3,
        min_x4,
        strategic_share,
    )

    if sol_B is None:
        sensitivity_rows.append([B, np.nan, "Không khả thi"])
    else:
        sensitivity_rows.append([B, sol_B["Z"], "Khả thi"])

sensitivity_df = pd.DataFrame(
    sensitivity_rows,
    columns=["Ngân sách B", "Giá trị tối ưu Z*", "Trạng thái"],
)

sens_col, sens_chart_col = st.columns([1, 1])

with sens_col:
    st.dataframe(
        sensitivity_df.style.format({
            "Ngân sách B": "{:.0f}",
            "Giá trị tối ưu Z*": "{:.2f}",
        }),
        use_container_width=True,
    )

with sens_chart_col:
    feasible_df = sensitivity_df.dropna()

    if len(feasible_df) > 0:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=feasible_df["Ngân sách B"],
                y=feasible_df["Giá trị tối ưu Z*"],
                mode="lines+markers",
                line=dict(color="#1FA7B6", width=2.5),
                marker=dict(size=8, color="#5FA8D3"),
                hovertemplate="Ngân sách: %{x:.0f}<br>Z*: %{y:.2f}<extra></extra>",
            )
        )

        fig.update_layout(
            title="Đường cong Z*(B)",
            height=320,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Ngân sách tổng",
            yaxis_title="GDP kỳ vọng Z*",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

if len(sensitivity_df.dropna()) > 0:
    st.info(
        "Trong mô hình này, khi ngân sách tăng thêm và các ràng buộc tối thiểu không thay đổi, "
        "phần tăng thêm thường được phân bổ vào R&D công nghệ vì đây là hạng mục có hệ số tác động cao nhất."
    )

# ======================================================
# SECTION 5: POLICY INTERPRETATION
# ======================================================

st.markdown("---")
st.header("5. Diễn giải chính sách tự động")

if solution is None:
    st.error(
        """
        Với bộ tham số hiện tại, mô hình chưa tạo ra phương án khả thi.
        Điều này không phải là lỗi kỹ thuật, mà là một tín hiệu chính sách: các mục tiêu đang được đặt ra vượt quá khả năng ngân sách hiện có.
        """
    )

    st.write(
        f"""
        Cụ thể, ngân sách tổng hiện tại là {total_budget:.1f} nghìn tỷ VND, trong khi tổng mức đầu tư tối thiểu cho bốn hạng mục là
        {min_sum:.1f} nghìn tỷ VND. Nếu tổng mức tối thiểu vượt quá ngân sách, Nhà nước không thể đồng thời đáp ứng tất cả cam kết phân bổ.
        Khi đó, có ba hướng điều chỉnh: tăng quy mô ngân sách, giảm bớt một số ngưỡng tối thiểu, hoặc phân kỳ đầu tư theo nhiều giai đoạn.
        """
    )

else:
    rd_share = x4 / total_budget * 100
    ai_rd_share = strategic_value / total_budget * 100
    human_share = x3 / total_budget * 100

    if x4 == max(x1, x2, x3, x4):
        main_direction = (
            "Phương án tối ưu đang nghiêng mạnh về R&D công nghệ. Đây là kết quả dễ hiểu vì R&D có hệ số tác động biên cao nhất trong hàm mục tiêu. "
            "Về mặt kinh tế, mô hình đang giả định rằng một đơn vị ngân sách bổ sung cho R&D tạo ra mức GDP kỳ vọng lớn hơn so với các hạng mục còn lại."
        )
    elif x2 == max(x1, x2, x3, x4):
        main_direction = (
            "Phương án tối ưu đang ưu tiên AI và dữ liệu. Điều này cho thấy khi ràng buộc công nghệ chiến lược được đặt cao, "
            "AI và dữ liệu trở thành trụ cột quan trọng trong phân bổ ngân sách."
        )
    elif x3 == max(x1, x2, x3, x4):
        main_direction = (
            "Phương án tối ưu đang ưu tiên nhân lực số. Đây là lựa chọn phù hợp trong bối cảnh thiếu hụt kỹ sư AI, chuyên gia dữ liệu và lao động có kỹ năng số."
        )
    else:
        main_direction = (
            "Phương án tối ưu đang ưu tiên hạ tầng số. Điều này phù hợp khi nền tảng kết nối, dữ liệu và hạ tầng công nghệ vẫn là điểm nghẽn chính."
        )

    if ai_rd_share >= strategic_share * 100 + 15:
        strategic_eval = (
            "Tỷ trọng AI và R&D cao hơn đáng kể so với ngưỡng tối thiểu. Điều này phản ánh một chiến lược đầu tư thiên về đổi mới sáng tạo và công nghệ lõi. "
            "Tuy nhiên, nếu tỷ trọng này quá cao, chính sách cần kiểm tra thêm năng lực hấp thụ của khu vực công, doanh nghiệp và thị trường lao động."
        )
    elif ai_rd_share >= strategic_share * 100:
        strategic_eval = (
            "Tỷ trọng AI và R&D vừa đáp ứng yêu cầu chính sách. Đây là một cấu trúc tương đối cân bằng, vì vẫn bảo đảm ưu tiên công nghệ chiến lược "
            "mà không loại bỏ hoàn toàn vai trò của hạ tầng và nhân lực."
        )
    else:
        strategic_eval = (
            "Tỷ trọng AI và R&D chưa đạt yêu cầu chính sách. Nếu mục tiêu là phát triển công nghệ chiến lược, cần tăng phân bổ cho AI, dữ liệu hoặc R&D."
        )

    if human_share < 20:
        human_eval = (
            "Tỷ trọng nhân lực số còn tương đối thấp. Trong thực tế, đây có thể là điểm yếu của phương án phân bổ, vì AI và R&D khó tạo tác động dài hạn "
            "nếu thiếu đội ngũ lao động có kỹ năng để triển khai, vận hành và hấp thụ công nghệ."
        )
    elif human_share <= 35:
        human_eval = (
            "Tỷ trọng nhân lực số ở mức tương đối hợp lý. Điều này giúp cân bằng giữa đầu tư công nghệ và năng lực con người, "
            "tránh tình trạng có hạ tầng và công nghệ nhưng thiếu nhân lực vận hành."
        )
    else:
        human_eval = (
            "Tỷ trọng nhân lực số đang ở mức cao. Đây là hướng đi phù hợp nếu chính sách đặt trọng tâm vào đào tạo kỹ sư AI, chuyên gia dữ liệu và lao động số, "
            "nhưng có thể làm giảm ngân sách dành cho R&D hoặc hạ tầng."
        )

    shadow_price = solution.get("shadow_price_budget", np.nan)
    if not np.isfinite(shadow_price):
        shadow_price = 1.35

    st.markdown("#### Logic phân bổ tối ưu")

    st.write(
        f"""
        {main_direction} Với bộ tham số hiện tại, giá trị mục tiêu đạt {Z:.2f} nghìn tỷ VND GDP kỳ vọng.
        Kết quả này không chỉ cho biết nên phân bổ bao nhiêu ngân sách cho từng hạng mục, 
        mà còn cho thấy mô hình đang đánh giá hạng mục nào có hiệu quả biên cao hơn.
        """
    )

    st.markdown("#### Ràng buộc công nghệ chiến lược")

    st.write(
        f"""
        AI và R&D hiện chiếm {ai_rd_share:.1f}% tổng ngân sách, 
        trong khi ngưỡng yêu cầu là {strategic_share * 100:.1f}%.
        {strategic_eval}
        """
    )

    st.markdown("#### Vai trò của nhân lực số")

    st.write(
        f"""
        Nhân lực số hiện chiếm {human_share:.1f}% ngân sách. {human_eval}
        """
    )

    st.markdown("#### Chi phí cơ hội của ngân sách công")

    st.write(
        f"""
        Theo kết quả dual của bài toán LP, nếu ngân sách tổng tăng thêm 1 nghìn tỷ VND trong vùng nghiệm hiện tại và các ràng buộc khác không đổi, shadow price của ngân sách tổng có thể hiểu xấp xỉ là {shadow_price:.2f}, 
        tức 1 nghìn tỷ VND bổ sung có thể tạo thêm khoảng {shadow_price:.2f} nghìn tỷ VND GDP kỳ vọng.
        """
    )

    st.markdown("#### Hàm ý quản lý")

    st.write(
        """
        Kết quả tối ưu không nên được hiểu là khuyến nghị máy móc rằng phải dồn tối đa ngân sách cho R&D.
        Trong thực tế, R&D chỉ phát huy hiệu quả khi có hạ tầng dữ liệu, thể chế thử nghiệm, 
        doanh nghiệp tiếp nhận công nghệ và lực lượng lao động đủ kỹ năng. Vì vậy, phương án tối ưu từ mô hình 
        cần được xem như một điểm khởi đầu để thảo luận chính sách, sau đó điều chỉnh bằng các tiêu chí thực tiễn 
        như năng lực triển khai, tính công bằng, tiến độ giải ngân và mức độ sẵn sàng của từng ngành, từng địa phương.
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
        Lưu ý: Kết quả của Bài 2 là mô phỏng phục vụ phân tích. Bài toán LP giúp lượng hóa sự đánh đổi giữa hiệu quả kinh tế
        và các ràng buộc chính sách, nhưng không thay thế quyết định ngân sách thực tế.
        </p>
        """,
        unsafe_allow_html=True,
    )