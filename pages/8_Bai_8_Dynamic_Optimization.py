import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

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

setup_page("Bài 8 - Tối ưu động")
render_sidebar("Bài 8 - Tối ưu động")

st.title("Bài 8. Tối ưu động phân bổ liên thời gian 2026–2035")

st.write(
    """
    Dashboard mô phỏng bài toán phân bổ đầu tư dài hạn giữa vốn vật chất, hạ tầng số, AI và nhân lực.
    Trọng tâm là lựa chọn quỹ đạo đầu tư theo thời gian, thay vì chỉ chọn mức đầu tư tối ưu cho một năm riêng lẻ.
    """
)

source_note(
    """
    Dữ liệu điều kiện ban đầu và tham số mô hình được xây dựng theo yêu cầu Bài 8 trong bộ đề.
    Kết quả là mô phỏng phục vụ phân tích, không thay thế dự báo kinh tế chính thức.
    """
)

# ======================================================
# MODEL SETUP
# ======================================================

years = np.arange(2026, 2036)
T = len(years)

K0 = 27500.0
L0 = 53.9
D0 = 20.9
AI0 = 86.0
H0 = 30.0
A0 = 1.0

delta_k = 0.05
delta_d = 0.12
delta_ai = 0.15
theta_h = 0.80
mu = 0.02

phi1 = 0.003
phi2 = 0.002
phi3 = 0.004

alpha_k = 0.33
alpha_l = 0.42
alpha_d = 0.10
alpha_ai = 0.08
alpha_h = 0.07

investment_labels = ["Vốn vật chất", "Hạ tầng số", "AI", "Nhân lực"]
investment_cols = ["sK", "sD", "sAI", "sH"]
investment_flow_cols = ["IK", "ID", "IAI", "IH"]

item_colors = {
    "Vốn vật chất": "#0B1D33",
    "Hạ tầng số": "#1FA7B6",
    "AI": "#FF6B6B",
    "Nhân lực": "#E6F7F5",
}

# ======================================================
# FUNCTIONS
# ======================================================

def production(A, K, L, D, AI, H):
    return A * (K ** alpha_k) * (L ** alpha_l) * (D ** alpha_d) * (AI ** alpha_ai) * (H ** alpha_h)


def utility(C, utility_type, gamma):
    C = np.maximum(C, 1e-9)
    if utility_type == "Log utility":
        return np.log(C)
    return (C ** (1 - gamma)) / (1 - gamma)


def simulate_path(
    shares,
    rho=0.97,
    utility_type="Log utility",
    gamma=1.5,
    shock=False,
    shock_year=2028,
    shock_size=0.08,
):
    shares = np.array(shares).reshape(T, 4)

    K = np.zeros(T + 1)
    D = np.zeros(T + 1)
    AI = np.zeros(T + 1)
    H = np.zeros(T + 1)
    A = np.zeros(T + 1)

    Y = np.zeros(T)
    C = np.zeros(T)
    welfare_terms = np.zeros(T)

    IK = np.zeros(T)
    ID = np.zeros(T)
    IAI = np.zeros(T)
    IH = np.zeros(T)

    K[0] = K0
    D[0] = D0
    AI[0] = AI0
    H[0] = H0
    A[0] = A0

    for t in range(T):
        Y[t] = production(A[t], K[t], L0, D[t], AI[t], H[t])

        if shock and years[t] == shock_year:
            Y[t] = Y[t] * (1 - shock_size)

        total_share = shares[t].sum()
        C[t] = (1 - total_share) * Y[t]

        IK[t] = shares[t, 0] * Y[t]
        ID[t] = shares[t, 1] * Y[t]
        IAI[t] = shares[t, 2] * Y[t]
        IH[t] = shares[t, 3] * Y[t]

        welfare_terms[t] = (rho ** t) * utility(C[t], utility_type, gamma)

        K[t + 1] = (1 - delta_k) * K[t] + IK[t]
        D[t + 1] = (1 - delta_d) * D[t] + ID[t]
        AI[t + 1] = (1 - delta_ai) * AI[t] + IAI[t]
        H[t + 1] = H[t] + theta_h * IH[t] - mu * H[t]

        A[t + 1] = A[t] * (1 + phi1 * D[t] + phi2 * AI[t] + phi3 * H[t])

    result = pd.DataFrame({
        "year": years,
        "Y": Y,
        "C": C,
        "K": K[:-1],
        "D": D[:-1],
        "AI": AI[:-1],
        "H": H[:-1],
        "A": A[:-1],
        "IK": IK,
        "ID": ID,
        "IAI": IAI,
        "IH": IH,
        "sK": shares[:, 0],
        "sD": shares[:, 1],
        "sAI": shares[:, 2],
        "sH": shares[:, 3],
        "total_investment_share": shares.sum(axis=1),
        "welfare_term": welfare_terms,
    })

    total_welfare = welfare_terms.sum()
    return result, total_welfare


def objective(x, rho, utility_type, gamma):
    _, welfare = simulate_path(
        x,
        rho=rho,
        utility_type=utility_type,
        gamma=gamma,
        shock=False,
    )
    return -welfare


def solve_dynamic_model(rho=0.97, utility_type="Log utility", gamma=1.5):
    x0 = np.tile(np.array([0.10, 0.08, 0.07, 0.08]), T)

    bounds = []
    for _ in range(T):
        bounds.extend([
            (0.04, 0.22),
            (0.03, 0.18),
            (0.02, 0.18),
            (0.03, 0.20),
        ])

    constraints = []

    for t in range(T):
        idx = 4 * t

        constraints.append({
            "type": "ineq",
            "fun": lambda x, idx=idx: 0.62 - np.sum(x[idx:idx + 4]),
        })

        constraints.append({
            "type": "ineq",
            "fun": lambda x, idx=idx: np.sum(x[idx:idx + 4]) - 0.18,
        })

        constraints.append({
            "type": "ineq",
            "fun": lambda x, idx=idx: x[idx + 3] - 0.04,
        })

        constraints.append({
            "type": "ineq",
            "fun": lambda x, idx=idx: (x[idx + 1] + x[idx + 2]) - 0.08,
        })

    result = minimize(
        objective,
        x0,
        args=(rho, utility_type, gamma),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1200,
            "ftol": 1e-8,
            "disp": False,
        },
    )

    path, welfare = simulate_path(
        result.x,
        rho=rho,
        utility_type=utility_type,
        gamma=gamma,
        shock=False,
    )

    return result, path, welfare


def build_equal_strategy():
    return np.tile(np.array([0.10, 0.08, 0.07, 0.08]), T)


def build_front_loaded_strategy():
    shares = np.zeros((T, 4))
    for t in range(T):
        if t <= 2:
            shares[t] = np.array([0.12, 0.12, 0.11, 0.11])
        elif t <= 5:
            shares[t] = np.array([0.10, 0.08, 0.07, 0.08])
        else:
            shares[t] = np.array([0.08, 0.06, 0.05, 0.07])
    return shares.reshape(-1)


def build_back_loaded_strategy():
    shares = np.zeros((T, 4))
    for t in range(T):
        if t <= 2:
            shares[t] = np.array([0.08, 0.06, 0.05, 0.07])
        elif t <= 5:
            shares[t] = np.array([0.10, 0.08, 0.07, 0.08])
        else:
            shares[t] = np.array([0.12, 0.12, 0.11, 0.11])
    return shares.reshape(-1)


def classify_loading(path):
    early = path[path["year"].between(2026, 2028)][investment_flow_cols].sum().sum()
    late = path[path["year"].between(2033, 2035)][investment_flow_cols].sum().sum()

    if early > late * 1.15:
        return "front-loaded"
    if late > early * 1.15:
        return "back-loaded"
    return "khá cân bằng"


def safe_pct_change(new, old):
    if abs(old) < 1e-9:
        return 0
    return (new - old) / abs(old) * 100


def add_line(fig, x, y, name, color, dash=None):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2.4, dash=dash),
            marker=dict(size=7),
            hovertemplate=f"{name}<br>Năm: %{{x}}<br>Giá trị: %{{y:,.3f}}<extra></extra>",
        )
    )


def format_fig(fig, title, height=340, x_title="Năm", y_title=""):
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

# ======================================================
# SETTINGS
# ======================================================

st.markdown("---")
st.header("Thiết lập mô hình")

section_caption(
    """
    Phần thiết lập được đặt cố định ở đầu bài để người dùng thay đổi tham số rồi quan sát ngay các tab kết quả bên dưới.
    Các tham số gồm hệ số chiết khấu, dạng hàm thỏa dụng, mức độ ngại rủi ro và cú sốc mô phỏng.
    """
)

setting_box = st.container(border=True)

with setting_box:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Phúc lợi liên thời gian")
        rho = st.number_input(
            "Hệ số chiết khấu rho",
            min_value=0.85,
            max_value=0.99,
            value=0.97,
            step=0.01,
            format="%.2f",
        )

        utility_type = st.selectbox(
            "Hàm thỏa dụng",
            ["Log utility", "CRRA utility"],
            index=0,
        )

    with col2:
        st.markdown("#### Tham số rủi ro")
        gamma_input = st.number_input(
            "Hệ số CRRA gamma",
            min_value=1.10,
            max_value=3.00,
            value=1.50,
            step=0.10,
            format="%.2f",
        )

        selected_strategy = st.selectbox(
            "Chiến lược so sánh chính",
            ["Đầu tư đều", "Front-loaded", "Back-loaded"],
            index=1,
        )

    with col3:
        st.markdown("#### Cú sốc mô phỏng")
        shock_year = st.selectbox(
            "Năm cú sốc",
            list(years),
            index=list(years).index(2028),
        )

        shock_size = st.number_input(
            "Mức giảm sản lượng do cú sốc",
            min_value=0.00,
            max_value=0.30,
            value=0.08,
            step=0.01,
            format="%.2f",
        )

# ======================================================
# SOLVE MODELS
# ======================================================

@st.cache_data(show_spinner=False)
def cached_solve(rho, utility_type, gamma):
    result, path, welfare = solve_dynamic_model(rho, utility_type, gamma)
    return result.success, result.message, path, welfare


with st.spinner("Đang giải bài toán tối ưu động..."):
    success, message, optimal_path, optimal_welfare = cached_solve(
        rho,
        utility_type,
        gamma_input,
    )

if not success:
    st.warning("Thuật toán chưa hội tụ hoàn toàn. Kết quả dưới đây được hiển thị như nghiệm xấp xỉ.")
    st.caption(str(message))

shock_path, shock_welfare = simulate_path(
    optimal_path[investment_cols].values.reshape(-1),
    rho=rho,
    utility_type=utility_type,
    gamma=gamma_input,
    shock=True,
    shock_year=shock_year,
    shock_size=shock_size,
)

equal_path, equal_welfare = simulate_path(
    build_equal_strategy(),
    rho=rho,
    utility_type=utility_type,
    gamma=gamma_input,
    shock=False,
)

front_path, front_welfare = simulate_path(
    build_front_loaded_strategy(),
    rho=rho,
    utility_type=utility_type,
    gamma=gamma_input,
    shock=False,
)

back_path, back_welfare = simulate_path(
    build_back_loaded_strategy(),
    rho=rho,
    utility_type=utility_type,
    gamma=gamma_input,
    shock=False,
)

success_short, message_short, short_path, short_welfare = cached_solve(
    0.90,
    utility_type,
    gamma_input,
)

if selected_strategy == "Đầu tư đều":
    selected_path = equal_path
    selected_welfare = equal_welfare
elif selected_strategy == "Front-loaded":
    selected_path = front_path
    selected_welfare = front_welfare
else:
    selected_path = back_path
    selected_welfare = back_welfare

loading_type = classify_loading(optimal_path)

ai_h_ratio = optimal_path["IAI"] / np.maximum(optimal_path["IH"], 1e-9)
ai_h_cv = ai_h_ratio.std() / ai_h_ratio.mean()

welfare_loss_shock = safe_pct_change(shock_welfare, optimal_welfare)
welfare_gain_selected = safe_pct_change(optimal_welfare, selected_welfare)

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card("Hệ số chiết khấu", f"{rho:.2f}", "Mức độ coi trọng phúc lợi tương lai.")
with k2:
    kpi_card("Kiểu đầu tư tối ưu", loading_type, "Phân loại front-loaded/back-loaded/cân bằng.")
with k3:
    kpi_card("Phúc lợi tối ưu", f"{optimal_welfare:,.4f}", "Tổng phúc lợi liên thời gian.")
with k4:
    kpi_card("Tác động cú sốc", f"{welfare_loss_shock:.2f}%", "Thay đổi phúc lợi khi có cú sốc mô phỏng.")

# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Tổng quan", "Quỹ đạo tối ưu", "Cú sốc và chiến lược", "Thảo luận chính sách"]
)

# ======================================================
# TAB 1
# ======================================================

with tab1:
    st.header("1. Tổng quan mô hình")

    st.write(
        """
        Bài toán lựa chọn tỷ lệ đầu tư hằng năm cho bốn loại vốn: vốn vật chất, hạ tầng số, AI và nhân lực.
        Phần tiêu dùng còn lại sau đầu tư tạo ra phúc lợi xã hội trong từng năm. Điểm quan trọng của bài toán là quyết định ở năm hiện tại
        có ảnh hưởng đến trạng thái vốn, năng lực số và sản lượng của nhiều năm sau.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Giai đoạn", "2026–2035", "Mô hình tối ưu động 10 năm.")
    with c2:
        kpi_card("Số biến quyết định", f"{T * 4}", "4 tỷ lệ đầu tư cho mỗi năm.")
    with c3:
        kpi_card("Hàm thỏa dụng", utility_type, "Log hoặc CRRA.")
    with c4:
        kpi_card("Chiến lược so sánh", selected_strategy, "Dùng làm benchmark ngoài nghiệm tối ưu.")

    st.markdown("#### Mô hình toán học tóm tắt")

    model_col, text_col = st.columns([0.9, 1.1])
    with model_col:
        st.latex(r"""
        \max \sum_{t=2026}^{2035} \rho^{t-2026}U(C_t)
        """)
        st.latex(r"""
        Y_t=A_tK_t^{0.33}L_t^{0.42}D_t^{0.10}AI_t^{0.08}H_t^{0.07}
        """)

    with text_col:
        st.write(
            """
            Mô hình tối đa hóa tổng phúc lợi liên thời gian, trong đó tiêu dùng C tạo ra phúc lợi hiện tại,
            còn đầu tư vào K, D, AI và H tạo ra năng lực sản xuất cho tương lai. Vì vậy, bài toán luôn có đánh đổi giữa tiêu dùng hôm nay
            và tích lũy năng lực tăng trưởng trong các năm sau.
            """
        )

        st.write(
            """
            Hệ số ρ càng cao thì mô hình càng coi trọng lợi ích dài hạn. Khi ρ thấp hơn, mô hình có xu hướng ưu tiên tiêu dùng hiện tại,
            từ đó làm giảm động lực đầu tư vào các tài sản có lợi ích xuất hiện muộn như AI, R&D và nhân lực.
            """
        )

    st.markdown("#### Điều kiện ban đầu")

    initial_df = pd.DataFrame({
        "Biến": ["K0", "L0", "D0", "AI0", "H0"],
        "Giá trị": [K0, L0, D0, AI0, H0],
        "Diễn giải": [
            "Vốn vật chất ban đầu",
            "Lao động",
            "Hạ tầng số ban đầu",
            "Năng lực AI ban đầu",
            "Vốn nhân lực ban đầu",
        ],
    })

    param_df = pd.DataFrame({
        "Tham số": ["delta_K", "delta_D", "delta_AI", "theta_H", "mu", "phi1", "phi2", "phi3"],
        "Giá trị": [delta_k, delta_d, delta_ai, theta_h, mu, phi1, phi2, phi3],
        "Ý nghĩa": [
            "Khấu hao vốn vật chất",
            "Khấu hao hạ tầng số",
            "Khấu hao năng lực AI",
            "Hiệu quả đầu tư nhân lực",
            "Brain drain",
            "Lan tỏa hạ tầng số vào TFP",
            "Lan tỏa AI vào TFP",
            "Lan tỏa nhân lực vào TFP",
        ],
    })

    col_init, col_param = st.columns([1, 1])
    with col_init:
        st.dataframe(initial_df, use_container_width=True)
    with col_param:
        st.dataframe(param_df.style.format({"Giá trị": "{:.3f}"}), use_container_width=True)

    st.markdown("#### Chỉ tiêu tổng hợp")

    summary_df = pd.DataFrame({
        "Chỉ tiêu": [
            "Phúc lợi tối ưu",
            f"Phúc lợi chiến lược {selected_strategy}",
            "Phúc lợi khi có cú sốc",
            f"Chênh lệch tối ưu so với {selected_strategy}",
            "Tác động cú sốc so với tối ưu",
        ],
        "Giá trị": [
            optimal_welfare,
            selected_welfare,
            shock_welfare,
            welfare_gain_selected,
            welfare_loss_shock,
        ],
    })

    st.dataframe(summary_df.style.format({"Giá trị": "{:.4f}"}), use_container_width=True)

# ======================================================
# TAB 2
# ======================================================

with tab2:
    st.header("2. Quỹ đạo tối ưu")

    st.write(
        """
        Phần này thể hiện cách các biến trạng thái và tỷ lệ đầu tư thay đổi qua thời gian.
        Nếu đầu tư tập trung nhiều ở giai đoạn đầu, quỹ đạo có tính front-loaded. Nếu đầu tư tăng mạnh về cuối kỳ,
        quỹ đạo có tính back-loaded.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("K năm 2035", f"{optimal_path['K'].iloc[-1]:,.0f}", "Vốn vật chất cuối kỳ.")
    with c2:
        kpi_card("D năm 2035", f"{optimal_path['D'].iloc[-1]:.2f}", "Hạ tầng số cuối kỳ.")
    with c3:
        kpi_card("AI năm 2035", f"{optimal_path['AI'].iloc[-1]:.2f}", "Năng lực AI cuối kỳ.")
    with c4:
        kpi_card("H năm 2035", f"{optimal_path['H'].iloc[-1]:.2f}", "Vốn nhân lực cuối kỳ.")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig = go.Figure()
        add_line(fig, optimal_path["year"], optimal_path["K"], "K", "#1FA7B6")
        add_line(fig, optimal_path["year"], optimal_path["D"], "D", "#5FA8D3")
        add_line(fig, optimal_path["year"], optimal_path["AI"], "AI", "#FF6B6B")
        add_line(fig, optimal_path["year"], optimal_path["H"], "H", "#8BC6EC")
        fig = format_fig(fig, "Quỹ đạo K, D, AI và H", height=340, y_title="Chỉ số/trạng thái")
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        fig = go.Figure()
        add_line(fig, optimal_path["year"], optimal_path["Y"], "Sản lượng", "#1FA7B6")
        add_line(fig, optimal_path["year"], optimal_path["C"], "Tiêu dùng", "#FF6B6B")
        fig = format_fig(fig, "Sản lượng và tiêu dùng", height=340, y_title="Giá trị")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cơ cấu tỷ lệ đầu tư tối ưu")

    fig = go.Figure()
    for label, col in zip(investment_labels, investment_cols):
        fig.add_trace(
            go.Bar(
                x=optimal_path["year"],
                y=optimal_path[col],
                name=label,
                marker_color=item_colors[label],
                hovertemplate=(
                    "Năm: %{x}<br>"
                    f"Hạng mục: {label}<br>"
                    "Tỷ lệ đầu tư: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Cơ cấu tỷ lệ đầu tư theo thời gian",
        height=360,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Năm",
        yaxis_title="Tỷ lệ trên sản lượng",
        barmode="stack",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Bảng quỹ đạo tối ưu")

    display_path = optimal_path[[
        "year", "Y", "C", "K", "D", "AI", "H",
        "sK", "sD", "sAI", "sH", "total_investment_share",
    ]].rename(columns={
        "year": "Năm",
        "Y": "Sản lượng",
        "C": "Tiêu dùng",
        "K": "K",
        "D": "D",
        "AI": "AI",
        "H": "H",
        "sK": "Tỷ lệ đầu tư K",
        "sD": "Tỷ lệ đầu tư D",
        "sAI": "Tỷ lệ đầu tư AI",
        "sH": "Tỷ lệ đầu tư H",
        "total_investment_share": "Tổng tỷ lệ đầu tư",
    })

    st.dataframe(
        display_path.style.format({
            "Sản lượng": "{:.3f}",
            "Tiêu dùng": "{:.3f}",
            "K": "{:.2f}",
            "D": "{:.2f}",
            "AI": "{:.2f}",
            "H": "{:.2f}",
            "Tỷ lệ đầu tư K": "{:.3f}",
            "Tỷ lệ đầu tư D": "{:.3f}",
            "Tỷ lệ đầu tư AI": "{:.3f}",
            "Tỷ lệ đầu tư H": "{:.3f}",
            "Tổng tỷ lệ đầu tư": "{:.3f}",
        }),
        use_container_width=True,
    )

# ======================================================
# TAB 3
# ======================================================

with tab3:
    st.header("3. Cú sốc và chiến lược")

    st.write(
        """
        Phần này so sánh quỹ đạo tối ưu với các chiến lược tham chiếu và mô phỏng cú sốc làm sản lượng suy giảm trong một năm.
        Các biểu đồ dùng Plotly để khi di chuột có thể thấy giá trị cụ thể tại từng điểm.
        """
    )

    compare_strategy = pd.DataFrame({
        "Chiến lược": [
            "Tối ưu động",
            "Đầu tư đều",
            "Front-loaded",
            "Back-loaded",
            f"Tối ưu động có cú sốc {shock_year}",
        ],
        "Phúc lợi": [
            optimal_welfare,
            equal_welfare,
            front_welfare,
            back_welfare,
            shock_welfare,
        ],
        "Sản lượng 2035": [
            optimal_path["Y"].iloc[-1],
            equal_path["Y"].iloc[-1],
            front_path["Y"].iloc[-1],
            back_path["Y"].iloc[-1],
            shock_path["Y"].iloc[-1],
        ],
        "Tiêu dùng 2035": [
            optimal_path["C"].iloc[-1],
            equal_path["C"].iloc[-1],
            front_path["C"].iloc[-1],
            back_path["C"].iloc[-1],
            shock_path["C"].iloc[-1],
        ],
    })

    table_col, chart_col = st.columns([1, 1])

    with table_col:
        st.dataframe(
            compare_strategy.style.format({
                "Phúc lợi": "{:.4f}",
                "Sản lượng 2035": "{:.3f}",
                "Tiêu dùng 2035": "{:.3f}",
            }),
            use_container_width=True,
        )

    with chart_col:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=compare_strategy["Chiến lược"],
            y=compare_strategy["Phúc lợi"],
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Chiến lược: %{x}<br>Phúc lợi: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title="So sánh phúc lợi theo chiến lược",
            height=330,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis=dict(tickangle=15),
            yaxis_title="Phúc lợi",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        fig = go.Figure()
        add_line(fig, optimal_path["year"], optimal_path["Y"], "Tối ưu", "#1FA7B6")
        add_line(fig, equal_path["year"], equal_path["Y"], "Đầu tư đều", "#1FA7B6")
        add_line(fig, front_path["year"], front_path["Y"], "Front-loaded", "#FF6B6B")
        add_line(fig, back_path["year"], back_path["Y"], "Back-loaded", "#E6F7F5")
        fig = format_fig(fig, "So sánh sản lượng theo chiến lược", height=350, y_title="Sản lượng")
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        fig = go.Figure()
        add_line(fig, optimal_path["year"], optimal_path["C"], "Không cú sốc", "#1FA7B6")
        add_line(fig, shock_path["year"], shock_path["C"], "Có cú sốc", "#F97316")
        fig.add_vline(
            x=shock_year,
            line_width=1,
            line_dash="dash",
            line_color="#334155",
            annotation_text="Cú sốc",
            annotation_position="top left",
        )
        fig = format_fig(fig, "Tác động cú sốc đến tiêu dùng", height=350, y_title="Tiêu dùng")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### So sánh khi rho hiện tại và rho = 0.90")

    rho_compare = pd.DataFrame({
        "Năm": years,
        f"Tổng đầu tư, rho={rho:.2f}": optimal_path["total_investment_share"],
        "Tổng đầu tư, rho=0.90": short_path["total_investment_share"],
        f"AI, rho={rho:.2f}": optimal_path["sAI"],
        "AI, rho=0.90": short_path["sAI"],
        f"H, rho={rho:.2f}": optimal_path["sH"],
        "H, rho=0.90": short_path["sH"],
    })

    comp_col, comp_chart_col = st.columns([1, 1])

    with comp_col:
        st.dataframe(
            rho_compare.style.format({
                f"Tổng đầu tư, rho={rho:.2f}": "{:.3f}",
                "Tổng đầu tư, rho=0.90": "{:.3f}",
                f"AI, rho={rho:.2f}": "{:.3f}",
                "AI, rho=0.90": "{:.3f}",
                f"H, rho={rho:.2f}": "{:.3f}",
                "H, rho=0.90": "{:.3f}",
            }),
            use_container_width=True,
        )

    with comp_chart_col:
        fig = go.Figure()
        add_line(fig, years, optimal_path["total_investment_share"], f"rho={rho:.2f}", "#1FA7B6")
        add_line(fig, years, short_path["total_investment_share"], "rho=0.90", "#FF6B6B", dash="dash")
        fig = format_fig(fig, "Tổng tỷ lệ đầu tư khi thay đổi hệ số chiết khấu", height=340, y_title="Tỷ lệ đầu tư")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 4
# ======================================================

with tab4:
    st.header("4. Thảo luận chính sách")

    early_investment = optimal_path[optimal_path["year"].between(2026, 2028)][investment_flow_cols].sum().sum()
    middle_investment = optimal_path[optimal_path["year"].between(2029, 2032)][investment_flow_cols].sum().sum()
    late_investment = optimal_path[optimal_path["year"].between(2033, 2035)][investment_flow_cols].sum().sum()

    avg_ai_h_early = ai_h_ratio.iloc[:3].mean()
    avg_ai_h_late = ai_h_ratio.iloc[-3:].mean()

    total_invest_current = optimal_path["total_investment_share"].mean()
    total_invest_090 = short_path["total_investment_share"].mean()

    ai_invest_current = optimal_path["sAI"].mean()
    ai_invest_090 = short_path["sAI"].mean()

    h_invest_current = optimal_path["sH"].mean()
    h_invest_090 = short_path["sH"].mean()

    st.markdown("#### Quỹ đạo tối ưu front-loaded hay back-loaded?")

    st.write(
        f"""
        Quỹ đạo tối ưu trong mô hình có xu hướng {loading_type}. Tổng đầu tư giai đoạn 2026–2028 đạt khoảng {early_investment:,.2f},
        giai đoạn 2029–2032 đạt khoảng {middle_investment:,.2f}, còn giai đoạn 2033–2035 đạt khoảng {late_investment:,.2f}.
        Nếu quỹ đạo nghiêng về front-loaded, mô hình đang khuyến nghị đầu tư tương đối sớm vì vốn số, AI và nhân lực đều có tính tích lũy.
        Đầu tư sớm không chỉ tạo sản lượng trong năm hiện tại mà còn làm tăng năng lực sản xuất, TFP và khả năng hấp thụ công nghệ cho các năm sau.
        """
    )

    st.write(
        """
        Với Việt Nam, logic này khá quan trọng. Hạ tầng số, dữ liệu, năng lực AI và nhân lực số đều cần thời gian hình thành.
        Nếu đầu tư bị trì hoãn đến cuối kỳ, nền kinh tế có thể không kịp tạo năng lực triển khai công nghệ ở quy mô lớn.
        Ngược lại, đầu tư sớm cũng làm giảm tiêu dùng hiện tại, nên chính sách cần giữ mức đầu tư đủ mạnh nhưng không làm suy giảm phúc lợi ngắn hạn quá mức.
        """
    )

    st.markdown("#### Tỷ lệ đầu tư AI/nhân lực có ổn định không?")

    st.write(
        f"""
        Tỷ lệ đầu tư AI so với đầu tư nhân lực không hoàn toàn cố định qua thời gian. Trong mô hình hiện tại,
        tỷ lệ AI/H trung bình giai đoạn 2026–2028 là khoảng {avg_ai_h_early:.2f}, còn giai đoạn 2033–2035 là khoảng {avg_ai_h_late:.2f}.
        Hệ số biến thiên của tỷ lệ này là khoảng {ai_h_cv:.2f}.
        """
    )

    st.write(
        """
        Kết quả này ngụ ý rằng AI và nhân lực phải đi cùng nhau. AI cần nhân lực để hấp thụ, vận hành, kiểm soát rủi ro dữ liệu và chuyển hóa thành năng suất.
        Nếu đầu tư AI đi quá nhanh trong khi đào tạo nhân lực chậm hơn, nền kinh tế có thể thiếu kỹ sư dữ liệu, thiếu năng lực triển khai và thiếu khả năng quản trị công nghệ.
        Ngược lại, nếu chỉ đào tạo nhân lực nhưng thiếu hạ tầng, dữ liệu và dự án AI, kỹ năng mới có thể không được sử dụng hiệu quả.
        Vì vậy, đào tạo nhân lực nên đi trước một bước ở giai đoạn đầu, sau đó diễn ra đồng thời với đầu tư AI trong giai đoạn mở rộng.
        """
    )

    st.markdown("#### Hệ số chiết khấu ρ và nguy cơ dưới đầu tư vào R&D")

    st.write(
        f"""
        Hệ số chiết khấu rho = {rho:.2f} thể hiện mức độ coi trọng phúc lợi tương lai. Khi so sánh với rho = 0.90,
        tỷ lệ đầu tư trung bình thay đổi từ {total_invest_current:.3f} xuống {total_invest_090:.3f}.
        Tỷ lệ đầu tư AI trung bình thay đổi từ {ai_invest_current:.3f} xuống {ai_invest_090:.3f}, còn tỷ lệ đầu tư nhân lực thay đổi từ {h_invest_current:.3f} xuống {h_invest_090:.3f}.
        """
    )

    st.write(
        """
        Khi rho thấp hơn, mô hình trở nên ngắn hạn hơn: tiêu dùng hiện tại được ưu tiên nhiều hơn so với lợi ích tương lai.
        Đây là một cách giải thích vì sao các chính phủ có thể dưới đầu tư vào R&D, AI và nhân lực số. Những khoản đầu tư này có chi phí hiện tại rõ ràng,
        nhưng lợi ích lại xuất hiện muộn, phân tán qua nhiều ngành và khó quy trực tiếp cho một nhiệm kỳ chính sách. Nếu áp lực ngắn hạn quá lớn,
        các khoản đầu tư dài hạn dễ bị cắt giảm hoặc trì hoãn dù có lợi cho tăng trưởng dài hạn.
        """
    )

    st.markdown("#### Cú sốc và hàm ý điều hành")

    st.write(
        f"""
        Khi giả sử năm {shock_year} xảy ra cú sốc làm sản lượng giảm {shock_size * 100:.0f}%, phúc lợi thay đổi khoảng {welfare_loss_shock:.2f}%
        so với quỹ đạo không có cú sốc. Kết quả này cho thấy một kế hoạch đầu tư tối ưu vẫn cần đi kèm khả năng điều chỉnh chính sách.
        Nếu toàn bộ nguồn lực bị khóa cứng vào kế hoạch dài hạn, nền kinh tế có thể thiếu dư địa ứng phó khi gặp khủng hoảng.
        """
    )

    st.write(
        """
        Một chiến lược thực tế nên kết hợp đầu tư dài hạn với cơ chế dự phòng tài khóa. Trong khủng hoảng, chính phủ có thể cần bảo vệ tiêu dùng,
        việc làm và an sinh ngắn hạn, nhưng vẫn nên duy trì các khoản đầu tư lõi vào hạ tầng số, AI và nhân lực.
        Đây là các tài sản có tính bảo hiểm dài hạn, giúp nền kinh tế phục hồi nhanh hơn sau cú sốc và tránh rơi vào bẫy tăng trưởng thấp.
        """
    )

    st.markdown("#### Kết luận")

    st.write(
        """
        Bài 8 cho thấy chính sách đầu tư cho AI và kinh tế số cần được nhìn theo quỹ đạo dài hạn.
        Lựa chọn tối ưu không chỉ là đầu tư bao nhiêu, mà còn là đầu tư vào lúc nào, phối hợp AI với nhân lực ra sao,
        và duy trì cam kết dài hạn như thế nào khi đối mặt với áp lực ngắn hạn.
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
        Lưu ý: Kết quả của Bài 8 là mô phỏng phục vụ phân tích. Mô hình tối ưu động giúp minh họa đánh đổi giữa tiêu dùng hiện tại
        và đầu tư dài hạn, nhưng không thay thế dự báo kinh tế chính thức hay quyết định ngân sách thực tế.
        </p>
        """,
        unsafe_allow_html=True,
    )
