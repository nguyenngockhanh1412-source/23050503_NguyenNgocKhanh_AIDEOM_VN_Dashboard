import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import gymnasium as gym
from gymnasium import spaces

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

setup_page("Bài 11 - Q-learning")

def safe_render_bai11_sidebar():
    """
    Dùng sidebar chung nếu tìm thấy đúng nhãn Bài 11 trong MENU_OPTIONS.
    Cách này tránh lỗi list.index nếu tên menu có khác dấu gạch ngang hoặc chữ hoa/thường.
    """
    try:
        import utils.aideom_ui as ui
        menu_options = getattr(ui, "MENU_OPTIONS", [])

        candidates = [
            "Bài 11 - Q-learning",
            "Bài 11 Q-learning",
            "Bài 11: Q-learning",
            "Bài 11 - Q Learning Policy",
            "Bài 11 Q Learning Policy",
        ]

        for name in candidates:
            if name in menu_options:
                render_sidebar(name)
                return

        for name in menu_options:
            normalized = str(name).lower()
            if "bài 11" in normalized or "bai 11" in normalized or "q-learning" in normalized or "q learning" in normalized:
                render_sidebar(name)
                return

        render_sidebar("Bài 11 - Q-learning")
    except Exception:
        st.sidebar.markdown("### AIDEOM-VN")
        st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI")
        st.sidebar.markdown("---")
        st.sidebar.info("Đang ở Bài 11. Nếu muốn hiện menu chung, kiểm tra lại tên Bài 11 trong MENU_OPTIONS của utils/aideom_ui.py.")

safe_render_bai11_sidebar()

PLOT_CONFIG = {
    "scrollZoom": False,
    "displayModeBar": False,
    "doubleClick": False,
    "responsive": True,
}

def show_plot(fig):
    fig.update_layout(
        dragmode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="#1f2937"),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

st.title("Bài 11. Học tăng cường Q-learning cho chính sách kinh tế thích nghi")

st.write(
    """
    Dashboard mô phỏng một agent chính sách học cách lựa chọn gói phân bổ ngân sách theo trạng thái kinh tế.
    Mỗi trạng thái phản ánh tăng trưởng GDP, chỉ số số hóa, năng lực AI và rủi ro thất nghiệp.
    Agent học qua nhiều episode để tìm chính sách thích nghi thay vì dùng một quy tắc cố định.
    """
)

source_note(
    """
    Bài 11 sử dụng môi trường mô phỏng dạng MDP với 81 trạng thái và 5 hành động chính sách theo yêu cầu bộ đề.
    Kết quả huấn luyện Q-learning là mô phỏng phục vụ phân tích, không thay thế quy trình thẩm định chính sách chính thức.
    """
)

# ======================================================
# BASIC DEFINITIONS
# ======================================================

levels = ["low", "medium", "high"]
level_vn = {
    "low": "Thấp",
    "medium": "Trung bình",
    "high": "Cao",
}

state_factors = ["GDP growth", "Digital index", "AI capacity", "Unemployment risk"]

actions = {
    0: {
        "name": "a0 - Truyền thống",
        "K": 0.70,
        "D": 0.10,
        "AI": 0.10,
        "H": 0.10,
        "description": "Ưu tiên vốn truyền thống, ít đầu tư vào số hóa, AI và nhân lực."
    },
    1: {
        "name": "a1 - Cân bằng",
        "K": 0.40,
        "D": 0.25,
        "AI": 0.15,
        "H": 0.20,
        "description": "Cân bằng giữa vốn truyền thống, chuyển đổi số, AI và nhân lực."
    },
    2: {
        "name": "a2 - Số hóa nhanh",
        "K": 0.25,
        "D": 0.45,
        "AI": 0.15,
        "H": 0.15,
        "description": "Ưu tiên hạ tầng và chuyển đổi số để nâng nền tảng kinh tế số."
    },
    3: {
        "name": "a3 - AI dẫn dắt",
        "K": 0.20,
        "D": 0.20,
        "AI": 0.45,
        "H": 0.15,
        "description": "Ưu tiên AI để thúc đẩy năng suất, nhưng có thể làm tăng rủi ro lao động và dữ liệu nếu nền tảng yếu."
    },
    4: {
        "name": "a4 - Bao trùm",
        "K": 0.30,
        "D": 0.20,
        "AI": 0.10,
        "H": 0.40,
        "description": "Ưu tiên nhân lực, đào tạo lại và giảm rủi ro thất nghiệp trong chuyển đổi số."
    },
}

n_states = 3 ** 4
n_actions = len(actions)


def state_to_index(state_tuple):
    gdp, digital, ai, unemp = state_tuple
    return gdp * 27 + digital * 9 + ai * 3 + unemp


def index_to_state(index):
    gdp = index // 27
    rem = index % 27
    digital = rem // 9
    rem = rem % 9
    ai = rem // 3
    unemp = rem % 3
    return int(gdp), int(digital), int(ai), int(unemp)


def state_label(index):
    gdp, digital, ai, unemp = index_to_state(index)
    return {
        "GDP growth": level_vn[levels[gdp]],
        "Digital index": level_vn[levels[digital]],
        "AI capacity": level_vn[levels[ai]],
        "Unemployment risk": level_vn[levels[unemp]],
    }


def move_level(value, delta):
    return int(np.clip(value + delta, 0, 2))


def action_vector(action_id):
    a = actions[action_id]
    return np.array([a["K"], a["D"], a["AI"], a["H"]])


# ======================================================
# GYMNASIUM ENVIRONMENT
# ======================================================

class VietnamPolicyEnv(gym.Env):
    def __init__(
        self,
        horizon=10,
        reward_weights=(0.40, 0.25, 0.20, 0.15),
        seed=42,
    ):
        super().__init__()

        self.horizon = horizon
        self.reward_weights = reward_weights
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_states)

        self.current_state = None
        self.t = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Trạng thái khởi đầu ngẫu nhiên nhưng nghiêng về trạng thái trung bình
        if self.rng.random() < 0.55:
            state_tuple = (1, 1, 1, 1)
        else:
            state_tuple = (
                self.rng.integers(0, 3),
                self.rng.integers(0, 3),
                self.rng.integers(0, 3),
                self.rng.integers(0, 3),
            )

        self.current_state = state_to_index(state_tuple)
        self.t = 0

        return self.current_state, {}

    def step(self, action):
        gdp, digital, ai, unemp = index_to_state(self.current_state)
        shares = action_vector(action)

        k_share, d_share, ai_share, h_share = shares

        # Tác động trực tiếp theo logic kinh tế
        gdp_score = (
            0.35 * k_share
            + 0.25 * d_share
            + 0.30 * ai_share
            + 0.10 * h_share
        )

        digital_score = (
            0.10 * k_share
            + 0.55 * d_share
            + 0.25 * ai_share
            + 0.10 * h_share
        )

        ai_score = (
            0.05 * k_share
            + 0.20 * d_share
            + 0.60 * ai_share
            + 0.15 * h_share
        )

        unemployment_pressure = (
            0.10 * k_share
            + 0.15 * d_share
            + 0.55 * ai_share
            - 0.45 * h_share
        )

        cyber_risk = (
            0.10 * d_share
            + 0.50 * ai_share
            - 0.25 * h_share
            - 0.05 * digital
        )

        emission = (
            0.55 * k_share
            + 0.25 * d_share
            + 0.20 * ai_share
            - 0.10 * h_share
        )

        # Trạng thái nền tảng ảnh hưởng đến hiệu quả chính sách
        gdp_outcome = (
            0.45 * gdp
            + 1.20 * gdp_score
            + 0.18 * digital
            + 0.15 * ai
            - 0.20 * unemp
        )

        unemployment_outcome = (
            0.55 * unemp
            + 1.10 * unemployment_pressure
            - 0.25 * h_share
            - 0.10 * digital
        )

        cyber_outcome = (
            0.45 * cyber_risk
            + 0.18 * ai
            - 0.12 * h_share
        )

        emission_outcome = (
            0.50 * emission
            + 0.12 * k_share
            - 0.08 * d_share
        )

        w_gdp, w_unemp, w_cyber, w_emission = self.reward_weights

        reward = (
            w_gdp * gdp_outcome
            - w_unemp * unemployment_outcome
            - w_cyber * cyber_outcome
            - w_emission * emission_outcome
        )

        # Thêm nhiễu nhỏ để mô phỏng bất định môi trường
        reward += self.rng.normal(0, 0.015)

        # Chuyển trạng thái
        digital_delta = 1 if digital_score > 0.28 else 0
        ai_delta = 1 if ai_score > 0.30 else 0

        if gdp_outcome > 1.45:
            gdp_delta = 1
        elif gdp_outcome < 0.55:
            gdp_delta = -1
        else:
            gdp_delta = 0

        if unemployment_outcome > 0.75:
            unemp_delta = 1
        elif unemployment_outcome < 0.30:
            unemp_delta = -1
        else:
            unemp_delta = 0

        next_gdp = move_level(gdp, gdp_delta)
        next_digital = move_level(digital, digital_delta)
        next_ai = move_level(ai, ai_delta)
        next_unemp = move_level(unemp, unemp_delta)

        next_state = state_to_index((next_gdp, next_digital, next_ai, next_unemp))

        self.current_state = next_state
        self.t += 1

        terminated = self.t >= self.horizon
        truncated = False

        info = {
            "gdp_outcome": gdp_outcome,
            "unemployment_outcome": unemployment_outcome,
            "cyber_outcome": cyber_outcome,
            "emission_outcome": emission_outcome,
        }

        return next_state, reward, terminated, truncated, info


# ======================================================
# Q-LEARNING
# ======================================================

def train_q_learning(
    episodes=10000,
    horizon=10,
    alpha=0.10,
    gamma=0.95,
    epsilon_start=1.00,
    epsilon_min=0.05,
    seed=42,
    reward_weights=(0.40, 0.25, 0.20, 0.15),
):
    env = VietnamPolicyEnv(
        horizon=horizon,
        reward_weights=reward_weights,
        seed=seed,
    )

    q_table = np.zeros((n_states, n_actions))
    rewards = np.zeros(episodes)
    epsilons = np.zeros(episodes)

    rng = np.random.default_rng(seed)

    epsilon_decay = (epsilon_min / epsilon_start) ** (1 / max(episodes - 1, 1))
    epsilon = epsilon_start

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(horizon):
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, info = env.step(action)

            best_next = np.max(q_table[next_state])

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * best_next - q_table[state, action]
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        rewards[ep] = total_reward
        epsilons[ep] = epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, rewards, epsilons


def evaluate_policy(
    q_table=None,
    policy_type="q_learning",
    fixed_action=None,
    episodes=800,
    horizon=10,
    seed=123,
    reward_weights=(0.40, 0.25, 0.20, 0.15),
):
    env = VietnamPolicyEnv(
        horizon=horizon,
        reward_weights=reward_weights,
        seed=seed,
    )

    rng = np.random.default_rng(seed + 100)
    episode_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(horizon):
            if policy_type == "q_learning":
                action = int(np.argmax(q_table[state]))
            elif policy_type == "fixed":
                action = fixed_action
            else:
                action = int(rng.integers(0, n_actions))

            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    return np.array(episode_rewards)


def moving_average(x, window=200):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def build_policy_table(q_table):
    rows = []

    for s in range(n_states):
        labels = state_label(s)
        best_action = int(np.argmax(q_table[s]))
        rows.append({
            "state_id": s,
            "GDP growth": labels["GDP growth"],
            "Digital index": labels["Digital index"],
            "AI capacity": labels["AI capacity"],
            "Unemployment risk": labels["Unemployment risk"],
            "Hành động khuyến nghị": actions[best_action]["name"],
            "Q-value": q_table[s, best_action],
        })

    return pd.DataFrame(rows)


def action_explanation(action_id, state_tuple):
    gdp, digital, ai, unemp = state_tuple
    action = actions[action_id]["name"]

    if action_id == 0:
        return (
            "Mô hình chọn hướng truyền thống khi cần giữ ổn định tăng trưởng ngắn hạn hoặc khi nền tảng số chưa đủ mạnh. "
            "Tuy nhiên, nếu duy trì quá lâu, chính sách này có thể làm chậm chuyển đổi số."
        )

    if action_id == 1:
        return (
            "Mô hình chọn phương án cân bằng khi trạng thái kinh tế không quá cực đoan. "
            "Đây là chiến lược dung hòa giữa tăng trưởng, số hóa, AI và nhân lực."
        )

    if action_id == 2:
        return (
            "Mô hình ưu tiên số hóa nhanh khi cần nâng nền tảng digital trước khi mở rộng AI. "
            "Điều này phù hợp nếu năng lực AI chưa đủ cao nhưng chuyển đổi số có thể tạo nền móng cho giai đoạn sau."
        )

    if action_id == 3:
        return (
            "Mô hình chọn AI dẫn dắt khi nền tảng số và năng lực AI đủ thuận lợi để tạo tăng trưởng nhanh. "
            "Nếu rủi ro thất nghiệp cao, chiến lược này cần đi kèm đào tạo lại để tránh tác động xã hội bất lợi."
        )

    return (
        "Mô hình chọn bao trùm khi rủi ro thất nghiệp hoặc năng lực hấp thụ công nghệ là vấn đề nổi bật. "
        "Chiến lược này ưu tiên nhân lực, đào tạo lại và giảm chi phí xã hội của chuyển đổi số."
    )


# ======================================================
# SETTINGS
# ======================================================

st.markdown("---")
st.header("1. Thiết lập mô hình và công thức Q-learning")

section_caption(
    """
    Phần thiết lập được đặt cố định ở đầu bài để người dùng thay đổi tham số rồi quan sát kết quả huấn luyện ở các mục bên dưới.
    Các tham số gồm số episode, số năm trong mỗi episode, learning rate, discount factor, epsilon tối thiểu và seed mô phỏng.
    """
)

setting_box = st.container(border=True)

with setting_box:
    left_control, right_formula = st.columns([1.05, 1])

    with left_control:
        st.markdown("#### Tham số huấn luyện")

        c1, c2 = st.columns(2)

        with c1:
            episodes = st.number_input(
                "Số episode",
                min_value=1000,
                max_value=30000,
                value=10000,
                step=1000,
            )

            alpha = st.number_input(
                "Learning rate alpha",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                format="%.2f",
            )

            gamma = st.number_input(
                "Discount gamma",
                min_value=0.50,
                max_value=0.99,
                value=0.95,
                step=0.01,
                format="%.2f",
            )

        with c2:
            horizon = st.number_input(
                "Số năm mỗi episode",
                min_value=5,
                max_value=20,
                value=10,
                step=1,
            )

            epsilon_min = st.number_input(
                "Epsilon tối thiểu",
                min_value=0.01,
                max_value=0.30,
                value=0.05,
                step=0.01,
                format="%.2f",
            )

            seed = st.number_input(
                "Seed mô phỏng",
                min_value=1,
                max_value=9999,
                value=42,
                step=1,
            )

    with right_formula:
        st.markdown("#### Công thức cập nhật Q-table")

        st.latex(r"""
        Q(s,a) \leftarrow Q(s,a) + \alpha
        \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
        """)

        st.write(
            """
            Trong đó, trạng thái s phản ánh điều kiện kinh tế hiện tại, hành động a là gói phân bổ ngân sách,
            r là phần thưởng chính sách, còn s' là trạng thái kinh tế sau khi chính sách tác động.
            """
        )

        st.markdown("#### Hàm thưởng")

        st.latex(r"""
        R = 0.40 \cdot GDP - 0.25 \cdot Unemployment
        - 0.20 \cdot CyberRisk - 0.15 \cdot Emission
        """)

k0, k1, k2, k3 = st.columns(4)

with k0:
    kpi_card("Số episode", f"{episodes:,.0f}", "Số vòng huấn luyện agent.")

with k1:
    kpi_card("Alpha", f"{alpha:.2f}", "Tốc độ cập nhật Q-table.")

with k2:
    kpi_card("Gamma", f"{gamma:.2f}", "Mức coi trọng phần thưởng tương lai.")

with k3:
    kpi_card("Epsilon cuối", f"{epsilon_min:.2f}", "Mức khám phá tối thiểu khi huấn luyện.")


# ======================================================
# TRAINING
# ======================================================

@st.cache_data(show_spinner=False)
def cached_training(episodes, horizon, alpha, gamma, epsilon_min, seed):
    return train_q_learning(
        episodes=int(episodes),
        horizon=int(horizon),
        alpha=float(alpha),
        gamma=float(gamma),
        epsilon_start=1.00,
        epsilon_min=float(epsilon_min),
        seed=int(seed),
        reward_weights=(0.40, 0.25, 0.20, 0.15),
    )


with st.spinner("Đang huấn luyện Q-learning agent..."):
    q_table, rewards, epsilons = cached_training(
        int(episodes),
        int(horizon),
        float(alpha),
        float(gamma),
        float(epsilon_min),
        int(seed),
    )

policy_df = build_policy_table(q_table)

q_eval = evaluate_policy(
    q_table=q_table,
    policy_type="q_learning",
    episodes=800,
    horizon=int(horizon),
    seed=int(seed) + 10,
)

balanced_eval = evaluate_policy(
    policy_type="fixed",
    fixed_action=1,
    episodes=800,
    horizon=int(horizon),
    seed=int(seed) + 20,
)

ai_led_eval = evaluate_policy(
    policy_type="fixed",
    fixed_action=3,
    episodes=800,
    horizon=int(horizon),
    seed=int(seed) + 30,
)

random_eval = evaluate_policy(
    policy_type="random",
    episodes=800,
    horizon=int(horizon),
    seed=int(seed) + 40,
)


# ======================================================
# SECTION 2: TRAINING RESULTS
# ======================================================

st.markdown("---")
st.header("2. Kết quả huấn luyện")

m1, m2, m3, m4 = st.columns(4)

with m1:
    kpi_card("Số trạng thái", f"{n_states}", "Tổng số trạng thái MDP.")

with m2:
    kpi_card("Số hành động", f"{n_actions}", "Số gói chính sách có thể chọn.")

with m3:
    kpi_card("Reward trung bình cuối", f"{np.mean(rewards[-500:]):.3f}", "Trung bình 500 episode cuối.")

with m4:
    kpi_card("Epsilon cuối", f"{epsilons[-1]:.3f}", "Mức khám phá ở cuối huấn luyện.")

chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    ma = moving_average(rewards, window=min(300, max(20, int(episodes // 30))))
    ma_x = np.arange(len(ma)) + (len(rewards) - len(ma))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards)),
            y=rewards,
            mode="lines",
            name="Reward từng episode",
            line=dict(color="#1FA7B6", width=1),
            opacity=0.45,
            hovertemplate="Episode: %{x}<br>Reward: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ma_x,
            y=ma,
            mode="lines",
            name="Moving average",
            line=dict(color="#1FA7B6", width=2),
            hovertemplate="Episode: %{x}<br>Reward TB: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Quá trình học qua episode",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Episode",
        yaxis_title="Reward",
        legend=dict(orientation="h", y=-0.25),
    )

    show_plot(fig)

with chart_col2:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(epsilons)),
            y=epsilons,
            mode="lines",
            line=dict(color="#1FA7B6", width=2),
            name="Epsilon",
            hovertemplate="Episode: %{x}<br>Epsilon: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Epsilon giảm dần theo thời gian",
        height=300,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Episode",
        yaxis_title="Epsilon",
    )

    show_plot(fig)


# ======================================================
# SECTION 3: POLICY MAP AND SCENARIO TESTER
# ======================================================

st.markdown("---")
st.header("3. Bản đồ chính sách học được")

policy_control, policy_chart = st.columns([1, 1])

with policy_control:
    st.markdown("#### Kiểm tra một trạng thái cụ thể")

    c1, c2 = st.columns(2)

    with c1:
        gdp_choice = st.selectbox("GDP growth", levels, format_func=lambda x: level_vn[x], index=1)
        digital_choice = st.selectbox("Digital index", levels, format_func=lambda x: level_vn[x], index=1)

    with c2:
        ai_choice = st.selectbox("AI capacity", levels, format_func=lambda x: level_vn[x], index=1)
        unemp_choice = st.selectbox("Unemployment risk", levels, format_func=lambda x: level_vn[x], index=1)

    selected_state_tuple = (
        levels.index(gdp_choice),
        levels.index(digital_choice),
        levels.index(ai_choice),
        levels.index(unemp_choice),
    )

    selected_state = state_to_index(selected_state_tuple)
    recommended_action = int(np.argmax(q_table[selected_state]))

    st.success(f"Hành động khuyến nghị: {actions[recommended_action]['name']}")

    st.write(action_explanation(recommended_action, selected_state_tuple))

    q_row = pd.DataFrame({
        "Hành động": [actions[a]["name"] for a in range(n_actions)],
        "Q-value": q_table[selected_state],
    })

    st.dataframe(
        q_row.style.format({"Q-value": "{:.4f}"}),
        use_container_width=True
    )

with policy_chart:
    action_counts = policy_df["Hành động khuyến nghị"].value_counts().reset_index()
    action_counts.columns = ["Hành động", "Số trạng thái"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=action_counts["Hành động"],
                y=action_counts["Số trạng thái"],
                marker_color="#E6F7F5",
                hovertemplate="Hành động: %{x}<br>Số trạng thái: %{y}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title="Tần suất hành động trong 81 trạng thái",
        height=310,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Số trạng thái",
    )

    show_plot(fig)

st.markdown("#### Bảng chính sách rút gọn")

filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    filter_gdp = st.selectbox("Lọc GDP", ["Tất cả", "Thấp", "Trung bình", "Cao"], index=0)

with filter_col2:
    filter_digital = st.selectbox("Lọc Digital", ["Tất cả", "Thấp", "Trung bình", "Cao"], index=0)

with filter_col3:
    filter_ai = st.selectbox("Lọc AI", ["Tất cả", "Thấp", "Trung bình", "Cao"], index=0)

with filter_col4:
    filter_unemp = st.selectbox("Lọc thất nghiệp", ["Tất cả", "Thấp", "Trung bình", "Cao"], index=0)

filtered_policy = policy_df.copy()

if filter_gdp != "Tất cả":
    filtered_policy = filtered_policy[filtered_policy["GDP growth"] == filter_gdp]

if filter_digital != "Tất cả":
    filtered_policy = filtered_policy[filtered_policy["Digital index"] == filter_digital]

if filter_ai != "Tất cả":
    filtered_policy = filtered_policy[filtered_policy["AI capacity"] == filter_ai]

if filter_unemp != "Tất cả":
    filtered_policy = filtered_policy[filtered_policy["Unemployment risk"] == filter_unemp]

st.dataframe(
    filtered_policy.head(30).style.format({"Q-value": "{:.4f}"}),
    use_container_width=True
)


# ======================================================
# SECTION 4: POLICY COMPARISON
# ======================================================

st.markdown("---")
st.header("4. So sánh với chính sách rule-based")

comparison_df = pd.DataFrame({
    "Chính sách": [
        "Q-learning policy",
        "Luôn chọn a1 - Cân bằng",
        "Luôn chọn a3 - AI dẫn dắt",
        "Random policy",
    ],
    "Reward trung bình": [
        q_eval.mean(),
        balanced_eval.mean(),
        ai_led_eval.mean(),
        random_eval.mean(),
    ],
    "Độ lệch chuẩn": [
        q_eval.std(),
        balanced_eval.std(),
        ai_led_eval.std(),
        random_eval.std(),
    ],
})

st.dataframe(
    comparison_df.style.format({
        "Reward trung bình": "{:.4f}",
        "Độ lệch chuẩn": "{:.4f}",
    }),
    use_container_width=True
)

fig = go.Figure(
    data=[
        go.Bar(
            x=comparison_df["Chính sách"],
            y=comparison_df["Reward trung bình"],
            error_y=dict(type="data", array=comparison_df["Độ lệch chuẩn"], visible=True),
            marker_color=["#1FA7B6", "#E6F7F5", "#FAD7D7", "#81D8D0"],
            hovertemplate="Chính sách: %{x}<br>Reward TB: %{y:.4f}<extra></extra>",
        )
    ]
)

fig.update_layout(
    title="Hiệu quả trung bình của các chính sách",
    height=320,
    margin=dict(l=10, r=10, t=45, b=20),
    xaxis=dict(tickangle=0),
    yaxis_title="Reward trung bình",
)

show_plot(fig)


# ======================================================
# SECTION 5: POLICY DISCUSSION
# ======================================================

st.markdown("---")
st.header("5. Diễn giải chính sách")

best_policy = comparison_df.sort_values("Reward trung bình", ascending=False).iloc[0]["Chính sách"]
q_advantage_balanced = q_eval.mean() - balanced_eval.mean()
q_advantage_ai = q_eval.mean() - ai_led_eval.mean()
q_advantage_random = q_eval.mean() - random_eval.mean()

unemp_high_policy = policy_df[policy_df["Unemployment risk"] == "Cao"]["Hành động khuyến nghị"].value_counts()
ai_low_policy = policy_df[policy_df["AI capacity"] == "Thấp"]["Hành động khuyến nghị"].value_counts()
digital_low_policy = policy_df[policy_df["Digital index"] == "Thấp"]["Hành động khuyến nghị"].value_counts()

top_unemp_action = unemp_high_policy.index[0] if len(unemp_high_policy) > 0 else "không xác định"
top_ai_low_action = ai_low_policy.index[0] if len(ai_low_policy) > 0 else "không xác định"
top_digital_low_action = digital_low_policy.index[0] if len(digital_low_policy) > 0 else "không xác định"

st.markdown("#### Agent học được gì?")

st.write(
    f"""
    Sau quá trình huấn luyện, chính sách có reward trung bình cao nhất trong mô phỏng là {best_policy}.
    So với chính sách luôn chọn cân bằng, Q-learning chênh lệch khoảng {q_advantage_balanced:.4f} điểm reward trung bình.
    So với chính sách luôn chọn AI dẫn dắt, Q-learning chênh lệch khoảng {q_advantage_ai:.4f}; so với random policy, chênh lệch khoảng {q_advantage_random:.4f}.

    Điểm quan trọng là Q-learning không cố định một hành động cho mọi hoàn cảnh. Agent học cách điều chỉnh hành động theo trạng thái.
    Khi rủi ro thất nghiệp cao, hành động xuất hiện nhiều nhất là {top_unemp_action}. Khi năng lực AI thấp, hành động xuất hiện nhiều nhất là {top_ai_low_action}.
    Khi chỉ số số hóa thấp, hành động xuất hiện nhiều nhất là {top_digital_low_action}.
    Điều này cho thấy chính sách thích nghi có thể linh hoạt hơn chính sách rule-based cố định.
    """
)

st.markdown("#### Liên hệ thực tiễn Việt Nam")

st.write(
    """
    Với Việt Nam, logic này phù hợp với thực tế là các vùng, ngành và giai đoạn phát triển không có cùng năng lực hấp thụ công nghệ.
    Khi nền tảng số còn thấp, ưu tiên chuyển đổi số cơ bản có thể hợp lý hơn việc dồn mạnh vào AI. Khi năng lực AI đã cao và rủi ro lao động được kiểm soát,
    chính sách AI dẫn dắt có thể tạo tăng trưởng nhanh hơn. Khi rủi ro thất nghiệp tăng, chính sách bao trùm và đào tạo lại trở nên quan trọng hơn.

    Điều này gắn với Quyết định 127/QĐ-TTg về Chiến lược AI quốc gia, Quyết định 749/QĐ-TTg về chuyển đổi số quốc gia,
    Quyết định 411/QĐ-TTg về kinh tế số, xã hội số và Nghị quyết 57-NQ/TW về khoa học công nghệ, đổi mới sáng tạo và chuyển đổi số.
    Các định hướng này đều nhấn mạnh công nghệ, dữ liệu và nhân lực, nhưng bài học từ Q-learning là mức độ ưu tiên giữa các trụ cột này nên thay đổi theo trạng thái nền kinh tế.
    """
)

st.markdown("#### Q-learning agent hành động ra sao ở trạng thái ban đầu?")

initial_state = state_to_index((1, 1, 1, 1))
initial_action = int(np.argmax(q_table[initial_state]))

st.write(
    f"""
    Với trạng thái khởi đầu trung bình của Việt Nam 2026 trong mô phỏng, tức GDP growth trung bình,
    Digital index trung bình, AI capacity trung bình và Unemployment risk trung bình, agent khuyến nghị hành động
    {actions[initial_action]["name"]}. Đây không phải mệnh lệnh chính sách, mà là kết quả của hàm thưởng đã thiết kế:
    mô hình cân bằng tăng trưởng GDP với rủi ro thất nghiệp, rủi ro an ninh mạng và phát thải.
    """
)

st.markdown("#### So sánh với chính sách rule-based")

st.write(
    f"""
    So với chính sách luôn chọn a1 - Cân bằng, Q-learning chênh lệch {q_advantage_balanced:.4f} điểm reward trung bình.
    So với chính sách luôn chọn a3 - AI dẫn dắt, Q-learning chênh lệch {q_advantage_ai:.4f} điểm reward trung bình.
    So với random policy, Q-learning chênh lệch {q_advantage_random:.4f} điểm reward trung bình.
    Nếu Q-learning có kết quả tốt hơn, lý do chính là mô hình có khả năng đổi hành động theo trạng thái thay vì áp dụng một quy tắc cố định cho mọi hoàn cảnh.
    """
)


st.markdown("#### Q-learning có thay thế quyết định chính trị không?")

st.write(
    """
    Q-learning không thay thế quyết định chính trị. Mô hình chỉ học từ môi trường mô phỏng và hàm thưởng do con người thiết kế.
    Nếu hàm thưởng đặt trọng số quá cao cho tăng trưởng, agent có thể ưu tiên hành động tạo GDP nhưng làm tăng rủi ro xã hội.
    Nếu hàm thưởng đặt trọng số quá cao cho giảm thất nghiệp, agent có thể quá thận trọng và làm chậm đổi mới công nghệ.

    Vì vậy, giá trị của Q-learning nằm ở khả năng mô phỏng các lựa chọn chính sách thích nghi và kiểm tra hệ quả của từng hệ trọng số.
    Nhà hoạch định chính sách vẫn phải quyết định mục tiêu nào quan trọng hơn trong từng giai đoạn: tăng trưởng, bao trùm, an ninh dữ liệu hay môi trường.
    Nói cách khác, Q-learning là công cụ hỗ trợ thảo luận chính sách, không phải cơ chế tự động ra quyết định thay cho Nhà nước.
    """
)

st.markdown("#### Mở rộng DQN")

st.write(
    """
    Với 81 trạng thái, Q-learning dạng bảng là phù hợp vì không gian trạng thái còn nhỏ và dễ giải thích.
    Nếu mở rộng sang nhiều biến liên tục hơn, ví dụ GRDP, tỷ lệ thất nghiệp, năng lực dữ liệu, mức phát thải và rủi ro an ninh mạng theo thang số,
    Q-table sẽ trở nên quá lớn. Khi đó có thể thay bằng Deep Q-Network, dùng mạng neural để xấp xỉ hàm Q(s,a).

    Tuy nhiên, DQN có nhược điểm là khó giải thích hơn Q-table. Trong bài toán chính sách công, khả năng giải thích rất quan trọng vì quyết định ngân sách
    cần minh bạch, có thể kiểm chứng và có thể thảo luận. Vì vậy, trong bài này, Q-learning tabular là lựa chọn phù hợp để minh họa logic học tăng cường
    mà vẫn giữ được tính minh bạch.
    """
)

st.success(
    """
    Kết luận: Bài 11 cho thấy học tăng cường có thể mô phỏng cách chính sách kinh tế thích nghi theo trạng thái.
    Chính sách tốt không nhất thiết là luôn chọn AI, luôn chọn cân bằng hay luôn chọn bao trùm, mà là biết thay đổi hành động khi trạng thái kinh tế thay đổi.
    """
)