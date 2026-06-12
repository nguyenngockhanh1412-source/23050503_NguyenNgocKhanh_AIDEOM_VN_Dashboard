import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.aideom_ui import (
    setup_page,
    render_sidebar,
    page_header,
    info_box,
    section_caption,
    source_note,
    kpi_card,
    note_box,
)

# ======================================================
# PAGE SETUP
# ======================================================

setup_page("Bài 1 - Cobb-Douglas")
render_sidebar("Bài 1 - Cobb-Douglas")

page_header(
    "Bài 1. Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa",
    "Bài 1 sử dụng hàm sản xuất Cobb-Douglas mở rộng để mô hình hóa GDP Việt Nam. Mô hình bổ sung thêm các yếu tố mới gồm số hóa, năng lực AI và nhân lực số."
)

st.write(
    """
    Trọng tâm của bài là tính năng suất nhân tố tổng hợp, so sánh GDP thực tế với GDP dự báo,
    phân rã tăng trưởng và mô phỏng kịch bản GDP Việt Nam đến năm 2030.
    """
)

# ======================================================
# DATA
# ======================================================

data = {
    "Năm": [2020, 2021, 2022, 2023, 2024, 2025],
    "Y_GDP": [8044.4, 8487.5, 9513.3, 10221.8, 11511.9, 12847.6],
    "K": [16500, 17800, 19600, 21300, 23500, 25900],
    "L": [53.6, 50.5, 51.7, 52.4, 52.9, 53.4],
    "D": [12.0, 12.7, 14.3, 16.5, 18.3, 19.5],
    "AI": [55.6, 60.2, 65.4, 67.0, 73.8, 80.1],
    "H": [24.1, 26.1, 26.2, 27.0, 28.4, 29.2],
}

df = pd.DataFrame(data)

# ======================================================
# SECTION 1: MODEL
# ======================================================

st.markdown("---")
st.header("1. Mô hình toán học")

model_col, explain_col = st.columns([0.9, 1.1])

with model_col:
    st.latex(
        r"""
        Y_t = A_t \cdot K_t^{\alpha} \cdot L_t^{\beta}
        \cdot D_t^{\gamma} \cdot AI_t^{\delta} \cdot H_t^{\theta}
        """
    )

with explain_col:
    st.write(
        """
        Trong mô hình này, Y là GDP; K là vốn vật chất; L là lao động; D là mức độ số hóa;
        AI là năng lực ứng dụng trí tuệ nhân tạo; H là nhân lực số; còn A là năng suất nhân tố tổng hợp.
        Các hệ số α, β, γ, δ và θ phản ánh độ co giãn của GDP theo từng yếu tố đầu vào.
        """
    )

    st.write(
        """
        Mục tiêu của mô hình là mô phỏng cách các yếu tố truyền thống và yếu tố mới có thể đóng góp vào tăng trưởng GDP,
        từ đó hỗ trợ thảo luận về vai trò của số hóa, AI và nhân lực số trong phát triển kinh tế Việt Nam.
        """
    )

# ======================================================
# SECTION 2: DATA + CHART
# ======================================================

st.markdown("---")
st.header("2. Dữ liệu đầu vào")

source_note(
    """
    Bộ dữ liệu mô phỏng/tổng hợp trong AIDEOM-VN cho giai đoạn 2020–2025, dùng để minh họa hàm sản xuất Cobb-Douglas mở rộng.
    Các biến có thể thay thế bằng số liệu chính thức khi triển khai nghiên cứu thực tế.
    """
)

data_col, chart_col = st.columns([1.1, 0.9])

with data_col:
    st.dataframe(
        df.style.format({
            "Y_GDP": "{:,.1f}",
            "K": "{:,.0f}",
            "L": "{:.1f}",
            "D": "{:.1f}",
            "AI": "{:.1f}",
            "H": "{:.1f}",
        }),
        use_container_width=True
    )

with chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Năm"],
            y=df["Y_GDP"],
            mode="lines+markers",
            name="GDP",
            line=dict(color="#1FA7B6", width=2.5),
            marker=dict(size=8, color="#5FA8D3"),
            hovertemplate="Năm: %{x}<br>GDP: %{y:,.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Xu hướng GDP giai đoạn 2020–2025",
        height=320,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Năm",
        yaxis_title="GDP",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 3: PARAMETERS
# ======================================================

st.markdown("---")
st.header("3. Thiết lập hệ số mô hình")

section_caption(
    """
    Người dùng có thể điều chỉnh các hệ số co giãn để kiểm tra độ nhạy của kết quả.
    Khi tổng hệ số bằng 1, mô hình giả định lợi suất không đổi theo quy mô. Nếu tổng lớn hơn 1 hoặc nhỏ hơn 1,
    mô hình lần lượt phản ánh lợi suất tăng hoặc giảm theo quy mô.
    """
)

coef_col, status_col = st.columns([1.25, 0.75])

with coef_col:
    c1, c2, c3 = st.columns(3)

    with c1:
        alpha = st.slider("α - Vốn vật chất K", 0.00, 1.00, 0.33, 0.01)
        beta = st.slider("β - Lao động L", 0.00, 1.00, 0.42, 0.01)

    with c2:
        gamma = st.slider("γ - Số hóa D", 0.00, 1.00, 0.10, 0.01)
        delta = st.slider("δ - Năng lực AI", 0.00, 1.00, 0.08, 0.01)

    with c3:
        theta = st.slider("θ - Nhân lực số H", 0.00, 1.00, 0.07, 0.01)

total_coef = alpha + beta + gamma + delta + theta

with status_col:
    kpi_card(
        "Tổng hệ số",
        f"{total_coef:.2f}",
        "Tổng α + β + γ + δ + θ phản ánh giả định lợi suất theo quy mô."
    )

    if abs(total_coef - 1.0) < 0.001:
        st.success("Mô hình thỏa mãn lợi suất không đổi theo quy mô.")
    elif total_coef > 1:
        st.warning("Mô hình đang giả định lợi suất tăng theo quy mô.")
    else:
        st.warning("Mô hình đang giả định lợi suất giảm theo quy mô.")

# ======================================================
# SECTION 4: TFP + GDP FORECAST
# ======================================================

st.markdown("---")
st.header("4. Tính TFP và GDP dự báo")

Y = df["Y_GDP"].values
K = df["K"].values
L = df["L"].values
D = df["D"].values
AI = df["AI"].values
H = df["H"].values

A = Y / (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
A_mean = A.mean()

Y_hat = A_mean * (K**alpha * L**beta * D**gamma * AI**delta * H**theta)
mape = np.mean(np.abs((Y - Y_hat) / Y)) * 100

df_result = df.copy()
df_result["TFP_A_t"] = A
df_result["GDP_dự_báo"] = Y_hat
df_result["Sai_số_%"] = np.abs((Y - Y_hat) / Y) * 100

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    kpi_card("TFP trung bình", f"{A_mean:.4f}", "Giá trị A trung bình được dùng để tính GDP dự báo.")

with metric_col2:
    kpi_card("MAPE", f"{mape:.2f}%", "Sai số phần trăm tuyệt đối trung bình của mô hình.")

with metric_col3:
    kpi_card("GDP thực tế 2025", f"{Y[-1]:,.1f}", "Nghìn tỷ VND, dùng làm mốc so sánh kịch bản 2030.")

result_col, chart_col = st.columns([1.1, 0.9])

with result_col:
    st.dataframe(
        df_result.style.format({
            "Y_GDP": "{:,.1f}",
            "K": "{:,.0f}",
            "L": "{:.1f}",
            "D": "{:.1f}",
            "AI": "{:.1f}",
            "H": "{:.1f}",
            "TFP_A_t": "{:.4f}",
            "GDP_dự_báo": "{:,.1f}",
            "Sai_số_%": "{:.2f}",
        }),
        use_container_width=True
    )

with chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_result["Năm"],
            y=df_result["Y_GDP"],
            mode="lines+markers",
            name="GDP thực tế",
            line=dict(color="#1FA7B6", width=2.5),
            marker=dict(size=8),
            hovertemplate="Năm: %{x}<br>GDP thực tế: %{y:,.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_result["Năm"],
            y=df_result["GDP_dự_báo"],
            mode="lines+markers",
            name="GDP dự báo",
            line=dict(color="#82C3EC", width=2.5, dash="dash"),
            marker=dict(size=8),
            hovertemplate="Năm: %{x}<br>GDP dự báo: %{y:,.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="GDP thực tế và GDP dự báo",
        height=340,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Năm",
        yaxis_title="GDP",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 5: TFP TREND
# ======================================================

st.markdown("---")
st.header("5. Xu hướng năng suất nhân tố tổng hợp")

tfp_text_col, tfp_chart_col = st.columns([0.9, 1.1])

with tfp_text_col:
    section_caption(
        """
        TFP phản ánh phần tăng trưởng không được giải thích trực tiếp bởi các yếu tố đầu vào trong mô hình.
        Nếu TFP tăng, có thể hiểu là chất lượng tăng trưởng, hiệu quả sử dụng nguồn lực hoặc tác động lan tỏa công nghệ đang cải thiện.
        """
    )

    if A[-1] > A[0]:
        st.info("TFP có xu hướng tăng trong giai đoạn 2020–2025.")
    else:
        st.warning("TFP chưa cho thấy xu hướng tăng rõ rệt trong giai đoạn 2020–2025.")

with tfp_chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_result["Năm"],
            y=df_result["TFP_A_t"],
            mode="lines+markers",
            line=dict(color="#1FA7B6", width=2.5),
            marker=dict(size=8, color="#1FA7B6"),
            hovertemplate="Năm: %{x}<br>TFP A_t: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Xu hướng TFP A_t",
        height=310,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Năm",
        yaxis_title="TFP",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 6: GROWTH DECOMPOSITION
# ======================================================

st.markdown("---")
st.header("6. Phân rã tăng trưởng GDP")

section_caption(
    """
    Phân rã tăng trưởng cho biết mỗi yếu tố đóng góp bao nhiêu điểm phần trăm vào tăng trưởng GDP bình quân.
    Đây là cách đọc chính sách quan trọng vì cùng một mức tăng trưởng GDP có thể đến từ mở rộng vốn, tăng lao động,
    cải thiện số hóa, mở rộng AI, nâng nhân lực số hoặc tăng TFP.
    """
)

growth_Y = np.diff(np.log(Y))
growth_K = np.diff(np.log(K))
growth_L = np.diff(np.log(L))
growth_D = np.diff(np.log(D))
growth_AI = np.diff(np.log(AI))
growth_H = np.diff(np.log(H))
growth_A = np.diff(np.log(A))

avg_growth_Y = growth_Y.mean()

contrib_K = alpha * growth_K.mean()
contrib_L = beta * growth_L.mean()
contrib_D = gamma * growth_D.mean()
contrib_AI = delta * growth_AI.mean()
contrib_H = theta * growth_H.mean()
contrib_A = growth_A.mean()

decomp = pd.DataFrame({
    "Yếu tố": [
        "Vốn vật chất K",
        "Lao động L",
        "Số hóa D",
        "Năng lực AI",
        "Nhân lực số H",
        "TFP A",
    ],
    "Đóng góp điểm %": [
        contrib_K * 100,
        contrib_L * 100,
        contrib_D * 100,
        contrib_AI * 100,
        contrib_H * 100,
        contrib_A * 100,
    ],
})

decomp["Tỷ trọng đóng góp (%)"] = decomp["Đóng góp điểm %"] / (avg_growth_Y * 100) * 100

decomp_col, decomp_chart_col = st.columns([1, 1])

with decomp_col:
    st.dataframe(
        decomp.style.format({
            "Đóng góp điểm %": "{:.2f}",
            "Tỷ trọng đóng góp (%)": "{:.2f}",
        }),
        use_container_width=True
    )

with decomp_chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=decomp["Yếu tố"],
            y=decomp["Đóng góp điểm %"],
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Yếu tố: %{x}<br>Đóng góp: %{y:.2f} điểm %<extra></extra>",
        )
    )

    fig.update_layout(
        title="Phân rã tăng trưởng GDP",
        height=330,
        margin=dict(l=10, r=10, t=45, b=30),
        xaxis=dict(tickangle=0),
        yaxis_title="Đóng góp điểm %",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 7: 2030 SIMULATION
# ======================================================

st.markdown("---")
st.header("7. Mô phỏng GDP Việt Nam năm 2030")

section_caption(
    """
    Phần này cho phép người dùng thay đổi giả định năm 2030 về kinh tế số, năng lực AI, nhân lực số,
    tăng trưởng vốn - lao động và tăng trưởng TFP. Kết quả GDP 2030 sẽ thay đổi theo các giả định này.
    """
)

sim_col, sim_result_col = st.columns([1.1, 0.9])

with sim_col:
    s1, s2 = st.columns(2)

    with s1:
        D_2030 = st.slider("D năm 2030 - Kinh tế số/GDP (%)", 15.0, 40.0, 30.0, 0.5)
        AI_2030 = st.slider("AI năm 2030 - Nghìn DN số", 70.0, 150.0, 100.0, 1.0)
        H_2030 = st.slider("H năm 2030 - Lao động qua đào tạo (%)", 25.0, 50.0, 35.0, 0.5)

    with s2:
        growth_KL = st.slider("Tăng trưởng K và L mỗi năm (%)", 2.0, 10.0, 6.0, 0.5)
        growth_TFP = st.slider("Tăng trưởng TFP mỗi năm (%)", 0.0, 5.0, 1.2, 0.1)

years_ahead = 5

K_2030 = K[-1] * (1 + growth_KL / 100) ** years_ahead
L_2030 = L[-1] * (1 + growth_KL / 100) ** years_ahead
A_2030 = A[-1] * (1 + growth_TFP / 100) ** years_ahead

Y_2030 = A_2030 * (
    K_2030**alpha
    * L_2030**beta
    * D_2030**gamma
    * AI_2030**delta
    * H_2030**theta
)

growth_2025_2030 = (Y_2030 / Y[-1] - 1) * 100

with sim_result_col:
    kpi_card(
        "GDP dự báo năm 2030",
        f"{Y_2030:,.1f}",
        "Nghìn tỷ VND, thay đổi theo giả định người dùng."
    )

    kpi_card(
        "Tăng so với năm 2025",
        f"{growth_2025_2030:.1f}%",
        "Mức tăng tích lũy từ 2025 đến 2030 theo kịch bản mô phỏng."
    )

sim_path = pd.DataFrame({
    "Năm": [2025, 2030],
    "GDP": [Y[-1], Y_2030],
})

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=sim_path["Năm"],
        y=sim_path["GDP"],
        mode="lines+markers",
        line=dict(color="#1FA7B6", width=3),
        marker=dict(size=10, color="#5FA8D3"),
        hovertemplate="Năm: %{x}<br>GDP: %{y:,.1f}<extra></extra>",
    )
)

fig.update_layout(
    title="Quỹ đạo GDP mô phỏng từ 2025 đến 2030",
    height=300,
    margin=dict(l=10, r=10, t=45, b=20),
    xaxis_title="Năm",
    yaxis_title="GDP",
    plot_bgcolor="white",
    paper_bgcolor="white",
)

st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 8: POLICY INTERPRETATION
# ======================================================

st.markdown("---")
st.header("8. Diễn giải và thảo luận chính sách")

new_factors = decomp[decomp["Yếu tố"].isin(["Số hóa D", "Năng lực AI", "Nhân lực số H"])]
top_new_factor = new_factors.sort_values("Đóng góp điểm %", ascending=False).iloc[0]

if A[-1] > A[0]:
    tfp_comment = (
        "TFP có xu hướng tăng trong giai đoạn 2020–2025. Điều này cho thấy chất lượng tăng trưởng có dấu hiệu cải thiện, "
        "tức là tăng trưởng không chỉ đến từ mở rộng vốn và lao động mà còn đến từ hiệu quả sử dụng nguồn lực."
    )
else:
    tfp_comment = (
        "TFP chưa cho thấy xu hướng tăng rõ rệt. Điều này hàm ý tăng trưởng vẫn phụ thuộc nhiều vào mở rộng đầu vào, "
        "trong khi hiệu quả tổng hợp của công nghệ, quản trị và năng suất chưa cải thiện đủ mạnh."
    )

if mape < 5:
    mape_comment = f"MAPE = {mape:.2f}%, cho thấy mô hình có sai số thấp trong giai đoạn quan sát."
elif mape < 10:
    mape_comment = f"MAPE = {mape:.2f}%, cho thấy sai số ở mức chấp nhận được đối với một mô hình mô phỏng đơn giản."
else:
    mape_comment = f"MAPE = {mape:.2f}%, cho thấy sai số còn khá cao, cần thận trọng khi diễn giải kết quả dự báo."

if D_2030 >= 30 and growth_TFP >= 1.2:
    scenario_comment = (
        "Kịch bản 2030 tương đối tích cực vì kinh tế số mở rộng đi kèm cải thiện năng suất. "
        "Điều này phản ánh một hướng tăng trưởng có chất lượng hơn, trong đó số hóa không chỉ làm tăng quy mô hoạt động kinh tế "
        "mà còn hỗ trợ cải thiện hiệu quả sản xuất."
    )
elif D_2030 >= 30 and growth_TFP < 1.2:
    scenario_comment = (
        "Kinh tế số đạt mức cao, nhưng TFP tăng chậm. Điều này hàm ý rằng mở rộng số hóa chưa chắc tạo ra tăng trưởng mạnh "
        "nếu công nghệ không được chuyển hóa thành năng suất thực tế trong doanh nghiệp, khu vực công và thị trường lao động."
    )
else:
    scenario_comment = (
        "Kinh tế số chưa đạt mốc 30%, cho thấy cần tiếp tục tăng đầu tư cho hạ tầng số, dữ liệu, dịch vụ số và nhân lực số "
        "nếu muốn tạo nền tảng vững chắc cho tăng trưởng dựa trên công nghệ."
    )

# So sánh với kịch bản mặc định của phần mô phỏng 2030
D_2030_base = 30.0
AI_2030_base = 100.0
H_2030_base = 35.0
growth_KL_base = 6.0
growth_TFP_base = 1.2

K_2030_base = K[-1] * (1 + growth_KL_base / 100) ** years_ahead
L_2030_base = L[-1] * (1 + growth_KL_base / 100) ** years_ahead
A_2030_base = A[-1] * (1 + growth_TFP_base / 100) ** years_ahead

Y_2030_base = A_2030_base * (
    K_2030_base**alpha
    * L_2030_base**beta
    * D_2030_base**gamma
    * AI_2030_base**delta
    * H_2030_base**theta
)

diff_from_base = Y_2030 - Y_2030_base
diff_from_base_pct = diff_from_base / Y_2030_base * 100

if diff_from_base > 0:
    comparison_comment = (
        f"Khi so với kịch bản mặc định, thiết lập hiện tại làm GDP 2030 cao hơn khoảng {diff_from_base:,.1f} nghìn tỷ VND, "
        f"tương đương tăng {diff_from_base_pct:.2f}%. Điều này cho thấy các giả định người dùng đang kéo theo hướng tích cực hơn so với cấu hình nền."
    )
elif diff_from_base < 0:
    comparison_comment = (
        f"Khi so với kịch bản mặc định, thiết lập hiện tại làm GDP 2030 thấp hơn khoảng {abs(diff_from_base):,.1f} nghìn tỷ VND, "
        f"tương đương giảm {abs(diff_from_base_pct):.2f}%. Điều này cho thấy các giả định người dùng đang thận trọng hơn hoặc chưa đủ mạnh để tạo tăng trưởng cao."
    )
else:
    comparison_comment = (
        "Thiết lập hiện tại cho kết quả GDP 2030 gần như trùng với kịch bản mặc định. "
        "Điều này xảy ra khi các thanh tham số vẫn giữ ở mức nền ban đầu."
    )

st.markdown("#### Độ phù hợp của mô hình")

st.write(
    f"""
    {mape_comment} Với bài toán này, MAPE được dùng để kiểm tra mức chênh lệch giữa GDP thực tế và GDP dự báo trong giai đoạn 2020–2025. 
    Nếu sai số ở mức chấp nhận được, mô hình có thể được dùng như một công cụ mô phỏng chính sách. Tuy nhiên, kết quả không nên được hiểu là dự báo chính thức, 
    vì các hệ số trong mô hình đang do người dùng thiết lập thay vì được ước lượng bằng mô hình kinh tế lượng.
    """
)

st.markdown("#### Vai trò của TFP và chất lượng tăng trưởng")

st.write(
    f"""
    {tfp_comment} Trong bối cảnh Việt Nam, điểm này có ý nghĩa chính sách quan trọng. Nếu tăng trưởng chủ yếu đến từ mở rộng vốn vật chất, 
    nền kinh tế có thể tăng quy mô nhưng chưa chắc tăng hiệu quả. Ngược lại, nếu TFP cải thiện, điều đó hàm ý chuyển đổi số, đổi mới công nghệ, 
    quản trị và chất lượng nhân lực đang đóng vai trò lớn hơn trong tăng trưởng.
    """
)

st.markdown("#### Đóng góp của số hóa, AI và nhân lực số")

st.write(
    f"""
    Trong nhóm số hóa, AI và nhân lực số, yếu tố đóng góp lớn nhất theo thiết lập hiện tại là {top_new_factor["Yếu tố"]}, 
    với mức đóng góp khoảng {top_new_factor["Đóng góp điểm %"]:.2f} điểm phần trăm. Kết quả này thay đổi khi người dùng kéo các hệ số α, β, γ, δ và θ. 
    Nếu tăng hệ số của số hóa, AI hoặc nhân lực số, mô hình sẽ nhấn mạnh hơn vai trò của các yếu tố mới trong tăng trưởng. 
    Ngược lại, nếu tăng hệ số của vốn vật chất hoặc lao động, mô hình sẽ phản ánh một cấu trúc tăng trưởng truyền thống hơn.
    """
)

st.markdown("#### So sánh kịch bản mô phỏng 2030")

st.write(
    f"""
    GDP dự báo năm 2030 theo thiết lập hiện tại đạt khoảng {Y_2030:,.1f} nghìn tỷ VND, tăng khoảng {growth_2025_2030:.1f}% so với năm 2025. 
    {scenario_comment}
    """
)

st.write(
    f"""
    {comparison_comment} Phần so sánh này giúp người dùng thấy rõ tác động của việc thay đổi các thanh tham số. 
    Nói cách khác, khi kéo các biến D, AI, H, tăng trưởng K-L hoặc tăng trưởng TFP, mô hình không chỉ thay đổi con số GDP 2030, 
    mà còn làm thay đổi cách diễn giải chính sách phía sau kết quả.
    """
)

st.markdown("#### Hàm ý chính sách")

st.write(
    """
    Từ góc nhìn chính sách, kết quả Bài 1 gợi ý rằng tăng trưởng GDP trong kỷ nguyên AI không nên chỉ dựa vào mở rộng vốn vật chất và lao động. 
    Nếu Việt Nam muốn nâng chất lượng tăng trưởng, các chính sách về chuyển đổi số, phát triển AI, đào tạo nhân lực số và cải thiện TFP cần được triển khai đồng thời. 
    Số hóa tạo nền tảng dữ liệu và hạ tầng vận hành; AI có thể nâng năng suất và đổi mới mô hình kinh doanh; nhân lực số quyết định khả năng hấp thụ công nghệ; 
    còn TFP phản ánh hiệu quả tổng hợp của các yếu tố này trong nền kinh tế.
    """
)

st.write(
    """
    Tuy nhiên, mô hình cũng cho thấy đầu tư vào công nghệ không tự động tạo tăng trưởng cao. Nếu kinh tế số tăng nhưng TFP không cải thiện, 
    điều đó hàm ý công nghệ chưa được chuyển hóa thành hiệu quả sản xuất thực tế. Vì vậy, chính sách cần tránh cách tiếp cận chỉ tăng đầu tư thiết bị hoặc nền tảng số, 
    mà phải đi kèm cải cách quản trị, nâng kỹ năng lao động, tăng khả năng ứng dụng công nghệ của doanh nghiệp và cải thiện chất lượng dữ liệu.
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
    Lưu ý: Kết quả của Bài 1 là mô phỏng phục vụ phân tích. Mô hình Cobb-Douglas mở rộng giúp minh họa quan hệ giữa GDP
    và các yếu tố đầu vào, nhưng không thay thế dự báo kinh tế chính thức. Khi sử dụng cho nghiên cứu thực tế, cần kiểm định
    dữ liệu, ước lượng hệ số bằng phương pháp kinh tế lượng và phân tích độ nhạy.
    </p>
    """,
    unsafe_allow_html=True
)