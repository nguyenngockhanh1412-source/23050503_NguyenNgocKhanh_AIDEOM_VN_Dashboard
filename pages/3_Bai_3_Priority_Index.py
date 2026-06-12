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

setup_page("Bài 3 - Priority Index")
render_sidebar("Bài 3 - Priority Index")

st.title("Bài 3. Chỉ số ưu tiên ngành Priorityᵢ cho 10 ngành Việt Nam")

st.write(
    """
    Bài 3 xây dựng chỉ số ưu tiên ngành nhằm xác định ngành nào nên được ưu tiên chuyển đổi số 
    và ứng dụng AI trước. Mô hình sử dụng 7 tiêu chí: tăng trưởng, năng suất, lan tỏa, xuất khẩu, 
    việc làm, AI Readiness và rủi ro tự động hóa.
    """
)

st.latex(r"""
Priority_i = a_1Growth_i + a_2Productivity_i + a_3Spillover_i + a_4Export_i + a_5Employment_i + a_6AIReadiness_i - a_7Risk_i
""")

st.write(
    """
    Trong công thức, Growth phản ánh tốc độ tăng trưởng ngành; Productivity phản ánh năng suất; Spillover thể hiện tác động lan tỏa;
    Export phản ánh năng lực xuất khẩu; Employment đo quy mô việc làm; AI Readiness phản ánh mức độ sẵn sàng ứng dụng AI;
    còn Risk là rủi ro tự động hóa. Các hệ số a₁ đến a₇ là trọng số chính sách, thể hiện mức ưu tiên tương đối của từng tiêu chí.
    """
)

source_note(
    """
    Bộ dữ liệu 10 ngành và 7 tiêu chí được dùng để minh họa mô hình Priority Index trong bộ đề.
    Các chỉ số là dữ liệu mô phỏng/tổng hợp phục vụ mô phỏng và phân tích, không thay thế xếp hạng chính thức của các ngành kinh tế Việt Nam.
    """
)

# ======================================================
# DATA
# ======================================================

sector_data = {
    "Ngành": [
        "Nông-Lâm-Thủy sản",
        "CN chế biến chế tạo",
        "Xây dựng",
        "Khai khoáng",
        "Bán buôn-bán lẻ",
        "Tài chính-Ngân hàng",
        "Logistics-Vận tải",
        "CNTT-Truyền thông",
        "Giáo dục-Đào tạo",
        "Y tế",
    ],
    "Tăng trưởng (%)": [3.27, 9.64, 7.45, -1.20, 7.10, 7.36, 9.93, 7.85, 6.42, 6.85],
    "Năng suất": [103.4, 241.2, 168.8, 1290.5, 145.3, 1072.4, 321.4, 713.8, 205.7, 437.1],
    "Lan tỏa": [0.35, 0.78, 0.42, 0.30, 0.55, 0.85, 0.72, 0.92, 0.65, 0.60],
    "Xuất khẩu": [40.5, 290.9, 2.5, 8.2, 5.5, 1.2, 3.1, 178.0, 0.0, 0.0],
    "Việc làm": [13.20, 11.50, 4.80, 0.30, 7.80, 0.55, 1.95, 0.62, 2.15, 0.75],
    "AI Readiness": [15, 55, 20, 30, 48, 72, 42, 88, 38, 45],
    "Rủi ro tự động hóa": [18, 42, 25, 55, 38, 52, 35, 28, 22, 18],
}

df_sector = pd.DataFrame(sector_data)

good_cols = [
    "Tăng trưởng (%)",
    "Năng suất",
    "Lan tỏa",
    "Xuất khẩu",
    "Việc làm",
    "AI Readiness",
]

risk_col = "Rủi ro tự động hóa"


def minmax_good(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


def minmax_risk(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())


X_norm = df_sector.copy()

for col in good_cols:
    X_norm[col] = minmax_good(df_sector[col])

X_norm[risk_col] = minmax_risk(df_sector[risk_col])


def normalize_weights(raw_weights):
    total = sum(raw_weights.values())
    if total == 0:
        return {k: 1 / len(raw_weights) for k in raw_weights}
    return {k: v / total for k, v in raw_weights.items()}


def compute_priority(weights):
    w = normalize_weights(weights)

    score = (
        w["Tăng trưởng"] * X_norm["Tăng trưởng (%)"]
        + w["Năng suất"] * X_norm["Năng suất"]
        + w["Lan tỏa"] * X_norm["Lan tỏa"]
        + w["Xuất khẩu"] * X_norm["Xuất khẩu"]
        + w["Việc làm"] * X_norm["Việc làm"]
        + w["AI Readiness"] * X_norm["AI Readiness"]
        - w["Rủi ro"] * X_norm["Rủi ro tự động hóa"]
    )

    result = df_sector.copy()
    result["Priority Score"] = score
    result["Xếp hạng"] = result["Priority Score"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("Priority Score", ascending=False)

    return result, w


# ======================================================
# SECTION 1: DATA AND NORMALIZATION
# ======================================================

st.markdown("---")
st.header("1. Dữ liệu và chuẩn hóa")

section_caption(
    """
    Các tiêu chí có đơn vị đo khác nhau nên cần chuẩn hóa min-max trước khi tính Priority Score.
    Các tiêu chí tăng trưởng, năng suất, lan tỏa, xuất khẩu, việc làm và AI Readiness được xem là tiêu chí lợi ích.
    Rủi ro tự động hóa là tiêu chí bất lợi nên được đưa vào công thức với dấu trừ.
    """
)

data_col, chart_col = st.columns([1.05, 0.95])

with data_col:
    st.markdown("#### Dữ liệu gốc của 10 ngành")

    st.dataframe(
        df_sector.style.format({
            "Tăng trưởng (%)": "{:.2f}",
            "Năng suất": "{:.1f}",
            "Lan tỏa": "{:.2f}",
            "Xuất khẩu": "{:.1f}",
            "Việc làm": "{:.2f}",
            "AI Readiness": "{:.0f}",
            "Rủi ro tự động hóa": "{:.0f}",
        }),
        use_container_width=True,
    )

with chart_col:
    st.markdown("#### Heatmap chuẩn hóa")

    heat_cols = good_cols + [risk_col]
    heat_df = X_norm[["Ngành"] + heat_cols].copy()

    fig = go.Figure(
        data=go.Heatmap(
            z=heat_df[heat_cols].values,
            x=heat_cols,
            y=heat_df["Ngành"],
            colorscale=[
                [0, "#F8FBFF"],
                [0.25, "#81D8D0"],
                [0.50, "#1FA7B6"],
                [0.75, "#FF6B6B"],
                [1, "#1FA7B6"],
            ],
            colorbar=dict(title="Chuẩn hóa"),
            hovertemplate="Ngành: %{y}<br>Tiêu chí: %{x}<br>Giá trị chuẩn hóa: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Ma trận chuẩn hóa min-max",
        height=360,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=25),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Ma trận chuẩn hóa min-max")

st.dataframe(
    X_norm.style.format({
        "Tăng trưởng (%)": "{:.3f}",
        "Năng suất": "{:.3f}",
        "Lan tỏa": "{:.3f}",
        "Xuất khẩu": "{:.3f}",
        "Việc làm": "{:.3f}",
        "AI Readiness": "{:.3f}",
        "Rủi ro tự động hóa": "{:.3f}",
    }),
    use_container_width=True,
)

# ======================================================
# SECTION 2: WEIGHTS
# ======================================================

st.markdown("---")
st.header("2. Thiết lập bộ trọng số")

section_caption(
    """
    Các thanh kéo thể hiện mức ưu tiên tương đối của từng tiêu chí. Hệ thống sẽ tự chuẩn hóa tổng trọng số về 100%.
    Khi trọng số thay đổi, Priority Score và thứ tự ưu tiên ngành cũng thay đổi theo.
    """
)

weight_mode = st.selectbox(
    "Chọn định hướng trọng số",
    [
        "Mặc định theo đề bài",
        "Định hướng tăng trưởng",
        "Định hướng bao trùm",
        "Ưu tiên AI Readiness",
        "Tùy chỉnh thủ công",
    ],
)

if weight_mode == "Mặc định theo đề bài":
    default_weights = {
        "Tăng trưởng": 15,
        "Năng suất": 15,
        "Lan tỏa": 20,
        "Xuất khẩu": 15,
        "Việc làm": 10,
        "AI Readiness": 20,
        "Rủi ro": 15,
    }
    scenario_text = "Bộ trọng số mặc định cân bằng giữa tăng trưởng, lan tỏa, xuất khẩu, AI Readiness và rủi ro."

elif weight_mode == "Định hướng tăng trưởng":
    default_weights = {
        "Tăng trưởng": 25,
        "Năng suất": 25,
        "Lan tỏa": 10,
        "Xuất khẩu": 25,
        "Việc làm": 5,
        "AI Readiness": 10,
        "Rủi ro": 5,
    }
    scenario_text = "Bộ trọng số này ưu tiên tăng trưởng, năng suất và xuất khẩu."

elif weight_mode == "Định hướng bao trùm":
    default_weights = {
        "Tăng trưởng": 10,
        "Năng suất": 10,
        "Lan tỏa": 25,
        "Xuất khẩu": 5,
        "Việc làm": 25,
        "AI Readiness": 10,
        "Rủi ro": 25,
    }
    scenario_text = "Bộ trọng số này ưu tiên lan tỏa, việc làm và kiểm soát rủi ro tự động hóa."

elif weight_mode == "Ưu tiên AI Readiness":
    default_weights = {
        "Tăng trưởng": 10,
        "Năng suất": 10,
        "Lan tỏa": 15,
        "Xuất khẩu": 10,
        "Việc làm": 10,
        "AI Readiness": 35,
        "Rủi ro": 10,
    }
    scenario_text = "Bộ trọng số này nhấn mạnh mức độ sẵn sàng ứng dụng AI của từng ngành."

else:
    default_weights = {
        "Tăng trưởng": 15,
        "Năng suất": 15,
        "Lan tỏa": 20,
        "Xuất khẩu": 15,
        "Việc làm": 10,
        "AI Readiness": 20,
        "Rủi ro": 15,
    }
    scenario_text = "Người dùng có thể điều chỉnh thủ công từng trọng số để kiểm tra độ nhạy của xếp hạng."

st.info(scenario_text)

weight_col, summary_col = st.columns([1.15, 0.85])

with weight_col:
    w1, w2 = st.columns(2)

    with w1:
        raw_growth = st.slider("Tăng trưởng", 0, 50, default_weights["Tăng trưởng"], 1)
        raw_productivity = st.slider("Năng suất", 0, 50, default_weights["Năng suất"], 1)
        raw_spillover = st.slider("Lan tỏa", 0, 50, default_weights["Lan tỏa"], 1)
        raw_export = st.slider("Xuất khẩu", 0, 50, default_weights["Xuất khẩu"], 1)

    with w2:
        raw_employment = st.slider("Việc làm", 0, 50, default_weights["Việc làm"], 1)
        raw_ai = st.slider("AI Readiness", 0, 50, default_weights["AI Readiness"], 1)
        raw_risk = st.slider("Rủi ro tự động hóa", 0, 50, default_weights["Rủi ro"], 1)

raw_weights = {
    "Tăng trưởng": raw_growth,
    "Năng suất": raw_productivity,
    "Lan tỏa": raw_spillover,
    "Xuất khẩu": raw_export,
    "Việc làm": raw_employment,
    "AI Readiness": raw_ai,
    "Rủi ro": raw_risk,
}

priority_result, normalized_weights = compute_priority(raw_weights)

weight_df = pd.DataFrame({
    "Tiêu chí": list(normalized_weights.keys()),
    "Trọng số chuẩn hóa": list(normalized_weights.values()),
})

with summary_col:
    max_weight_name = max(normalized_weights, key=normalized_weights.get)
    max_weight_value = normalized_weights[max_weight_name]

    kpi_card(
        "Tiêu chí chi phối mạnh nhất",
        max_weight_name,
        f"Trọng số chuẩn hóa khoảng {max_weight_value * 100:.1f}%."
    )

    if max_weight_value >= 0.35:
        st.warning(
            f"Trọng số {max_weight_name} đang khá cao. Kết quả xếp hạng có thể bị chi phối mạnh bởi một tiêu chí."
        )
    elif max_weight_value >= 0.25:
        st.info(
            f"Tiêu chí nổi bật hiện tại là {max_weight_name}. Đây là lựa chọn phù hợp nếu muốn nhấn mạnh định hướng này."
        )
    else:
        st.success("Bộ trọng số hiện tại tương đối cân bằng.")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=weight_df["Tiêu chí"],
            y=weight_df["Trọng số chuẩn hóa"],
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Tiêu chí: %{x}<br>Trọng số: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Trọng số chuẩn hóa",
        height=260,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=25),
        yaxis_title="Trọng số",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### Bảng trọng số sau chuẩn hóa")

st.dataframe(
    weight_df.style.format({"Trọng số chuẩn hóa": "{:.3f}"}),
    use_container_width=True,
)

# ======================================================
# SECTION 3: RANKING
# ======================================================

st.markdown("---")
st.header("3. Xếp hạng ngành theo Priority Score")

show_cols = [
    "Xếp hạng",
    "Ngành",
    "Priority Score",
    "Tăng trưởng (%)",
    "Năng suất",
    "Lan tỏa",
    "Xuất khẩu",
    "Việc làm",
    "AI Readiness",
    "Rủi ro tự động hóa",
]

top3 = priority_result.head(3)

m1, m2, m3 = st.columns(3)

with m1:
    kpi_card("Top 1", top3.iloc[0]["Ngành"], f'Priority Score: {top3.iloc[0]["Priority Score"]:.3f}')

with m2:
    kpi_card("Top 2", top3.iloc[1]["Ngành"], f'Priority Score: {top3.iloc[1]["Priority Score"]:.3f}')

with m3:
    kpi_card("Top 3", top3.iloc[2]["Ngành"], f'Priority Score: {top3.iloc[2]["Priority Score"]:.3f}')

rank_col, rank_chart_col = st.columns([1.05, 0.95])

with rank_col:
    st.dataframe(
        priority_result[show_cols].style.format({
            "Priority Score": "{:.3f}",
            "Tăng trưởng (%)": "{:.2f}",
            "Năng suất": "{:.1f}",
            "Lan tỏa": "{:.2f}",
            "Xuất khẩu": "{:.1f}",
            "Việc làm": "{:.2f}",
            "AI Readiness": "{:.0f}",
            "Rủi ro tự động hóa": "{:.0f}",
        }),
        use_container_width=True,
    )

with rank_chart_col:
    sorted_plot = priority_result.sort_values("Priority Score", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=sorted_plot["Priority Score"],
            y=sorted_plot["Ngành"],
            orientation="h",
            marker_color="#1FA7B6",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Ngành: %{y}<br>Priority Score: %{x:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Xếp hạng ngành theo Priority Score",
        height=420,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Priority Score",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 4: AI READINESS SENSITIVITY
# ======================================================

st.markdown("---")
st.header("4. Phân tích độ nhạy theo AI Readiness")

st.write(
    """
    Phần này thay đổi trọng số AI Readiness từ 0,05 đến 0,40. Các trọng số còn lại được điều chỉnh theo tỷ lệ tương ứng 
    để tổng trọng số vẫn bằng 1. Mục tiêu là kiểm tra xem top 3 ngành có ổn định khi mức ưu tiên AI thay đổi hay không.
    """
)

ai_weight_values = np.arange(0.05, 0.45, 0.05)
sensitivity_rows = []

base_no_ai_sum = 1 - normalized_weights["AI Readiness"]

for ai_w in ai_weight_values:
    adjusted_weights = {}

    for key, value in normalized_weights.items():
        if key == "AI Readiness":
            adjusted_weights[key] = ai_w
        else:
            if base_no_ai_sum > 0:
                adjusted_weights[key] = value / base_no_ai_sum * (1 - ai_w)
            else:
                adjusted_weights[key] = (1 - ai_w) / 6

    temp_result, _ = compute_priority(adjusted_weights)
    top_sectors = temp_result.head(3)["Ngành"].tolist()

    sensitivity_rows.append({
        "Trọng số AI Readiness": ai_w,
        "Top 1": top_sectors[0],
        "Top 2": top_sectors[1],
        "Top 3": top_sectors[2],
    })

sensitivity_df = pd.DataFrame(sensitivity_rows)

sens_col, sens_chart_col = st.columns([1, 1])

with sens_col:
    st.dataframe(
        sensitivity_df.style.format({"Trọng số AI Readiness": "{:.2f}"}),
        use_container_width=True,
    )

with sens_chart_col:
    top1_counts = sensitivity_df["Top 1"].value_counts().reset_index()
    top1_counts.columns = ["Ngành", "Số lần xuất hiện Top 1"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=top1_counts["Ngành"],
            y=top1_counts["Số lần xuất hiện Top 1"],
            marker_color="#0B1D33",
            marker_line=dict(color="#1FA7B6", width=1),
            hovertemplate="Ngành: %{x}<br>Số lần Top 1: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Độ ổn định của Top 1 khi thay đổi trọng số AI",
        height=320,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=15),
        yaxis_title="Số lần xuất hiện",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 5: COMPARISON BETWEEN WEIGHT ORIENTATIONS
# ======================================================

st.markdown("---")
st.header("5. So sánh hai định hướng trọng số")

growth_weights = {
    "Tăng trưởng": 25,
    "Năng suất": 25,
    "Lan tỏa": 10,
    "Xuất khẩu": 25,
    "Việc làm": 5,
    "AI Readiness": 10,
    "Rủi ro": 5,
}

inclusive_weights = {
    "Tăng trưởng": 10,
    "Năng suất": 10,
    "Lan tỏa": 25,
    "Xuất khẩu": 5,
    "Việc làm": 25,
    "AI Readiness": 10,
    "Rủi ro": 25,
}

growth_result, _ = compute_priority(growth_weights)
inclusive_result, _ = compute_priority(inclusive_weights)

st.write(
    """
    Hai bảng dưới đây so sánh top 3 ngành theo hai cách nhìn khác nhau: một bên nhấn mạnh tăng trưởng, năng suất và xuất khẩu;
    bên còn lại nhấn mạnh việc làm, lan tỏa và giảm rủi ro tự động hóa.
    """
)

compare_df = pd.DataFrame({
    "Hạng": [1, 2, 3],
    "Định hướng tăng trưởng": growth_result.head(3)["Ngành"].tolist(),
    "Score tăng trưởng": growth_result.head(3)["Priority Score"].tolist(),
    "Định hướng bao trùm": inclusive_result.head(3)["Ngành"].tolist(),
    "Score bao trùm": inclusive_result.head(3)["Priority Score"].tolist(),
})

compare_col, compare_chart_col = st.columns([1, 1])

with compare_col:
    st.dataframe(
        compare_df.style.format({
            "Score tăng trưởng": "{:.3f}",
            "Score bao trùm": "{:.3f}",
        }),
        use_container_width=True,
    )

with compare_chart_col:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=compare_df["Hạng"].astype(str),
            y=compare_df["Score tăng trưởng"],
            name="Định hướng tăng trưởng",
            marker_color="#1FA7B6",
            hovertemplate="Hạng: %{x}<br>Score tăng trưởng: %{y:.3f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=compare_df["Hạng"].astype(str),
            y=compare_df["Score bao trùm"],
            name="Định hướng bao trùm",
            marker_color="#E6F7F5",
            hovertemplate="Hạng: %{x}<br>Score bao trùm: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="So sánh Priority Score theo hai định hướng",
        height=320,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Hạng",
        yaxis_title="Priority Score",
        barmode="group",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 6: POLICY INTERPRETATION
# ======================================================

st.markdown("---")
st.header("6. Diễn giải và thảo luận chính sách")

top1 = priority_result.iloc[0]
top2 = priority_result.iloc[1]
top3_sector = priority_result.iloc[2]

ai_weight = normalized_weights["AI Readiness"]
growth_orientation = (
    normalized_weights["Tăng trưởng"]
    + normalized_weights["Năng suất"]
    + normalized_weights["Xuất khẩu"]
)

inclusive_orientation = (
    normalized_weights["Việc làm"]
    + normalized_weights["Lan tỏa"]
    + normalized_weights["Rủi ro"]
)

if growth_orientation > inclusive_orientation + 0.10:
    orientation_comment = (
        "Bộ trọng số hiện tại nghiêng về định hướng tăng trưởng. Mô hình đang ưu tiên các ngành có tốc độ tăng trưởng cao, "
        "năng suất tốt và khả năng xuất khẩu lớn. Cách tiếp cận này phù hợp nếu mục tiêu chính là thúc đẩy GDP, năng lực cạnh tranh và hiệu quả kinh tế."
    )
elif inclusive_orientation > growth_orientation + 0.10:
    orientation_comment = (
        "Bộ trọng số hiện tại nghiêng về định hướng bao trùm. Mô hình không chỉ quan tâm đến hiệu quả kinh tế, "
        "mà còn chú trọng khả năng tạo việc làm, tác động lan tỏa và hạn chế rủi ro tự động hóa."
    )
else:
    orientation_comment = (
        "Bộ trọng số hiện tại tương đối cân bằng giữa mục tiêu tăng trưởng và mục tiêu bao trùm. Đây là cách tiếp cận thận trọng, "
        "vì không để một nhóm tiêu chí duy nhất chi phối kết quả xếp hạng."
    )

if ai_weight >= 0.30:
    ai_comment = (
        "Trọng số AI Readiness đang cao, nên kết quả có xu hướng ưu tiên những ngành đã có nền tảng công nghệ tốt. "
        "Điều này phù hợp nếu chính sách muốn tạo các cực tăng trưởng nhanh, nhưng có thể khiến các ngành truyền thống bị xếp thấp hơn."
    )
elif ai_weight >= 0.15:
    ai_comment = (
        "Trọng số AI Readiness ở mức vừa phải. Mô hình vẫn ghi nhận vai trò của năng lực công nghệ, "
        "nhưng không bỏ qua các tiêu chí khác như việc làm, xuất khẩu, năng suất và lan tỏa."
    )
else:
    ai_comment = (
        "Trọng số AI Readiness đang thấp. Kết quả xếp hạng lúc này ít phản ánh năng lực sẵn sàng AI hiện tại, "
        "mà thiên nhiều hơn về các đặc điểm kinh tế - xã hội truyền thống."
    )

sensitivity_top1_unique = sensitivity_df["Top 1"].nunique()

if sensitivity_top1_unique == 1:
    sensitivity_comment = (
        "Kết quả Top 1 khá ổn định khi thay đổi trọng số AI Readiness. Điều này cho thấy ngành dẫn đầu không chỉ phụ thuộc vào tiêu chí AI, "
        "mà có lợi thế tương đối đồng đều trên nhiều tiêu chí."
    )
else:
    sensitivity_comment = (
        "Top 1 thay đổi khi trọng số AI Readiness thay đổi. Điều này cho thấy kết quả xếp hạng khá nhạy với cách xác định ưu tiên chính sách."
    )

khai_khoang_row = priority_result[priority_result["Ngành"] == "Khai khoáng"].iloc[0]
khai_khoang_rank = int(khai_khoang_row["Xếp hạng"])
khai_khoang_score = khai_khoang_row["Priority Score"]

st.markdown("#### Định hướng chính sách của bộ trọng số")

st.write(
    f"""
    {orientation_comment} Vì vậy, cùng một bộ dữ liệu ngành nhưng khi thay đổi trọng số, mô hình có thể tạo ra thứ tự ưu tiên khác nhau.
    Đây là điểm quan trọng của Priority Index: kết quả xếp hạng không chỉ phản ánh dữ liệu, mà còn phản ánh lựa chọn mục tiêu chính sách.
    """
)

st.markdown("#### Vai trò của AI Readiness")

st.write(
    f"""
    {ai_comment} Trong bối cảnh chuyển đổi số và phát triển AI, AI Readiness là tiêu chí quan trọng vì nó cho biết ngành nào có khả năng hấp thụ công nghệ nhanh hơn.
    Tuy nhiên, nếu quá nhấn mạnh AI Readiness, chính sách có thể ưu tiên các ngành đã mạnh sẵn về công nghệ và bỏ qua các ngành cần hỗ trợ chuyển đổi để tránh tụt hậu.
    """
)

st.markdown("#### Ngành đứng đầu và nhóm ngành ưu tiên")

st.write(
    f"""
    Với bộ trọng số hiện tại, ngành có Priority Score cao nhất là {top1["Ngành"]}, đạt {top1["Priority Score"]:.3f} điểm.
    Top 3 hiện tại gồm {top1["Ngành"]}, {top2["Ngành"]} và {top3_sector["Ngành"]}.
    Nhóm này có thể được xem là nhóm ngành nên ưu tiên triển khai chuyển đổi số và AI trước, vì kết hợp được nhiều lợi thế về tăng trưởng,
    năng suất, lan tỏa hoặc mức độ sẵn sàng công nghệ.
    """
)

st.write(
    """
    Liên hệ với tinh thần của Nghị quyết 57-NQ/TW, kết quả này có thể được hiểu theo hướng ưu tiên những ngành có khả năng tạo động lực mới
    từ khoa học công nghệ, đổi mới sáng tạo và chuyển đổi số. Tuy nhiên, Priority Index không nên được xem là danh sách ưu tiên chính thức.
    Trong thực tế, việc lựa chọn ngành ưu tiên còn cần xét thêm vai trò chiến lược của ngành, năng lực doanh nghiệp, mức độ sẵn sàng dữ liệu,
    khả năng triển khai chính sách và tác động xã hội.
    """
)

st.markdown("#### Vì sao Khai khoáng có năng suất cao nhưng không nhất thiết vào nhóm ưu tiên?")

st.write(
    f"""
    Trong bộ dữ liệu hiện tại, Khai khoáng có năng suất cao nhưng xếp hạng tổng hợp là hạng {khai_khoang_rank}, với Priority Score khoảng {khai_khoang_score:.3f}.
    Lý do là Priority Index không chỉ xét riêng năng suất. Một ngành có năng suất cao vẫn có thể bị xếp thấp hơn nếu điểm lan tỏa thấp,
    quy mô việc làm nhỏ, rủi ro tự động hóa cao, mức độ sẵn sàng AI không nổi bật hoặc không phù hợp với định hướng chuyển đổi số rộng hơn.
    """
)

st.write(
    """
    Điều này cho thấy ưu tiên chính sách không nên chỉ dựa trên một tiêu chí đơn lẻ. Nếu chỉ nhìn vào năng suất, Khai khoáng có thể có vẻ hấp dẫn.
    Nhưng nếu mục tiêu là chuyển đổi số, lan tỏa công nghệ, tạo việc làm chất lượng và phát triển năng lực AI dài hạn, các ngành như công nghiệp chế biến chế tạo,
    CNTT - truyền thông, tài chính - ngân hàng hoặc logistics có thể phù hợp hơn tùy bộ trọng số.
    """
)

st.markdown("#### Độ nhạy của kết quả")

st.write(
    f"""
    {sensitivity_comment} Vì vậy, Priority Score không nên được hiểu như một xếp hạng tuyệt đối,
    mà là công cụ hỗ trợ thảo luận chính sách. Khi mục tiêu chính sách thay đổi, thứ tự ưu tiên ngành cũng có thể thay đổi theo.
    """
)

st.markdown("#### Ai nên quyết định bộ trọng số?")

st.write(
    """
    Bộ trọng số không nên do riêng mô hình hay riêng chuyên gia kỹ thuật quyết định. Chuyên gia dữ liệu có thể đề xuất tiêu chí, phương pháp chuẩn hóa
    và kiểm định độ nhạy. Chuyên gia ngành có thể đánh giá tính thực tế của từng tiêu chí. Cơ quan hoạch định chính sách cần xác định mục tiêu ưu tiên
    trong từng giai đoạn, ví dụ ưu tiên tăng trưởng, xuất khẩu, đổi mới sáng tạo, việc làm hay kiểm soát rủi ro tự động hóa.
    """
)

st.write(
    """
    Dưới góc độ chính phủ, trọng số nên được quyết định thông qua một quy trình minh bạch hơn là một lựa chọn kỹ thuật thuần túy.
    Có thể kết hợp hội đồng chuyên gia, cơ quan quản lý, đại diện doanh nghiệp, tổ chức đào tạo và tham vấn công khai.
    Cách làm này giúp tăng tính chính danh của kết quả xếp hạng, vì phân bổ ưu tiên ngành có thể ảnh hưởng đến nguồn lực đầu tư,
    cơ hội của doanh nghiệp và định hướng lao động trong nền kinh tế.
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
    Lưu ý: Kết quả của Bài 3 là mô phỏng phục vụ phân tích. Priority Index giúp lượng hóa thứ tự ưu tiên ngành theo một bộ trọng số nhất định,
    nhưng không thay thế quá trình đánh giá chính sách thực tế, vốn cần thêm dữ liệu chính thức, tham vấn chuyên gia và kiểm định độ nhạy.
    </p>
    """,
    unsafe_allow_html=True,
)