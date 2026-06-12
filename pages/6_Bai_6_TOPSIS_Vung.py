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

setup_page("Bài 6 - TOPSIS vùng")
render_sidebar("Bài 6 - TOPSIS vùng")

st.title("Bài 6. Regional AI Investment Map bằng TOPSIS")

st.write(
    """
    Dashboard này đánh giá mức độ ưu tiên đầu tư AI và chuyển đổi số cho 6 vùng kinh tế - xã hội của Việt Nam.
    Kết quả được trình bày theo bốn lớp: tổng quan xếp hạng, định hướng phân bổ chính sách, so sánh kịch bản
    và cảnh báo rủi ro vùng.
    """
)

source_note(
    """
    File vietnam_regions_2024.csv trong thư mục data. Đây là bộ dữ liệu đầu vào theo đề bài, dùng để tính TOPSIS,
    trọng số Entropy, AHP đơn giản và phân tích rủi ro chính sách.
    """
)

# ======================================================
# LOAD DATA
# ======================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_FILE = DATA_DIR / "vietnam_regions_2024.csv"

if not DATA_FILE.exists():
    st.error(f"Không tìm thấy file dữ liệu: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)

required_cols = [
    "region",
    "grdp_per_capita",
    "fdi",
    "digital_index",
    "ai_readiness",
    "trained_labor",
    "rd_grdp",
    "internet",
    "gini",
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error("File vietnam_regions_2024.csv đang thiếu các cột sau:")
    st.write(missing_cols)
    st.stop()

criteria = [
    "grdp_per_capita",
    "fdi",
    "digital_index",
    "ai_readiness",
    "trained_labor",
    "rd_grdp",
    "internet",
    "gini",
]

criteria_vn = {
    "grdp_per_capita": "GRDP/người",
    "fdi": "FDI",
    "digital_index": "Digital Index",
    "ai_readiness": "AI Readiness",
    "trained_labor": "LĐ đào tạo",
    "rd_grdp": "R&D/GRDP",
    "internet": "Internet",
    "gini": "Gini",
}

benefit_criteria = [
    "grdp_per_capita",
    "fdi",
    "digital_index",
    "ai_readiness",
    "trained_labor",
    "rd_grdp",
    "internet",
]

cost_criteria = ["gini"]

expert_weights = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])

BLUE = "#1FA7B6"
BLUE_2 = "#5FA8D3"
BLUE_3 = "#1FA7B6"
BLUE_4 = "#81D8D0"
GREEN = "#E6F7F5"
SKY = "#EAF7FF"
PINK = "#FDE2E4"


# ======================================================
# FUNCTIONS
# ======================================================

def topsis(data, weights, benefit_cols, cost_cols):
    X = data.astype(float).values
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    denominator = np.sqrt((X ** 2).sum(axis=0))
    denominator[denominator == 0] = 1

    normalized = X / denominator
    weighted = normalized * weights

    ideal_best = []
    ideal_worst = []

    for j, col in enumerate(data.columns):
        if col in benefit_cols:
            ideal_best.append(weighted[:, j].max())
            ideal_worst.append(weighted[:, j].min())
        else:
            ideal_best.append(weighted[:, j].min())
            ideal_worst.append(weighted[:, j].max())

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    d_plus = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_minus = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = d_minus / (d_plus + d_minus)

    return score, d_plus, d_minus, normalized, weighted


def entropy_weights(data, benefit_cols, cost_cols):
    X = data.astype(float).copy()
    norm = pd.DataFrame(index=X.index)

    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()

        if max_val == min_val:
            norm[col] = 1
        else:
            if col in benefit_cols:
                norm[col] = (X[col] - min_val) / (max_val - min_val)
            else:
                norm[col] = (max_val - X[col]) / (max_val - min_val)

    norm = norm + 1e-12
    P = norm / norm.sum(axis=0)

    n = len(norm)
    k = 1 / np.log(n)

    entropy = -k * (P * np.log(P)).sum(axis=0)
    diversity = 1 - entropy

    if diversity.sum() == 0:
        weights = np.ones(len(diversity)) / len(diversity)
    else:
        weights = diversity / diversity.sum()

    detail = pd.DataFrame({
        "Tiêu chí": [criteria_vn[c] for c in X.columns],
        "Entropy": entropy.values,
        "Mức phân hóa": diversity.values,
        "Trọng số Entropy": weights.values,
    })

    return weights.values, detail


def ahp_from_expert_weights(base_weights):
    w = np.array(base_weights, dtype=float)
    n = len(w)

    pairwise = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            pairwise[i, j] = w[i] / w[j]

    eigenvalues, eigenvectors = np.linalg.eig(pairwise)
    max_index = np.argmax(eigenvalues.real)

    principal_value = eigenvalues[max_index].real
    principal_vector = np.abs(eigenvectors[:, max_index].real)
    ahp_w = principal_vector / principal_vector.sum()

    ci = (principal_value - n) / (n - 1)

    ri_table = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }

    ri = ri_table.get(n, 1.49)
    cr = 0 if ri == 0 else ci / ri

    return ahp_w, pairwise, ci, cr


def weighted_sum(data, weights, benefit_cols, cost_cols):
    X = data.astype(float).copy()
    norm = pd.DataFrame(index=X.index)

    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()

        if max_val == min_val:
            norm[col] = 1
        else:
            if col in benefit_cols:
                norm[col] = (X[col] - min_val) / (max_val - min_val)
            else:
                norm[col] = (max_val - X[col]) / (max_val - min_val)

    score = norm.values @ weights
    return score, norm


def make_topsis_result(dataframe, weights, prefix):
    score, d_plus, d_minus, normalized, weighted = topsis(
        dataframe[criteria],
        weights,
        benefit_criteria,
        cost_criteria,
    )

    result = dataframe.copy()
    result[f"{prefix}_score"] = score
    result[f"{prefix}_d_plus"] = d_plus
    result[f"{prefix}_d_minus"] = d_minus
    result[f"{prefix}_rank"] = result[f"{prefix}_score"].rank(
        ascending=False,
        method="dense",
    ).astype(int)

    result = result.sort_values(f"{prefix}_rank").reset_index(drop=True)

    normalized_df = pd.DataFrame(normalized, columns=criteria)
    weighted_df = pd.DataFrame(weighted, columns=criteria)

    normalized_df.insert(0, "region", dataframe["region"])
    weighted_df.insert(0, "region", dataframe["region"])

    return result, normalized_df, weighted_df


def classify_group(row):
    score = row["expert_score"]
    ai = row["ai_readiness"]
    digital = row["digital_index"]
    gini = row["gini"]

    if score >= 0.70 and ai >= 65 and digital >= 70:
        return "Cực tăng trưởng AI"
    if ai < 30 or digital < 45:
        return "Vùng cần bắt kịp số"
    if gini >= 0.40:
        return "Ưu tiên bao trùm"
    return "Nâng cấp có chọn lọc"


def policy_direction(group):
    if group == "Cực tăng trưởng AI":
        return (
            "Phù hợp triển khai các dự án AI có yêu cầu nền tảng cao như dữ liệu dùng chung, dịch vụ công thông minh, "
            "sandbox AI, ứng dụng AI trong công nghiệp, logistics và tài chính."
        )

    if group == "Vùng cần bắt kịp số":
        return (
            "Nên ưu tiên hạ tầng số, Internet, dữ liệu cơ bản, kỹ năng số và hỗ trợ doanh nghiệp nhỏ trước khi mở rộng "
            "các dự án AI phức tạp."
        )

    if group == "Ưu tiên bao trùm":
        return (
            "Cần gắn đầu tư số với đào tạo lại lao động, hỗ trợ nhóm dễ bị tổn thương và kiểm soát nguy cơ gia tăng "
            "chênh lệch trong quá trình chuyển đổi số."
        )

    return (
        "Có thể triển khai nâng cấp từng bước, ưu tiên các lĩnh vực có khả năng hấp thụ công nghệ và có tác động lan tỏa rõ."
    )


def risk_flags(row):
    flags = []

    if row["ai_readiness"] < 30:
        flags.append("AI readiness thấp")
    if row["digital_index"] < 45:
        flags.append("Digital Index thấp")
    if row["trained_labor"] < 22:
        flags.append("Lao động đào tạo thấp")
    if row["internet"] < 75:
        flags.append("Internet thấp")
    if row["gini"] >= 0.40:
        flags.append("Gini cao")

    if len(flags) == 0:
        return "Không có cảnh báo lớn theo ngưỡng mô hình"

    return "; ".join(flags)


def risk_level(row):
    flag_text = risk_flags(row)

    if "Không có" in flag_text:
        return "Theo dõi"

    count = len(flag_text.split(";"))

    if count >= 3:
        return "Cao"
    if count == 2:
        return "Trung bình"
    return "Thấp"


def list_text(items):
    return ", ".join(items) if len(items) > 0 else "không có vùng nào theo ngưỡng hiện tại"


def format_display_df(dataframe):
    return dataframe.style.format({
        "GRDP/người": "{:.1f}",
        "FDI": "{:.1f}",
        "Digital Index": "{:.0f}",
        "AI Readiness": "{:.0f}",
        "LĐ đào tạo": "{:.1f}",
        "R&D/GRDP": "{:.2f}",
        "Internet": "{:.0f}",
        "Gini": "{:.3f}",
    })


# ======================================================
# COMPUTE RESULTS
# ======================================================

expert_result, expert_norm, expert_weighted = make_topsis_result(df, expert_weights, "expert")

entropy_w, entropy_detail = entropy_weights(
    df[criteria],
    benefit_criteria,
    cost_criteria,
)

entropy_result, entropy_norm, entropy_weighted = make_topsis_result(df, entropy_w, "entropy")

ahp_w, ahp_matrix, ahp_ci, ahp_cr = ahp_from_expert_weights(expert_weights)

ahp_score, ahp_norm = weighted_sum(
    df[criteria],
    ahp_w,
    benefit_criteria,
    cost_criteria,
)

ahp_result = df.copy()
ahp_result["ahp_score"] = ahp_score
ahp_result["ahp_rank"] = ahp_result["ahp_score"].rank(
    ascending=False,
    method="dense",
).astype(int)
ahp_result = ahp_result.sort_values("ahp_rank").reset_index(drop=True)

expert_result["policy_group"] = expert_result.apply(classify_group, axis=1)
expert_result["policy_direction"] = expert_result["policy_group"].apply(policy_direction)
expert_result["risk_flags"] = expert_result.apply(risk_flags, axis=1)
expert_result["risk_level"] = expert_result.apply(risk_level, axis=1)

comparison = pd.DataFrame({
    "Vùng": df["region"],
    "Hạng chuyên gia": df["region"].map(dict(zip(expert_result["region"], expert_result["expert_rank"]))),
    "Điểm chuyên gia": df["region"].map(dict(zip(expert_result["region"], expert_result["expert_score"]))),
    "Hạng Entropy": df["region"].map(dict(zip(entropy_result["region"], entropy_result["entropy_rank"]))),
    "Điểm Entropy": df["region"].map(dict(zip(entropy_result["region"], entropy_result["entropy_score"]))),
    "Hạng AHP": df["region"].map(dict(zip(ahp_result["region"], ahp_result["ahp_rank"]))),
    "Điểm AHP": df["region"].map(dict(zip(ahp_result["region"], ahp_result["ahp_score"]))),
})

comparison["Chênh lệch Expert - Entropy"] = (
    comparison["Hạng Entropy"] - comparison["Hạng chuyên gia"]
)

# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Tổng quan", "Phân bổ", "Kịch bản so sánh", "Cảnh báo rủi ro"]
)

# ======================================================
# TAB 1
# ======================================================

with tab1:
    st.header("1. Tổng quan")

    section_caption(
        """
        Phần này trình bày dữ liệu đầu vào, trọng số chuyên gia và kết quả xếp hạng 6 vùng theo điểm ưu tiên.
        Gini được xử lý là tiêu chí chi phí, các tiêu chí còn lại được xử lý là tiêu chí lợi ích.
        """
    )

    display_df = df.rename(columns={
        "region": "Vùng",
        "grdp_per_capita": "GRDP/người",
        "fdi": "FDI",
        "digital_index": "Digital Index",
        "ai_readiness": "AI Readiness",
        "trained_labor": "LĐ đào tạo",
        "rd_grdp": "R&D/GRDP",
        "internet": "Internet",
        "gini": "Gini",
    })

    data_col, weight_col = st.columns([1.15, 0.85])

    with data_col:
        st.markdown("#### Dữ liệu đầu vào")
        st.dataframe(format_display_df(display_df), use_container_width=True)

    with weight_col:
        st.markdown("#### Trọng số chuyên gia")

        weight_df = pd.DataFrame({
            "Tiêu chí": [criteria_vn[c] for c in criteria],
            "Trọng số": expert_weights,
            "Loại tiêu chí": [
                "Lợi ích" if c in benefit_criteria else "Chi phí"
                for c in criteria
            ],
        })

        st.dataframe(
            weight_df.style.format({"Trọng số": "{:.2f}"}),
            use_container_width=True,
        )

    st.markdown("#### Kết quả xếp hạng")

    top_region = expert_result.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Vùng đứng đầu", top_region["region"], "Xếp hạng theo TOPSIS trọng số chuyên gia.")

    with c2:
        kpi_card("Điểm ưu tiên", f"{top_region['expert_score']:.3f}", "Cᵢ* càng cao thì càng gần nghiệm lý tưởng.")

    with c3:
        kpi_card("AI Readiness", f"{top_region['ai_readiness']:.0f}", "Mức độ sẵn sàng AI của vùng đứng đầu.")

    with c4:
        kpi_card("Digital Index", f"{top_region['digital_index']:.0f}", "Nền tảng chuyển đổi số của vùng đứng đầu.")

    result_table = expert_result[
        [
            "expert_rank",
            "region",
            "expert_score",
            "expert_d_plus",
            "expert_d_minus",
            "policy_group",
        ]
    ].rename(columns={
        "expert_rank": "Hạng",
        "region": "Vùng",
        "expert_score": "Điểm ưu tiên",
        "expert_d_plus": "Khoảng cách tới lý tưởng tốt",
        "expert_d_minus": "Khoảng cách tới lý tưởng xấu",
        "policy_group": "Nhóm chính sách",
    })

    rank_col, chart_col = st.columns([1.1, 0.9])

    with rank_col:
        st.dataframe(
            result_table.style.format({
                "Điểm ưu tiên": "{:.4f}",
                "Khoảng cách tới lý tưởng tốt": "{:.4f}",
                "Khoảng cách tới lý tưởng xấu": "{:.4f}",
            }),
            use_container_width=True,
        )

    with chart_col:
        chart_df = expert_result.sort_values("expert_score", ascending=True)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=chart_df["region"],
                x=chart_df["expert_score"],
                orientation="h",
                marker_color=BLUE_3,
                marker_line=dict(color=BLUE, width=1),
                hovertemplate="Vùng: %{y}<br>Điểm ưu tiên: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Xếp hạng vùng theo điểm ưu tiên",
            height=360,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Điểm ưu tiên Cᵢ*",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# TAB 2
# ======================================================

with tab2:
    st.header("2. Phân bổ")

    section_caption(
        """
        Phần này chuyển kết quả xếp hạng thành định hướng phân bổ chính sách. Cách hiểu không phải là chia đều ngân sách
        cho các vùng, mà là xác định vùng nào phù hợp với đầu tư AI bứt phá, vùng nào cần đầu tư nền tảng và vùng nào
        cần chính sách bao trùm đi kèm.
        """
    )

    policy_table = expert_result[[
        "expert_rank",
        "region",
        "expert_score",
        "digital_index",
        "ai_readiness",
        "trained_labor",
        "gini",
        "policy_group",
        "policy_direction",
    ]].rename(columns={
        "expert_rank": "Hạng",
        "region": "Vùng",
        "expert_score": "Điểm",
        "digital_index": "Digital Index",
        "ai_readiness": "AI Readiness",
        "trained_labor": "LĐ đào tạo",
        "gini": "Gini",
        "policy_group": "Nhóm chính sách",
        "policy_direction": "Định hướng",
    })

    table_col, map_col = st.columns([1.15, 0.85])

    with table_col:
        st.dataframe(
            policy_table.style.format({
                "Điểm": "{:.4f}",
                "Digital Index": "{:.0f}",
                "AI Readiness": "{:.0f}",
                "LĐ đào tạo": "{:.1f}",
                "Gini": "{:.3f}",
            }),
            use_container_width=True,
        )

    with map_col:
        color_map = {
            "Cực tăng trưởng AI": BLUE,
            "Vùng cần bắt kịp số": "#F4B860",
            "Ưu tiên bao trùm": "#F497A9",
            "Nâng cấp có chọn lọc": "#9AD9B5",
        }

        fig = go.Figure()

        for group in expert_result["policy_group"].unique():
            sub = expert_result[expert_result["policy_group"] == group]
            fig.add_trace(
                go.Scatter(
                    x=sub["digital_index"],
                    y=sub["ai_readiness"],
                    mode="markers+text",
                    text=sub["expert_rank"].astype(str),
                    textposition="top center",
                    name=group,
                    marker=dict(
                        size=np.sqrt(sub["grdp_per_capita"]) * 5,
                        color=color_map.get(group, BLUE_2),
                        line=dict(color="#334155", width=0.8),
                        opacity=0.85,
                    ),
                    customdata=np.stack(
                        [
                            sub["region"],
                            sub["expert_score"],
                            sub["trained_labor"],
                            sub["gini"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "Vùng: %{customdata[0]}<br>"
                        "Điểm TOPSIS: %{customdata[1]:.4f}<br>"
                        "Digital Index: %{x:.0f}<br>"
                        "AI Readiness: %{y:.0f}<br>"
                        "LĐ đào tạo: %{customdata[2]:.1f}%<br>"
                        "Gini: %{customdata[3]:.3f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Digital Index và AI Readiness theo vùng",
            height=390,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Digital Index",
            yaxis_title="AI Readiness",
            legend=dict(orientation="h", y=-0.22),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Diễn giải chính sách")

    ai_growth = expert_result[expert_result["policy_group"] == "Cực tăng trưởng AI"]["region"].tolist()
    catch_up = expert_result[expert_result["policy_group"] == "Vùng cần bắt kịp số"]["region"].tolist()
    inclusive = expert_result[expert_result["policy_group"] == "Ưu tiên bao trùm"]["region"].tolist()
    selective = expert_result[expert_result["policy_group"] == "Nâng cấp có chọn lọc"]["region"].tolist()

    st.write(
        f"""
        Kết quả cho thấy chính sách AI theo vùng không nên được thiết kế theo một công thức giống nhau cho mọi địa bàn.
        Quyết định 127/QĐ-TTg đặt mục tiêu phát triển nghiên cứu, phát triển và ứng dụng AI đến năm 2030; tuy nhiên,
        năng lực hấp thụ AI giữa các vùng không đồng đều. Vì vậy, nếu triển khai chính sách AI chỉ dựa trên tham vọng công nghệ
        mà không xét hạ tầng số, nhân lực, R&D và mức độ bất bình đẳng, chính sách có thể tạo ra khoảng cách triển khai rất lớn.

        Nhóm cực tăng trưởng AI gồm: {list_text(ai_growth)}. Đây là nhóm có điều kiện thuận lợi hơn để triển khai các dự án AI
        có tính bứt phá. Cách phân bổ phù hợp là ưu tiên các dự án có yêu cầu cao về dữ liệu, nhân lực và năng lực quản trị,
        ví dụ dịch vụ công thông minh, nền tảng dữ liệu dùng chung, ứng dụng AI trong sản xuất, logistics và tài chính.

        Nhóm vùng cần bắt kịp số gồm: {list_text(catch_up)}. Với nhóm này, vấn đề chính không phải là chưa nên đầu tư, mà là nên đầu tư khác cách.
        Nếu đưa ngay các dự án AI phức tạp vào vùng có Digital Index và AI Readiness thấp, rủi ro là dự án khó vận hành, thiếu dữ liệu,
        thiếu nhân lực triển khai và khó tạo tác động thực tế. Do đó, chính sách nên bắt đầu từ hạ tầng số, Internet, dữ liệu cơ bản,
        kỹ năng số và hỗ trợ doanh nghiệp nhỏ.

        Nhóm ưu tiên bao trùm gồm: {list_text(inclusive)}. Với nhóm này, đầu tư AI cần đi kèm chính sách xã hội và nhân lực.
        Nếu chỉ tập trung vào công nghệ, lợi ích chuyển đổi số có thể tập trung vào nhóm lao động có kỹ năng cao, trong khi nhóm yếu thế
        bị bỏ lại phía sau. Đây là đánh đổi quan trọng giữa hiệu quả công nghệ và công bằng phát triển vùng.

        Nhóm nâng cấp có chọn lọc gồm: {list_text(selective)}. Các vùng này có thể chọn một số lĩnh vực có khả năng hấp thụ tốt để thử nghiệm,
        đo lường kết quả và mở rộng sau khi hạ tầng và nhân lực được cải thiện.
        """
    )

# ======================================================
# TAB 3
# ======================================================

with tab3:
    st.header("3. Kịch bản so sánh")

    section_caption(
        """
        Phần này so sánh ba cách đánh giá: trọng số chuyên gia, trọng số khách quan theo Entropy và AHP đơn giản.
        Mục tiêu không phải chọn một phương pháp duy nhất, mà là xem thứ hạng vùng có ổn định hay phụ thuộc mạnh vào cách xác định trọng số.
        """
    )

    entropy_weight_df = pd.DataFrame({
        "Tiêu chí": [criteria_vn[c] for c in criteria],
        "Trọng số chuyên gia": expert_weights,
        "Trọng số Entropy": entropy_w,
        "Trọng số AHP": ahp_w,
    })

    weight_col, rank_col = st.columns([1, 1])

    with weight_col:
        st.markdown("#### So sánh trọng số")
        st.dataframe(
            entropy_weight_df.style.format({
                "Trọng số chuyên gia": "{:.3f}",
                "Trọng số Entropy": "{:.3f}",
                "Trọng số AHP": "{:.3f}",
            }),
            use_container_width=True,
        )

    with rank_col:
        st.markdown("#### So sánh thứ hạng")
        st.dataframe(
            comparison.sort_values("Hạng chuyên gia").style.format({
                "Điểm chuyên gia": "{:.4f}",
                "Điểm Entropy": "{:.4f}",
                "Điểm AHP": "{:.4f}",
            }),
            use_container_width=True,
        )

    plot_df = comparison.sort_values("Hạng chuyên gia")

    fig = go.Figure()
    for col, name, color in [
        ("Hạng chuyên gia", "Expert", BLUE),
        ("Hạng Entropy", "Entropy", BLUE_3),
        ("Hạng AHP", "AHP", GREEN),
    ]:
        fig.add_trace(
            go.Bar(
                x=plot_df["Vùng"],
                y=plot_df[col],
                name=name,
                marker_color=color,
                hovertemplate=f"Vùng: %{{x}}<br>{name}: %{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="So sánh thứ hạng giữa các phương pháp",
        height=360,
        margin=dict(l=10, r=10, t=45, b=20),
        yaxis_title="Hạng",
        yaxis=dict(autorange="reversed", dtick=1),
        xaxis=dict(tickangle=15),
        barmode="group",
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Độ nhạy theo trọng số AI Readiness")

    ai_weight_values = np.round(np.arange(0.10, 0.401, 0.05), 2)
    sensitivity_rows = []

    ai_index = criteria.index("ai_readiness")
    other_indices = [i for i in range(len(criteria)) if i != ai_index]
    other_base_sum = expert_weights[other_indices].sum()

    for ai_w in ai_weight_values:
        new_w = expert_weights.copy()
        new_w[ai_index] = ai_w

        remaining_weight = 1 - ai_w

        for idx in other_indices:
            new_w[idx] = expert_weights[idx] / other_base_sum * remaining_weight

        sens_result, _, _ = make_topsis_result(df, new_w, "sens")
        top3 = sens_result.head(3)["region"].tolist()

        for _, row in sens_result.iterrows():
            sensitivity_rows.append({
                "w_AI": ai_w,
                "region": row["region"],
                "rank": row["sens_rank"],
                "score": row["sens_score"],
                "top3": row["region"] in top3,
            })

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    top3_by_weight = sensitivity_df[sensitivity_df["top3"]].groupby("w_AI")["region"].apply(
        lambda x: ", ".join(x)
    ).reset_index()

    sens_col, sens_chart_col = st.columns([0.9, 1.1])

    with sens_col:
        st.dataframe(top3_by_weight, use_container_width=True)

    with sens_chart_col:
        fig = go.Figure()
        for region in df["region"]:
            sub = sensitivity_df[sensitivity_df["region"] == region]
            fig.add_trace(
                go.Scatter(
                    x=sub["w_AI"],
                    y=sub["rank"],
                    mode="lines+markers",
                    name=region,
                    hovertemplate=(
                        "Vùng: " + region + "<br>"
                        "w_AI: %{x:.2f}<br>"
                        "Hạng: %{y}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Độ nhạy thứ hạng khi thay đổi trọng số AI",
            height=360,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Trọng số AI Readiness",
            yaxis_title="Hạng",
            yaxis=dict(autorange="reversed", dtick=1),
            legend=dict(orientation="h", y=-0.30),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    stable_top3 = top3_by_weight["region"].nunique() == 1

    if stable_top3:
        st.success("Top-3 ổn định trong toàn bộ khoảng trọng số AI từ 0.10 đến 0.40.")
    else:
        st.warning("Top-3 thay đổi khi tăng trọng số AI. Điều này cho thấy kết quả nhạy với ưu tiên chính sách về AI Readiness.")

    st.markdown("#### Diễn giải so sánh")

    st.write(
        """
        So sánh giữa trọng số chuyên gia và Entropy cho thấy thứ hạng không chỉ phụ thuộc vào dữ liệu, mà còn phụ thuộc vào cách nhìn chính sách.
        Trọng số chuyên gia phản ánh ưu tiên chủ quan của nhà hoạch định chính sách, trong khi Entropy nhấn mạnh tiêu chí có độ phân hóa dữ liệu cao hơn
        giữa các vùng. Nếu một vùng có thứ hạng cao trong cả ba phương pháp, kết quả có tính ổn định tương đối và có thể được xem là ứng viên rõ ràng
        cho đầu tư ưu tiên.
        """
    )

# ======================================================
# TAB 4
# ======================================================

with tab4:
    st.header("4. Cảnh báo rủi ro")

    section_caption(
        """
        Phần này bổ sung góc nhìn rủi ro để tránh hiểu bảng xếp hạng như một kết luận tuyệt đối.
        Một vùng có điểm ưu tiên cao vẫn có thể đối mặt với rủi ro về bất bình đẳng, nhân lực hoặc khả năng lan tỏa công nghệ.
        """
    )

    risk_table = expert_result[
        [
            "expert_rank",
            "region",
            "expert_score",
            "digital_index",
            "ai_readiness",
            "trained_labor",
            "internet",
            "gini",
            "risk_level",
            "risk_flags",
        ]
    ].rename(columns={
        "expert_rank": "Hạng",
        "region": "Vùng",
        "expert_score": "Điểm ưu tiên",
        "digital_index": "Digital Index",
        "ai_readiness": "AI Readiness",
        "trained_labor": "LĐ đào tạo",
        "internet": "Internet",
        "gini": "Gini",
        "risk_level": "Mức cảnh báo",
        "risk_flags": "Nội dung cảnh báo",
    })

    risk_col, risk_chart_col = st.columns([1.1, 0.9])

    with risk_col:
        st.dataframe(
            risk_table.style.format({
                "Điểm ưu tiên": "{:.4f}",
                "Digital Index": "{:.0f}",
                "AI Readiness": "{:.0f}",
                "LĐ đào tạo": "{:.1f}",
                "Internet": "{:.0f}",
                "Gini": "{:.3f}",
            }),
            use_container_width=True,
        )

    with risk_chart_col:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=expert_result["gini"],
                y=expert_result["expert_score"],
                mode="markers+text",
                text=expert_result["expert_rank"].astype(str),
                textposition="top center",
                marker=dict(
                    size=expert_result["ai_readiness"] * 0.65,
                    color=BLUE_3,
                    line=dict(color=BLUE, width=1),
                    opacity=0.85,
                ),
                customdata=np.stack(
                    [
                        expert_result["region"],
                        expert_result["ai_readiness"],
                        expert_result["risk_level"],
                        expert_result["risk_flags"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Vùng: %{customdata[0]}<br>"
                    "Gini: %{x:.3f}<br>"
                    "Điểm ưu tiên: %{y:.4f}<br>"
                    "AI Readiness: %{customdata[1]:.0f}<br>"
                    "Mức cảnh báo: %{customdata[2]}<br>"
                    "Cảnh báo: %{customdata[3]}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Gini và điểm ưu tiên",
            height=360,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Gini",
            yaxis_title="Điểm ưu tiên",
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    high_gini_regions = expert_result[expert_result["gini"] >= 0.40]["region"].tolist()
    low_ai_regions = expert_result[expert_result["ai_readiness"] < 30]["region"].tolist()
    low_digital_regions = expert_result[expert_result["digital_index"] < 45]["region"].tolist()
    low_labor_regions = expert_result[expert_result["trained_labor"] < 22]["region"].tolist()

    st.markdown("#### Diễn giải rủi ro chính sách")

    top_expert_region = expert_result.iloc[0]["region"]
    top3_expert_regions = expert_result.head(3)["region"].tolist()

    rank_change_df = comparison.copy()
    rank_change_df["Mức thay đổi hạng"] = (
        rank_change_df["Hạng Entropy"] - rank_change_df["Hạng chuyên gia"]
    ).abs()

    max_change = rank_change_df["Mức thay đổi hạng"].max()
    most_changed_regions = rank_change_df[
        rank_change_df["Mức thay đổi hạng"] == max_change
    ]["Vùng"].tolist()

    corr_ai_internet = df["ai_readiness"].corr(df["internet"])
    corr_ai_digital = df["ai_readiness"].corr(df["digital_index"])

    st.write(
        f"""
        Kết quả cảnh báo rủi ro cho thấy bảng xếp hạng TOPSIS không nên được hiểu như một quyết định đầu tư tự động.
        Theo trọng số chuyên gia, vùng dẫn đầu là {top_expert_region}. Đây là vùng có điểm ưu tiên cao nhất do kết hợp được
        nhiều điều kiện thuận lợi như mức độ sẵn sàng AI, hạ tầng số, lao động đào tạo và năng lực kinh tế tương đối tốt.
        Tuy nhiên, việc vùng này đứng đầu không có nghĩa là nên mặc định đặt trung tâm AI quốc gia đầu tiên tại đây.
        TOPSIS chỉ đo mức độ gần với cấu hình lý tưởng theo 8 tiêu chí trong mô hình; còn quyết định đặt trung tâm AI quốc gia
        cần xét thêm các điều kiện ngoài mô hình như vai trò đầu mối quản trị quốc gia, liên kết viện - trường - doanh nghiệp,
        an ninh dữ liệu, hạ tầng điện toán, kết nối quốc tế, quỹ đất, khả năng thu hút nhân tài và tác động lan tỏa liên vùng.
        """
    )

    st.write(
        f"""
        Khi chuyển từ trọng số chuyên gia sang trọng số Entropy, vùng có mức thay đổi hạng lớn nhất là {list_text(most_changed_regions)}
        với mức thay đổi {int(max_change)} bậc. Trong bộ dữ liệu hiện tại, mức thay đổi này tương đối nhỏ nếu các vùng dẫn đầu
        có ưu thế đồng thời trên nhiều tiêu chí, thay vì chỉ mạnh ở một biến riêng lẻ.
        """
    )

    st.write(
        f"""
        Một hạn chế quan trọng của TOPSIS là mô hình giả định các tiêu chí đóng góp tương đối độc lập và được cộng gộp tuyến tính sau chuẩn hóa.
        Trong thực tế, AI Readiness, Digital Index và Internet có thể tương quan cao. Với bộ dữ liệu này, tương quan giữa AI Readiness và Internet
        là khoảng {corr_ai_internet:.2f}, còn tương quan giữa AI Readiness và Digital Index là khoảng {corr_ai_digital:.2f}.
        Khi các tiêu chí tương quan cao cùng xuất hiện trong mô hình, một năng lực nền tảng có thể bị đếm lặp.
        Cách xử lý là kiểm tra tương quan trước khi chạy mô hình, gộp các tiêu chí trùng thông tin thành một nhóm năng lực số chung,
        hoặc giảm trọng số của các biến có nội dung gần nhau.
        """
    )

    st.write(
        f"""
        Nếu dựa trên kết quả TOPSIS để đề xuất 3 vùng cho mục tiêu xây dựng các trung tâm AI lớn,
        ba ứng viên theo điểm chuyên gia là: {list_text(top3_expert_regions)}.
        Cách chọn này hợp lý nếu mục tiêu là tối đa hóa năng lực hấp thụ công nghệ và khả năng triển khai sớm.
        Tuy nhiên, quyết định cuối cùng không nên chỉ dựa vào TOPSIS. Cần bổ sung tiêu chí địa - chính trị,
        vị trí chiến lược Bắc - Trung - Nam, khả năng liên kết vùng, an ninh dữ liệu, năng lực đại học - viện nghiên cứu,
        hạ tầng điện toán, khả năng thu hút chuyên gia và mức độ lan tỏa sang các vùng lân cận.
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
        Lưu ý: Kết quả của Bài 6 là mô phỏng phục vụ phân tích. TOPSIS, Entropy và AHP giúp lượng hóa mức độ ưu tiên vùng,
        nhưng không thay thế quá trình đánh giá chính sách thực tế, vốn cần thêm dữ liệu chính thức, tham vấn chuyên gia và kiểm định độ nhạy.
        </p>
        """,
        unsafe_allow_html=True,
    )
