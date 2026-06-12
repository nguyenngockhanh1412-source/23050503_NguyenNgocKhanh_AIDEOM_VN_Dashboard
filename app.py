import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.aideom_ui import (
    setup_page,
    render_sidebar,
    page_header,
    info_box,
    section_caption,
    source_note,
    kpi_card,
)

# ======================================================
# PAGE SETUP
# ======================================================

setup_page("AIDEOM-VN Dashboard", page_icon="📊")
render_sidebar("Trang chủ")

# ======================================================
# HOME HEADER
# ======================================================

st.markdown(
    '<div class="main-title">AIDEOM-VN Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Dashboard mô phỏng quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Hệ thống trình bày các mô hình phân tích định lượng phục vụ đánh giá tăng trưởng, phân bổ nguồn lực,
    lựa chọn ngành/vùng ưu tiên, tác động lao động và rủi ro chính sách trong bối cảnh chuyển đổi số và AI.
    """
)

info_box(
    """
    Trang chủ cung cấp bức tranh tổng quan về dữ liệu nền và các kịch bản mô phỏng.
    Các trang sau đi vào từng mô hình cụ thể, gồm dự báo kinh tế, tối ưu ngân sách, xếp hạng ưu tiên,
    mô phỏng lao động, tối ưu ngẫu nhiên, học tăng cường và dashboard tích hợp.
    """
)

# ======================================================
# DATA BY YEAR
# ======================================================

year_data = pd.DataFrame({
    "Năm": [2020, 2021, 2022, 2023, 2024, 2025],
    "GDP danh nghĩa": [6293, 6479, 8096, 9430, 11350, 12848],
    "Tỷ trọng kinh tế số": [12.0, 13.5, 14.8, 16.5, 18.3, 19.5],
    "Doanh nghiệp ứng dụng AI": [28.0, 35.5, 45.0, 58.2, 70.4, 80.1],
    "Lao động qua đào tạo": [24.1, 25.1, 26.2, 27.5, 28.4, 29.2],
    "FDI": [20.0, 19.7, 22.4, 23.2, 25.4, 27.6],
    "Xuất khẩu": [282, 336, 371, 355, 405, 475],
    "Số vùng phân tích": [6, 6, 6, 6, 6, 6],
    "Số ngành phân tích": [8, 8, 8, 8, 8, 8],
})

# ======================================================
# SECTION 1: KPI OVERVIEW
# ======================================================

st.markdown("---")
st.header("1. Bức tranh kinh tế - số hóa - AI")

section_caption(
    """
    Chọn năm để xem các chỉ tiêu nền thay đổi theo thời gian. Các chỉ tiêu này đóng vai trò biến nền cho những mô hình phía sau,
    đặc biệt là dự báo tăng trưởng, chuyển đổi số, AI readiness và mô phỏng lao động.
    """
)

year_col, source_col = st.columns([0.32, 0.68])

with year_col:
    selected_year = st.selectbox(
        "Chọn năm phân tích",
        year_data["Năm"].tolist(),
        index=len(year_data) - 1,
    )

selected_row = year_data[year_data["Năm"] == selected_year].iloc[0]

with source_col:
    source_note(
        """
        Bộ dữ liệu mô phỏng/tổng hợp trong AIDEOM-VN được tổng hợp từ nhiều nguồn như Tổng cục thống kế, Bộ kế hoạch và đầu tư, World Bank, OECD, WIPO, Stanford HAI, IMF
        """
    )

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    kpi_card(
        "GDP danh nghĩa tham chiếu",
        f"{selected_row['GDP danh nghĩa']:,.0f}",
        "Nghìn tỷ VND, dùng làm nền mô phỏng tăng trưởng."
    )

with kpi2:
    kpi_card(
        "Tỷ trọng kinh tế số",
        f"{selected_row['Tỷ trọng kinh tế số']:.1f}%",
        "Biến đại diện cho mức độ số hóa của nền kinh tế."
    )

with kpi3:
    kpi_card(
        "Doanh nghiệp ứng dụng AI",
        f"{selected_row['Doanh nghiệp ứng dụng AI']:.1f}",
        "Nghìn doanh nghiệp, dùng trong nhóm biến AI readiness."
    )

with kpi4:
    kpi_card(
        "Lao động qua đào tạo",
        f"{selected_row['Lao động qua đào tạo']:.1f}%",
        "Nền tảng hấp thụ công nghệ và chuyển đổi việc làm."
    )

kpi5, kpi6, kpi7, kpi8 = st.columns(4)

with kpi5:
    kpi_card(
        "FDI tham chiếu",
        f"{selected_row['FDI']:.1f}",
        "Tỷ USD, phản ánh liên kết quốc tế và vốn công nghệ."
    )

with kpi6:
    kpi_card(
        "Xuất khẩu tham chiếu",
        f"{selected_row['Xuất khẩu']:.0f}",
        "Tỷ USD, phản ánh độ mở và mức phụ thuộc thị trường ngoài."
    )

with kpi7:
    kpi_card(
        "Số vùng phân tích",
        f"{selected_row['Số vùng phân tích']:.0f}",
        "Phục vụ TOPSIS vùng và bản đồ đầu tư AI."
    )

with kpi8:
    kpi_card(
        "Số ngành phân tích",
        f"{selected_row['Số ngành phân tích']:.0f}",
        "Phục vụ mô phỏng tác động AI tới thị trường lao động."
    )

# ======================================================
# SECTION 2: POLICY SCENARIOS
# ======================================================

st.markdown("---")
st.header("2. Khung kịch bản chính sách")

section_caption(
    """
    Năm kịch bản dưới đây mô phỏng các hướng phân bổ ngân sách khác nhau giữa vốn truyền thống, chuyển đổi số,
    AI và nhân lực. Mục tiêu là cho thấy mỗi định hướng chính sách tạo ra một cấu trúc đánh đổi khác nhau.
    """
)

scenario_df = pd.DataFrame({
    "Kịch bản": [
        "Truyền thống",
        "Số hóa nhanh",
        "AI dẫn dắt",
        "Bao trùm số",
        "Tối ưu cân bằng",
    ],
    "K": [0.70, 0.25, 0.20, 0.30, 0.35],
    "D": [0.10, 0.45, 0.20, 0.20, 0.25],
    "AI": [0.10, 0.15, 0.45, 0.10, 0.20],
    "H": [0.10, 0.15, 0.15, 0.40, 0.20],
    "Tăng trưởng": [76, 82, 88, 74, 85],
    "Bao trùm": [55, 68, 60, 90, 78],
    "Đổi mới": [52, 78, 92, 62, 80],
    "Chống chịu": [60, 70, 63, 85, 82],
})

source_note(
    """
    Các kịch bản chính sách được xây dựng phục vụ mô phỏng AIDEOM-VN.
    Các điểm tăng trưởng, bao trùm, đổi mới và chống chịu là chỉ số chuẩn hóa minh họa để so sánh kịch bản,
    không phải số liệu thống kê chính thức.
    """
)

chart1, chart2 = st.columns([1, 1])

with chart1:
    fig = go.Figure()

    colors = {
        "K": "#0B1D33",
        "D": "#81D8D0",
        "AI": "#1FA7B6",
        "H": "#E6F7F5",
    }

    for col in ["K", "D", "AI", "H"]:
        fig.add_trace(
            go.Bar(
                x=scenario_df["Kịch bản"],
                y=scenario_df[col],
                name=col,
                marker_color=colors[col],
                hovertemplate=(
                    "Kịch bản: %{x}<br>"
                    "Hạng mục: " + col + "<br>"
                    "Tỷ trọng: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        barmode="stack",
        title="Cơ cấu ngân sách theo 5 kịch bản",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis=dict(tickangle=0),
        yaxis_title="Tỷ trọng",
        legend=dict(orientation="h", y=-0.22),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

with chart2:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scenario_df["Đổi mới"],
            y=scenario_df["Bao trùm"],
            mode="markers+text",
            text=scenario_df["Kịch bản"],
            textposition="top center",
            marker=dict(
                size=scenario_df["Tăng trưởng"] / 2,
                color=scenario_df["Chống chịu"],
                colorscale=[
                    [0, "#EAF7FF"],
                    [0.4, "#1FA7B6"],
                    [0.7, "#5FA8D3"],
                    [1, "#1FA7B6"],
                ],
                showscale=True,
                colorbar=dict(title="Chống chịu"),
                line=dict(color="#1e3a8a", width=1),
            ),
            hovertemplate=(
                "Kịch bản: %{text}<br>"
                "Đổi mới: %{x}<br>"
                "Bao trùm: %{y}<br>"
                "Tăng trưởng: %{marker.size}<br>"
                "Chống chịu: %{marker.color}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Đánh đổi giữa đổi mới và bao trùm",
        height=330,
        margin=dict(l=10, r=10, t=45, b=20),
        xaxis_title="Điểm đổi mới",
        yaxis_title="Điểm bao trùm",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ======================================================
# SECTION 3: ANALYTICAL STRUCTURE
# ======================================================

st.markdown("---")
st.header("3. Cấu trúc phân tích của hệ thống")

section_caption(
    """
    Phần này tóm tắt vai trò của từng nhóm mô hình trong hệ thống. Cách trình bày này giúp người đọc hiểu vì sao
    các bài toán riêng lẻ được kết nối với nhau trong một dashboard hỗ trợ ra quyết định.
    """
)

structure_df = pd.DataFrame({
    "Nhóm phân tích": [
        "Dự báo tăng trưởng",
        "Lựa chọn ngành và vùng ưu tiên",
        "Tối ưu phân bổ nguồn lực",
        "Lao động và an sinh",
        "Bất định và thích nghi",
        "Tích hợp quyết định",
    ],
    "Câu hỏi chính sách": [
        "Tăng trưởng thay đổi thế nào khi tăng vốn, số hóa, AI và nhân lực?",
        "Ngành/vùng nào nên được ưu tiên đầu tư?",
        "Nguồn lực hữu hạn nên phân bổ thế nào giữa các mục tiêu?",
        "AI tạo thêm việc làm hay làm tăng dịch chuyển lao động?",
        "Chính sách có đủ linh hoạt trước cú sốc và bất định không?",
        "Kịch bản nào cân bằng tốt hơn giữa tăng trưởng, bao trùm và rủi ro?",
    ],
    "Mô hình sử dụng": [
        "Cobb-Douglas, tối ưu động",
        "Priority Index, TOPSIS",
        "LP, MIP, NSGA-II",
        "Mô phỏng NetJob và đào tạo lại",
        "Stochastic Programming, Q-learning",
        "Dashboard AIDEOM-VN",
    ],
})

st.dataframe(structure_df, use_container_width=True)

# ======================================================
# SECTION 4: POLICY IMPLICATIONS
# ======================================================

st.markdown("---")
st.header("4. Hàm ý")

st.markdown(
    """
    <div class="note-box">
    AIDEOM-VN cho thấy chính sách phát triển kinh tế trong kỷ nguyên AI không nên chỉ tối đa hóa tăng trưởng ngắn hạn.
    Một chiến lược phù hợp với Việt Nam cần phối hợp bốn trụ cột: vốn và hạ tầng, chuyển đổi số, AI và nhân lực.
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    Nếu đầu tư quá chậm, Việt Nam có thể bỏ lỡ cơ hội năng suất và đổi mới. Nếu đầu tư quá nhanh vào AI mà thiếu nhân lực,
    dữ liệu và quản trị rủi ro, chính sách có thể làm tăng bất bình đẳng lao động, rủi ro an ninh dữ liệu và phụ thuộc công nghệ.
    Vì vậy, định hướng hợp lý hơn là chọn lộ trình cân bằng có điều chỉnh theo trạng thái kinh tế.
    Số hóa là nền tảng, AI là động lực tăng năng suất, nhân lực là điều kiện hấp thụ công nghệ,
    còn quản trị rủi ro là điều kiện để phát triển bền vững.
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
    Lưu ý: Toàn bộ tham số, điểm số, kịch bản và khuyến nghị trong dashboard là kết quả mô phỏng phục vụ phân tích.
    Các kết quả này chỉ mang tính tham khảo, không được hiểu là dự báo chính thức hay khuyến nghị chính sách cuối cùng.
    Khi áp dụng thực tế, cần thay thế bằng số liệu thống kê chính thức, tham vấn chuyên gia và kiểm định độ nhạy của mô hình.
    </p>
    """,
    unsafe_allow_html=True
)