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
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize
except ImportError:
    st.error("Bạn chưa cài pymoo. Hãy chạy trong Terminal: python -m pip install pymoo")
    st.stop()


# ======================================================
# PAGE SETUP
# ======================================================

setup_page("Bài 7 - NSGA-II Pareto")
render_sidebar("Bài 7 - NSGA-II Pareto")

st.title("Bài 7. Tối ưu đa mục tiêu Pareto với NSGA-II")

st.write(
    """
    Bài này mô phỏng bài toán phân bổ nguồn lực phát triển kinh tế số và AI theo vùng trong điều kiện nhiều mục tiêu cùng tồn tại.
    Thay vì tìm một nghiệm tối ưu duy nhất, mô hình tạo ra một tập nghiệm Pareto để thể hiện các đánh đổi giữa tăng trưởng, bao trùm,
    môi trường và an ninh dữ liệu.
    """
)

source_note(
    """
    Dữ liệu vùng được lấy từ vietnam_regions_2024.csv; các tham số phát thải và rủi ro được sử dụng theo bảng tham số bổ sung của Bài 7.
    Kết quả là mô phỏng phục vụ phân tích, không thay thế quy trình thẩm định ngân sách chính thức.
    """
)


# ======================================================
# LOAD DATA
# ======================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGION_FILE = DATA_DIR / "vietnam_regions_2024.csv"

if not REGION_FILE.exists():
    st.error(f"Không tìm thấy file dữ liệu: {REGION_FILE}")
    st.stop()

regions_df = pd.read_csv(REGION_FILE)

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

missing_cols = [c for c in required_cols if c not in regions_df.columns]

if missing_cols:
    st.error("File vietnam_regions_2024.csv đang thiếu các cột sau:")
    st.write(missing_cols)
    st.stop()

regions = regions_df["region"].tolist()
n_regions = len(regions)

items = ["Hạ tầng số", "Dữ liệu", "AI", "Nhân lực"]
item_codes = ["I", "D", "A", "H"]
n_items = len(items)

# Bảng tham số bổ sung theo đề Bài 7
extra_params = pd.DataFrame({
    "region": regions,
    "emission": [0.42, 0.55, 0.48, 0.32, 0.62, 0.38],
    "ai_risk": [0.18, 0.45, 0.28, 0.12, 0.52, 0.22],
    "risk_reduction": [0.32, 0.28, 0.30, 0.35, 0.25, 0.30],
})


# ======================================================
# HELPER FUNCTIONS
# ======================================================

def minmax(series):
    series = pd.Series(series, dtype=float)
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def build_beta_matrix(df):
    digital = minmax(df["digital_index"])
    internet = minmax(df["internet"])
    ai_ready = minmax(df["ai_readiness"])
    trained = minmax(df["trained_labor"])
    grdp = minmax(df["grdp_per_capita"])
    fdi = minmax(df["fdi"])

    digital_gap = 1 - digital
    trained_gap = 1 - trained

    beta_infra = 0.70 + 0.30 * digital_gap + 0.10 * (1 - internet)
    beta_data = 0.65 + 0.20 * digital + 0.15 * internet
    beta_ai = 0.75 + 0.25 * ai_ready + 0.15 * fdi + 0.10 * grdp
    beta_human = 0.60 + 0.30 * trained_gap + 0.10 * digital_gap

    beta = np.vstack([
        beta_infra.values,
        beta_data.values,
        beta_ai.values,
        beta_human.values,
    ])

    return beta


beta_matrix = build_beta_matrix(regions_df)


def topsis_on_pareto(pareto_df, weights):
    data = pareto_df[["growth", "inclusion_cost", "emission_cost", "security_risk"]].copy()

    benefit_cols = ["growth"]
    cost_cols = ["inclusion_cost", "emission_cost", "security_risk"]

    norm = pd.DataFrame(index=data.index)

    for col in data.columns:
        min_v = data[col].min()
        max_v = data[col].max()

        if max_v == min_v:
            norm[col] = 1
        else:
            norm[col] = (data[col] - min_v) / (max_v - min_v)

    weighted = norm.values * np.array(weights)

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

    out = pareto_df.copy()
    out["topsis_score"] = score
    out["topsis_rank"] = out["topsis_score"].rank(ascending=False, method="dense").astype(int)

    return out.sort_values("topsis_rank").reset_index(drop=True)


def evaluate_solution(X, beta, emission, ai_risk, risk_reduction):
    region_total = X.sum(axis=0)
    avg_region_budget = region_total.mean()

    growth = np.sum(beta * X) / 1000
    inclusion_cost = np.mean(np.abs(region_total - avg_region_budget)) / avg_region_budget
    emission_cost = np.sum(emission * (X[0, :] + X[2, :])) / 1000
    security_risk = (np.sum(ai_risk * X[2, :]) - np.sum(risk_reduction * X[3, :])) / 1000

    return growth, inclusion_cost, emission_cost, security_risk


# ======================================================
# STREAMLIT SETTINGS
# ======================================================

st.markdown("---")
st.header("Thiết lập mô phỏng")

section_caption(
    """
    Phần thiết lập được đặt cố định ở đầu bài để người dùng thay đổi tham số rồi quan sát các tab kết quả bên dưới.
    Các tham số gồm ngân sách tổng, sàn/trần vùng, đầu tư AI tối thiểu và cấu hình thuật toán NSGA-II.
    """
)

setting_box = st.container(border=True)

with setting_box:
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### Nguồn lực")
        total_budget = st.number_input(
            "Tổng ngân sách",
            min_value=30000,
            max_value=90000,
            value=60000,
            step=5000,
        )

        region_floor = st.number_input(
            "Sàn ngân sách mỗi vùng",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=500,
        )

    with col_b:
        st.markdown("#### Ràng buộc phân bổ")
        region_cap = st.number_input(
            "Trần ngân sách mỗi vùng",
            min_value=8000,
            max_value=25000,
            value=14000,
            step=1000,
        )

        min_ai = st.number_input(
            "Đầu tư AI tối thiểu",
            min_value=1000,
            max_value=30000,
            value=7000,
            step=1000,
        )

    with col_c:
        st.markdown("#### Cấu hình NSGA-II")
        pop_size = st.number_input(
            "Population size",
            min_value=50,
            max_value=300,
            value=100,
            step=50,
        )

        n_gen = st.number_input(
            "Số thế hệ",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
        )

    seed = st.number_input(
        "Seed",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
    )

k0, k1, k2, k3 = st.columns(4)

with k0:
    kpi_card("Tổng ngân sách", f"{total_budget:,.0f}", "Quy mô ngân sách phân bổ cho 6 vùng.")

with k1:
    kpi_card("Sàn mỗi vùng", f"{region_floor:,.0f}", "Mức phân bổ tối thiểu để bảo đảm bao trùm.")

with k2:
    kpi_card("Trần mỗi vùng", f"{region_cap:,.0f}", "Giới hạn để tránh tập trung vốn quá mức.")

with k3:
    kpi_card("Đầu tư AI tối thiểu", f"{min_ai:,.0f}", "Ràng buộc bảo đảm AI vẫn là trọng tâm của mô hình.")

min_human = 9000
min_infra_data = 15000
max_ai = 22000

weak_region_idx = regions_df.sort_values("digital_index").head(2).index.tolist()
min_weak_region_support = 9000


# ======================================================
# NSGA-II PROBLEM
# ======================================================

@st.cache_data(show_spinner=False)
def run_nsga2_model(
    total_budget,
    region_floor,
    region_cap,
    min_ai,
    pop_size,
    n_gen,
    seed,
    beta_matrix,
    emission_array,
    ai_risk_array,
    risk_reduction_array,
    weak_region_idx,
):
    class ParetoBudgetProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=n_items * n_regions,
                n_obj=4,
                n_constr=2 + n_regions + n_regions + 5,
                xl=np.zeros(n_items * n_regions),
                xu=np.ones(n_items * n_regions) * region_cap,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            X = x.reshape((n_items, n_regions))

            region_total = X.sum(axis=0)
            item_total = X.sum(axis=1)
            total = X.sum()

            growth, inclusion_cost, emission_cost, security_risk = evaluate_solution(
                X,
                beta_matrix,
                emission_array,
                ai_risk_array,
                risk_reduction_array,
            )

            out["F"] = np.array([
                -growth,
                inclusion_cost,
                emission_cost,
                security_risk,
            ])

            constraints = []
            constraints.append(total - total_budget)
            constraints.append(0.98 * total_budget - total)
            constraints.extend(region_floor - region_total)
            constraints.extend(region_total - region_cap)
            constraints.append(min_human - item_total[3])
            constraints.append(min_ai - item_total[2])
            constraints.append(item_total[2] - max_ai)
            constraints.append(min_infra_data - (item_total[0] + item_total[1]))

            weak_support = region_total[weak_region_idx].sum()
            constraints.append(min_weak_region_support - weak_support)

            out["G"] = np.array(constraints)

    problem = ParetoBudgetProblem()

    algorithm = NSGA2(
        pop_size=int(pop_size),
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_gen", int(n_gen))

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=int(seed),
        verbose=False,
    )

    if result.X is None or result.F is None:
        return pd.DataFrame(), pd.DataFrame()

    X_arr = result.X
    F_arr = result.F

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
        F_arr = F_arr.reshape(1, -1)

    rows = []
    allocations = []

    for idx, x in enumerate(X_arr):
        X = x.reshape((n_items, n_regions))

        growth, inclusion_cost, emission_cost, security_risk = evaluate_solution(
            X,
            beta_matrix,
            emission_array,
            ai_risk_array,
            risk_reduction_array,
        )

        row = {
            "solution_id": idx,
            "growth": growth,
            "inclusion_cost": inclusion_cost,
            "emission_cost": emission_cost,
            "security_risk": security_risk,
            "total_budget": X.sum(),
        }

        for i, item in enumerate(item_codes):
            for r, region in enumerate(regions):
                row[f"{item}_{r}"] = X[i, r]

                allocations.append({
                    "solution_id": idx,
                    "item": items[i],
                    "region": region,
                    "allocation": X[i, r],
                })

        rows.append(row)

    pareto_df = pd.DataFrame(rows)
    allocation_df = pd.DataFrame(allocations)

    return pareto_df, allocation_df


with st.spinner("Đang chạy mô hình NSGA-II..."):
    pareto_df, allocation_long = run_nsga2_model(
        total_budget,
        region_floor,
        region_cap,
        min_ai,
        pop_size,
        n_gen,
        seed,
        beta_matrix,
        extra_params["emission"].values,
        extra_params["ai_risk"].values,
        extra_params["risk_reduction"].values,
        weak_region_idx,
    )

if pareto_df.empty:
    st.error("Mô hình chưa tìm được nghiệm khả thi. Hãy nới ngân sách, giảm sàn ngân sách vùng hoặc giảm yêu cầu tối thiểu cho AI.")
    st.stop()

policy_weights = np.array([0.40, 0.25, 0.20, 0.15])
pareto_scored = topsis_on_pareto(pareto_df, policy_weights)

compromise = pareto_scored.iloc[0]
growth_best = pareto_scored.sort_values("growth", ascending=False).iloc[0]

compromise_id = int(compromise["solution_id"])
growth_best_id = int(growth_best["solution_id"])

compromise_alloc = allocation_long[allocation_long["solution_id"] == compromise_id]
growth_best_alloc = allocation_long[allocation_long["solution_id"] == growth_best_id]

compromise_matrix = compromise_alloc.pivot(index="item", columns="region", values="allocation").reindex(items)
growth_best_matrix = growth_best_alloc.pivot(index="item", columns="region", values="allocation").reindex(items)

growth_sacrifice = (growth_best["growth"] - compromise["growth"]) / growth_best["growth"] * 100
inclusion_gain = (growth_best["inclusion_cost"] - compromise["inclusion_cost"]) / growth_best["inclusion_cost"] * 100
emission_gain = (growth_best["emission_cost"] - compromise["emission_cost"]) / growth_best["emission_cost"] * 100

risk_denominator = abs(growth_best["security_risk"]) if abs(growth_best["security_risk"]) > 1e-9 else 1
security_gain = (growth_best["security_risk"] - compromise["security_risk"]) / risk_denominator * 100


# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Tổng quan", "Pareto", "Nghiệm thỏa hiệp", "Thảo luận chính sách"]
)


# ======================================================
# TAB 1
# ======================================================

with tab1:
    st.header("1. Tổng quan bài toán")

    st.write(
        """
        Mô hình phân bổ ngân sách cho 4 hạng mục ở 6 vùng. Mỗi nghiệm là một phương án phân bổ khác nhau.
        Thay vì chỉ tối đa hóa tăng trưởng, bài toán đồng thời xem xét mức độ bao trùm giữa các vùng, chi phí môi trường
        và rủi ro an ninh dữ liệu.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Số biến quyết định", f"{n_items * n_regions}", "4 hạng mục × 6 vùng.")

    with c2:
        kpi_card("Số mục tiêu", "4", "Tăng trưởng, bao trùm, môi trường, an ninh.")

    with c3:
        kpi_card("Số nghiệm Pareto", f"{len(pareto_df)}", "Số phương án không bị lấn át.")

    with c4:
        kpi_card("Tổng ngân sách", f"{total_budget:,.0f}", "Ngân sách theo thiết lập hiện tại.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Các hạng mục đầu tư")

        item_df = pd.DataFrame({
            "Ký hiệu": item_codes,
            "Hạng mục": items,
            "Vai trò trong mô hình": [
                "Tăng năng lực hạ tầng số, nhưng có phát thải gián tiếp",
                "Củng cố dữ liệu, nền tảng cho chuyển đổi số và AI",
                "Tạo tác động tăng trưởng nhưng làm tăng rủi ro dữ liệu",
                "Giảm rủi ro an ninh dữ liệu và cải thiện năng lực hấp thụ công nghệ",
            ],
        })

        st.dataframe(item_df, use_container_width=True)

    with col2:
        st.markdown("#### Bốn mục tiêu của mô hình")

        objective_df = pd.DataFrame({
            "Mục tiêu": [
                "Tăng trưởng",
                "Bao trùm",
                "Môi trường",
                "An ninh dữ liệu",
            ],
            "Chiều tối ưu": [
                "Tối đa hóa",
                "Tối thiểu hóa",
                "Tối thiểu hóa",
                "Tối thiểu hóa",
            ],
            "Cách hiểu": [
                "Tác động kỳ vọng của phân bổ ngân sách đến tăng trưởng",
                "Mức lệch phân bổ ngân sách giữa các vùng",
                "Phát thải gián tiếp từ hạ tầng số và AI",
                "Rủi ro dữ liệu tăng từ AI, giảm nhờ đầu tư nhân lực",
            ],
        })

        st.dataframe(objective_df, use_container_width=True)

    st.markdown("#### Tham số phát thải và an ninh dữ liệu")

    extra_display = extra_params.rename(columns={
        "region": "Vùng",
        "emission": "e_r",
        "ai_risk": "p_r",
        "risk_reduction": "sigma_r",
    })

    st.dataframe(
        extra_display.style.format({
            "e_r": "{:.2f}",
            "p_r": "{:.2f}",
            "sigma_r": "{:.2f}",
        }),
        use_container_width=True,
    )


# ======================================================
# TAB 2
# ======================================================

with tab2:
    st.header("2. Tập nghiệm Pareto")

    st.write(
        """
        Tập nghiệm Pareto thể hiện các phương án không bị lấn át. Một nghiệm Pareto không thể cải thiện một mục tiêu
        mà không làm ít nhất một mục tiêu khác xấu đi.
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card("Tăng trưởng cao nhất", f"{pareto_df['growth'].max():.2f}", "Giá trị lớn nhất trên tập Pareto.")

    with c2:
        kpi_card("Bao trùm tốt nhất", f"{pareto_df['inclusion_cost'].min():.3f}", "Chi phí bất cân bằng thấp nhất.")

    with c3:
        kpi_card("Môi trường tốt nhất", f"{pareto_df['emission_cost'].min():.2f}", "Chi phí phát thải thấp nhất.")

    with c4:
        kpi_card("Rủi ro thấp nhất", f"{pareto_df['security_risk'].min():.2f}", "Rủi ro an ninh dữ liệu thấp nhất.")

    st.markdown("#### Scatter 3D của các nghiệm Pareto")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pareto_df["growth"],
            y=pareto_df["inclusion_cost"],
            z=pareto_df["emission_cost"],
            mode="markers",
            marker=dict(
                size=5,
                color=pareto_df["security_risk"],
                colorscale=[
                    [0, "#81D8D0"],
                    [0.35, "#1FA7B6"],
                    [0.70, "#FF6B6B"],
                    [1, "#1FA7B6"],
                ],
                colorbar=dict(title="Rủi ro dữ liệu"),
                opacity=0.78,
            ),
            text=pareto_df["solution_id"],
            hovertemplate=(
                "Nghiệm: %{text}<br>"
                "Tăng trưởng: %{x:.3f}<br>"
                "Bao trùm: %{y:.4f}<br>"
                "Môi trường: %{z:.3f}<br>"
                "Rủi ro dữ liệu: %{marker.color:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[compromise["growth"]],
            y=[compromise["inclusion_cost"]],
            z=[compromise["emission_cost"]],
            mode="markers",
            marker=dict(size=8, color="#F97316"),
            name="Nghiệm thỏa hiệp",
            hovertemplate=(
                "Nghiệm thỏa hiệp<br>"
                "Tăng trưởng: %{x:.3f}<br>"
                "Bao trùm: %{y:.4f}<br>"
                "Môi trường: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Đường biên Pareto 3D",
        height=430,
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis_title="Tăng trưởng",
            yaxis_title="Bao trùm",
            zaxis_title="Môi trường",
        ),
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Parallel coordinates")

    plot_sample = pareto_scored.copy()

    if len(plot_sample) > 120:
        plot_sample = plot_sample.sample(120, random_state=42)

    parallel_cols = ["growth", "inclusion_cost", "emission_cost", "security_risk"]
    normalized_plot = pd.DataFrame(index=plot_sample.index)

    for col in parallel_cols:
        min_v = pareto_scored[col].min()
        max_v = pareto_scored[col].max()

        if max_v == min_v:
            normalized_plot[col] = 0.5
        else:
            if col == "growth":
                normalized_plot[col] = (plot_sample[col] - min_v) / (max_v - min_v)
            else:
                normalized_plot[col] = 1 - (plot_sample[col] - min_v) / (max_v - min_v)

    normalized_plot["solution_id"] = plot_sample["solution_id"].values
    normalized_plot["topsis_score"] = plot_sample["topsis_score"].values

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=normalized_plot["topsis_score"],
                colorscale=[
                    [0, "#81D8D0"],
                    [0.5, "#FF6B6B"],
                    [1, "#1FA7B6"],
                ],
                showscale=True,
                colorbar=dict(title="TOPSIS"),
            ),
            dimensions=[
                dict(label="Tăng trưởng", values=normalized_plot["growth"]),
                dict(label="Bao trùm", values=normalized_plot["inclusion_cost"]),
                dict(label="Môi trường", values=normalized_plot["emission_cost"]),
                dict(label="An ninh", values=normalized_plot["security_risk"]),
            ],
        )
    )

    fig.update_layout(
        title="So sánh nhiều mục tiêu trên tập Pareto",
        height=360,
        margin=dict(l=40, r=40, t=50, b=30),
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Bảng nghiệm Pareto rút gọn")

    st.dataframe(
        pareto_scored[
            [
                "solution_id",
                "growth",
                "inclusion_cost",
                "emission_cost",
                "security_risk",
                "topsis_score",
                "topsis_rank",
            ]
        ].head(20).style.format({
            "growth": "{:.3f}",
            "inclusion_cost": "{:.4f}",
            "emission_cost": "{:.3f}",
            "security_risk": "{:.3f}",
            "topsis_score": "{:.4f}",
        }),
        use_container_width=True,
    )


# ======================================================
# TAB 3
# ======================================================

with tab3:
    st.header("3. Nghiệm thỏa hiệp theo TOPSIS")

    st.write(
        """
        Từ tập nghiệm Pareto, mô hình sử dụng trọng số chính sách 0.40 cho tăng trưởng, 0.25 cho bao trùm,
        0.20 cho môi trường và 0.15 cho an ninh dữ liệu để chọn một nghiệm thỏa hiệp.
        """
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        kpi_card("Điểm TOPSIS", f"{compromise['topsis_score']:.3f}", "Điểm chọn nghiệm thỏa hiệp.")

    with m2:
        kpi_card("Tăng trưởng", f"{compromise['growth']:.2f}", "Mục tiêu được tối đa hóa.")

    with m3:
        kpi_card("Bao trùm", f"{compromise['inclusion_cost']:.3f}", "Càng thấp càng tốt.")

    with m4:
        kpi_card("Môi trường", f"{compromise['emission_cost']:.2f}", "Càng thấp càng tốt.")

    with m5:
        kpi_card("An ninh", f"{compromise['security_risk']:.2f}", "Càng thấp càng tốt.")

    st.markdown("#### Ma trận phân bổ nghiệm thỏa hiệp")

    matrix_col, chart_col = st.columns([1, 1])

    with matrix_col:
        st.dataframe(
            compromise_matrix.style.format("{:,.0f}"),
            use_container_width=True,
        )

    with chart_col:
        region_alloc = compromise_matrix.T

        fig = go.Figure()

        item_color_map = {
            "Hạ tầng số": "#0B1D33",
            "Dữ liệu": "#1FA7B6",
            "AI": "#FF6B6B",
            "Nhân lực": "#E6F7F5",
        }

        for item in items:
            fig.add_trace(
                go.Bar(
                    y=region_alloc.index,
                    x=region_alloc[item],
                    name=item,
                    orientation="h",
                    marker_color=item_color_map[item],
                    hovertemplate=(
                        "Vùng: %{y}<br>"
                        f"Hạng mục: {item}<br>"
                        "Ngân sách: %{x:,.0f}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title="Phân bổ ngân sách theo vùng và hạng mục",
            height=390,
            margin=dict(l=10, r=10, t=45, b=20),
            xaxis_title="Ngân sách",
            yaxis_title="",
            barmode="stack",
            legend=dict(orientation="h", y=-0.18),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### So sánh với nghiệm tăng trưởng cao nhất")

    compare_df = pd.DataFrame({
        "Chỉ tiêu": [
            "Tăng trưởng",
            "Bao trùm",
            "Môi trường",
            "An ninh dữ liệu",
        ],
        "Nghiệm tăng trưởng cao nhất": [
            growth_best["growth"],
            growth_best["inclusion_cost"],
            growth_best["emission_cost"],
            growth_best["security_risk"],
        ],
        "Nghiệm thỏa hiệp": [
            compromise["growth"],
            compromise["inclusion_cost"],
            compromise["emission_cost"],
            compromise["security_risk"],
        ],
    })

    comp_col, comp_chart_col = st.columns([1, 1])

    with comp_col:
        st.dataframe(
            compare_df.style.format({
                "Nghiệm tăng trưởng cao nhất": "{:.4f}",
                "Nghiệm thỏa hiệp": "{:.4f}",
            }),
            use_container_width=True,
        )

    with comp_chart_col:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=compare_df["Chỉ tiêu"],
                y=compare_df["Nghiệm tăng trưởng cao nhất"],
                name="Tăng trưởng cao nhất",
                marker_color="#0B1D33",
                hovertemplate="Chỉ tiêu: %{x}<br>Giá trị: %{y:.4f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                x=compare_df["Chỉ tiêu"],
                y=compare_df["Nghiệm thỏa hiệp"],
                name="Nghiệm thỏa hiệp",
                marker_color="#FF6B6B",
                hovertemplate="Chỉ tiêu: %{x}<br>Giá trị: %{y:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="So sánh hai nghiệm tiêu biểu",
            height=330,
            margin=dict(l=10, r=10, t=45, b=20),
            barmode="group",
            legend=dict(orientation="h", y=-0.22),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"""
        So với nghiệm tăng trưởng cao nhất, nghiệm thỏa hiệp hy sinh khoảng {growth_sacrifice:.2f}% tăng trưởng.
        Đổi lại, chỉ số bất cân bằng vùng cải thiện khoảng {inclusion_gain:.2f}%, chi phí môi trường thay đổi khoảng {emission_gain:.2f}%,
        và rủi ro an ninh dữ liệu thay đổi khoảng {security_gain:.2f}% theo thang đo của mô hình.
        """
    )


# ======================================================
# TAB 4
# ======================================================

with tab4:
    st.header("4. Thảo luận chính sách")

    corr_growth_inclusion = pareto_scored["growth"].corr(pareto_scored["inclusion_cost"])
    corr_growth_emission = pareto_scored["growth"].corr(pareto_scored["emission_cost"])
    corr_growth_security = pareto_scored["growth"].corr(pareto_scored["security_risk"])

    if corr_growth_inclusion > 0.25:
        tradeoff_growth_inclusion = (
            "Đánh đổi giữa tăng trưởng và bao trùm thể hiện tương đối rõ. Khi điểm tăng trưởng tăng, chi phí bất cân bằng vùng có xu hướng tăng theo."
        )
    elif corr_growth_inclusion < -0.25:
        tradeoff_growth_inclusion = (
            "Trong mô phỏng này, tăng trưởng và bao trùm không mâu thuẫn trực tiếp; một số nghiệm vừa cải thiện tăng trưởng vừa giảm bất cân bằng vùng."
        )
    else:
        tradeoff_growth_inclusion = (
            "Đánh đổi giữa tăng trưởng và bao trùm không hoàn toàn tuyến tính. Điều này cho thấy vẫn tồn tại một số phương án trung gian có thể dung hòa hai mục tiêu."
        )

    st.markdown("#### Đánh đổi giữa tăng trưởng và bao trùm")

    st.write(
        f"""
        Khi quan sát đường biên Pareto, có thể thấy các mục tiêu phát triển không di chuyển cùng một chiều.
        Hệ số tương quan giữa tăng trưởng và chi phí bao trùm trong tập nghiệm là khoảng {corr_growth_inclusion:.2f}.
        {tradeoff_growth_inclusion}
        """
    )

    st.write(
        """
        Mức đánh đổi này phản ánh một đặc điểm quan trọng của cơ cấu kinh tế Việt Nam: năng lực hấp thụ công nghệ, hạ tầng số,
        nhân lực số và khả năng triển khai AI giữa các vùng không đồng đều. Các vùng có nền tảng tốt hơn thường tạo tác động tăng trưởng
        nhanh hơn khi nhận thêm đầu tư, nhưng nếu ngân sách dồn quá mạnh vào các vùng này thì mục tiêu bao trùm vùng miền có thể bị suy yếu.
        Ngược lại, nếu phân bổ quá đều, mục tiêu công bằng được cải thiện nhưng hiệu quả tăng trưởng ngắn hạn có thể giảm.
        """
    )

    st.write(
        f"""
        So với nghiệm tăng trưởng cao nhất, nghiệm thỏa hiệp hy sinh khoảng {growth_sacrifice:.2f}% tăng trưởng.
        Đổi lại, chỉ số bất cân bằng vùng cải thiện khoảng {inclusion_gain:.2f}%, chi phí môi trường thay đổi khoảng {emission_gain:.2f}%,
        và rủi ro an ninh dữ liệu thay đổi khoảng {security_gain:.2f}% theo thang đo của mô hình.
        Điều này cho thấy nghiệm thỏa hiệp không tối đa hóa một mục tiêu duy nhất, mà cố gắng cân bằng giữa tăng trưởng, bao trùm,
        môi trường và an ninh dữ liệu.
        """
    )

    st.markdown("#### Trọng số chính sách và ưu tiên phát triển")

    st.write(
        """
        Bộ trọng số 0.40 cho tăng trưởng, 0.25 cho bao trùm, 0.20 cho môi trường và 0.15 cho an ninh dữ liệu phản ánh cách tiếp cận
        đặt tăng trưởng ở vị trí ưu tiên nhưng vẫn dành vai trò đáng kể cho phát triển bao trùm, chuyển đổi xanh và an ninh số.
        Cách đặt trọng số này tương đối phù hợp với mục tiêu phát triển nhanh và bền vững trong định hướng phát triển của Việt Nam,
        vì tăng trưởng vẫn là điều kiện vật chất để đầu tư cho hạ tầng, nhân lực, đổi mới sáng tạo và chuyển đổi số.
        """
    )

    st.write(
        """
        Tuy nhiên, nếu liên hệ với cam kết phát thải ròng bằng 0 theo COP26, trọng số môi trường có thể cần tăng lên.
        Một cấu hình phù hợp hơn với ưu tiên xanh có thể là 0.30 cho tăng trưởng, 0.25 cho bao trùm, 0.30 cho môi trường
        và 0.15 cho an ninh dữ liệu. Khi đó, mô hình sẽ chấp nhận giảm một phần tăng trưởng kỳ vọng để hạn chế phát thải gián tiếp
        từ hạ tầng số, trung tâm dữ liệu và đầu tư AI.
        """
    )

    st.write(
        """
        Nếu gắn mạnh hơn với Chiến lược quốc gia về nghiên cứu, phát triển và ứng dụng AI theo Quyết định 127/QĐ-TTg,
        trọng số cho an ninh dữ liệu và năng lực nhân lực có thể cần được nâng lên. Một cấu hình thay thế là 0.35 cho tăng trưởng,
        0.20 cho bao trùm, 0.20 cho môi trường và 0.25 cho an ninh dữ liệu. Lý do là phát triển AI không chỉ cần đầu tư vào thuật toán
        hay hạ tầng tính toán, mà còn cần dữ liệu an toàn, nhân lực có năng lực và khả năng quản trị rủi ro.
        """
    )

    st.write(
        """
        Vì vậy, trọng số không phải là một thao tác kỹ thuật trung lập. Nó phản ánh lựa chọn chính sách trong từng giai đoạn:
        ưu tiên tăng trưởng nhanh, ưu tiên bao trùm vùng miền, ưu tiên chuyển đổi xanh hay ưu tiên an ninh dữ liệu.
        """
    )

    st.markdown("#### Vai trò của NSGA-II so với LP đơn mục tiêu")

    st.write(
        """
        NSGA-II khác LP đơn mục tiêu ở chỗ nó không ép toàn bộ bài toán về một hàm mục tiêu duy nhất.
        Trong LP đơn mục tiêu, nhà phân tích thường phải chọn trước một mục tiêu chính, ví dụ tối đa hóa tăng trưởng.
        Cách làm đó rõ ràng và dễ giải thích, nhưng dễ che khuất các đánh đổi giữa các mục tiêu chính sách.
        """
    )

    st.write(
        """
        NSGA-II tạo ra một tập nghiệm Pareto, tức là nhiều phương án hợp lý khác nhau.
        Mỗi phương án đại diện cho một cấu trúc đánh đổi riêng: có nghiệm thiên về tăng trưởng, có nghiệm thiên về bao trùm,
        có nghiệm giảm phát thải tốt hơn và có nghiệm giảm rủi ro dữ liệu tốt hơn.
        Vì vậy, NSGA-II phù hợp với các bài toán phát triển trong kỷ nguyên AI, nơi chính sách không chỉ cần hiệu quả kinh tế
        mà còn phải xét công bằng, môi trường, dữ liệu và năng lực quản trị.
        """
    )

    st.write(
        """
        Tuy nhiên, NSGA-II không thay thế được quyết định chính trị. Mô hình chỉ cho thấy các phương án khả thi và chi phí cơ hội giữa chúng.
        Việc chọn nghiệm cuối cùng vẫn phụ thuộc vào ưu tiên xã hội, cam kết quốc tế, khả năng ngân sách, mức chấp nhận rủi ro
        và mục tiêu phát triển từng giai đoạn. Nói cách khác, NSGA-II là công cụ hỗ trợ ra quyết định, không phải cơ chế tự động thay thế
        nhà hoạch định chính sách.
        """
    )

    st.markdown("#### Kết luận")

    st.write(
        """
        Bài 7 cho thấy trong chính sách AI và kinh tế số, không tồn tại một nghiệm tối ưu tuyệt đối cho mọi mục tiêu.
        Cách tiếp cận phù hợp hơn là nhận diện tập nghiệm Pareto, sau đó chọn nghiệm thỏa hiệp dựa trên trọng số chính sách,
        thảo luận xã hội và mục tiêu phát triển của từng giai đoạn.
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
        Lưu ý: Kết quả của Bài 7 là mô phỏng phục vụ phân tích. NSGA-II giúp minh họa đánh đổi giữa các mục tiêu chính sách,
        nhưng không thay thế quy trình thẩm định, tham vấn và ra quyết định chính sách thực tế.
        </p>
        """,
        unsafe_allow_html=True,
    )
