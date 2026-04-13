"""
Adaptive Video Inference — 정량 평가 분석 스크립트

사용법:
    python analyze.py --adaptive eval_adaptive.csv --baseline eval_baseline.csv --out results/

CSV 컬럼: frame, timestamp_ms, tier, latency_ms, diff_score
tier 값: ZERO (T0), ONE (T1), TWO (T2)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── 공통 설정 ──────────────────────────────────────────────────────────────────
TIER_ORDER = ["ZERO", "ONE", "TWO"]
TIER_LABELS = {"ZERO": "T0 (Cache)", "ONE": "T1 (Fast Update)", "TWO": "T2 (Full)"}
TIER_COLORS = {"ZERO": "#4CAF50", "ONE": "#2196F3", "TWO": "#F44336"}
FIGSIZE = (7, 4)
DPI = 150


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"frame", "timestamp_ms", "tier", "latency_ms", "diff_score"}
    missing = expected - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] {path} 에 컬럼 누락: {missing}")
    return df


# ── 통계 요약 ─────────────────────────────────────────────────────────────────
def summary(df: pd.DataFrame, label: str) -> dict:
    total = len(df)
    counts = {t: (df["tier"] == t).sum() for t in TIER_ORDER}
    pcts = {t: counts[t] / total * 100 for t in TIER_ORDER}
    avg_lat = df["latency_ms"].mean()
    median_lat = df["latency_ms"].median()
    total_time_s = df["latency_ms"].sum() / 1000
    return dict(
        label=label,
        total_frames=total,
        t0_count=counts["ZERO"],
        t1_count=counts["ONE"],
        t2_count=counts["TWO"],
        t0_pct=pcts["ZERO"],
        t1_pct=pcts["ONE"],
        t2_pct=pcts["TWO"],
        avg_latency_ms=avg_lat,
        median_latency_ms=median_lat,
        total_time_s=total_time_s,
    )


def print_summary_table(ad: dict, bl: dict):
    speedup = bl["avg_latency_ms"] / ad["avg_latency_ms"] if ad["avg_latency_ms"] > 0 else float("inf")
    frame_ratio = ad["total_frames"] / bl["total_frames"] if bl["total_frames"] > 0 else float("inf")

    print("\n" + "=" * 60)
    print("  Quantitative Evaluation Summary")
    print("=" * 60)
    rows = [
        ("Total frames processed", f"{ad['total_frames']}", f"{bl['total_frames']}",
         f"{frame_ratio:.1f}×"),
        ("T0 Cache (%)", f"{ad['t0_pct']:.1f}%", f"{bl['t0_pct']:.1f}%", "—"),
        ("T1 Fast Update (%)", f"{ad['t1_pct']:.1f}%", f"{bl['t1_pct']:.1f}%", "—"),
        ("T2 Full Inference (%)", f"{ad['t2_pct']:.1f}%", f"{bl['t2_pct']:.1f}%", "—"),
        ("Avg latency / frame (ms)", f"{ad['avg_latency_ms']:.0f}", f"{bl['avg_latency_ms']:.0f}",
         f"{speedup:.1f}×"),
        ("Median latency / frame (ms)", f"{ad['median_latency_ms']:.0f}", f"{bl['median_latency_ms']:.0f}",
         "—"),
        ("Total inference time (s)", f"{ad['total_time_s']:.1f}", f"{bl['total_time_s']:.1f}", "—"),
    ]
    header = f"{'Metric':<30} {'Adaptive':>12} {'Baseline':>12} {'Gain':>8}"
    print(header)
    print("-" * 60)
    for name, a_val, b_val, gain in rows:
        print(f"{name:<30} {a_val:>12} {b_val:>12} {gain:>8}")
    print("=" * 60)
    print(f"  Speedup: {speedup:.2f}×  |  Frames processed: {frame_ratio:.2f}×")
    print("=" * 60 + "\n")


# ── Figure 1: Tier 분포 비교 (Grouped Bar) ────────────────────────────────────
def plot_tier_distribution(ad: dict, bl: dict, out: Path):
    tiers = TIER_ORDER
    labels = [TIER_LABELS[t] for t in tiers]
    ad_vals = [ad[f"t{i}_pct"] for i, t in enumerate(tiers) for _ in [None] if t == TIER_ORDER[i]]
    ad_vals = [ad["t0_pct"], ad["t1_pct"], ad["t2_pct"]]
    bl_vals = [bl["t0_pct"], bl["t1_pct"], bl["t2_pct"]]

    x = np.arange(len(tiers))
    w = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    bars_a = ax.bar(x - w / 2, ad_vals, w, label="Adaptive", color=[TIER_COLORS[t] for t in tiers], alpha=0.9)
    bars_b = ax.bar(x + w / 2, bl_vals, w, label="Baseline", color=[TIER_COLORS[t] for t in tiers], alpha=0.4,
                    edgecolor="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Tier Distribution: Adaptive vs Baseline")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.set_ylim(0, 105)

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=8,
                color="gray")

    fig.tight_layout()
    path = out / "fig1_tier_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── Figure 2: 프레임별 레이턴시 (Adaptive, 시계열) ───────────────────────────
def plot_latency_timeline(df_ad: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(9, 3.5), dpi=DPI)

    colors = df_ad["tier"].map(TIER_COLORS)
    ax.scatter(df_ad["frame"], df_ad["latency_ms"], c=colors, s=6, alpha=0.7)

    # 범례
    for t in TIER_ORDER:
        ax.scatter([], [], c=TIER_COLORS[t], label=TIER_LABELS[t], s=30)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Frame Latency — Adaptive Mode")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    path = out / "fig2_latency_timeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── Figure 3: 레이턴시 CDF 비교 ───────────────────────────────────────────────
def plot_latency_cdf(df_ad: pd.DataFrame, df_bl: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for df, label, color, ls in [
        (df_ad, "Adaptive", "#2196F3", "-"),
        (df_bl, "Baseline", "#F44336", "--"),
    ]:
        sorted_lat = np.sort(df["latency_ms"].values)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
        ax.plot(sorted_lat, cdf, color=color, linestyle=ls, label=label, linewidth=1.8)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF: Adaptive vs Baseline")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xscale("log")

    fig.tight_layout()
    path = out / "fig3_latency_cdf.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── Figure 4: diff_score 분포 히스토그램 + 임계값 표시 ───────────────────────
def plot_diff_score_histogram(df_ad: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.hist(df_ad["diff_score"], bins=80, color="#607D8B", alpha=0.8, edgecolor="none")

    low_thresh = 0.01
    high_thresh = 0.05
    ax.axvline(low_thresh, color=TIER_COLORS["ONE"], linestyle="--", linewidth=1.5,
               label=f"T0/T1 boundary ({low_thresh})")
    ax.axvline(high_thresh, color=TIER_COLORS["TWO"], linestyle="--", linewidth=1.5,
               label=f"T1/T2 boundary ({high_thresh})")

    ax.set_xlabel("diff_score (32×32 MAD)")
    ax.set_ylabel("Frame count")
    ax.set_title("diff_score Distribution (Adaptive)")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    path = out / "fig4_diff_score_histogram.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── Figure 5: Ablation — T0+T2 vs T0+T1+T2 평균 레이턴시 비교 ────────────────
def plot_ablation(df_ad: pd.DataFrame, out: Path):
    """
    T0+T2만 사용했을 때를 시뮬레이션:
    T1 프레임을 T2로 처리했다고 가정 (T1 latency → T2 평균 latency)
    """
    t2_avg = df_ad.loc[df_ad["tier"] == "TWO", "latency_ms"].mean()
    if np.isnan(t2_avg):
        print("[skip] T2 프레임 없음, ablation 스킵")
        return

    df_sim = df_ad.copy()
    df_sim.loc[df_sim["tier"] == "ONE", "latency_ms"] = t2_avg

    avg_full = df_ad["latency_ms"].mean()
    avg_no_t1 = df_sim["latency_ms"].mean()

    fig, ax = plt.subplots(figsize=(5, 4), dpi=DPI)
    bars = ax.bar(
        ["T0+T1+T2\n(Proposed)", "T0+T2\n(w/o T1)"],
        [avg_full, avg_no_t1],
        color=["#2196F3", "#FF9800"],
        width=0.5,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f} ms", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Avg latency per frame (ms)")
    ax.set_title("Ablation: Effect of T1 Tier")
    ax.set_ylim(0, avg_no_t1 * 1.2)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    path = out / "fig5_ablation_t1.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[saved] {path}")


# ── LaTeX 테이블 출력 ─────────────────────────────────────────────────────────
def print_latex_table(ad: dict, bl: dict):
    speedup = bl["avg_latency_ms"] / ad["avg_latency_ms"]
    frame_ratio = ad["total_frames"] / bl["total_frames"]
    print("\n--- LaTeX Table ---")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Quantitative evaluation on CAVIAR dataset.}")
    print(r"\label{tab:eval}")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"Metric & Adaptive & Baseline \\")
    print(r"\hline")
    print(f"Frames processed & {ad['total_frames']} & {bl['total_frames']} ({frame_ratio:.1f}$\\times$) \\\\")
    print(f"T0 ratio (\\%) & {ad['t0_pct']:.1f} & {bl['t0_pct']:.1f} \\\\")
    print(f"T1 ratio (\\%) & {ad['t1_pct']:.1f} & {bl['t1_pct']:.1f} \\\\")
    print(f"T2 ratio (\\%) & {ad['t2_pct']:.1f} & {bl['t2_pct']:.1f} \\\\")
    print(f"Avg latency (ms) & {ad['avg_latency_ms']:.0f} & {bl['avg_latency_ms']:.0f} ({speedup:.1f}$\\times$) \\\\")
    print(f"Total time (s) & {ad['total_time_s']:.1f} & {bl['total_time_s']:.1f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("-------------------\n")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Adaptive Video Inference eval analysis")
    parser.add_argument("--adaptive", required=True, help="Adaptive 모드 CSV 경로")
    parser.add_argument("--baseline", required=True, help="Baseline 모드 CSV 경로")
    parser.add_argument("--out", default="results", help="출력 디렉토리 (기본: results/)")
    parser.add_argument("--latex", action="store_true", help="LaTeX 테이블 출력")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df_ad = load(args.adaptive)
    df_bl = load(args.baseline)

    ad = summary(df_ad, "Adaptive")
    bl = summary(df_bl, "Baseline")

    print_summary_table(ad, bl)

    plot_tier_distribution(ad, bl, out)
    plot_latency_timeline(df_ad, out)
    plot_latency_cdf(df_ad, df_bl, out)
    plot_diff_score_histogram(df_ad, out)
    plot_ablation(df_ad, out)

    if args.latex:
        print_latex_table(ad, bl)

    print(f"\n분석 완료. 결과물: {out.resolve()}/")


if __name__ == "__main__":
    main()
