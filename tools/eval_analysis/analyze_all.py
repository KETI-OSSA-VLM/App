"""
전체 클립 통합 분석 — 논문용 figures 생성
출력: results_all/ 디렉토리
"""

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 데이터 정의 ────────────────────────────────────────────────────────────────
BASE = Path("C:/android/accv/tmp")
ALL  = BASE / "all_csvs"
OUT  = Path("C:/android/accv/tools/eval_analysis/results_all")
OUT.mkdir(parents=True, exist_ok=True)

CLIPS = [
    # (이름, 분류, adaptive CSV, baseline CSV)
    ("Browse1",        "Static", ALL/"eval_adaptive_20260406_020318.csv",  ALL/"eval_baseline_20260406_021957.csv"),
    ("OneShopOneWait", "Static", ALL/"eval_adaptive_20260406_024302.csv",  ALL/"eval_baseline_20260406_024547.csv"),
    ("ShopAssistant",  "Static", ALL/"eval_adaptive_20260406_023649.csv",  ALL/"eval_baseline_20260406_023950.csv"),
    ("Browse2",        "Static", ALL/"eval_adaptive_20260406_025721.csv",  ALL/"eval_baseline_20260406_025923.csv"),
    ("EnterExit",      "Dynamic", ALL/"eval_adaptive_20260406_025302.csv", ALL/"eval_baseline_20260406_025404.csv"),
    ("Fight_RunAway",  "Dynamic", ALL/"eval_adaptive_20260406_030032.csv", ALL/"eval_baseline_20260406_030531.csv"),
    ("Meet_Crowd",     "Dynamic", ALL/"eval_adaptive_20260406_024936.csv", ALL/"eval_baseline_20260406_025145.csv"),
    ("Fight_Chase",    "Dynamic", ALL/"eval_adaptive_20260406_021748.csv", ALL/"eval_baseline_20260406_023317.csv"),
    ("KR_Street",      "Dynamic", BASE/"korean_street_adaptive.csv",       BASE/"korean_street_baseline.csv"),
    ("Halloween",      "Dynamic", BASE/"E05_016_adaptive.csv",             BASE/"E05_016_baseline.csv"),
]

COLORS = {"Static": "#2196F3", "Dynamic": "#F44336"}
T_COLORS = {"ZERO": "#4CAF50", "ONE": "#2196F3", "TWO": "#F44336"}
DPI = 150


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────
def load(path):
    rows = list(csv.DictReader(open(path)))
    return rows

def stats(rows):
    n = len(rows)
    t = {"ZERO": 0, "ONE": 0, "TWO": 0}
    lat = 0.0
    lats = []
    for r in rows:
        t[r["tier"]] += 1
        v = float(r["latency_ms"])
        lat += v
        lats.append(v)
    return {"n": n, "t0": t["ZERO"]/n*100, "t1": t["ONE"]/n*100, "t2": t["TWO"]/n*100,
            "avg": lat/n, "lats": lats,
            "diffs": [float(r["diff_score"]) for r in rows],
            "tiers": [r["tier"] for r in rows]}


# ── 데이터 로드 ────────────────────────────────────────────────────────────────
data = []
for name, cat, af, bf in CLIPS:
    a = stats(load(af))
    b = stats(load(bf))
    speedup = b["avg"] / a["avg"]
    data.append({"name": name, "cat": cat, "a": a, "b": b, "speedup": speedup})

static  = [d for d in data if d["cat"] == "Static"]
dynamic = [d for d in data if d["cat"] == "Dynamic"]


# ── Fig 1: Speedup 바 차트 (정적/동적 그룹) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=DPI)

names   = [d["name"] for d in data]
speedups = [d["speedup"] for d in data]
cats    = [d["cat"] for d in data]
bar_colors = [COLORS[c] for c in cats]

bars = ax.bar(names, speedups, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.0f}×", ha="center", va="bottom", fontsize=8, fontweight="bold")

# 그룹 구분선
ax.axvline(3.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.text(1.5, max(speedups)*0.92, "Static Scenes", ha="center", fontsize=9, color=COLORS["Static"])
ax.text(7.0, max(speedups)*0.92, "Dynamic Scenes", ha="center", fontsize=9, color=COLORS["Dynamic"])

ax.set_ylabel("Speedup vs. Baseline (×)", fontsize=11)
ax.set_title("Adaptive Inference Speedup per Scene", fontsize=12)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}×"))
ax.grid(True, axis="y", linestyle="--", alpha=0.3, which="both")
ax.tick_params(axis="x", rotation=20)

patches = [mpatches.Patch(color=COLORS["Static"], label="Static"),
           mpatches.Patch(color=COLORS["Dynamic"], label="Dynamic")]
ax.legend(handles=patches, fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig1_speedup_bar.png")
plt.close(fig)
print("[saved] fig1_speedup_bar.png")


# ── Fig 2: Tier 분포 누적 바 (Stacked) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI)

names = [d["name"] for d in data]
t0s = [d["a"]["t0"] for d in data]
t1s = [d["a"]["t1"] for d in data]
t2s = [d["a"]["t2"] for d in data]
x = np.arange(len(names))

ax.bar(x, t0s, label="T0 (Cache Reuse)", color=T_COLORS["ZERO"], alpha=0.9)
ax.bar(x, t1s, bottom=t0s, label="T1 (Text-Only Regen)", color=T_COLORS["ONE"], alpha=0.9)
ax.bar(x, t2s, bottom=[a+b for a,b in zip(t0s,t1s)], label="T2 (Full Inference)", color=T_COLORS["TWO"], alpha=0.9)

ax.axvline(3.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Percentage (%)")
ax.set_title("Tier Distribution per Scene (Adaptive Mode)")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, axis="y", linestyle="--", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / "fig2_tier_distribution.png")
plt.close(fig)
print("[saved] fig2_tier_distribution.png")


# ── Fig 3: 평균 레이턴시 비교 (Adaptive vs Baseline) ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 4), dpi=DPI)

names = [d["name"] for d in data]
a_avgs = [d["a"]["avg"] for d in data]
b_avgs = [d["b"]["avg"] for d in data]
x = np.arange(len(names))
w = 0.35

ax.bar(x - w/2, a_avgs, w, label="Adaptive", color="#2196F3", alpha=0.9)
ax.bar(x + w/2, b_avgs, w, label="Baseline", color="#F44336", alpha=0.6)

ax.axvline(3.5, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.set_ylabel("Avg Latency per Frame (ms)")
ax.set_title("Average Latency: Adaptive vs Baseline")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}ms"))
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle="--", alpha=0.3, which="both")

fig.tight_layout()
fig.savefig(OUT / "fig3_latency_comparison.png")
plt.close(fig)
print("[saved] fig3_latency_comparison.png")


# ── Fig 4: 전체 레이턴시 CDF (정적/동적 그룹 별) ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=DPI, sharey=True)

for ax, group, title in [(axes[0], static, "Static Scenes"), (axes[1], dynamic, "Dynamic Scenes")]:
    for d in group:
        for lats, label, ls, alpha in [(d["a"]["lats"], d["name"]+"\n(Adaptive)", "-", 0.8),
                                        (d["b"]["lats"], None, "--", 0.3)]:
            sl = sorted(lats)
            cdf = np.arange(1, len(sl)+1) / len(sl)
            ax.plot(sl, cdf, linestyle=ls, alpha=alpha, linewidth=1.2,
                    label=label if label else "_")
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")

axes[0].set_ylabel("CDF")
fig.suptitle("Latency CDF — Adaptive (solid) vs Baseline (dashed)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig4_latency_cdf.png")
plt.close(fig)
print("[saved] fig4_latency_cdf.png")


# ── Fig 5: T0 비율 vs Speedup 스캐터 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5), dpi=DPI)

for d in data:
    color = COLORS[d["cat"]]
    ax.scatter(d["a"]["t0"], d["speedup"], color=color, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax.annotate(d["name"], (d["a"]["t0"], d["speedup"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7, color="dimgray")

ax.set_xlabel("T0 (Cache Reuse) Ratio (%)", fontsize=11)
ax.set_ylabel("Speedup vs. Baseline (×)", fontsize=11)
ax.set_title("T0 Ratio vs. Speedup", fontsize=12)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}×"))
ax.grid(True, linestyle="--", alpha=0.3, which="both")

patches = [mpatches.Patch(color=COLORS["Static"], label="Static"),
           mpatches.Patch(color=COLORS["Dynamic"], label="Dynamic")]
ax.legend(handles=patches, fontsize=9)

fig.tight_layout()
fig.savefig(OUT / "fig5_t0_vs_speedup.png")
plt.close(fig)
print("[saved] fig5_t0_vs_speedup.png")


# ── 요약 테이블 출력 ───────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Clip':<16} {'Cat':<8} {'Frames':>6} {'T0%':>6} {'T1%':>6} {'T2%':>6} {'A-avg':>7} {'B-avg':>7} {'Speedup':>8}")
print("-"*75)
for d in data:
    print(f"{d['name']:<16} {d['cat']:<8} {d['a']['n']:>6} "
          f"{d['a']['t0']:>6.1f} {d['a']['t1']:>6.1f} {d['a']['t2']:>6.1f} "
          f"{d['a']['avg']:>7.0f} {d['b']['avg']:>7.0f} {d['speedup']:>8.1f}×")

all_speedups = [d["speedup"] for d in data]
s_speedups   = [d["speedup"] for d in static]
d_speedups   = [d["speedup"] for d in dynamic]
print("="*75)
print(f"Static  avg speedup: {sum(s_speedups)/len(s_speedups):.1f}×  "
      f"(min {min(s_speedups):.1f}× / max {max(s_speedups):.1f}×)")
print(f"Dynamic avg speedup: {sum(d_speedups)/len(d_speedups):.1f}×  "
      f"(min {min(d_speedups):.1f}× / max {max(d_speedups):.1f}×)")
print(f"Overall avg speedup: {sum(all_speedups)/len(all_speedups):.1f}×  "
      f"(min {min(all_speedups):.1f}× / max {max(all_speedups):.1f}×)")
print(f"\nFigures saved to: {OUT.resolve()}")
