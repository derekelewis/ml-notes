import io
import os

import pandas as pd
import plotly.express as px


def parse_data(file_path):
    """
    Reads a genai_perf.csv, skipping lines 6 & 7, returns the resulting DataFrame.
    """
    buffer = io.StringIO()
    with open(file_path, "rt") as f:
        for i, line in enumerate(f):
            if i not in [6, 7]:
                buffer.write(line)
    buffer.seek(0)
    return pd.read_csv(buffer)


def extract_tps_ttft(csv_path):
    """
    From a genai_perf.csv, return (TPS, TTFT_in_seconds).
    - TPS is row[6]['avg'] (with commas stripped).
    - TTFT is row[0]['avg']/1000 (with commas stripped).
    """
    df = parse_data(csv_path)
    print(df)
    tps_str = df.iloc[6]["avg"].replace(",", "")
    ttft_str = df.iloc[0]["avg"].replace(",", "")
    return float(tps_str), float(ttft_str) / 1000.0


concurrencies = [1, 2, 5, 10, 50, 100, 250, 500]

# Root paths for each system type
systems = {
    # "2×A100 PCIE": "./artifacts/2xa100pcie",
    # "2×H100 SXM": "./artifacts/2xh100sxm",
    "4×A100 PCIe": "./artifacts/4xa100pcie",
    "4×H100 SXM": "./artifacts/4xh100sxm",
}

# Which CSVs to plot
csv_variants = {
    "200_200": "200_200_genai_perf.csv",
    "1000_200": "1000_200_genai_perf.csv",
}

all_dfs = {}

for variant_label, csv_name in csv_variants.items():
    rows = []
    for system_label, root_dir in systems.items():
        for con in concurrencies:
            sub_dir = f"meta_llama-3.3-70b-instruct-openai-chat-concurrency{con}"
            csv_path = os.path.join(root_dir, sub_dir, csv_name)
            tps, ttft = extract_tps_ttft(csv_path)
            rows.append(
                {
                    "System": system_label,
                    "Concurrency": con,
                    "TPS": tps,
                    "TTFT": ttft,
                }
            )
    df = pd.DataFrame(rows)

    df["TPS_per_user"] = df["TPS"] / df["Concurrency"]
    df["TPS_per_user"] = df["TPS_per_user"].round(2)

    all_dfs[variant_label] = df

fig_200_200 = px.line(
    all_dfs["200_200"],
    x="TTFT",
    y="TPS",
    color="System",
    markers=True,
    text="Concurrency",
    title="Performance (200/200 ISL/OSL) Across 4×A100 PCIe & 4×H100 SXM",
)
fig_200_200.update_layout(
    xaxis_title="Single User TTFT (seconds)",
    yaxis_title="Total System TPS",
)
fig_200_200.update_xaxes(type="log")
fig_200_200.update_traces(textposition="top center", textfont={"size": 15})
fig_200_200.show()

fig_1000_200 = px.line(
    all_dfs["1000_200"],
    x="TTFT",
    y="TPS",
    color="System",
    markers=True,
    text="Concurrency",
    title="Performance (1000/200 ISL/OSL) Across 4×A100 PCIe & 4×H100 SXM",
)
fig_1000_200.update_layout(
    xaxis_title="Single User TTFT (seconds)",
    yaxis_title="Total System TPS",
)
fig_1000_200.update_xaxes(type="log")
fig_1000_200.update_traces(textposition="top center", textfont={"size": 15})
fig_1000_200.show()

fig_200_200_per_user = px.line(
    all_dfs["200_200"],
    x="Concurrency",
    y="TTFT",
    color="System",
    markers=True,
    text="TPS_per_user",
    title="TTFT vs Concurrency (200/200 ISL/OSL) TPS/user",
)
fig_200_200_per_user.update_layout(
    xaxis_title="Concurrency (users)",
    yaxis_title="TTFT (seconds)",
)
fig_200_200_per_user.update_xaxes(type="log")
fig_200_200.update_traces(textfont={"size": 15})
fig_200_200_per_user.update_traces(
    textposition="top center",
    textfont={"size": 15},
    selector=dict(name="4×A100 PCIe"),
)
fig_200_200_per_user.update_traces(
    textposition="bottom center",
    textfont={"size": 15},
    selector=dict(name="4×H100 SXM"),
)
fig_200_200_per_user.show()

fig_1000_200_per_user = px.line(
    all_dfs["1000_200"],
    x="Concurrency",
    y="TTFT",
    color="System",
    markers=True,
    text="TPS_per_user",
    title="TTFT vs Concurrency (1000/200 ISL/OSL) TPS/user",
)
fig_1000_200_per_user.update_layout(
    xaxis_title="Concurrency (users)",
    yaxis_title="TTFT (seconds)",
)
fig_1000_200_per_user.update_xaxes(type="log")
fig_1000_200_per_user.update_traces(
    textposition="top center",
    textfont={"size": 15},
    selector=dict(name="4×A100 PCIe"),
)
fig_1000_200_per_user.update_traces(
    textposition="bottom center",
    textfont={"size": 15},
    selector=dict(name="4×H100 SXM"),
)
fig_1000_200_per_user.show()
