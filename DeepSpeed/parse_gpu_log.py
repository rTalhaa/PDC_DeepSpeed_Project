import argparse
from pathlib import Path
import pandas as pd

def num(s):
    return s.astype(str).str.replace(" MiB","", regex=False).str.replace(" %","", regex=False).str.replace(" W","", regex=False).astype(float)

p = argparse.ArgumentParser()
p.add_argument("--log", default="logs/gpu_log_zero2_fair.csv")
p.add_argument("--out", default="results/gpu_summary_zero2_fair.csv")
args = p.parse_args()

df = pd.read_csv(args.log, skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
mem = [c for c in df.columns if "memory.used" in c][0]
util = [c for c in df.columns if "utilization.gpu" in c][0]
power = [c for c in df.columns if "power.draw" in c][0]

df["memory_used_mib_num"] = num(df[mem])
df["gpu_util_percent_num"] = num(df[util])
df["power_draw_w_num"] = num(df[power])

summary = df.groupby("index").agg(
    max_memory_used_mib=("memory_used_mib_num","max"),
    avg_memory_used_mib=("memory_used_mib_num","mean"),
    max_gpu_util_percent=("gpu_util_percent_num","max"),
    avg_gpu_util_percent=("gpu_util_percent_num","mean"),
    max_power_draw_w=("power_draw_w_num","max"),
    avg_power_draw_w=("power_draw_w_num","mean"),
).reset_index()

Path(args.out).parent.mkdir(exist_ok=True)
summary.to_csv(args.out, index=False)
print(summary)
print("Saved:", args.out)
