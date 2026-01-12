import re
import os

import matplotlib
matplotlib.use("Agg")   # 必须放在 import pyplot 之前

import matplotlib.pyplot as plt


LOG_FILE = "/mnt/pfs/3zpd5q/code/train/DexVLA/server_logs/20260104_173542.txt"     
SAVE_DIR = "/mnt/pfs/3zpd5q/code/train/DexVLA/server_logs"                 # 保存目录
SAVE_NAME = "duration_plot.png"      # 图片名


def parse_log(file_path):
    recv_durations = []
    model_durations = []
    x_idx = []

    recv_pattern = re.compile(r"\[RecvDuration\]:\s*([\d\.]+)s")
    model_pattern = re.compile(r"\[ModelDuration\]:\s*([\d\.]+)s")

    line_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            recv_match = recv_pattern.search(line)
            model_match = model_pattern.search(line)

            if recv_match and model_match:
                recv = float(recv_match.group(1))
                model = float(model_match.group(1))

                line_count += 1
                x_idx.append(line_count)
                recv_durations.append(recv)
                model_durations.append(model)

    return x_idx, recv_durations, model_durations


def plot_and_save(x, recv, model, save_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ax1: RecvDuration
    ax1.plot(x, recv, marker='o')
    ax1.set_ylabel("RecvDuration (s)")
    ax1.set_title("RecvDuration over Log Lines")
    ax1.grid(True)

    # ax2: ModelDuration
    ax2.plot(x, model, marker='o')
    ax2.set_ylabel("ModelDuration (s)")
    ax2.set_xlabel("Log Line Index (Accumulated)")
    ax2.set_title("ModelDuration over Log Lines")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)

    x, recv, model = parse_log(LOG_FILE)

    if len(x) == 0:
        print("❌ No valid log entries found, nothing saved.")
    else:
        plot_and_save(x, recv, model, save_path)
        print(f"✅ Plot saved to: {save_path}")
