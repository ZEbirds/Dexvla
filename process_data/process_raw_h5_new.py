import h5py
import numpy as np
import os
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors

# ----------------【复刻老数据的插值函数】----------------
def interpolate_rel_path(rel_path: np.ndarray,
                         chunk_size: int,
                         max_dist: float) -> np.ndarray:
    """
    把 (x,z,yaw) 路径插值 / 截断到固定长度 chunk_size.
    rel_path : (...,8) 或 (...,3)
    """
    if rel_path.ndim != 2 or rel_path.shape[1] not in (3, 8):
        raise ValueError("rel_path shape must be (N,3) or (N,8)")

    # 1. 取 x,z,yaw
    data = rel_path[:, [0, 2, 7]] if rel_path.shape[1] == 8 else rel_path.copy()

    # 2. 特例：全 0 或空
    if data.size == 0 or np.allclose(data, 0):
        return np.zeros((chunk_size, 3), np.float32)

    # 3. 计算沿线累积距离
    diffs  = np.diff(data[:, :2], axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    s_full = np.concatenate(([0], np.cumsum(dists)))        # len = N
    total  = s_full[-1]

    # 若超过 max_dist → 找截断点 (总长≥max_dist)
    if total > max_dist:
        idx = np.searchsorted(s_full, max_dist)
        if idx == len(s_full):
            idx -= 1
        # 截断为 idx+1 个点，并在 idx 点上插入精确 max_dist 位置
        excess = s_full[idx] - max_dist
        if excess > 1e-6 and idx > 0:
            ratio = (dists[idx-1] - excess) / dists[idx-1]
            interp_pt = data[idx-1] + ratio * (data[idx] - data[idx-1])
            data = np.vstack([data[:idx], interp_pt])
            s_full = np.concatenate(([0], np.cumsum(np.linalg.norm(
                np.diff(data[:, :2], axis=0), axis=1))))

        total = max_dist

    # 4. 等间距采样到 chunk_size
    if chunk_size == 1:
        samples = np.array([[0, 0, 0]], np.float32)
    else:
        s_samples = np.linspace(0, total, chunk_size)
        samples = np.zeros((chunk_size, 3), np.float32)
        yaw_src = np.unwrap(data[:, 2])    
        for k, s in enumerate(s_samples):
            idx = np.searchsorted(s_full, s) - 1
            idx = np.clip(idx, 0, len(s_full) - 2)
            seg_len = s_full[idx+1] - s_full[idx]
            if seg_len < 1e-8:
                samples[k] = data[idx]
            else:
                t = (s - s_full[idx]) / seg_len
                samples[k, :2] = data[idx, :2] + t * (data[idx+1, :2] - data[idx, :2])
                # yaw 带环绕，简单线插足够（前提 Δyaw 不跨 ±π）
                # samples[k, 2] = data[idx, 2] + t * (data[idx+1, 2] - data[idx, 2])
                yaw_lin = yaw_src[idx] + t * (yaw_src[idx+1] - yaw_src[idx])   # ② 线性插值
                samples[k, 2] = (yaw_lin + np.pi) % (2 * np.pi) - np.pi   

    return samples.astype(np.float32)

# ----------------【可视化函数 - 修正 ROS X朝前】----------------
def visualize_follow_path_new(obs_idx: int,
                              actions: np.ndarray,
                              human_local_2d: np.ndarray,
                              out_png: Path,
                              cmap_name: str = "viridis"):
    """
    actions : (30, 3) 
    """
    forward_val = actions[:, 0]  # X 朝前
    lateral_val = actions[:, 1]  # Y 朝左
    traj_plot = np.stack([lateral_val, forward_val], axis=-1)
    yaw_deg = np.degrees(actions[:, 2])            
    steps   = np.arange(len(actions))              

    cmap   = cm.get_cmap(cmap_name)
    norm   = colors.Normalize(vmin=0, vmax=max(1, len(actions)-1)) 
    colors_arr = cmap(norm(steps))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 8),
        gridspec_kw={"height_ratios": [2, 1]},
        sharex=False
    )

    if len(traj_plot) > 1:
        segs = np.concatenate(
            [traj_plot[:-1, None, :], traj_plot[1:, None, :]], axis=1
        )
        lc = LineCollection(segs, colors=colors_arr[:-1], linewidths=2)
        ax_top.add_collection(lc)
    
    ax_top.scatter(0, 0, c="red", marker="*", s=150, label="Robot (0,0)")
    
    if human_local_2d is not None:
        ax_top.scatter(human_local_2d[1], human_local_2d[0],
                       c="blue", marker="*", s=150, label="Human")
        
    ax_top.set_aspect("equal")
    ax_top.set_xlim(-3, 3)   
    ax_top.set_ylim(-1, 5)   
    ax_top.set_xlabel("Local Y (Lateral / Left) [m]")
    ax_top.set_ylabel("Local X (Forward) [m]")
    ax_top.legend(fontsize="small")
    ax_top.set_title(f"obs_idx = {obs_idx}")

    if len(yaw_deg) > 1:
        for i in range(len(yaw_deg)-1):
            ax_bot.plot(steps[i:i+2], yaw_deg[i:i+2],
                        color=colors_arr[i], linewidth=2)
    else:
        ax_bot.plot(steps, yaw_deg, 'o')
        
    ax_bot.set_xlabel("timestep")
    ax_bot.set_ylabel("Yaw (deg)")
    ax_bot.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_top, ax_bot], orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label("future time step")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# -----------------------------------------------------------

def process_new_data_split(src_file: Path, frames_root: Path, dst_root: Path, viz_root: Path = None, history_len: int = 10):
    ep_name = src_file.stem 
    
    with h5py.File(src_file, "r") as fin:
        traj_keys = [k for k in fin.keys() if k.startswith('traj_')]
        
        if not traj_keys:
            return
            
        print(f"📦 发现集装箱文件 {ep_name}.h5，开始拆分、补齐原点并插值 30 步...")
        
        for traj_name in traj_keys:
            dst_h5 = dst_root / f"{ep_name}_{traj_name}_proc.hdf5" 
            
            traj_group = fin[traj_name]
            rgbs = traj_group["rgb"][()]
            local_trajs = traj_group["local_traj"][()]
            local_points = traj_group["local_points"][()] if "local_points" in traj_group else None
            instruction = traj_group["instruction"][()].decode('utf-8') 
            num_frames = rgbs.shape[0]
            
            frames_dir = frames_root / ep_name / traj_name
            frames_dir.mkdir(parents=True, exist_ok=True)
            
            traj_frame_paths = []
            for i in range(num_frames):
                png_path = frames_dir / f"frame_{i:06d}.png"
                Image.fromarray(rgbs[i, ..., :3]).save(png_path)
                traj_frame_paths.append(png_path.as_posix())
            
            with h5py.File(dst_h5, "w") as fout:
                dgrp_all = fout.create_group("follow_paths")
                
                for i in range(num_frames):
                    sub_name = f"{i:06d}" 
                    d = dgrp_all.create_group(sub_name)
                    
                    d.create_dataset("obs_idx", data=i)
                    d.create_dataset("observations/images", data=traj_frame_paths[i], dtype=h5py.string_dtype())
                    
                    start = max(0, i - history_len)
                    hist_paths = traj_frame_paths[start:i]
                    if len(hist_paths) < history_len:
                        pad_frame = hist_paths[0] if len(hist_paths) > 0 else traj_frame_paths[i]
                        hist_paths = [pad_frame] * (history_len - len(hist_paths)) + hist_paths
                        
                    d.create_dataset("observations/history_images", data=np.array(hist_paths, dtype=h5py.string_dtype()))
                    
                    # =======================================================
                    # 核心修改：补齐 (0,0,0) 原点并插值成 30 步！
                    # =======================================================
                    raw_action = local_trajs[i].astype(np.float32) # 原本 (5, 3)

                    # 【新增：强行把 ROS坐标 转换成 旧数据的习惯 [左Y, 前X, Yaw]】
                    swapped_action = np.zeros_like(raw_action)
                    swapped_action[:, 0] = raw_action[:, 1]  # 第 0 维放 左右(Y)
                    swapped_action[:, 1] = raw_action[:, 0]  # 第 1 维放 前后(X)
                    swapped_action[:, 2] = raw_action[:, 2]  # Yaw 保持不变
                    
                    # 1. 强行在开头插入 [0,0,0]
                    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
                    action_with_origin = np.vstack([origin, raw_action]) # 变成 (6, 3)
                    
                    # 2. 插值并截断，生成完美的 30 步密集轨迹
                    final_action = interpolate_rel_path(action_with_origin, chunk_size=30, max_dist=3.0) # 变成 (30, 3)
                    
                    d.create_dataset('action', data=final_action, compression='gzip')
                    d.create_dataset('language_raw', data=instruction)
                    
                    qposes = np.zeros_like(final_action)
                    d.create_dataset('qpos', data=qposes, compression='gzip')
                    
                    if viz_root is not None:
                        viz_png_path = viz_root / ep_name / traj_name / f"action_viz_{i:06d}.png"
                        human_pos_2d = local_points[i] if local_points is not None else None
                        visualize_follow_path_new(obs_idx=i, 
                                                  actions=final_action, 
                                                  human_local_2d=human_pos_2d, 
                                                  out_png=viz_png_path)
                
                fout.create_dataset("frame_paths", data=np.array(traj_frame_paths, dtype=h5py.string_dtype()))
                
            print(f"  ✅ 拆分完成: {dst_h5.name} ({num_frames} 帧)")

if __name__ == "__main__":
    # 存放你 5 个原始大 .h5 文件的文件夹
    raw_h5_dir = Path("/mnt/pfs/3zpd5q/code/zf/raw_data/render_data/person_follow/")
    
    # 图片和中间文件的根目录
    frames_root = Path("data/frames/multi_follow")
    dst_root = Path("data/proc_data/multi_follow_temp")
    viz_root = Path("results_multi/test/viz_temp_30steps")
    
    # 自动找到目录下所有的 .h5 文件并挨个处理
    all_h5_files = list(raw_h5_dir.glob("*.h5"))
    print(f"一共发现了 {len(all_h5_files)} 个数据包，准备开始批量流水线...")
    
    for src_file in all_h5_files:
        # 为每个包创建一个专属的输出子文件夹，比如 floor_1_xxx, floor_2_xxx
        ep_name = src_file.stem 
        if ep_name != "yongshuiqiao_zhongting_person_follow":
            continue
        dst_dir = dst_root / ep_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=====================================")
        print(f"正在处理第 {all_h5_files.index(src_file)+1}/{len(all_h5_files)} 个包: {ep_name}")
        process_new_data_split(src_file, frames_root, dst_dir, viz_root=viz_root)