import h5py
import numpy as np
import os
from pathlib import Path
import shutil
import zarr
import numcodecs

def process_file(s, new_h5_path, cam_name, n_frames=10):
    try:
        with h5py.File(new_h5_path, 'w') as fout:
            # ========================================================
            # 核心修改：在存入最终文件夹前，智能拦截并纠正坐标！
            # ========================================================
            raw_action = s["action"][()]
            raw_qpos = s["qpos"][()]
            
            # 自动检测哪一维是“前方”：跟随任务中，前后的移动波动绝对最大
            max_dim_0 = np.max(np.abs(raw_action[:, 0]))
            max_dim_1 = np.max(np.abs(raw_action[:, 1]))
            
            final_action = np.copy(raw_action)
            final_qpos = np.copy(raw_qpos)
            
            # 如果第 0 维波动 > 第 1 维，实锤第 0 维是前后(X)，触发翻转机制！
            if max_dim_0 > max_dim_1:
                final_action[:, 0] = raw_action[:, 1]  # 第 0 维放 Y(左右)
                final_action[:, 1] = raw_action[:, 0]  # 第 1 维放 X(前后)
                
                # 同步翻转 qpos (如果存在前两维)
                if final_qpos.shape[-1] >= 2:
                    final_qpos[:, 0] = raw_qpos[:, 1]
                    final_qpos[:, 1] = raw_qpos[:, 0]
            
            # 写入纠正后的安全数据
            fout.create_dataset("/action", data=final_action)
            fout.create_dataset("/observations/qpos", data=final_qpos)
            # ========================================================

            language_raw = s["language_raw"]
            fout.create_dataset("language_raw", data=language_raw)

            history_paths = s["observations/history_images"][()]

            # 处理历史图片路径
            if int(s["obs_idx"][()]) <= n_frames:
                if len(history_paths) > 1:
                    history_paths[0] = history_paths[1]
                else:
                    history_paths[0] = s["observations/images"][()]
                history_paths = history_paths.tolist()

                if n_frames > len(history_paths):
                    history_paths = [history_paths[0]] * (n_frames - len(history_paths)) + history_paths

            fout.create_dataset("/observations/history_images", data=np.array(history_paths), compression='gzip')

            obs = s["observations/images"]
            fout.create_dataset(f"/observations/images/{cam_name}", data=obs)

    except Exception as e:
        print(f"Error while processing {new_h5_path}: {e}")
        if os.path.exists(new_h5_path):
            os.remove(new_h5_path)
        print(f"File {new_h5_path} has been removed due to the error.")
        return False
    return True

# process_file_zarr 保留你的原版，如果你不跑 zarr 也可以不改
def process_file_zarr(s, new_zarr_path, cam_name, n_frames=10):
    pass # 这里保留你原版的代码，为了排版简洁我折叠了

def save_selected_keys_as_individual_h5(src_h5_path, dst_dir):
    try:
        with h5py.File(src_h5_path, 'r') as fin:
            if "follow_paths" not in fin:
                print(f"[SKIP] No 'follow_paths' group in {src_h5_path}")
                return 0

            sgrp_all = fin["follow_paths"]
            count = 0
            base_filename = os.path.basename(src_h5_path).replace('.hdf5', '')
            cam_name = 'cam_high'

            for sub in sgrp_all:
                s = sgrp_all[sub]
                new_h5_filename = f"{base_filename}_{sub}.hdf5"
                new_h5_path = os.path.join(dst_dir, new_h5_filename)

                try:
                    success = process_file(s, new_h5_path, cam_name, n_frames=10)
                    if success:
                        print(f"[OK] {new_h5_path}")
                        count += 1
                    else:
                        print(f"[FAIL] Process error in {new_h5_path}, skipping.")
                except Exception as e:
                    print(f"[ERROR] Exception processing {new_h5_path}: {e}")
                    continue

            return count
    except OSError as e:
        print(f"[ERROR] Failed to open HDF5: {src_h5_path} — {e}")
        return 0
    except Exception as e:
        print(f"[ERROR] Unexpected error: {src_h5_path} — {e}")
        return 0

def process_all_hdf5_in_directory(src_dir, dst_dir):
    h5_files = sorted(Path(src_dir).glob("*.hdf5"))
    
    if not h5_files:
        print(f"[WARNING] No HDF5 files found in {src_dir}")
        return

    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    total = 0
    for h5_file in h5_files:
        print(f"\n>>> Processing {h5_file}")
        try:
            count = save_selected_keys_as_individual_h5(h5_file, dst_dir)
            total += count
            print(f"Current total episodes: {total}")
        except Exception as e:
            print(f"[ERROR] Skipping {h5_file} due to: {e}")
            continue

if __name__ == "__main__":
    src_dir = "./data/proc_data/multi_follow_temp/yongshuiqiao_library_person_follow"
    dst_dir = "./data/split_data/yongshuiqiao_library_person_follow"
    
    # 建议运行前手动清空一下旧的 split_data，防止有没被覆盖到的漏网之鱼
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
        print(f"已清空旧的 {dst_dir}，准备生成新数据！")

    process_all_hdf5_in_directory(src_dir, dst_dir)