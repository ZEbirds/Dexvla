import h5py
import numpy as np

def print_h5_tree(name, node):
    """递归打印 HDF5 文件的树状结构和 Dataset 的详细信息"""
    indent = "  " * name.count('/')
    
    if isinstance(node, h5py.Group):
        print(f"{indent}📁 Group: /{name}")
    elif isinstance(node, h5py.Dataset):
        dtype = node.dtype
        shape = node.shape
        print(f"{indent}📄 Dataset: /{name}  | Shape: {shape} | Type: {dtype}")
        
        # 对于标量或小数组，打印出具体的值以便检查
        if len(shape) == 0 or (len(shape) == 1 and shape[0] <= 10):
             print(f"{indent}    └─ Value: {node[()]}")

def inspect_file(file_path):
    print(f"========== Inspecting: {file_path} ==========")
    try:
        with h5py.File(file_path, 'r') as f:
            f.visititems(print_h5_tree)
    except Exception as e:
         print(f"读取失败: {e}")

if __name__ == "__main__":
    # 替换为你新采集的一个原始 HDF5 文件的真实路径
    test_file = "/mnt/pfs/3zpd5q/code/zf/raw_data/render_data/person_follow/floor_1_person_follow.h5" 
    inspect_file(test_file)