import h5py
import numpy as np

with h5py.File("/home/xul6106/ur10_demonstrations/1755163423_1213498/low_dim.hdf5", "r") as f:
    # 列出所有群組和資料集
    def print_h5_items(name, obj):
        print(name, obj)
    f.visititems(print_h5_items)

    # 假設 rewards 和 dones 在 'data' 群組下的某個示範中
    # 請根據您檔案的實際結構修改路徑
    demo_key = list(f['data'].keys())[0] # 以第一個示範為例
    
    rewards = f[f'data/{demo_key}/rewards'][:]
    dones = f[f'data/{demo_key}/dones'][:]

    print("Rewards (first 100 values):", rewards[:100])
    print("Dones (first 100 values):", dones[:100])
    print("Rewards shape:", rewards.shape)
    print("Dones shape:", dones.shape)