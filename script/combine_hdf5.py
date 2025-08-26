import h5py
import numpy as np

def combine_robosuite_datasets_with_attrs(file_paths, output_file_path):
    """
    將多個 Robosuite HDF5 資料集合併成一個，並複製所有屬性。

    Args:
        file_paths (list): 包含所有要合併的 .h5 檔案路徑。
        output_file_path (str): 合併後的新 .h5 檔案路徑。
    """
    with h5py.File(output_file_path, 'w') as f_out:
        total_demos = 0
        
        for i, file_path in enumerate(file_paths):
            print(f"處理檔案：{file_path}")
            with h5py.File(file_path, 'r') as f_in:
                if 'data' in f_in:
                    input_data_group = f_in['data']
                    
                    # === 步驟 1: 在第一次處理時，複製根群組的屬性 ===
                    if i == 0:
                        for attr_name, attr_value in f_in.attrs.items():
                            f_out.attrs[attr_name] = attr_value
                        # 創建根群組，並複製其屬性
                        if 'data' in f_in:
                            f_out.create_group('data')
                            for attr_name, attr_value in f_in['data'].attrs.items():
                                f_out['data'].attrs[attr_name] = attr_value
                                
                    # === 步驟 2: 遍歷每個示範並複製資料和屬性 ===
                    for demo_key, demo_group in input_data_group.items():
                        if demo_key.startswith('demo'):
                            new_demo_key = f"demo_{total_demos}"
                            new_demo_group = f_out['data'].create_group(new_demo_key)
                            
                            # 複製 demo 群組上的屬性
                            for attr_name, attr_value in demo_group.attrs.items():
                                new_demo_group.attrs[attr_name] = attr_value
                            
                            # 複製每個 dataset (例如：states, actions...)
                            for dataset_name, dataset in demo_group.items():
                                if isinstance(dataset, h5py.Dataset):
                                    new_dataset = new_demo_group.create_dataset(
                                        name=dataset_name,
                                        data=dataset[()],
                                        dtype=dataset.dtype
                                    )
                                    # 複製 dataset 上的屬性
                                    for attr_name, attr_value in dataset.attrs.items():
                                        new_dataset.attrs[attr_name] = attr_value
                                        
                            total_demos += 1
        
        # (可選) 在合併結束後，更新 total 屬性
        f_out.attrs['total'] = total_demos
        
        print(f"成功合併 {total_demos} 個示範到 {output_file_path}")

# 範例使用：
file_paths_to_combine = ['/home/xul6106/ur10_demonstrations/combined_dataset.hdf5', 
                         '/home/xul6106/ur10_demonstrations/1755754821_4817817/demo.hdf5']
output_combined_file = '/home/xul6106/ur10_demonstrations/combined_dataset_800.hdf5'
combine_robosuite_datasets_with_attrs(file_paths_to_combine, output_combined_file)

