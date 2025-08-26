combined_dataset_800.hdf5 是由以下三個hdf5所組成

- - -
ur10_dualtable_dualsense_205、ur10_dualtable_dualsense_220、ur10_dualtable_dualsense_381
- - -

詳見：run.sh
收集dataset請用
```Python
script/collect_human_demonstrations_rewards.py
```
要組合不同hdf5檔案請用
```Python
script/combine_hdf5.py
```
