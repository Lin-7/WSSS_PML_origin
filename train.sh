#!/bin/bash

echo "Current working directory: $(pwd)"
# List of Python scripts to run in serial order along with their arguments
python_scripts_with_args=(
    "train_cls_loc_jointly_merge_infer.py --session_name=patchcls-mergeTV-epoch4-bs32-patchlossweight0.2"
    "train_cls_loc_jointly_merge_infer.py --session_name=base-mergeTV-epoch4-bs32-patchlossweight0.2"
)

# Loop through the scripts and run them one by one with arguments
for script_with_args in "${python_scripts_with_args[@]}"; do
    script="${script_with_args%% *}"     # Extract the script name
    args="${script_with_args#* }"        # Extract the arguments after the space

    echo "Running $script with arguments: $args"
    nohup python "$script" $args &
    wait
done

# # 停40min保证上面代码eval完成
# sleep 2400s

# # 切换到另一个目录
# cd "../wsss_pml"

# # 在脚本所在目录下执行命令
# echo "Current working directory: $(pwd)"

# echo "nohup python train_cls_loc_jointly_new.py --session_name=patchcls-fgmidselect0.3-patchweight0.2 --patch_select_cri=fgratio --patch_select_part_fg=mid &"
# nohup python train_cls_loc_jointly_new.py --session_name=patchcls-fgmidselect0.3-patchweight0.2 --patch_select_cri=fgratio --patch_select_part_fg=mid &


echo "All scripts have been executed."