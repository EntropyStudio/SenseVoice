# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

workspace=`pwd`

# 设置单GPU
export CUDA_VISIBLE_DEVICES="0"
gpu_num=1  # 直接设置为1，替代原来的计算方式

# 模型设置保持原样
model_name_or_model_dir="iic/SenseVoiceSmall"

# 数据路径保持原样
train_data=${workspace}/data/train_example.jsonl
val_data=${workspace}/data/val_example.jsonl

# 输出目录保持原样
output_dir="./outputs"
log_file="${output_dir}/log.txt"

# 修改为适用于单卡的deepspeed配置（或关闭deepspeed）
deepspeed_config=${workspace}/deepspeed_conf/ds_stage0.json  # 使用stage 0配置
# 或者完全禁用deepspeed:
# use_deepspeed=false

mkdir -p ${output_dir}
echo "log_file: ${log_file}"

# 单卡训练时简化分布式参数
DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --standalone
"

# funasr trainer路径保持原样
train_tool=/kaggle/working/mysitepackages/funasr/bin

# 修改后的执行命令
torchrun $DISTRIBUTED_ARGS \
${train_tool} \
++model="${model_name_or_model_dir}" \
++trust_remote_code=true \
++train_data_set_list="${train_data}" \
++valid_data_set_list="${val_data}" \
++dataset_conf.data_split_num=1 \
++dataset_conf.batch_sampler="BatchSampler" \
++dataset_conf.batch_size=3000  \  # 根据显存适当减小
++dataset_conf.sort_size=512 \    # 适当减小排序缓冲区
++dataset_conf.batch_type="token" \
++dataset_conf.num_workers=4 \
++train_conf.max_epoch=50 \
++train_conf.log_interval=1 \
++train_conf.resume=true \
++train_conf.validate_interval=2000 \
++train_conf.save_checkpoint_interval=2000 \
++train_conf.keep_nbest_models=20 \
++train_conf.avg_nbest_model=10 \
++train_conf.use_deepspeed=true \  # 如果使用deepspeed stage 0则保持true
++train_conf.deepspeed_config=${deepspeed_config} \
++optim_conf.lr=0.0002 \
++output_dir="${output_dir}" &> ${log_file}

# 或者更简单的单卡启动方式（如果支持）：
# python ${train_tool} ... 参数同上
