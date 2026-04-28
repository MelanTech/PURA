# ☆☆☆☆☆☆☆!!!  设置 一阶段 预训练模型    !!!☆☆☆☆☆☆☆
# ☆☆☆☆☆☆☆!!!  设置 显卡               !!!☆☆☆☆☆☆☆
# ☆☆☆☆☆☆☆!!!  设置 模型的保存路径       !!!☆☆☆☆☆☆☆
# 选择显卡
export CUDA_VISIBLE_DEVICES=2
GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
((GPU_NUM = GPU_NUM + 1))
/data/zzf/anaconda3/envs/seqtrack/bin/python -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" --master_port=26956 \
  /data/zzf/proj/TBSI/lib/train/run_training.py \
  --script=tbsi_track \
  --config=vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls \
  --save_dir=/data/zzf/proj/TBSI/train_output_二阶段 2>&1 |
  tee /data/zzf/proj/TBSI/train_output_二阶段.log



export CUDA_VISIBLE_DEVICES=2
GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
((GPU_NUM = GPU_NUM + 1))
# lasher_test vtuav_test rgbt234
/data/zzf/anaconda3/envs/seqtrack/bin/python /data/zzf/proj/TBSI/tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls \
  --threads=5 --num_gpus="${GPU_NUM}" \
  --dataset_name='lasher_test' |&
  tee /data/zzf/proj/TBSI/output/test.log