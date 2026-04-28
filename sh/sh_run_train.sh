# 选择显卡
export CUDA_VISIBLE_DEVICES=4,5,6,7

GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
((GPU_NUM = GPU_NUM + 1))
/data/zzf/anaconda3/envs/seqtrack/bin/python -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" --master_port=23956 \
  /data/zzf/proj/TBSI/lib/train/run_training.py \
  --script=tbsi_track \
  --config=vitb_256_tbsi_32x1_1e4_lasher_15ep_sot \
  --save_dir=/data/zzf/proj/TBSI/train_output_一阶段 2>&1 |
  tee /data/zzf/proj/TBSI/train_output_一阶段.log
