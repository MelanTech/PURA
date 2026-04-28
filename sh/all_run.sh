# 选择显卡
export CUDA_VISIBLE_DEVICES=6,7,8,9
((GPU_NUM = $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c) + 1))
/home/szk/anaconda3/envs/seqtrack/bin/python -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" --master_port=23956 \
  /data/szk/Project/TBSI_TU_0509/lib/train/run_training.py \
  --script=tbsi_track \
  --config=vitb_256_tbsi_32x1_1e4_lasher_15ep_sot \
  --save_dir=/data/szk/Project/TBSI_TU_0509/train_output_一阶段 2>&1 |
  tee /data/szk/Project/TBSI_TU_0509/train_output_一阶段.log

export CUDA_VISIBLE_DEVICES=6
((GPU_NUM = $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c) + 1))
/home/szk/anaconda3/envs/seqtrack/bin/python -m torch.distributed.launch --nproc_per_node="${GPU_NUM}" --master_port=26956 \
  /data/szk/Project/TBSI_TU_0509/lib/train/run_training.py \
  --script=tbsi_track \
  --config=vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls \
  --save_dir=/data/szk/Project/TBSI_TU_0509/train_output_二阶段 2>&1 |
  tee /data/szk/Project/TBSI_TU_0509/train_output_二阶段.log

export CUDA_VISIBLE_DEVICES=6,7,8,9
((GPU_NUM = $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c) + 1))
# lasher_test vtuav_test rgbt234
rm -r "/data/szk/Project/TBSI_TU_0509/output/"
/home/szk/anaconda3/envs/seqtrack/bin/python /data/szk/Project/TBSI_TU_0509/tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls \
  --threads=5 --num_gpus="${GPU_NUM}" \
  --dataset_name='lasher_test' |&
  tee /data/szk/Project/TBSI_TU_0509/output/test.log
