mkdir -p /data/zzf/proj/TBSI/output

# ☆☆☆☆☆☆☆!!!  设置模型路径  !!!☆☆☆☆☆☆☆
GPU_NUM=2
# lasher_test vtuav_test rgbt234
/data/zzf/anaconda3/envs/seqtrack/bin/python /data/zzf/proj/TBSI/tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot \
  --threads=5 --num_gpus=${GPU_NUM} \
  --dataset_name='vtuav_test' |&
  tee /data/zzf/proj/TBSI/output/test.log
