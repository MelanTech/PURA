mkdir -p /data/szk/Project/PURA/output

# ☆☆☆☆☆☆☆!!!  设置模型路径  !!!☆☆☆☆☆☆☆
# ☆☆☆☆☆☆☆!!!  查看output文件是否为空  !!!☆☆☆☆☆☆☆
# ☆☆☆☆☆☆☆!!!  设置 显卡    !!!☆☆☆☆☆☆☆

# 选择显卡
export CUDA_VISIBLE_DEVICES=1,2
((GPU_NUM = $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c) + 1))
# lasher_test vtuav_test rgbt234
#rm -r "/data1/szk/Project/PURA/output/"
/home/szk/anaconda3/envs/seqtrack/bin/python /data1/szk/Project/PURA/tracking/test.py tbsi_track vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls \
  --threads=2 --num_gpus="${GPU_NUM}" \
  --dataset_name='rgbt234' |&
  tee /data1/szk/Project/PURA/output/test.log