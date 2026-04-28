import os
import cv2
import numpy as np
from lib_mine.mutils.gpu_name import get_hostname
from tqdm import tqdm


def draw_bbox(seq_name, bbox_txt_type, img_root, mode=None, save_path=None, concat_mode=None):
    # seq_name = '10runone'
    bbox_txt = '/data/zzf/proj/TBSI/' + bbox_txt_type + \
               '/test/tracking_results/tbsi_track/vitb_256_tbsi_32x1_1e4_lasher_15ep_sot_cls/lasher/' \
               + seq_name + '.txt'
    save_path = '/data/zzf/proj/TBSI/draw_bbox/' + seq_name if save_path is None else save_path
    os.makedirs(save_path, exist_ok=True)

    bbox_coord = np.loadtxt(bbox_txt)
    path_rgb = '/' + img_root + '/RGBT_Dataset/LasHeR0428/testingset/' + seq_name + '/visible'
    path_tir = '/' + img_root + '/RGBT_Dataset/LasHeR0428/testingset/' + seq_name + '/infrared'
    path_gt = '/' + img_root + '/RGBT_Dataset/LasHeR0428/testingset/' + seq_name + '/init.txt'
    img_rgb_list = [os.path.join(path_rgb, file) for file in sorted(os.listdir(path_rgb))]
    img_tir_list = [os.path.join(path_tir, file) for file in sorted(os.listdir(path_tir))]
    gt = np.loadtxt(path_gt, delimiter=',')

    for index, cur_bbox in tqdm(enumerate(bbox_coord, start=0), total=len(bbox_coord)):
        img_rgb = cv2.imread(img_rgb_list[index])
        img_tir = cv2.imread(img_tir_list[index])
        cur_bbox[2:] = cur_bbox[:2] + cur_bbox[2:]
        cur_bbox = cur_bbox.astype(int)

        gt[index][2:] = gt[index][:2] + gt[index][2:]
        gt_bbox = gt[index].astype(int)

        color = (0, 0, 255)  # 蓝色
        color_gt = (0, 255, 0)  # 绿色
        line_width = 2

        if mode == 'RGB':
            cv2.rectangle(img_rgb, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), color_gt, line_width)
            cv2.rectangle(img_rgb, (cur_bbox[0], cur_bbox[1]), (cur_bbox[2], cur_bbox[3]), color, line_width)
            cv2.imwrite(save_path + '/{}.png'.format(os.path.basename(img_rgb_list[index])), img_rgb)
        elif mode == 'TIR':
            cv2.rectangle(img_tir, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), color_gt, line_width)
            cv2.rectangle(img_tir, (cur_bbox[0], cur_bbox[1]), (cur_bbox[2], cur_bbox[3]), color, line_width)
            cv2.imwrite(save_path + '/{}.png'.format(os.path.basename(img_tir_list[index])), img_tir)
        else:
            cv2.rectangle(img_rgb, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), color_gt, line_width)
            cv2.rectangle(img_rgb, (cur_bbox[0], cur_bbox[1]), (cur_bbox[2], cur_bbox[3]), color, line_width)
            cv2.rectangle(img_tir, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), color_gt, line_width)
            cv2.rectangle(img_tir, (cur_bbox[0], cur_bbox[1]), (cur_bbox[2], cur_bbox[3]), color, line_width)
            if concat_mode == 'shu':
                img = cv2.vconcat([img_rgb, img_tir])
            else:
                img = cv2.hconcat([img_rgb, img_tir])
            cv2.imwrite(save_path + '/{}'.format(os.path.basename(img_rgb_list[index])), img)


if __name__ == '__main__':
    hostname = get_hostname()
    img_root = 'data2' if hostname in ['ICML', 'ICLR'] else 'data'

    # mode RGB TIR
    draw_bbox(seq_name='11runtwo', bbox_txt_type='output', img_root=img_root, mode='All', concat_mode='shu')
