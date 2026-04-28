import os

import cv2
import numpy as np

dataset_path = '/data/RGBT_Dataset/RGBT234'
result_base = './results'
save_base = './vis_result'

seq_name = 'man7'

trackers = [
    ('ViPT', 'ViPT'),
    ('TBSI', 'TBSI'),
    ('MPLT', 'MPLT'),
    ('Baseline', 'Source'),
    ('PURA', 'PURA'),
]

box_color = {
    'GT': (0, 255, 0),  # Green
    'PURA': (255, 0, 255),  # Red
    'TBSI': (255, 0, 0),  # Blue
    'ViPT': (0, 255, 255),  # Yellow
    'MPLT': (255, 255, 0),  #
    'Baseline': (0, 0, 255),  #
}

if __name__ == '__main__':
    v_base = os.path.join(dataset_path, seq_name, 'visible')
    i_base = os.path.join(dataset_path, seq_name, 'infrared')

    # Read GT Box
    boxes = []

    boxes.append(
        ('GT', np.loadtxt(os.path.join(dataset_path, seq_name, 'visible.txt'), delimiter=',').astype(np.int32)))

    # Read Tracker Result
    for name, tracker_path in trackers:
        try:
            boxes.append(
                (name,
                 np.loadtxt(os.path.join(result_base, tracker_path, seq_name + '.txt'), delimiter='\t').astype(
                     np.int32))
            )
        except:
            try:
                boxes.append(
                    (name,
                     np.loadtxt(os.path.join(result_base, tracker_path, seq_name + '.txt'), delimiter=',').astype(
                         np.int32))
                )
            except:
                boxes.append(
                    (name,
                     np.loadtxt(os.path.join(result_base, tracker_path, seq_name + '.txt'), delimiter=' ').astype(
                         np.int32))
                )

    v_imgs = os.listdir(v_base)
    i_imgs = os.listdir(i_base)

    v_save_path = os.path.join(save_base, seq_name, 'visible')
    i_save_path = os.path.join(save_base, seq_name, 'infrared')

    if not os.path.exists(v_save_path):
        os.makedirs(v_save_path)
    if not os.path.exists(i_save_path):
        os.makedirs(i_save_path)

    v_imgs.sort()
    i_imgs.sort()

    idx = 0
    for v_img_name, i_img_name in zip(v_imgs, i_imgs):
        print(idx + 1, '/', len(v_imgs))
        v_img = cv2.imread(os.path.join(v_base, v_img_name))
        i_img = cv2.imread(os.path.join(i_base, i_img_name))

        for name, box in boxes:
            bbox = box[idx]
            x1, y1, w, h = list(map(int, bbox))
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(v_img, (x1, y1), (x2, y2), color=box_color[name], thickness=2)
            cv2.rectangle(i_img, (x1, y1), (x2, y2), color=box_color[name], thickness=2)

        cv2.imwrite(os.path.join(v_save_path, v_img_name), v_img)
        cv2.imwrite(os.path.join(i_save_path, i_img_name), i_img)

        idx += 1
