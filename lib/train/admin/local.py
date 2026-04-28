import socket


class EnvironmentSettings:
    def __init__(self):
        hostname = socket.gethostname()
        self.workspace_dir = '/data/szk/Project/PURA/'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/szk/Project/PURA/train_output/tensorboard'  # Directory for tensorboard files.
        self.pretrained_networks = '/data/szk/Project/PURA/pretrained_networks'

        self.vtuav_train_dir = '/data/RGBT_Dataset/VTUAV/train'
        self.vtuav_test_dir = '/data/RGBT_Dataset/VTUAV/test'
        self.lasher_train_dir = '/data/RGBT_Dataset/LasHeR0428/trainingset'
        self.lasher_test_dir = '/data/RGBT_Dataset/LasHeR0428/testingset'
        self.rgbt234_dir = '/data/RGBT_Dataset/RGBT234'
        self.rgbt210_dir = '/home/szk/Dataset/RGBT210'
        if hostname in ['ICLR', 'ICML']:
            self.vtuav_train_dir = '/data2/RGBT_Dataset/VTUAV/train'
            self.vtuav_test_dir = '/data2/RGBT_Dataset/VTUAV/test'
            self.lasher_train_dir = '/data2/RGBT_Dataset/LasHeR0428/trainingset'
            self.lasher_test_dir = '/data2/RGBT_Dataset/LasHeR0428/testingset'

        self.lasot_dir = '/data/szk/Project/PURA/data/lasot'
        self.got10k_dir = '/data/szk/Project/PURA/data/got10k/train'
        self.got10k_val_dir = '/data/szk/Project/PURA/data/got10k/val'
        self.lasot_lmdb_dir = '/data/szk/Project/PURA/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data/szk/Project/PURA/data/got10k_lmdb'
        self.trackingnet_dir = '/data/szk/Project/PURA/data/trackingnet'
        self.trackingnet_lmdb_dir = '/data/szk/Project/PURA/data/trackingnet_lmdb'
        self.coco_dir = '/data/szk/Project/PURA/data/coco'
        self.coco_lmdb_dir = '/data/szk/Project/PURA/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data/szk/Project/PURA/data/vid'
        self.imagenet_lmdb_dir = '/data/szk/Project/PURA/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
