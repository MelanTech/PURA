import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class VTUAVDataset(BaseDataset):
    def __init__(self, split='ST_test'):
        super().__init__()
        self.base_path = self.env_settings.vtuav_path + '/test'

        self.sequence_list = self._get_sequence_list(split=None)

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/rgb.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=' ', dtype=np.float64)

        # anno_path_tir = '{}/{}/ir.txt'.format(self.base_path, sequence_name)
        # ground_truth_rect_tir = load_text(str(anno_path_tir), delimiter=' ', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frames_list_v = sorted([os.path.join(frames_path + '/rgb', frame) for frame in os.listdir(frames_path + '/rgb') if frame.endswith(".jpg")])
        frames_list_i = sorted([os.path.join(frames_path + '/ir', frame) for frame in os.listdir(frames_path + '/ir') if frame.endswith(".jpg")])
        frames_list = [frames_list_v, frames_list_i]
        # frame_list = {
        #     'rgb': sorted([os.path.join(frames_path + '/rgb', frame) for frame in os.listdir(frames_path + '/rgb') if frame.endswith(".jpg")]),
        #     'tir': sorted([os.path.join(frames_path + '/ir', frame) for frame in os.listdir(frames_path + '/ir') if frame.endswith(".jpg")])}

        return Sequence(sequence_name, frames_list, 'vtuav', ground_truth_rect.reshape(-1, 4), )

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split=None):
        sequence_list = []
        if split in ['ST_test', 'LT_test']:
            f = open(self.base_path + "/../{}.txt".format(split), 'r')
            for line in f:
                sequence_list.append(line.strip())
        else:
            sequence_list = sorted(os.listdir(self.base_path))
        return sequence_list
