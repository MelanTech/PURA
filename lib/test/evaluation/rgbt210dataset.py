import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os


class RGBT210DATASET(BaseDataset):
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.rgbt210_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        anno_path_tir = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        ground_truth_rect_tir = load_text(str(anno_path_tir), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frames_list_v = sorted([os.path.join(frames_path + '/visible', frame)
                                for frame in os.listdir(frames_path + '/visible') if frame.endswith(".jpg")])
        frames_list_i = sorted([os.path.join(frames_path + '/infrared', frame)
                                for frame in os.listdir(frames_path + '/infrared') if frame.endswith(".jpg")])
        frames_list = [frames_list_v, frames_list_i]
        # frame_list = {
        #     'rgb': sorted([os.path.join(frames_path + '/rgb', frame) for frame in os.listdir(frames_path + '/rgb') if frame.endswith(".jpg")]),
        #     'tir': sorted([os.path.join(frames_path + '/ir', frame) for frame in os.listdir(frames_path + '/ir') if frame.endswith(".jpg")])}

        return Sequence(sequence_name, frames_list, 'rgbt210', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split=None):
        # sequence_list = sorted(os.listdir(self.base_path))
        file = os.path.join(self.base_path, 'list_210.txt')
        text = open(file, "r", encoding="utf-8").readlines()
        sequence_list = [line.strip("\n") for line in text]

        return sequence_list
