import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
import re

class UAV135_darkDataset(BaseDataset):
    """ UAV135 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav135_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = []

        seq_root = os.path.join(self.base_path, "Sequences")
        anno_root = os.path.join(self.base_path, "anno_revise")

        object_list = sorted(os.listdir(seq_root))
        for object_file in object_list:
            img_path = os.path.join(seq_root, object_file)
            if not os.path.isdir(img_path):
                continue

            anno_path = os.path.join(anno_root, object_file + ".txt")
            if not os.path.isfile(anno_path):
                continue

            img_files = []
            for fn in os.listdir(img_path):
                base, ext = os.path.splitext(fn)
                ext = ext.lower()
                if ext in [".jpg", ".jpeg", ".png"] and base.isdigit():
                    img_files.append(fn)

            cnt = len(img_files)
            if cnt == 0:
                continue

            sample = sorted(img_files)[0]
            sample_base = os.path.splitext(sample)[0]
            nz = len(sample_base)

            ext = os.path.splitext(sample)[1].lstrip(".").lower()

            object_class = re.sub(r"\d+$", "", object_file)  # basketballplayer1 -> basketballplayer
            if object_class == "":
                object_class = "unknown"

            sequence = {
                "name": object_file,
                "path": os.path.join("Sequences", object_file),   # 注意这里不加 /img
                "startFrame": 1,
                "endFrame": cnt,                                  # 假设从1开始连续
                "nz": nz,
                "ext": ext,
                "anno_path": os.path.join("anno_revise", object_file + ".txt"),
                "object_class": object_class
            }
            sequence_info_list.append(sequence)

        return sequence_info_list