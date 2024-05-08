# -*- coding: utf-8 -*-

import pandas as pd
import glob, numpy, os, random, soundfile, torch
from scipy import signal

class train_loader(object):
    def __init__(self, train_list, num_frames, **kwargs):
        self.num_frames = num_frames
        # Load data & labels
        df_train = pd.read_csv(train_list)
        self.data_list = df_train['wav_path'].tolist()
        self.data_label = df_train['label'].tolist()

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])
        length = self.num_frames * 80
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio],axis=0)
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)
