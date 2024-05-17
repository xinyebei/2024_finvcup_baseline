# -*- coding: utf-8 -*-

import argparse, glob, os, warnings, time
from utils.tools import *
from utils.FakeModel import FakeModel
from utils.CNN import CNN
import soundfile
import torch
import math
import numpy as np
import tqdm
import pandas as pd
import argparse

class Inferencer(object):
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = CNN(in_features=1)
        self_state = model.state_dict()
        loaded_state = torch.load(model_path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("speaker_encoder.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

        model.load_state_dict(self_state)
        model.eval().cuda()
        return model

    def eval_embedding(self, speech_path, num_frames=200, max_audio=48000, channel='left'):
        #处理音频
        audio, _  = soundfile.read(speech_path)

        max_audio = num_frames*80
        if audio.shape[0] <= max_audio:
            shortage = max_audio - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        feats = []
        startframe = np.linspace(0, audio.shape[0]-max_audio, num=5)    
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

        feats = np.stack(feats, axis = 0).astype(np.float)
        data = torch.FloatTensor(feats).cuda()
        
        #推断
        with torch.no_grad():
            outputs = self.model.forward(data)
            outputs = torch.mean(outputs, dim=0).view(1, -1)

        return outputs.detach().cpu().numpy().argmax(axis=1)[0]
    
def main(speech_list, infer, res_path):
    result = []
    for idx, file in tqdm.tqdm(enumerate(speech_list), total = len(speech_list)):
        output = infer.eval_embedding(file)
        result.append([os.path.basename(file), output])
    df_result = pd.DataFrame(result, columns=["speech_name", "pred_label"])
    df_result.to_csv(res_path, index=False, header=None)
    return df_result
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "DeepFake audio")
    parser.add_argument('--model_path',      type=str,   default='exps/model/model_0001.model',       help='Model checkpoint path')
    parser.add_argument('--test_path',      type=str,   default='data/finvcup9th_1st_ds5/finvcup9th_1st_ds5_test_data.csv',       help='Path of test file, strictly same with the original file')
    parser.add_argument('--save_path', type=str, default='./submit/submit.csv', help='Path of result')
    args = parser.parse_args()
    print('loading model...')
    infer = Inferencer(args.model_path)
    df_test = pd.read_csv(args.test_path)
   
    print('model inferring...')
    main(df_test["wav_path"].tolist(), infer, res_path=args.save_path)
    print('done!')
    
        
    
    
    
    
    
    
    
