# -*- coding: utf-8 -*-

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from utils.tools import *
from utils.CNN import CNN
import pandas as pd


class FakeModel(nn.Module):
    def __init__(self, lr, lr_decay, n_class, device, test_step, **kwargs):
        super(FakeModel, self).__init__()
        ## ResNet
        self.device = device
        self.speaker_encoder = CNN(in_features=1, nclass=n_class).to(self.device)
        
        ## Classifier
        self.speaker_loss = nn.CrossEntropyLoss()
        
        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

    def train_network(self, epoch, loader):
        self.train()
        ## Update the learning rate based on the current epcoh
        self.scheduler.step(epoch - 1)
        index, acc, loss, recall, F1 = 0, 0, 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start = 1):
            self.zero_grad()
            labels            = torch.LongTensor(labels).to(self.device)
            outputs = self.speaker_encoder.forward(data.to(self.device))
            nloss = self.speaker_loss(outputs, labels)
            acc_t, recall_t, prec_t, F1_t = metrics_scores(outputs, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            acc += acc_t
            recall += recall_t
            F1 += F1_t
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
            " Loss: %.5f, Acc: %2.2f%%, Recall: %2.2f%%, F1: %2.2f%%\r" %(loss/(num), acc/index*len(labels), recall/index*len(labels), F1/index*len(labels)*100))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss/num, lr, acc/index*len(labels), recall/index*len(labels), F1/index*len(labels)*100 

    def eval_network(self, eval_list, num_frames):
        self.eval()
        files = []
        outputs = torch.tensor([]).to(self.device)
        df_test = pd.read_csv(eval_list)
        label_list = df_test["label"].tolist()
        setfiles = df_test["wav_path"].tolist()
        loss, top1 = 0, 0
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio, _ = soundfile.read(file)

            # Spliited utterance matrix
            max_audio = num_frames * 80
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(numpy.float)
            data_2 = torch.FloatTensor(feats).to(self.device)
            # Speaker embeddings
            with torch.no_grad():
                output = self.speaker_encoder.forward(data_2)
                output = torch.mean(output, dim=0).view(1, -1)
            outputs = torch.cat((outputs, output), 0)
        acc, recall, prec, F1 = metrics_scores(outputs, torch.tensor(label_list).to(self.device))
                
        return acc, recall, F1*100
    
    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)