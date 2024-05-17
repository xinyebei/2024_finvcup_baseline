# 第九届信也科技杯baseline


这是第九届信也科技杯baseline。 
本届全球算法大赛聚焦语音深度鉴伪，挑战前沿AI技术，以推动国际合作，引导科技向善。


## Environments
Implementing environment:  
- python=3.6
- torch==1.9.0
- torchaudio==0.9.0
- pydub==0.21.0
- numba==0.48.0
- numpy==1.15.4
- pandas==0.23.3
- scipy==1.2.1
- scikit-learn==0.19.1
- tqdm
- SoundFile==0.12.1
- librosa==0.6.2

- GPU: Tesla V100 32G  



## Dataset

新建./submit目录,请将下载好的数据文件 finvcup9th_1st_ds5.tar.gz 解压到./data目录下，执行
```bash
python data_prepare.py
```


## Training

```bash
python train.py --max_epoch 80 --device cuda:0 --save_path ./exps/
```


## Inference

新建./submit目录，执行
```bash
python inference.py --model_path exps/model/model_0001.model --save_path ./submit/submit.csv
```
会在根目录下生成提交所需的submit.csv文件。
注：csv文件共两列(文件名，预测标签)以逗号分隔，格式参照./submit/submit_sample.csv。
    不符合格式提交结果可能会出错。

