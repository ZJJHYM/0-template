# Code for Algo 2021 Track 2 (Multi Tagging)  

## Setup

> All the setup has only been tested on Tencent Ti-one platform.  

Open `terminal`.

```powershell
conda create -n algo python=3.8
conda activate algo
pip install -r requirements.txt

# install PyTorch 1.8 + cu101, for the CUDA version on the platform is 10.1
wget https://download.pytorch.org/whl/cu101/torch-1.8.1%2Bcu101-cp38-cp38-linux_x86_64.whl
pip install ./torch-1.8.1%2Bcu101-cp38-cp38-linux_x86_64.whl
```

## Run Model  

All code for training and testing is in `main.py`, use different flag to control.  

Flag list:

`--train`: start training model  
`--test`: run testing script  
`--report`: run validation epoch and save results as report  
`--model`: choose the model (now only `ctx_gate`, no need to specify)  
`--restore_path`: restore model from checkpoint, must be set when testing  

---

`config` folder contains all configurations for model and training.  

`trainer` configuration in `base.yaml` is for [PyTorch Lightning](https://www.pytorchlightning.ai/) Trainer.

---

Notice: Some pretrained model need to be downloaded during traing, however, there might be error due to network issue. You need to download them manually and store them in the correct dir.  

### Train

```powershell
python main.py --train
```

### Test  

```powershell
# use best checkpoint (default)
python main.py --test
# specify checkpoint
python main.py --test --restore_path %YOUR_CKPT_PATH%
```

This will end up with `tagging_5k_test_%DATE%_ckpt_%ITER_CNT%_with_model.json`.  

## Details  

### Models  

### Training Pipeline  

The whole training pipeline is based on [PyTorch Lightning](https://www.pytorchlightning.ai/).  
