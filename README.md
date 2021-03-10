# Wildfire detection with neural networks

![fire](docs/fire.gif)

## Quickstart

1. Download the repository and pretrained weights
```bash
git clone git@github.com:tomek-l/fire-detect-nn.git
cd fire-detect-nn
wget https://dl.dropbox.com/s/6t17srif65vzqfn/firedetect-densenet121-pretrained.pt --directory-prefix=weights/
```

2. Inference on pretrained weights
```
pip3 install -r requirements.txt
python3 inference.py 
```

3. Training & Testing

- [download the dataset](https://drive.google.com/drive/folders/1j0Smp1ALUdZiAt4a3qEFH_85oMc17vsV?usp=sharing).
- unzip the folder to `data/fire-detect-nn-public-combo`

```bash
python3 train.py # training 
python3 train-with-gradcam.py # for the model with gradcam (for the heatmap output)
python3 test.py # evaluation
```
For the GradCAM heatmap generation use jupyter notebook and navigate to `inference-video.ipynb`

<!-- ## Unit Tests
(Well, kind of...)
```bash
cd test
python3 test_models.py
``` -->