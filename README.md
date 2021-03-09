# Wildfire detection with neural networks

![fire](docs/fire.gif)

## Quickstart
```bash
git clone git@github.com:tomek-l/fire-detect-nn.git
wget https://dl.dropbox.com/s/6t17srif65vzqfn/firedetect-densenet121-pretrained.pt --directory-prefix=weights/
pip3 install -r requirements.txt
python3 inference.py # inference
```

```bash
python3 train.py # training 
python3 train-with-gradcam.py # for final model with gradcam
python3 inference.py # evaluation
# For the GradCAM heatmap generation use jupyter notebook and navigate to inference-video.ipynb
```

## Unit Tests
(Well, kind of...)
```bash
cd test
python3 test_models.py
```