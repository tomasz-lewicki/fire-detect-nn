# Wildfire detection with neural networks

![fire](docs/fire.gif)


# Training 
```bash
python3 train.py
python3 train-with-gradcam.py # for final model with gradcam
```

# Evaluation
```bash
python3 inference.py
```

# Inference

Basic inference:
```bash
python3 inference.py
```

Inference on a video:
```bash
jupyter notebook
```
navigate to inference-video.ipynb

# Unit Tests
```bash
cd test
python3 test_models.py
```


# FAQ

If training on a custom dataset, make sure to set appropriate class to label matching.
For example, if dataset folders are called: `fire` and `nofire` do:s
```
dataset.class_to_idx = {'fire': 1, 'nofire': 0} # class mapping
```
Otherwise pytorch will match classes to labels alphabetically and the result may be reversed!