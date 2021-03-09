from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image

from datasets.combo import transform


def img_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


if __name__ == "__main__":

    device = "cuda:0"
    m = torch.load("weights/firedetect-densenet121-pretrained.pt")
    m = m.to(device)
    m.eval()

    # Load an out-of-sample image from the internets
    img = img_from_url(
        "https://s.abcnews.com/images/US/northern-california-fire-09-gty-jc-181109_hpMain_16x9_992.jpg"
        # "https://cdn.cnn.com/cnnnext/dam/assets/200927234512-02-glass-fire-0927-exlarge-169.jpg"
    )

    # Alternatively, read from a local file
    # path = "data/dunnings_dataset/fire-dataset-dunnings/images-224x224/test/fire/Ogdenhousefire849.png"
    # path = "data/dunnings_dataset/fire-dataset-dunnings/images-224x224/test/nofire/CarInFlames-FireFighterHelmetCam2359.png"
    # img = Image.open(path)

    tensor_in = transform(img).to(device)
    batch_in = torch.unsqueeze(tensor_in, dim=0)
    batch_out = m(batch_in)
    print(f"Fire score: {float(batch_out[0])}")
