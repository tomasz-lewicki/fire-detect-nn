import torch
import torchvision

from PIL import Image
import requests
from io import BytesIO

url = "https://cdn.cnn.com/cnnnext/dam/assets/200927234512-02-glass-fire-0927-exlarge-169.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

device = "cuda:0"
m = torch.load('../weights/resnet50-epoch-8-valid_acc=0.97-test_acc=-1.00.pt')
m = m.to(device)
m.eval()

tr = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
)

tensor_in = tr(img).to(device)
batch_in = torch.unsqueeze(tensor_in, dim=0)
batch_out = m(batch_in)
print(f"Fire score: {float(batch_out[0])}")
