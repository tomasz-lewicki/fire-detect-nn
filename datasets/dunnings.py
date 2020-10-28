import torch
import torchvision

tr = torchvision.transforms.Compose(
    [torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()]
)


def make_dunnings_train_loader(path, batch_size=16):
    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=tr
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, shuffle=True
    )

    return loader

def make_dunnings_test_loader(path, batch_size=16):
    dataset = torchvision.datasets.ImageFolder(
        root=path, transform=tr
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4, shuffle=True
    )

    return loader