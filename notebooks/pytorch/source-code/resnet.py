from torchvision import models
from torchvision import transforms
from PIL import Image
import torch

resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

img = Image.open('../data/bobby.jpg')
# img.show()
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)
percentage = torch.softmax(out, dim=1)[0] * 100

with open('../data/imagenet_classes.txt') as file:
    labels = [line.strip() for line in file.readlines()]

# _, index = torch.max(out, 1)
# print(labels[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])