import time
import cv2
import torch

from model import get_number_net
from datasets import get_dataset
import torch.optim as optim
import torch.nn as nn

# 确定设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# 获取数据和初始化模型，优化器，损失函数
train_dataset, test_dataset = get_dataset()
model = get_number_net().to(device=device)
train_dataset = train_dataset
test_dataset = test_dataset
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epoch = 1

start = time.time()
for epoch in range(num_epoch):
    running_loss = 0.0
    for idx, data in enumerate(train_dataset):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = torch.tensor([labels]).to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if idx % 1000 == 0:
            print('epoch:%d idx :%d loss:%f' % (epoch, idx, running_loss / 500))
            running_loss = 0.0

print("epoch end")
end = time.time()
print("time cost: %d " % (end - start))
correct = 0
total = 0

with torch.no_grad():
    # 展示图片
    data = next(iter(test_dataset))
    show = data[0].squeeze(0)
    cv2.imshow('hah', show.numpy())

    out = model(data[0].to(device))
    print(out, torch.argmax(out))

    for idx, data in enumerate(test_dataset):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = torch.tensor([labels]).to(device)
        out = model(inputs)
        result = torch.argmax(out, dim=0)
        total += labels.size(0)
        correct += (result == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))
cv2.waitKey()
