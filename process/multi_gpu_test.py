from setproctitle import setproctitle
setproctitle("multigpu test")
from tqdm import tqdm

import torch
import torchvision
import horovod.torch as hvd

hvd.init()

torch.cuda.set_device(hvd.local_rank())

train_dataset = torchvision.datasets.CIFAR100(
    # download=True,
    root='data',
    train=True,
    transform=torchvision.transforms.ToTensor())
# breakpoint()
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    sampler=train_sampler)

model = torchvision.models.resnet18(num_classes=100).cuda()
# model.cuda()

optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=1e-4)
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters())
criterion = torch.nn.CrossEntropyLoss()
hvd.broadcast_parameters(
    model.state_dict(),
    root_rank=0)

for epoch in range(20):
    acc = 0
    loss = 0
    model.train()
    for data, target in tqdm(train_loader):

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        logits = model(data)
        step_loss = criterion(logits, target)
        step_loss.backward()
        optimizer.step()

        pred = torch.argmax(logits, axis=1)
        pred = pred.eq(target).sum().item() / data.shape[0]

        loss += step_loss.item()
        acc += pred

    print(f'loss : {loss / len(train_loader)}')
    print(f'acc : {acc / len(train_loader)}')
