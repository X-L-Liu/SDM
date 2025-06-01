from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from model_class import *
from attacks import *
import argparse
import time


def train_epoch():
    model.train()
    top1 = AverageMeter()
    with tqdm(total=len(train_loader), desc='Train-Progress', ncols=100) as pbar:
        for k, (image, label) in enumerate(train_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image[::2] = att(image[::2], label[::2])
            if config.use_amp:
                with torch.cuda.amp.autocast():
                    logit = model(image)
                    loss = F.cross_entropy(logit, label)
                loss = scaler.scale(loss)
            else:
                logit = model(image)
                loss = F.cross_entropy(logit, label)
            optimizer.zero_grad()
            loss.backward()
            if config.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.rate


def test_epoch():
    model.eval()
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc='Test-Progress ', ncols=100) as pbar:
        for k, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            image[::2] = att(image[::2], label[::2])
            logit = model(image)
            top1.update((logit.max(1)[1] == label).sum().item(), len(label))
            pbar.update(1)

    return top1.rate


def main(Reload):
    global best_model_file, best_acc
    for epoch in range(config.epochs):
        start = time.time()
        train_acc = train_epoch()
        test_acc = test_epoch()
        scheduler.step()
        if test_acc > best_acc:
            best_model_file = os.path.join(
                config.model_save_path, f'{config.dataset_name}_{config.model_name}_{config.attack}_{test_acc}.pt'
            )
            torch.save(model.state_dict(), best_model_file)
            if os.path.exists(best_model_file.replace(str(test_acc), str(best_acc))):
                os.remove(best_model_file.replace(str(test_acc), str(best_acc)))
            best_acc = test_acc
        print(f'Attack: {config.attack}  Model: {config.model_name}  '
              f'Reload: {Reload + 1}/{config.reload}  Epoch: {epoch + 1}/{config.epochs}  '
              f'Train-Top1: {train_acc * 100:.2f}%  Test-Top1: {test_acc * 100:.2f}%  '
              f'Best-Top1: {best_acc * 100:.2f}%  Time: {time.time() - start:.0f}s')


def load_model():
    classifier = globals()[config.model_name](num_classes)
    if best_model_file != '':
        classifier.load_state_dict(torch.load(best_model_file, map_location=device))
    classifier.to(device)
    classifier.eval()

    return classifier


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--milestones', type=tuple, default=(35, 65, 85))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--reload', type=int, default=3)
    parser.add_argument('--attack', type=str, default='PGD', choices=['PGD', 'SDM'])
    parser.add_argument('--model_name', type=str, default='WideResNet28x10')
    parser.add_argument('--model_save_path', type=str, default='AT')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_amp', type=bool, default=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    config = parse_args()
    device = torch.device(f'cuda:{config.device}')
    assert config.attack in ['PGD', 'SDM']

    if config.dataset_name == 'cifar10':
        num_classes = 10
        trainSet = datasets.CIFAR10(root=config.data_path, train=True, transform=transform_cifar10_train)
        testSet = datasets.CIFAR10(root=config.data_path, train=False, transform=transform_cifar10_test)
    else:
        num_classes = 100
        trainSet = datasets.CIFAR100(root=config.data_path, train=True, transform=transform_cifar100_train)
        testSet = datasets.CIFAR100(root=config.data_path, train=False, transform=transform_cifar100_test)

    train_loader = DataLoader(trainSet, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    best_model_file = ''
    best_acc = 0

    for reload in range(config.reload):
        model = load_model()
        att = torchattacks.PGD(model) if config.attack == 'PGD' else SDM(model, total_steps=10)
        print('>' * 100)
        print(f'Attack: {config.attack}  Model: {config.model_name}  '
              f'Total params: {sum(p.numel() for p in model.parameters()) / 1000000.0:.2f}M')
        optimizer = optim.SGD(model.parameters(), config.lr, config.momentum, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config.milestones)
        scaler = torch.cuda.amp.GradScaler()
        main(reload)
