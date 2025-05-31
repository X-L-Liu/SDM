import argparse
from utils import *
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_class import *
from attacks import *
from torchattacks import PGD, PGDL2, APGD


def eval_model(ATT):
    top1 = AverageMeter()
    with tqdm(total=len(test_loader), desc=f'Evaluate-{att.__class__.__name__}', ncols=100) as pbar:
        for k, (image, label) in enumerate(test_loader):
            image, label = torch.Tensor(image).to(device), torch.Tensor(label).to(device)
            adv_img = ATT(image, label)
            logit = model(adv_img)
            top1.update((logit.max(1)[1] != label).sum().item(), len(label))
            pbar.update(1)

    return top1.err_rate


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default=r'dataset')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--total_steps', type=int, default=20)
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    seed_torch()
    config = parse_args()
    device = torch.device(f'cuda:{config.device}')

    if config.dataset_name == 'cifar10':
        num_classes = 10
        testSet = datasets.CIFAR10(root=config.data_path, train=False, download=True, transform=transform_cifar10_test)
    else:
        num_classes = 100
        testSet = datasets.CIFAR100(root=config.data_path, train=False, download=True, transform=transform_cifar100_test)

    test_loader = DataLoader(testSet, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    model = globals()[config.model_name](num_classes)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)
    model.eval()

    atts = [
        PGD(model, steps=config.total_steps, random_start=False),
        CW(model, steps=config.total_steps, random_start=False),
        APGD(model, steps=config.total_steps, n_restarts=5),
        APGD(model, loss="dlr", steps=config.total_steps, n_restarts=5),
        SDM(model, total_steps=config.total_steps, random_start=False),
        PGDL2(model, steps=config.total_steps, random_start=False),
        CWL2(model, steps=config.total_steps),
        APGD(model, norm="L2", eps=1.0, steps=config.total_steps, n_restarts=5),
        APGD(model, norm="L2", eps=1.0, loss="dlr", steps=config.total_steps, n_restarts=5),
        SDML2(model, total_steps=config.total_steps, random_start=False),
    ]

    for att in atts:
        Err = eval_model(att)
        print(f"{att.__class__.__name__}: {Err * 100:.2f}%\n")
