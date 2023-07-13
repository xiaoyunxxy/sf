
# numpy package
import numpy as np

# torch package
import torch
import torchvision
from torchvision import transforms
from torchvision import utils as vutils
from torch.nn.functional import cross_entropy, softmax
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize

# basic package
import os
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# grad_cam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# custom package
from loader.loader import dataset_loader, network_loader, attack_loader
from subclass import create_sublass

# torchcam
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask



# cudnn enable
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Adversarial Test')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--data_root', default='../data', type=str, help='path to dataset')
parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--model_dir', default='/data/xuxx/experiment_MI/Plain_resnet18_cifar10_09120918.pth', type=str, help='save directory')
parser.add_argument('--eps', default=0.03, type=float, help='max norm')
parser.add_argument('--network', default='resnet18', type=str, help='network name')
parser.add_argument('--steps', default=10, type=int, help='adv. steps')
parser.add_argument('--attack', default='pgd', type=str, help='attack type')
parser.add_argument('--pretrained', default='false', type=str2bool, help='pretrained boolean')
parser.add_argument('--batchnorm', default='true', type=str2bool, help='batchnorm boolean')
parser.add_argument('--fc', default=False, type=str2bool, help='feature channels')
parser.add_argument('--cwsteps', default=200, type=int, help='adv. steps')
parser.add_argument('--attack_mi', default=False, type=str2bool, help='using MI in PGD')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id


# loader trained network
args.mean=0.5
args.std=0.25
trainloader, testloader = dataset_loader(args)
net = network_loader(args, mean=args.mean, std=args.std).cuda()
net.load_state_dict(torch.load(args.model_dir)['model_state_dict'])

# attack
attack = attack_loader(args, net)

    
classes = ('plane', 'car', 'bird', 'cat',
       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def test():
    correct = 0
    total = 0
    conf_sum = 0.0
    soft_conf_sum = 0.0

    net.eval()
    print('\n\n[Plain/Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):

        # dataloader parsing and generate adversarial examples
        inputs, targets = inputs.cuda(), targets.cuda()

        # Evaluation
        outputs = net(inputs).detach()

        # Test
        conf_values, predicted = torch.max(outputs, dim=1)
        conf_sum += conf_values.sum().item()
        total += targets.numel()
        correct += (predicted == targets).sum().item() 

        soft_out = softmax(outputs, dim=1)
        conf_values, predicted = torch.max(soft_out, dim=1)
        soft_conf_sum += conf_values.sum().item()
        
    print('[Plain/Test] Acc: {:.3f}'.format(100.*correct / total))
    print('Confidence value: ', conf_sum / total)
    print('Soft max Confidence value: ', soft_conf_sum / total)
    return 100.*correct / total


def adv_test():
    correct = 0.0
    total = 0
    conf_sum = 0.0
    soft_conf_sum = 0.0
    # validation loop

    net.eval()
    print('\n\n[Plain/Adv_Test] Under Testing ... Wait PLZ')
    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
        inputs, targets = inputs.cuda(), targets.cuda()

        adv_input = attack(inputs, targets)
        pred = net(adv_input)

        conf_values, predicted = torch.max(pred, dim=1)
        conf_sum += conf_values.sum().item()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        soft_out = softmax(pred, dim=1)
        conf_values, predicted = torch.max(soft_out, dim=1)
        soft_conf_sum += conf_values.sum().item()


    print('[Adv/Test] Acc: {:.3f}'.format(100.*correct / total))
    print('Confidence value: ', conf_sum / total)
    print('Soft max Confidence value: ', soft_conf_sum / total)
    return 100.*correct / total

def subclass_data(sub_cla=0, ori_dataset=torchvision.datasets.MNIST, num_classes=10, train=True):
    exclude_list = list(range(num_classes))
    exclude_list.remove(sub_cla)

    transform_train = Compose([
        ToTensor()
    ])

    subloader = create_sublass(ori_dataset)
    subtest = subloader(args.data_root, train=train, transform=transform_train, download=True, exclude_list=exclude_list)
    class_data = torch.utils.data.DataLoader(subtest, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    return class_data


def grad_cam_test():
    model = net.eval().cuda()
    # Get your input
    # img = read_image("path/to/your/image.png")
    # # Preprocess it for your chosen model
    # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    cls_id = 4
    subl = subclass_data(sub_cla=cls_id, train=False, ori_dataset=torchvision.datasets.CIFAR10)


    
    act_idx = 4
    count = 1
    tar_dir = './at_class'+str(cls_id)
    for batch_idx, (inputs, targets) in enumerate(tqdm(subl)):
        if batch_idx == count:
            break

        input_tensor = inputs.cuda()

        cam_extractor = SmoothGradCAMpp(model, target_layer=model.layer4)

        adv_input = attack(inputs, targets).cuda()

        # Preprocess your data and feed it to the model
        out = model(input_tensor)
        conf_values, predicted = torch.max(out, dim=1)

        for i in range(5):

            y_c = torch.ones_like(out)
            y_c[act_idx] *= 2

            # print('y_c: ', y_c)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(list(predicted), out)

            # break

            result = overlay_mask(to_pil_image(input_tensor[act_idx]), to_pil_image(activation_map[0][act_idx].squeeze(0), mode='F'), alpha=0.5)

            plt.imshow(result); plt.axis('off'); plt.tight_layout();
            # plt.savefig(tar_dir+'/test_'+str(cls_id)+'_'+str(batch_idx)+'.jpg')
            plt.savefig('./'+str(i)+'_test_'+str(cls_id)+'_'+str(batch_idx)+'.jpg', bbox_inches='tight', pad_inches = -0.1)

            frame= plt.gca()
            frame.axes.get_yaxis().set_visible(False)
            frame.axes.get_xaxis().set_visible(False)

            imgplot = plt.imshow(input_tensor[act_idx].permute(1,2,0).cpu())
            # plt.savefig(tar_dir+'/test_ori_'+str(cls_id)+'_'+str(batch_idx)+'.jpg')
            plt.savefig('./test_ori_'+str(cls_id)+'_'+str(batch_idx)+'.jpg', bbox_inches='tight', pad_inches = -0.1)


if __name__ == '__main__':
    grad_cam_test()


    
