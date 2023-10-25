from matplotlib import pyplot as plt
import torch, cv2
from torch import nn
from data.data import ForceData
from torchvision.models.segmentation import fcn_resnet50
import torchvision.transforms as T


def to_pil(force_map):
    toPIL = T.ToPILImage()
    force_map = force_map.squeeze(0)
    return toPIL(force_map * 255)

if __name__ == "__main__":
    stats_dir = "stats"
    data_path = "src/data2"
    checkout_path = "/home/zjw/learn/cv_self/force_map/backbones/cifar10_swin_t_deformable_best_model_backbone.pt"
    batch_size = 4
    epoch = 100
    lr = 0.01
    lr_min = 0.00001
    num_workers = 8
    decoder_params = {"depth": 4, "hidden_dim": 96, "norm_type": dict(type="BN"), "act_type": dict(type="GELU")}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ForceData(data_path)

    rgb, inf, _, mask = dataset[80]

    checkpoint = torch.load("stats/10_04/best.pth", map_location="cpu")
    model = fcn_resnet50(num_classes=1)
    model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    rgb = rgb.to(device)
    inf = inf.to(device)
    x = torch.cat((rgb, inf), dim=0)
    out = model(x.unsqueeze(0))['out']
    m = nn.Sigmoid()
    out = m(out)

    out = out.squeeze(0)
    out = out.cpu().detach().numpy()
    out *= 255
    out = out.transpose(2,1,0)
    cv2.imwrite("out_10_4.png", out)

    mask = mask.numpy()
    mask *= 255
    mask = mask.transpose(2,1,0)
    cv2.imwrite("mask_10_4.png", mask)
