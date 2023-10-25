import json, os
import numpy as np
import torch, cv2
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as T

def read_mask(label_path):
    label_data = json.load(open(label_path))
    segs = label_data["shapes"][0]["points"]
    img_shape = [label_data["imageHeight"], label_data["imageWidth"]]
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask_image = Image.fromarray(mask)
    xy = list(map(tuple, segs))
    ImageDraw.Draw(mask_image).polygon(xy=xy, outline=1, fill=255)
    mask_image = np.array(mask_image)
    mask = mask_image.astype(np.float32) 
    return mask

class ForceData(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.force = np.loadtxt(open(data_path+"/data.csv","rb"),delimiter=",")
        self.rgb = glob(os.path.join(data_path, "rgb_cut", "*.jpg"))
        self.inf = glob(os.path.join(data_path, "inf_cut", "*"))
        self.lable = glob(os.path.join(data_path, "rgb_cut", "*.json"))
        self.rgb.sort()
        self.inf.sort()
        self.lable.sort()

        self.scale = [1,1,6]

        self.transform1 = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])
        self.transform2 = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.ToTensor()
        ])

    def totensor(self, img):
        '''
        convert numpy array to tensor
        '''
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        img = img.astype(np.float32) / 255.0
        img -= img.mean()
        img = img.transpose((2,0,1))

        return torch.from_numpy(img.astype(np.float32))
    
    def mask_transform(self, mask):
        mask = Image.fromarray(mask)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask =  torch.from_numpy(np.array(mask).astype(np.float32) / 255)
        return mask.unsqueeze(0)

    def __getitem__(self, index):
        rgb = cv2.imread(self.rgb[index])
        inf = cv2.imread(self.inf[index])
        mask = read_mask(self.lable[index])
        x_force = (mask * (self.force[index][0] + self.scale[0]) / self.scale[0])
        y_force = (mask * (self.force[index][1] + self.scale[1]) / self.scale[1])
        z_force = (mask * (self.force[index][2] + self.scale[2]) / self.scale[2])

        rgb = cv2.resize(rgb, (256, 256))
        inf = cv2.resize(inf, (256, 256))
        rgb = self.totensor(rgb)
        inf = self.totensor(inf)
        # rgb = self.transform1(rgb)
        # inf = self.transform1(inf)
        force_map_x = self.mask_transform(x_force)
        force_map_y = self.mask_transform(y_force)
        force_map_z = self.mask_transform(z_force)

        mask = self.mask_transform(mask)

        return rgb, inf, (force_map_x, force_map_y, force_map_z), mask
    
    def __len__(self):
        return len(self.rgb)
