
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
import torch


class dicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir,  self.Speckles_frame.iloc[idx, 1])
        dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])

        Ref_img = np.genfromtxt(Ref_name, delimiter=',')
        Def_img = np.genfromtxt(Def_name, delimiter=',')
        dispx = np.genfromtxt(dispx_name, delimiter=',')
        dispy = np.genfromtxt(dispy_name, delimiter=',')
        sample = {'Ref':Ref_img, 'Def': Def_img, 'Dispx': dispx, 'Dispy': dispy}
        if self.transform:
            sample = self.transform(sample)
        return  sample



class Augmentation(object):
    """Convert ndarrays in sample to Tensors."""
    # 翻转 水平 垂直
    def __init__(self,rho = 10,crop_size = (256, 256),is_Aug = False):
        self.rho = rho
        self.crop_size =crop_size
        self.is_Aug = is_Aug
    def Flip(self,sample):
        Ref, Def = sample['Ref'], sample['Def']
        k=np.random.choice([-1,0,1], 1)
        Ref = cv.flip(Ref, k[0])
        Def = cv.flip(Def, k[0])
        Ref =  Ref [np.newaxis, ...]
        Def =  Def [np.newaxis, ...]
        sample = {'Ref': Ref , 'Def': Def}
        Ref_croped,Def_croped,start=self.random_crop(sample)
        return self.Normalization(Ref),self.Normalization(Def),self.Normalization(Ref_croped),self.Normalization(Def_croped),start

    # 旋转
    def Rotation(self,sample):
        Ref, Def = sample['Ref'], sample['Def']

        cols,rows = Ref.shape
        center=(cols / 2, rows / 2)
        angle = np.random.uniform(-360, 360)
        rotMat = cv.getRotationMatrix2D(center, angle, 1);
        Ref = cv.warpAffine( Ref ,rotMat, dsize=(rows, cols))
        Def = cv.warpAffine( Def ,rotMat, dsize=(rows, cols))
        Ref = Ref[np.newaxis, ...]
        Def = Def[np.newaxis, ...]
        sample = {'Ref': Ref, 'Def': Def}
        Ref_croped, Def_croped, start = self.random_crop(sample)
        return self.Normalization(Ref), self.Normalization(Def), self.Normalization(Ref_croped), self.Normalization(Def_croped), start

    # 裁剪
    def Crop(self,sample):
        Ref, Def = sample['Ref'], sample['Def']
        cols, rows = Ref.shape
        borad_x = np.random.randint(low=10, high=35, size=1)[0]
        borad_y = np.random.randint(low=10, high=35, size=1)[0]

        borad1 = cols-borad_x*2
        borad2 = rows- borad_y * 2
        Ref = Ref[borad_x:borad_x+borad1,borad_y:borad_y+borad2]
        Ref = cv.copyMakeBorder( Ref, borad_x, borad_x, borad_y, borad_y, cv.BORDER_CONSTANT, value=(0, 0, 0))
        Def = Def[borad_x:borad_x + borad1, borad_y:borad_y + borad2]
        Def = cv.copyMakeBorder(Def, borad_x, borad_x, borad_y, borad_y, cv.BORDER_CONSTANT, value=(0, 0, 0))
        Ref = Ref[np.newaxis, ...]
        Def = Def[np.newaxis, ...]
        sample = {'Ref': Ref, 'Def': Def}
        Ref_croped, Def_croped, start = self.random_crop(sample)
        return self.Normalization(Ref), self.Normalization(Def), self.Normalization(Ref_croped), self.Normalization(Def_croped), start




    def Scale(self,sample):
        Ref, Def = sample['Ref'], sample['Def']
        cols, rows = Ref.shape
        center = (np.random.uniform(rows/5, rows/2), np.random.uniform(rows/5, rows/2+rows/6))
        scale = np.random.uniform(0.3, 2)
        M = cv.getRotationMatrix2D(center, 0,  scale);
        Ref = cv.warpAffine(Ref, M, dsize=(rows, cols), flags=cv.INTER_CUBIC)
        Def = cv.warpAffine(Def, M, dsize=(rows, cols), flags=cv.INTER_CUBIC)

        Ref = Ref[np.newaxis, ...]
        Def = Def[np.newaxis, ...]
        sample = {'Ref': Ref, 'Def': Def}
        Ref_croped, Def_croped, start = self.random_crop(sample)
        return self.Normalization(Ref), self.Normalization(Def), self.Normalization(Ref_croped), self.Normalization(Def_croped), start
    #归一化


    def Normalization(self,img):
        self.mean = 0.0
        self.std = 255.0
        return torch.from_numpy((img - self.mean) / self.std).float()

    # def random_crop(self, sample):
    #     Ref, Def = sample['Ref'], sample['Def']
    #     height, width = Ref.shape[1:]
    #     patch_size_h, patch_size_w = self.crop_size
    #     x = np.random.randint(self.rho, width - self.rho - patch_size_w)
    #     # print(self.rho, height - self.rho - patch_size_h)
    #     y = np.random.randint(self.rho, height - self.rho - patch_size_h)
    #     start = np.array([x, y])
    #     start = np.expand_dims(np.expand_dims(start, 1), 2)
    #     img_1_patch = Ref[:, y: y + patch_size_h, x: x + patch_size_w]
    #     img_2_patch = Def[:, y: y + patch_size_h, x: x + patch_size_w]
    #     return img_1_patch, img_2_patch, start
    def random_crop(self, sample):
        Ref, Def = sample['Ref'], sample['Def']
        height, width = Ref.shape[1:]
        patch_size_h, patch_size_w = self.crop_size

        # 如果图像尺寸小于裁剪尺寸，直接返回原图
        if height <= patch_size_h or width <= patch_size_w:
            start = np.array([0, 0])
            img_1_patch = Ref
            img_2_patch = Def
        else:
            try:
                # 确保裁剪区域合法
                x = np.random.randint(self.rho, width - self.rho - patch_size_w)
                y = np.random.randint(self.rho, height - self.rho - patch_size_h)
            except ValueError:
                # 如果随机裁剪失败，回退到中心裁剪
                x = (width - patch_size_w) // 2
                y = (height - patch_size_h) // 2

            start = np.array([x, y])
            start = np.expand_dims(np.expand_dims(start, 1), 2)

            img_1_patch = Ref[:, y: y + patch_size_h, x: x + patch_size_w]
            img_2_patch = Def[:, y: y + patch_size_h, x: x + patch_size_w]

        return img_1_patch, img_2_patch, start

    def NoAugmentation(self, sample):
        Ref, Def ,Dispx,Dispy= sample['Ref'], sample['Def'],sample['Dispx'], sample['Dispy']
        Ref = Ref[np.newaxis, ...]
        Def = Def[np.newaxis, ...]

        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy}
        Ref_croped, Def_croped, start = self.random_crop(sample)
        return self.Normalization(Ref), self.Normalization(Def), self.Normalization(Ref_croped), self.Normalization(Def_croped), start, self.Normalization(Dispx), self.Normalization(Dispy)
    def __call__(self, sample):
        if  self.is_Aug:
            Ref1, Def1, Ref_croped1, Def_croped1, start1 = self.Flip(sample)
            Ref2, Def2, Ref_croped2, Def_croped2, start2 = self.Rotation(sample)
            Ref3, Def3, Ref_croped3, Def_croped3, start3 = self.Crop(sample)
            Ref4, Def4, Ref_croped4, Def_croped4, start4= self.Scale(sample)


            Ref = [Ref1,Ref2,Ref3,Ref4]
            Def = [Def1, Def2, Def3, Def4]
            Ref_croped = [Ref_croped1, Ref_croped2, Ref_croped3, Ref_croped4]
            Def_croped = [Def_croped1, Def_croped2, Def_croped3, Def_croped4]
            start = [start1, start2, start3, start4]
        else:
            Ref, Def, Ref_croped, Def_croped, start,Dispx,Dispy = self.NoAugmentation(sample)


        return {'Ref': Ref, 'Def': Def,
                'Ref_croped':Ref_croped,'Def_croped': Def_croped,
                'start':start, 'Dispx':Dispx, 'Dispy':Dispy
                }


def dic(batch):

    return {'Ref': torch.cat(batch['Ref'],0), 'Def': torch.cat(batch['Def'],0),
                'Ref_croped':torch.cat(batch['Ref_croped'],0),'Def_croped':torch.cat(batch['Def_croped'],0),
                'start':torch.cat(batch['start'],0), 'Dispx':torch.cat(batch['Dispx'],0), 'Dispy':torch.cat(batch['Dispy'],0)}

def main():
    # p = (256,256)
    # H,W = p
    # print( H)
    transform = transforms.Compose([Augmentation()])
    #
    test_data = dicDataset(csv_file="/home/dell/DATA/wh/DATASET/Test_annotations1.csv", root_dir="/home/dell/DATA/wh/DATASET/Test1/", transform=transform)

    test_loader = DataLoader(test_data, batch_size=1, num_workers=0, pin_memory=True, shuffle=True)

    for i, batch in enumerate(test_loader):
        batch=dic(batch)
        # measure data loading timeimg1
        print(i)
        print(batch['Ref'].shape)
        print(batch['Def'].shape)
        print(batch['Ref_croped'].shape)
        print(batch['Def_croped'].shape)
        print(batch['start'].shape)




if __name__ == '__main__':
    main()


#SpeckleDataset(csv_file)
