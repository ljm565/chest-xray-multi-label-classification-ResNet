import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image



class DLoader(Dataset):
    def __init__(self, base_path, data, state, transform):
        self.base_path = base_path
        self.data = data
        self.state = state
        self.transform = transform
        assert self.state in ['train', 'test']

        self.img, self.label = self.make_data(self.state, self.data)
        
        # sanity check
        assert len(self.img) == len(self.label)
        self.length = len(self.img)

    
    def make_data(self, state, data):
        total_img, total_label = [], []
        if state == 'train':
            for sub in tqdm(data.keys()):
                stu, img, label = data[sub]
                if not stu[-1] in ['8', '9']:
                    img_path = self.base_path + '/data/' + self.state + '/' + img + '.jpg'
                    img = Image.open(img_path)            
                    img = self.transform(img)
                    label = torch.Tensor(label)
                    total_img.append(img)
                    total_label.append(label)
        else:
            for sub in tqdm(data.keys()):
                stu, img, label = data[sub]
                if stu[-1] in ['8', '9']:
                    img_path = self.base_path + '/data/' + self.state + '/' + img + '.jpg'
                    img = Image.open(img_path)
                    img = self.transform(img)
                    label = torch.Tensor(label)
                    total_img.append(img)
                    total_label.append(label)
        return total_img, total_label


    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


    def __len__(self):
        return self.length