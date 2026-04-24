from torch.utils.data.sampler import RandomSampler
import cv2 as cv
import json
import os
import torch
import torchvision.transforms as T


DATASET_INFOS = {
    'mvtec': [
        ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile','toothbrush', 'transistor', 'wood', 'zipper'], 
        ['bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper'], 
        ['carpet', 'grid', 'leather', 'tile', 'wood']
    ],  # all, obj, texture
    'mvtec_3d': [
        ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"], 
        ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"], 
        []
    ],
    'btad': [
        ["01", "02", "03"], 
        ["01", "03"], 
        ["02"]
    ]
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def register_custom_dataset(data_dir, sub_categories, object_categories=None):
    """Register a custom AOI dataset into DATASET_INFOS.
    
    Supports two directory layouts:
    
    1. MVTec-style (default, auto-detected):
       data_dir/<category>/train/good/...
       data_dir/<category>/test/good/...
       data_dir/<category>/test/<defect_type>/...
       data_dir/<category>/ground_truth/<defect_type>/...
    
    2. JSON annotation style:
       data_dir/<category>.json with structure:
       {
         "train": [{"image_path": "...", "anomaly_class": "OK"}, ...],
         "test":  [{"image_path": "...", "mask_path": "...", "anomaly_class": "OK|<defect>"}, ...]
       }
    """
    if object_categories is None:
        object_categories = list(sub_categories)  # default: treat all as objects
    texture_categories = [c for c in sub_categories if c not in object_categories]
    DATASET_INFOS['custom'] = [
        list(sub_categories),
        list(object_categories),
        texture_categories
    ]
    return DATASET_INFOS['custom']


def setup_custom_from_json(json_path, data_dir, sub_category):
    """Convert JSON annotation to MVTec-style directory structure for CPR compatibility.
    
    JSON format:
    {
      "train": [{"image_path": "path/to/img.png", "anomaly_class": "OK"}, ...],
      "test":  [{"image_path": "path/to/img.png", "mask_path": "path/to/mask.png", "anomaly_class": "OK|defect_name"}, ...]
    }
    
    Creates symlinked MVTec-style layout under data_dir/custom/<sub_category>/
    """
    import shutil
    
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    root = os.path.join(data_dir, 'custom', sub_category)
    
    # Create train/good
    train_dir = os.path.join(root, 'train', 'good')
    os.makedirs(train_dir, exist_ok=True)
    for idx, sample in enumerate(annotations.get('train', [])):
        src = sample['image_path']
        if not os.path.isabs(src):
            src = os.path.join(os.path.dirname(json_path), src)
        ext = os.path.splitext(src)[1]
        dst = os.path.join(train_dir, f'{idx:06d}{ext}')
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    
    # Create test/<class> and ground_truth/<class>
    for idx, sample in enumerate(annotations.get('test', [])):
        src = sample['image_path']
        if not os.path.isabs(src):
            src = os.path.join(os.path.dirname(json_path), src)
        anomaly_class = sample.get('anomaly_class', 'OK')
        class_dir = 'good' if anomaly_class == 'OK' else anomaly_class
        
        ext = os.path.splitext(src)[1]
        test_dir = os.path.join(root, 'test', class_dir)
        os.makedirs(test_dir, exist_ok=True)
        dst = os.path.join(test_dir, f'{idx:06d}{ext}')
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
        
        # Ground truth mask
        mask_path = sample.get('mask_path')
        if mask_path and anomaly_class != 'OK':
            if not os.path.isabs(mask_path):
                mask_path = os.path.join(os.path.dirname(json_path), mask_path)
            gt_dir = os.path.join(root, 'ground_truth', class_dir)
            os.makedirs(gt_dir, exist_ok=True)
            mask_dst = os.path.join(gt_dir, f'{idx:06d}_mask{os.path.splitext(mask_path)[1]}')
            if not os.path.exists(mask_dst):
                os.symlink(os.path.abspath(mask_path), mask_dst)
    
    return root


def read_image(path, resize = None):
    img = cv.imread(path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if resize:
        img = cv.resize(img, dsize=resize)
    return img

def read_mask(path, resize = None):
    mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f'Cannot read mask: {path}')
    if resize:
        mask = cv.resize(mask, dsize=resize, interpolation=cv.INTER_NEAREST)
    return mask

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def inverse_test_transform(image):
    denormalized = image * torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1) + \
                   torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    img = denormalized * 255.0
    img = img.to(torch.uint8)
    return img.cpu().numpy().transpose(1, 2, 0)

class InfiniteSampler(RandomSampler):
    def __iter__(self):
        while True: yield from super().__iter__()

from dataset.base import CPRDataset