import torch
import json
import pandas as pd
import csv
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor

class HatefulMemesDataset(Dataset):
    def __init__(self,args, data_path, img_dir, processor, split='train', max_length=77):
        self.data = [json.loads(line) for line in open(data_path)]
        self.img_dir = img_dir
        self.processor = processor
        self.split = split
        self.max_length = max_length  # CLIP默认最大长度77


        # 添加caption加载
        self.caption_dict = {}
        with open('/root/autodl-tmp/hateful_memes/caption.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                filename = row[0].split('/')[-1]  # 提取纯文件名
                self.caption_dict[filename] = row[1]
        """
        # 添加数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ])
        self.val_transform = transforms.ToTensor()
        """
        
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(f"{self.img_dir}/{item['img']}").convert("RGB")
        """
        if self.split == 'train':
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        image = transforms.ToPILImage()(image).resize((224, 224))
        """
        # 获取原始文本和caption
        original_text = item['text']
        filename = item['img'].split('/')[-1]  # 提取纯文件名
        caption = self.caption_dict.get(filename, "")  # 处理缺失情况
        
        # 拼接文本
        #combined_text = f"{original_text} [SEP] {caption}"  # 使用分隔符
        combined_text = f"{original_text}{caption}"  # 使用分隔符

        inputs = self.processor(
            text=original_text,
            images=image,
            return_tensors="pt",
            padding='max_length',  # 关键修改
            max_length=self.max_length,
            truncation=True
        )
       
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'text_aug': item['text_aug'],  
        }

class Collator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.stack([x['label'] for x in batch]),
            'text_aug':torch.stack([x['text_aug'] for x in batch])
        }

def get_dataloaders(args):
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    collator = Collator(processor)
    
    train_ds = HatefulMemesDataset(
        f"{args.data_dir}/merged/train_merged.jsonl", 
        args.data_dir,
        processor
    )
    val_ds = HatefulMemesDataset(
        f"{args.data_dir}/merged/val_merged.jsonl",
        args.data_dir,
        processor
    )
    test_ds = HatefulMemesDataset(
        f"{args.data_dir}/merged/test_merged.jsonl",
        args.data_dir,
        processor
    )
    
    return (
        DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                  num_workers=args.num_workers,  pin_memory=True,collate_fn=collator),
        DataLoader(val_ds, batch_size=args.batch_size,
                  num_workers=args.num_workers, collate_fn=collator),
        DataLoader(test_ds, batch_size=args.batch_size,
                  num_workers=args.num_workers, collate_fn=collator)
    )