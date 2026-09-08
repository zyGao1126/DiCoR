import os
import sys
import json
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bert.tokenization_bert import BertTokenizer
from pycocotools import mask


class ReferDataset(data.Dataset):
    def __init__(self,
                 args,
                 cfg,
                 image_transforms=None,
                 split='train'):

        self.image_transforms = image_transforms
        self.split = split
        self.data_root = args.refer_data_root
        self.img_size = int(args.img_size)

        if args.dataset == "rrsisd":
            self.ann_file = f"datainfo/rrsisd_{split}.jsonl"
            self.image_root = os.path.join(self.data_root, 'images/rrsisd/JPEGImages')
            self.max_tokens = 22
        elif args.dataset == "refsegrs":
            self.ann_file = f"datainfo/refsegrs_{split}.jsonl"
            self.image_root = os.path.join(self.data_root, 'images')
            self.max_tokens = 20
        else:
            self.ann_file = f"datainfo/risbench_{split}.jsonl"
            self.image_root = os.path.join(self.data_root, 'img_rgb')
            self.max_tokens = 50

        if os.path.exists(self.ann_file):
            ann_path = self.ann_file
        else:
            ann_path = os.path.join(self.data_root, self.ann_file)
        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"Annotation file not found: {self.ann_file}. "
                f"Place it under ./datainfo or {os.path.join(self.data_root, 'datainfo')}."
            )
        self.ann_path = os.path.abspath(ann_path)

        with open(self.ann_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f if line.strip()]
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.processed_data = []
        self._preprocess_all_data()
    
    def create_text_inputs(self, sentence):
        max_len = self.max_tokens
        tokens = ['[CLS]'] + self.tokenizer.tokenize(sentence) + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]

        attention_mask = [1] * len(input_ids)
        pad_len = max_len - len(input_ids)
        if pad_len:
            input_ids += [0] * pad_len
            attention_mask += [0] * pad_len

        return {
            'input_ids'     : torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask]),
        }
    
    def _preprocess_all_data(self):
        tmp_items = []

        RRSISD_exclude = ['22187.jpg', '20203.jpg', '00413.jpg', '01072.jpg', '01664.jpg', '03661.jpg', '05125.jpg', '06728.jpg',
                          '06861.jpg', '09319.jpg', '10579.jpg', '10653.jpg', '11147.jpg', '11898.jpg',
                          '12492.jpg', '12630.jpg', '14464.jpg', '14915.jpg', '15357.jpg', '15584.jpg',
                          '15737.jpg', '17068.jpg', '18552.jpg', '18845.jpg', '20235.jpg', '21126.jpg', '07239.jpg']
        RISBench_exclude = ['train_12443_2.png', 'train_11904_8.png', 'train_11818_2.png', 'train_11785_0.png', 'train_11021_1.png',
                            'train_10698_0.png', 'train_10222_0.png', 'train_9598_7.png', 'train_9405_1.png', 'train_8518_1.png',
                            'train_7581_3.png', 'train_7222_1.png', 'train_7008_0.png', 'train_6194_0.png', 'train_6194_1.png', 'train_819_1.png']        

        for idx, item in enumerate(self.dataset):
            if item['file_name'] in RRSISD_exclude:
                print('Skipping image:', item['file_name'])
                continue
            if item['file_name'] in RISBench_exclude:
                print('Skipping image:', item['file_name'])
                continue
            
            sentence = item['sent']
            text_inputs = self.create_text_inputs(sentence)

            seg_rle = item['segmentation'][0] if isinstance(item['segmentation'], list) else item['segmentation']
            ref = mask.decode(seg_rle)
            h, w = ref.shape[:2]
            fg_area = float((ref == 1).sum())
            area_ratio = fg_area / (float(h) * float(w))
            sam3_list = item.get('sam3') or []

            tmp_items.append({
                'idx': idx,
                'file_name': item['file_name'],
                'sentence': sentence,
                'segmentation': seg_rle,
                'text_inputs': text_inputs,
                'area_ratio': area_ratio,
                'sam3': sam3_list,
            })
        
        assert len(tmp_items) > 0, "tmp_items is empty, Please check ann_file."

        self.processed_data = list(tmp_items)
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        return self.get_item(index)
    
    def get_item(self, index):
        item = self.processed_data[index]
        img_path = os.path.join(self.image_root, item['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        seg_mask_rle = item['segmentation']
        ref_mask = mask.decode(seg_mask_rle)
        h, w = ref_mask.shape[:2]
        
        annot = np.zeros(ref_mask.shape, dtype=np.uint8)
        annot[ref_mask == 1] = 1
        annot = Image.fromarray(annot, mode="P")

        sam3_entries = item.get('sam3', [])
        sam3_masks = []
        for inst in sam3_entries:
            rle = {
                'size': [h, w],
                'counts': inst['counts']
            }
            m = mask.decode(rle)
            if m.ndim == 3:
                m = m[..., 0]
            sam3_masks.append(m.astype(np.uint8))
        sam3_masks = [Image.fromarray(m.astype(np.uint8), mode='P') for m in sam3_masks]         
        target_dict = {'gold_masks': annot, 'sam3_masks': sam3_masks}

        if self.image_transforms is not None:
            img, target_dict = self.image_transforms(img, target_dict)
            target = target_dict['gold_masks']
            sam3_masks = target_dict['sam3_masks']
        else:
            target = annot
            sam3_masks = [torch.from_numpy(np.array(m)) for m in sam3_masks]

        text_inputs = item['text_inputs']
        save_prefix = f"{item['idx']}_{item['sentence'][:50]}"

        result = {
            'image': img,
            'target': target,
            'tensor_embeddings': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'save_prefix': save_prefix,
            'sentence': item['sentence'],
            'area_ratio': item['area_ratio'],
            'index': index,
            'sam3_masks': sam3_masks,
        }
        
        return result

    def get_coarse_context_item(self, index):
        """Load only the image and language used by frozen coarse inference."""
        if self.image_transforms is None:
            raise RuntimeError("Coarse-context preprocessing requires image transforms")

        item = self.processed_data[index]
        img_path = os.path.join(self.image_root, item['file_name'])
        image = Image.open(img_path).convert("RGB")
        image, _ = self.image_transforms(image, None)
        text_inputs = item['text_inputs']
        return {
            'image': image,
            'tensor_embeddings': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'index': index,
        }
