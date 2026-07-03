import os
import sys
import json
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image
import random
import re
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bert.tokenization_bert import BertTokenizer
from pycocotools import mask

RISBENCH_ALIASES = {
    "airplane": ["plane", "aircraft"],
    "baseball diamond": ["baseball field"],
    "soccer ball field": ["soccer-ball-shaped field", "soccer-ball field", "soccer field"],
    "vehicle": ["car", "truck", "bus"],
    "ground track field": [
        "ground-track field", "field and track", "track and field",
        "ground track and field", "ground track-and-field", "ground and track field"
    ],
    "expressway toll station": [
        "toll station", "expressway-toll station", "expressway-service area", "service area"
    ],
    "golf field": ["golf course", "large green area"],
    "ship": ["vessel", "boat"],
    "storage tank": ["storage is tank"],
    "running track field": ["U-shaped running track", "U-shaped ground track"],
    "chimney": ["cooling tower"],
    "harbor": ["dock", "pier"],
}


class ReferDataset(data.Dataset):
    def __init__(self,
                 args,
                 cfg,
                 image_transforms=None,
                 split='train'):

        self.classes = []
        self.image_transforms = image_transforms
        self.dataset_name = args.dataset
        self.split = split
        self.data_root = args.refer_data_root

        if args.dataset == "rrsisd":
            self.ann_file = f"datainfo/rrsisd_{split}.jsonl"
            self.target_cls = {
                "airplane", "airport", "golf field", "expressway service area", "baseball field",
                "stadium", "ground track field", "storage tank", "basketball court", "chimney",
                "tennis court", "overpass", "train station", "ship", "expressway toll station",
                "dam", "harbor", "bridge", "vehicle", "windmill"
            }
            self.image_root = os.path.join(self.data_root, 'images/rrsisd/JPEGImages')
            self.max_tokens = 22
        elif args.dataset == "refsegrs":
            self.ann_file = f"datainfo/refsegrs_{split}.jsonl"
            self.target_cls = {
                "road", "vehicle", "car", "van", "building", "truck", "trailer", "bus",
                "road marking", "bikeway", "sidewalk", "tree", "low vegetation", "impervious surface"
            }
            self.image_root = os.path.join(self.data_root, 'images')
            self.max_tokens = 20
        else:
            self.ann_file = f"datainfo/risbench_{split}.jsonl"
            self.target_cls = {
                "expressway service area", "expressway toll station", "ground track field",
                "basketball court", "container crane", "roundabout", "windmill", "overpass",
                "stadium", "bridge", "soccer ball field", "baseball diamond", "train station",
                "golf field", "airport", "harbor", "dam", "ship", "helipad", "vehicle",
                "chimney", "airplane", "helicopter", "tennis court", "storage tank", "swimming pool", "running track field"
            }
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

        with open(ann_path, "r", encoding="utf-8") as f:
            self.dataset = [json.loads(line) for line in f if line.strip()]
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.processed_data = []
        self._preprocess_all_data()

    def find_first_category(self, sentence):
        """
        在句子中找到第一个出现的类别（按字符位置最靠前）。
        返回 (category_name, start_pos, end_pos)
        """
        alias_map = RISBENCH_ALIASES if self.dataset_name == 'risbench' else None
        
        s = (sentence or "").lower()
        best_cat = None
        best_pos = None
        best_end = None

        for cat in self.target_cls:
            base = cat.lower()

            variants = [base]

            # 例如 "golf field" -> "golf-field", "golffield"
            if " " in base:
                variants.append(base.replace(" ", "-"))
                variants.append(base.replace(" ", ""))

            # 数据集特定的别名
            if alias_map is not None and cat in alias_map:
                variants.extend([a.lower() for a in alias_map[cat]])

            for v in variants:
                pattern = r"\b" + re.escape(v) + r"s?\b"
                m = re.search(pattern, s)
                if m:
                    pos = m.start()
                    if best_pos is None or pos < best_pos:
                        best_pos = pos
                        best_cat = cat
                        best_end = m.end()

        if best_cat is None:
            return None, float('inf'), None
        
        return best_cat, best_pos, best_end
    
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
        missing_category_count = 0

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
            # Normalize optional fields
            sam3_list = item.get('sam3') or []
            category_name = item.get('category_name', None)
            category_id = item.get('category_id', None)
            if category_name is None or category_id is None:
                missing_category_count += 1
                category_name = '0' if category_name is None else category_name
                category_id = 0 if category_id is None else category_id

            tmp_items.append({
                'idx': idx,
                'file_name': item['file_name'],
                'sentence': sentence,
                'segmentation': seg_rle,
                'text_inputs': text_inputs,
                'category_name': category_name,
                'category_id': category_id,
                'area_ratio': area_ratio,
                'sam3': sam3_list,
            })
        
        assert len(tmp_items) > 0, "tmp_items is empty, Please check ann_file."
        if missing_category_count > 0:
            warnings.warn(
                f"{self.ann_file}: {missing_category_count} samples have no category info; "
                "defaulting missing category_name to '0' and missing category_id to 0.",
                UserWarning,
            )

        self.processed_data = list(tmp_items)
    
    def get_classes(self):
        return self.classes
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, index):
        try:
            return self.get_item(index)
        except Exception as e:
            print('Error in ReferDataset.__getitem__:', e)
            return self.get_item(random.randint(0, len(self.processed_data) - 1))
    
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
            'category_name': item['category_name'],
            'class_ids': item['category_id'],
            'area_ratio': item['area_ratio'],
            'index': index,
            'sam3_masks': sam3_masks,
        }
        
        return result

        import traceback
        traceback.print_exc()
        
        print("\n=== Troubleshooting Tips ===")
        print("1. Check if the data path is correct")
        print("2. Ensure BERT tokenizer is properly installed")
        print("3. Verify that the JSONL files exist in datainfo/ directory")
        print("4. Make sure nltk punkt tokenizer is downloaded: nltk.download('punkt')")
        print("5. Check if pycocotools is properly installed")
