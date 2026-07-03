from torch.utils.data.dataloader import default_collate

def colllate_fn_custom(batch):
    batch_dict = {}
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'sam3_masks':
            batch_dict[key] = [d[key] for d in batch]
        else:
            values = [d.get(key, None) for d in batch]
            if any(v is None for v in values):
                sample_ids = [d.get('index', None) for d in batch]
                types = [type(v).__name__ for v in values]
                raise TypeError(
                    f"collate failed: key={key} contains None. "
                    f"sample_ids={sample_ids}, types={types}"
                )
            batch_dict[key] = default_collate(values)
            
    return batch_dict
