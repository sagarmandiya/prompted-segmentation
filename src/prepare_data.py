import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil


def parse_coco_to_masks(coco_json_path, image_dir, output_dir, target_size=(640, 640)):
    """Parse COCO JSON and generate binary masks."""
    coco = COCO(coco_json_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    data_entries = []
    skipped_no_anns = 0
    skipped_empty_mask = 0
    
    img_ids = coco.getImgIds()
    print(f"Processing {len(img_ids)} images from {coco_json_path}")
    
    for img_id in tqdm(img_ids, desc="Generating masks"):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if len(anns) == 0:
            skipped_no_anns += 1
            continue
        
        original_h, original_w = img_info['height'], img_info['width']
        
        combined_mask = np.zeros((original_h, original_w), dtype=np.uint8)
        
        for ann in anns:
            mask = None
            
            if 'segmentation' in ann and ann['segmentation']:
                if isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                    rle = mask_utils.frPyObjects(ann['segmentation'], original_h, original_w)
                    if isinstance(rle, list):
                        rle = mask_utils.merge(rle)
                    mask = mask_utils.decode(rle)
                elif isinstance(ann['segmentation'], dict):
                    mask = mask_utils.decode(ann['segmentation'])
            
            if mask is None and 'bbox' in ann:
                x, y, w, h = map(int, ann['bbox'])
                mask = np.zeros((original_h, original_w), dtype=np.uint8)
                mask[y:y+h, x:x+w] = 1
            
            if mask is not None:
                combined_mask = np.maximum(combined_mask, mask)
        
        if combined_mask.max() == 0:
            skipped_empty_mask += 1
            continue
        
        resized_mask = cv2.resize(combined_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        mask_filename = Path(img_info['file_name']).stem + '_mask.png'
        mask_path = os.path.join(output_dir, mask_filename)
        cv2.imwrite(mask_path, resized_mask * 255)
        
        image_path = os.path.join(image_dir, img_info['file_name'])
        
        category_ids = [ann['category_id'] for ann in anns]
        category_names = [coco.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
        
        data_entries.append({
            'image_id': img_id,
            'image_path': image_path,
            'mask_path': mask_path,
            'original_size': [original_w, original_h],
            'resized_size': list(target_size),
            'categories': category_names,
            'num_annotations': len(anns)
        })
    
    print(f"Generated {len(data_entries)} mask entries")
    if skipped_no_anns > 0:
        print(f"WARNING: Skipped {skipped_no_anns} images with no annotations")
    if skipped_empty_mask > 0:
        print(f"WARNING: Skipped {skipped_empty_mask} images with empty masks")
    return data_entries


def create_splits(data_entries, dataset_name, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create train/val/test splits."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(seed)
    
    indices = np.arange(len(data_entries))
    
    train_idx, temp_idx = train_test_split(
        indices, 
        train_size=train_ratio, 
        random_state=seed
    )
    
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        train_size=val_size, 
        random_state=seed
    )
    
    splits = {
        'train': [data_entries[i] for i in train_idx],
        'val': [data_entries[i] for i in val_idx],
        'test': [data_entries[i] for i in test_idx]
    }
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} entries to {output_path}")
    
    return splits


def create_drywall_splits(train_entries, valid_entries, dataset_name, output_dir, seed=42):
    """Create splits for drywall dataset using existing train/valid."""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(valid_entries) == 0:
        print(f"WARNING: No valid entries for {dataset_name} dataset")
        val_entries = []
        test_entries = []
    else:
        val_entries, test_entries = train_test_split(
            valid_entries, 
            test_size=0.5, 
            random_state=seed
        )
    
    splits = {
        'train': train_entries,
        'val': val_entries,
        'test': test_entries
    }
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} entries to {output_path}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Prepare COCO datasets for prompted segmentation")
    parser.add_argument("--cracks_dir", type=str, required=True, help="Path to cracks dataset directory")
    parser.add_argument("--drywall_dir", type=str, required=True, help="Path to drywall dataset directory")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    cracks_base = Path(args.cracks_dir)
    drywall_base = Path(args.drywall_dir)
    
    print("\nProcessing cracks dataset...")
    
    cracks_coco = cracks_base / "train" / "_annotations.coco.json"
    cracks_images = cracks_base / "train"
    cracks_masks_dir = data_dir / "processed" / "cracks_masks"
    
    cracks_entries = parse_coco_to_masks(
        str(cracks_coco),
        str(cracks_images),
        str(cracks_masks_dir),
        target_size=(640, 640)
    )
    
    cracks_splits = create_splits(
        cracks_entries,
        "cracks",
        str(data_dir / "splits"),
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    print(f"\nCracks dataset splits:")
    print(f"  Train: {len(cracks_splits['train'])} samples")
    print(f"  Val:   {len(cracks_splits['val'])} samples")
    print(f"  Test:  {len(cracks_splits['test'])} samples")
    
    print("\nProcessing drywall dataset...")
    
    drywall_train_coco = drywall_base / "train" / "_annotations.coco.json"
    drywall_train_images = drywall_base / "train"
    drywall_train_masks_dir = data_dir / "processed" / "drywall_masks_train"
    
    drywall_train_entries = parse_coco_to_masks(
        str(drywall_train_coco),
        str(drywall_train_images),
        str(drywall_train_masks_dir),
        target_size=(640, 640)
    )
    
    drywall_valid_coco = drywall_base / "valid" / "_annotations.coco.json"
    drywall_valid_images = drywall_base / "valid"
    drywall_valid_masks_dir = data_dir / "processed" / "drywall_masks_valid"
    
    drywall_valid_entries = parse_coco_to_masks(
        str(drywall_valid_coco),
        str(drywall_valid_images),
        str(drywall_valid_masks_dir),
        target_size=(640, 640)
    )
    
    # drywall dataset only has train/valid splits, so we split valid 50/50 to get val/test
    drywall_splits = create_drywall_splits(
        drywall_train_entries,
        drywall_valid_entries,
        "drywall",
        str(data_dir / "splits")
    )
    
    print(f"\nDrywall dataset splits:")
    print(f"  Train: {len(drywall_splits['train'])} samples")
    print(f"  Val:   {len(drywall_splits['val'])} samples")
    print(f"  Test:  {len(drywall_splits['test'])} samples")
    
    print("\nData preparation complete!")
    print(f"\nProcessed masks saved to: {data_dir / 'processed'}")
    print(f"Split files saved to: {data_dir / 'splits'}")


if __name__ == "__main__":
    main()
