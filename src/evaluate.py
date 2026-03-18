import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import precision_recall_fscore_support

from dataset import PromptedSegmentationDataset, get_clipseg_transforms


def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU) for binary segmentation."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    if target_binary.sum() == 0 and pred_binary.sum() == 0:
        return 1.0
    if target_binary.sum() == 0:
        return 0.0
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient for this binary segmentation."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    if target_binary.sum() == 0 and pred_binary.sum() == 0:
        return 1.0
    if target_binary.sum() == 0:
        return 0.0
    
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection + 1e-8) / (pred_binary.sum() + target_binary.sum() + 1e-8)
    
    return dice.item()


def compute_f1(pred, target, threshold=0.5):
    """Compute F1 score (pixel-wise) for binary segmentation."""
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float().cpu().numpy().flatten()
    target_binary = target.float().cpu().numpy().flatten()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_binary, pred_binary, average='binary', zero_division=0
    )
    
    return f1


def find_optimal_threshold(model, dataloader, processor, device, task_name):
    """Find optimal threshold on validation set by grid search."""
    model.eval()
    # grid search from 0.1 to 0.9 with 0.02 steps
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_threshold = 0.5
    best_dice = 0.0
    
    all_preds = []
    all_targets = []
    
    print(f"\nCollecting predictions for {task_name}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference [{task_name}]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']
            
            text_inputs = processor(
                text=list(prompts),
                return_tensors="pt",
                padding=True
            ).to(device)
            
            outputs = model(
                pixel_values=images,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            logits = outputs.logits
            
            logits = logits.unsqueeze(1)
            logits = F.interpolate(logits, size=(640, 640), mode='bilinear', align_corners=False)
            logits = logits.squeeze(1)
            
            all_preds.append(logits.cpu())
            all_targets.append(masks.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"Finding optimal threshold for {task_name}...")
    for threshold in tqdm(thresholds, desc="Threshold search"):
        dice = compute_dice(all_preds, all_targets, threshold=threshold)
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    print(f"Optimal threshold for {task_name}: {best_threshold:.2f} (Dice: {best_dice:.4f})")
    return best_threshold


def evaluate_model(model, dataloader, processor, device, threshold, task_name):
    """Evaluate model on test set with given threshold."""
    model.eval()
    
    all_ious = []
    all_dices = []
    all_f1s = []
    
    print(f"\nEvaluating {task_name} on test set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluation [{task_name}]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            prompts = batch['prompt']
            
            text_inputs = processor(
                text=list(prompts),
                return_tensors="pt",
                padding=True
            ).to(device)
            
            outputs = model(
                pixel_values=images,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            logits = outputs.logits
            
            logits = logits.unsqueeze(1)
            logits = F.interpolate(logits, size=(640, 640), mode='bilinear', align_corners=False)
            logits = logits.squeeze(1)
            
            for i in range(logits.shape[0]):
                iou = compute_iou(logits[i], masks[i], threshold=threshold)
                dice = compute_dice(logits[i], masks[i], threshold=threshold)
                f1 = compute_f1(logits[i], masks[i], threshold=threshold)
                
                all_ious.append(iou)
                all_dices.append(dice)
                all_f1s.append(f1)
    
    metrics = {
        'iou': np.mean(all_ious),
        'dice': np.mean(all_dices),
        'f1': np.mean(all_f1s),
        'iou_std': np.std(all_ious),
        'dice_std': np.std(all_dices),
        'f1_std': np.std(all_f1s),
        'threshold': threshold,
        'num_samples': len(all_ious)
    }
    
    return metrics


def save_predictions(model, dataloader, processor, device, threshold, prompt_slug, output_dir):
    """Save prediction masks as PNG files."""
    pred_dir = Path(output_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Saving [{prompt_slug}]"):
            images = batch['image'].to(device)
            prompts = batch['prompt']
            image_paths = batch['image_path']
            
            text_inputs = processor(
                text=list(prompts),
                return_tensors="pt",
                padding=True
            ).to(device)
            
            outputs = model(
                pixel_values=images,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            logits = outputs.logits
            
            logits = logits.unsqueeze(1)
            logits = F.interpolate(logits, size=(640, 640), mode='bilinear', align_corners=False)
            logits = logits.squeeze(1)
            
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).cpu().numpy().astype(np.uint8) * 255
            
            for i, img_path in enumerate(image_paths):
                image_id = Path(img_path).stem
                out_name = f"{image_id}__{prompt_slug}.png"
                cv2.imwrite(str(pred_dir / out_name), preds[i])
    
    print(f"Saved predictions to: {pred_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIPSeg model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory with splits")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading model and processor...")
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Checkpoint avg val Dice: {checkpoint.get('avg_val_dice', 'N/A')}")
    
    print("\nLoading datasets...")
    
    splits_dir = Path(args.data_dir) / "splits"
    transform = get_clipseg_transforms(processor)
    
    cracks_val_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/cracks_val.json",
        prompt_text="segment crack",
        transform=transform,
        augment=False
    )
    cracks_test_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/cracks_test.json",
        prompt_text="segment crack",
        transform=transform,
        augment=False
    )
    
    drywall_val_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/drywall_val.json",
        prompt_text="segment taping area",
        transform=transform,
        augment=False
    )
    drywall_test_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/drywall_test.json",
        prompt_text="segment taping area",
        transform=transform,
        augment=False
    )
    
    cracks_val_loader = DataLoader(cracks_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    cracks_test_loader = DataLoader(cracks_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    drywall_val_loader = DataLoader(drywall_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    drywall_test_loader = DataLoader(drywall_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print("\nFinding optimal thresholds...")
    
    cracks_threshold = find_optimal_threshold(model, cracks_val_loader, processor, device, "Cracks")
    drywall_threshold = find_optimal_threshold(model, drywall_val_loader, processor, device, "Drywall")
    
    print("\nEvaluating on test sets...")
    
    cracks_metrics = evaluate_model(model, cracks_test_loader, processor, device, cracks_threshold, "Cracks")
    drywall_metrics = evaluate_model(model, drywall_test_loader, processor, device, drywall_threshold, "Drywall")
    
    print("\nResults:")
    
    print("\nCracks Detection:")
    print(f"  Threshold: {cracks_metrics['threshold']:.2f}")
    print(f"  IoU:       {cracks_metrics['iou']:.4f} ± {cracks_metrics['iou_std']:.4f}")
    print(f"  Dice:      {cracks_metrics['dice']:.4f} ± {cracks_metrics['dice_std']:.4f}")
    print(f"  F1:        {cracks_metrics['f1']:.4f} ± {cracks_metrics['f1_std']:.4f}")
    print(f"  Samples:   {cracks_metrics['num_samples']}")
    
    print("\nDrywall Taping Area:")
    print(f"  Threshold: {drywall_metrics['threshold']:.2f}")
    print(f"  IoU:       {drywall_metrics['iou']:.4f} ± {drywall_metrics['iou_std']:.4f}")
    print(f"  Dice:      {drywall_metrics['dice']:.4f} ± {drywall_metrics['dice_std']:.4f}")
    print(f"  F1:        {drywall_metrics['f1']:.4f} ± {drywall_metrics['f1_std']:.4f}")
    print(f"  Samples:   {drywall_metrics['num_samples']}")
    
    avg_dice = (cracks_metrics['dice'] + drywall_metrics['dice']) / 2
    avg_iou = (cracks_metrics['iou'] + drywall_metrics['iou']) / 2
    avg_f1 = (cracks_metrics['f1'] + drywall_metrics['f1']) / 2
    
    print("\nAverage Across Tasks:")
    print(f"  IoU:  {avg_iou:.4f}")
    print(f"  Dice: {avg_dice:.4f}")
    print(f"  F1:   {avg_f1:.4f}")
    
    results = {
        'cracks': cracks_metrics,
        'drywall': drywall_metrics,
        'average': {
            'iou': avg_iou,
            'dice': avg_dice,
            'f1': avg_f1
        },
        'checkpoint_path': str(args.checkpoint),
        'checkpoint_epoch': checkpoint['epoch']
    }
    
    print("\nSaving predictions...")
    
    save_predictions(model, cracks_test_loader, processor, device, cracks_threshold, "segment_crack", output_dir)
    save_predictions(model, drywall_test_loader, processor, device, drywall_threshold, "segment_taping_area", output_dir)
    
    def convert_numpy(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
