import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import argparse
from tqdm import tqdm
import random

from dataset import PromptedSegmentationDataset, get_clipseg_transforms


def create_triptych(image, gt_mask, pred_mask, title="", save_path=None):
    """Create a triptych visualization: Image | Ground Truth | Prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(image)
    axes[1].imshow(gt_mask, alpha=0.5, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='Blues', vmin=0, vmax=1)
    axes[2].set_title("Prediction", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_triptychs(model, dataloader, processor, device, threshold, task_name, output_dir, num_samples=4, seed=42):
    """Generate triptych visualizations for random samples."""
    model.eval()
    
    random.seed(seed)
    np.random.seed(seed)
    
    all_samples = []
    
    print(f"\nCollecting samples for {task_name}...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Inference [{task_name}]"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
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
            preds = (probs > threshold).float()
            
            for i in range(images.shape[0]):
                # normalize image to [0,1] for matplotlib - CLIPSeg processor gives weird ranges
                img_np = images[i].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                gt_np = masks[i].cpu().numpy()
                pred_np = preds[i].cpu().numpy()
                
                # Make sure all arrays have the same shape for proper overlay
                # Image is (H, W, 3), masks should be (H, W)
                if img_np.shape[:2] != gt_np.shape:
                    print(f"Warning: Size mismatch - Image: {img_np.shape[:2]}, GT mask: {gt_np.shape}, resizing mask")
                    gt_np = cv2.resize(gt_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                if img_np.shape[:2] != pred_np.shape:
                    print(f"Warning: Size mismatch - Image: {img_np.shape[:2]}, Pred mask: {pred_np.shape}, resizing mask")
                    pred_np = cv2.resize(pred_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                all_samples.append({
                    'image': img_np,
                    'gt_mask': gt_np,
                    'pred_mask': pred_np,
                    'image_path': image_paths[i]
                })
    
    if len(all_samples) < num_samples:
        num_samples = len(all_samples)
        print(f"Warning: Only {num_samples} samples available")
    
    selected_indices = random.sample(range(len(all_samples)), num_samples)
    
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating {num_samples} triptychs for {task_name}...")
    for idx, sample_idx in enumerate(selected_indices):
        sample = all_samples[sample_idx]
        image_id = Path(sample['image_path']).stem
        
        title = f"{task_name} - Sample {idx+1}\nImage: {image_id}"
        save_path = figures_dir / f"{task_name.lower().replace(' ', '_')}_triptych_{idx+1}.png"
        
        create_triptych(
            sample['image'],
            sample['gt_mask'],
            sample['pred_mask'],
            title=title,
            save_path=save_path
        )
        
        print(f"  Saved: {save_path.name}")
    
    print(f"Triptychs saved to: {figures_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate triptych visualizations")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory with splits")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for visualizations")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of triptychs per task")
    parser.add_argument("--cracks_threshold", type=float, default=0.5, help="Threshold for cracks (use optimal from eval)")
    parser.add_argument("--drywall_threshold", type=float, default=0.5, help="Threshold for drywall (use optimal from eval)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection")
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
    
    print("\nLoading datasets...")
    
    splits_dir = Path(args.data_dir) / "splits"
    transform = get_clipseg_transforms(processor)
    
    cracks_test_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/cracks_test.json",
        prompt_text="segment crack",
        transform=transform,
        augment=False
    )
    
    drywall_test_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/drywall_test.json",
        prompt_text="segment taping area",
        transform=transform,
        augment=False
    )
    
    cracks_test_loader = DataLoader(cracks_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    drywall_test_loader = DataLoader(drywall_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print("\nGenerating visualizations...")
    
    print(f"\nThresholds:")
    print(f"  Cracks:  {args.cracks_threshold:.2f}")
    print(f"  Drywall: {args.drywall_threshold:.2f}")
    
    generate_triptychs(
        model, cracks_test_loader, processor, device,
        args.cracks_threshold, "Crack Detection",
        args.output_dir, args.num_samples, args.seed
    )
    
    generate_triptychs(
        model, drywall_test_loader, processor, device,
        args.drywall_threshold, "Drywall Taping Area",
        args.output_dir, args.num_samples, args.seed
    )
    
    print("\nVisualization complete!")
    print(f"Figures saved to: {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
