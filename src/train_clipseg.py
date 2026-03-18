import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from dataset import PromptedSegmentationDataset, get_clipseg_transforms


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def freeze_clip_encoder(model):
    """Freeze CLIP encoder, train decoder only."""
    for name, param in model.named_parameters():
        if name.startswith('clip.') or name.startswith('model.clip'):
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Frozen CLIP encoder")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def compute_dice_score(pred, target, threshold=0.5):
    """Compute Dice score for evaluation."""
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    
    return dice.item()


def train_epoch(model, dataloader, criterion, optimizer, processor, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        prompts = batch['prompt']
        
        optimizer.zero_grad()
        
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
        
        # Check for empty or invalid logits
        if logits.numel() == 0:
            print(f"Warning: Empty logits tensor in training, skipping batch")
            continue
        
        # Check tensor shape before processing
        if len(logits.shape) < 3:
            print(f"Warning: Invalid logits shape {logits.shape} in training, skipping batch")
            continue
        
        # Handle different output shapes
        if len(logits.shape) == 3:
            logits = logits.unsqueeze(1)  # CLIPSeg returns [B,H,W], need [B,1,H,W] for interpolate
        # CLIPSeg always returns [B,H,W], unsqueeze to [B,1,H,W] for interpolate
        
        logits = F.interpolate(logits, size=(640, 640), mode='bilinear', align_corners=False)
        logits = logits.squeeze(1)
        
        loss = criterion(logits, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        with torch.no_grad():
            dice = compute_dice_score(logits, masks)
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


def validate(model, dataloader, criterion, processor, device, epoch, split_name="Val"):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [{split_name}]")
        for batch in pbar:
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
            
            # Check for empty or invalid logits
            if logits.numel() == 0:
                print(f"Warning: Empty logits tensor, skipping batch")
                continue
            
            # Check tensor shape before processing
            if len(logits.shape) < 3:
                print(f"Warning: Invalid logits shape {logits.shape}, skipping batch")
                continue
            
            # Handle different output shapes
            if len(logits.shape) == 3:
                logits = logits.unsqueeze(1)  # CLIPSeg returns [B,H,W], need [B,1,H,W] for interpolate
            # CLIPSeg always returns [B,H,W], unsqueeze to [B,1,H,W] for interpolate
            
            logits = F.interpolate(logits, size=(640, 640), mode='bilinear', align_corners=False)
            logits = logits.squeeze(1)
            
            loss = criterion(logits, masks)
            dice = compute_dice_score(logits, masks)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train CLIPSeg for prompted segmentation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory with splits")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="BCE loss weight")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Dice loss weight")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args_temp = parser.parse_known_args()[0] if '--device' in os.sys.argv else None
    default_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=default_device, help="Device")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "mps" and torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading model and processor...")
    
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    model = freeze_clip_encoder(model)
    model = model.to(device)
    
    print("\nLoading datasets...")
    
    splits_dir = Path(args.data_dir) / "splits"
    
    transform = get_clipseg_transforms(processor)
    
    cracks_train_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/cracks_train.json",
        prompt_text="segment crack",
        transform=transform,
        augment=True
    )
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
    
    drywall_train_dataset = PromptedSegmentationDataset(
        f"{splits_dir}/drywall_train.json",
        prompt_text="segment taping area",
        transform=transform,
        augment=True
    )
    # drywall val set is only 202 images, so 50/50 val/test split keeps both usable
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
    
    combined_train = ConcatDataset([cracks_train_dataset, drywall_train_dataset])
    
    train_loader = DataLoader(combined_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    cracks_val_loader = DataLoader(cracks_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    drywall_val_loader = DataLoader(drywall_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    criterion = CombinedLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Combined training samples: {len(combined_train)}")
    print(f"Cracks validation samples: {len(cracks_val_dataset)}")
    print(f"Drywall validation samples: {len(drywall_val_dataset)}")
    
    history = {
        'train_loss': [], 'train_dice': [],
        'cracks_val_loss': [], 'cracks_val_dice': [],
        'drywall_val_loss': [], 'drywall_val_dice': [],
        'avg_val_dice': [],
        'lr': []
    }
    
    history_path = output_dir / "training_history.json"
    best_avg_dice = 0.0
    epochs_no_improve = 0
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch}/{args.epochs} | LR: {current_lr:.2e}")
        print(f"{'='*80}")
        
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, processor, device, epoch
        )
        
        print(f"\nTrain - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        
        print("\n--- Validation ---")
        cracks_val_loss, cracks_val_dice = validate(
            model, cracks_val_loader, criterion, processor, device, epoch, "Cracks Val"
        )
        drywall_val_loss, drywall_val_dice = validate(
            model, drywall_val_loader, criterion, processor, device, epoch, "Drywall Val"
        )
        
        print(f"Cracks Val  - Loss: {cracks_val_loss:.4f}, Dice: {cracks_val_dice:.4f}")
        print(f"Drywall Val - Loss: {drywall_val_loss:.4f}, Dice: {drywall_val_dice:.4f}")
        
        avg_val_dice = (cracks_val_dice + drywall_val_dice) / 2
        print(f"Average Val Dice: {avg_val_dice:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['cracks_val_loss'].append(cracks_val_loss)
        history['cracks_val_dice'].append(cracks_val_dice)
        history['drywall_val_loss'].append(drywall_val_loss)
        history['drywall_val_dice'].append(drywall_val_dice)
        history['avg_val_dice'].append(avg_val_dice)
        history['lr'].append(current_lr)
        
        if avg_val_dice > best_avg_dice:
            best_avg_dice = avg_val_dice
            epochs_no_improve = 0
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avg_val_dice': avg_val_dice,
                'history': history
            }, checkpoint_path)
            print(f"\n✓ Saved best model (avg val Dice: {avg_val_dice:.4f}) to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"\nNo improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        scheduler.step()
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best average validation Dice: {best_avg_dice:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
