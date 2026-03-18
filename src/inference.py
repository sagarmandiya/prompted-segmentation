import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import argparse


def load_model(checkpoint_path, device):
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
    return model, processor


def run_inference(image_path, prompt, model, processor, device, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0).unsqueeze(0)
        elif len(logits.shape) == 3:
            logits = logits.unsqueeze(1)
        
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        logits = logits.squeeze(0).squeeze(0)
        
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
    
    return np.array(image), mask.cpu().numpy(), probs.cpu().numpy()


def show_result(image, mask, prompt, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(image)
    ax1.set_title("Original", fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image)
    ax2.imshow(mask, alpha=0.6, cmap='jet')
    ax2.set_title(f'"{prompt}"', fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Segment images with text prompts")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--prompt", required=True, help='Text prompt like "segment crack"')
    parser.add_argument("--checkpoint", required=True, help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold (default: 0.5)")
    parser.add_argument("--output", help="Save path (optional)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Device: {device}\n")
    
    model, processor = load_model(args.checkpoint, device)
    
    print(f'Running: "{args.prompt}" on {args.image}')
    image, mask, probs = run_inference(args.image, args.prompt, model, processor, device, args.threshold)
    
    coverage = 100 * mask.mean()
    print(f"✓ Done! Segmented {coverage:.2f}% of image\n")
    
    show_result(image, mask, args.prompt, args.output)


if __name__ == "__main__":
    main()
