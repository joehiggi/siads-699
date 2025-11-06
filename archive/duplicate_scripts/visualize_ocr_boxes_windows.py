"""
Visualize OCR Bounding Boxes - Windows Compatible Version
Shows an image with bounding boxes from Tesseract OCR detection
(YOLO disabled to avoid PyTorch DLL issues on Windows)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import io
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract


def extract_image_from_row(image_data):
    """Extract PIL Image from parquet row data"""
    if isinstance(image_data, dict):
        image_bytes = image_data.get("bytes")
    else:
        image_bytes = image_data
    return Image.open(io.BytesIO(image_bytes))


def tesseract_ocr(image):
    """Use Tesseract to extract text from image"""
    try:
        if image.mode not in ["RGB", "L", "RGBA"]:
            image = image.convert("RGB")

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        full_text = pytesseract.image_to_string(image)

        confidences = [int(conf) for conf in data["conf"] if conf != "-1"]
        avg_confidence = np.mean(confidences) if confidences else 0

        words = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if int(data["conf"][i]) > 0:
                word = {
                    "text": data["text"][i],
                    "confidence": int(data["conf"][i]),
                    "bbox": [
                        data["left"][i],
                        data["top"][i],
                        data["left"][i] + data["width"][i],
                        data["top"][i] + data["height"][i],
                    ],
                }
                if word["text"].strip():
                    words.append(word)

        return {
            "full_text": full_text.strip(),
            "words": words,
            "avg_confidence": float(avg_confidence),
            "total_words": len(words),
        }
    except Exception as e:
        print(f"Error in Tesseract OCR: {e}")
        return {
            "full_text": "",
            "words": [],
            "avg_confidence": 0,
            "total_words": 0,
            "error": str(e),
        }


def visualize_image_with_boxes(image_data, tesseract_words=None, label=None, save_path=None):
    """Visualize an image with bounding boxes from OCR detection"""
    # Extract image
    image = extract_image_from_row(image_data)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Create figure with matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'Original Image\nLabel: {label}', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Image with Tesseract boxes
    axes[1].imshow(image)
    if tesseract_words:
        for word in tesseract_words:
            bbox = word['bbox']
            confidence = word['confidence']
            text = word['text']

            # Color based on confidence
            if confidence > 80:
                color = 'green'
            elif confidence > 60:
                color = 'blue'
            else:
                color = 'orange'

            # Draw rectangle
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1.5,
                edgecolor=color,
                facecolor='none',
                alpha=0.7
            )
            axes[1].add_patch(rect)

            # Add text label for high confidence words
            if confidence > 50:
                axes[1].text(
                    bbox[0], bbox[1] - 2,
                    f'{text[:15]}',
                    color=color,
                    fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
                )

    axes[1].set_title(f'Tesseract OCR Detection\n{len(tesseract_words) if tesseract_words else 0} words detected',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='green', label='High confidence (>80%)'),
        Patch(facecolor='none', edgecolor='blue', label='Medium confidence (60-80%)'),
        Patch(facecolor='none', edgecolor='orange', label='Low confidence (<60%)')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {save_path}")

    plt.close()
    return save_path


def visualize_from_parquet(parquet_path, image_index=0, save_output=True):
    """Load an image from parquet and visualize with OCR bounding boxes"""
    print(f"\nLoading image {image_index} from {Path(parquet_path).name}")

    # Load parquet
    df = pd.read_parquet(parquet_path)
    print(f"Total images in file: {len(df)}")

    if image_index >= len(df):
        print(f"Error: Image index {image_index} out of range (max: {len(df)-1})")
        return

    # Get image and label
    row = df.iloc[image_index]
    image_data = row['image']
    label = row.get('label', 'Unknown')

    print(f"Processing image with label: {label}")
    print("Running Tesseract OCR detection...")

    # Process image with Tesseract
    image = extract_image_from_row(image_data)
    tesseract_data = tesseract_ocr(image)
    tesseract_words = tesseract_data.get('words', [])

    print(f"\n{'='*60}")
    print(f"Detection Results:")
    print(f"  • Words detected: {len(tesseract_words)}")
    print(f"  • Average confidence: {tesseract_data.get('avg_confidence', 0):.1f}%")
    print(f"{'='*60}")

    if tesseract_data.get('full_text'):
        print(f"\nExtracted Text Preview:")
        print(f"  {tesseract_data['full_text'][:300]}")
        if len(tesseract_data['full_text']) > 300:
            print("  ...")

    # Visualize
    save_path = None
    if save_output:
        output_dir = Path("data/output/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ocr_viz_{Path(parquet_path).stem}_img{image_index}.png"
        save_path = output_dir / filename

    visualize_image_with_boxes(
        image_data=image_data,
        tesseract_words=tesseract_words,
        label=label,
        save_path=save_path
    )

    return save_path


def main():
    """Main execution"""
    # Default parameters
    parquet_file = r"data\raw\train-00000-of-00005.parquet"
    image_index = 0

    # Parse command line arguments
    if len(sys.argv) > 1:
        parquet_file = sys.argv[1]
    if len(sys.argv) > 2:
        image_index = int(sys.argv[2])

    print("=" * 70)
    print("OCR Bounding Box Visualization (Windows Version - Tesseract Only)")
    print("=" * 70)

    try:
        save_path = visualize_from_parquet(
            parquet_path=parquet_file,
            image_index=image_index,
            save_output=True
        )

        print("\n" + "=" * 70)
        print("✓ Visualization complete!")
        print("=" * 70)

        if save_path and save_path.exists():
            print(f"\nOpen the image: {save_path.absolute()}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: File not found - {e}")
        print("\nUsage:")
        print(f"  python {Path(__file__).name} [parquet_file] [image_index]")
        print("\nExample:")
        print(f"  python {Path(__file__).name} data\\raw\\train-00000-of-00005.parquet 5")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
