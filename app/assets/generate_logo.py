"""Helper script to generate a medical-grade logo for the Streamlit dashboard."""

from pathlib import Path
from PIL import Image, ImageDraw


def generate_logo():
    # Create a 200x200 RGB image with a dark-blue background
    img = Image.new("RGBA", (200, 200), color=(44, 62, 80, 255))
    draw = ImageDraw.Draw(img)

    # Draw a stylized medical cross in white/teal
    # Vertical bar
    draw.rectangle([85, 30, 115, 170], fill=(24, 188, 156, 255))
    # Horizontal bar
    draw.rectangle([30, 85, 170, 115], fill=(24, 188, 156, 255))

    # Draw a inner circle representing scan zone
    draw.ellipse([60, 60, 140, 140], outline=(255, 255, 255, 255), width=4)

    # Save to app/assets/logo.png
    assets_dir = Path(__file__).resolve().parent
    assets_dir.mkdir(parents=True, exist_ok=True)
    img.save(assets_dir / "logo.png")
    print(f"Logo generated at {assets_dir / 'logo.png'}")


if __name__ == "__main__":
    generate_logo()
