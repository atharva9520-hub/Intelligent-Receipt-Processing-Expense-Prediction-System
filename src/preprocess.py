import cv2
from pathlib import Path

def preprocess_image(image_path: Path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.medianBlur(gray, 3)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Save processed image
    output_dir = image_path.parent.parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_path.stem}_processed.jpg"
    cv2.imwrite(str(output_path), thresh)

    return thresh
