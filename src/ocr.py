import easyocr

# Initialize with both English and Malay
reader = easyocr.Reader(['en', 'ms'])

def extract_text(image_array, min_confidence=0.5):
    # detail=1 forces EasyOCR to return the bounding box, text, and confidence
    results = reader.readtext(image_array, detail=1)

    extracted_data = []

    for (bbox, text, prob) in results:
        if prob >= min_confidence:
            # Convert numpy.int32 coordinates to standard Python ints
            clean_bbox = [[int(point[0]), int(point[1])] for point in bbox]
            
            extracted_data.append({
                "text": text,
                "bbox": clean_bbox,
                "confidence": float(prob) # Convert to standard float
            })

    return extracted_data