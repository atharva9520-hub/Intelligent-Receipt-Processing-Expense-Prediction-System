from transformers import pipeline

# Load Document QA model fine-tuned on receipts
nlp_pipeline = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa"
)

def extract_total(image_path):
    try:
        # First attempt: Ask in English
        result = nlp_pipeline(image_path, "What is the total amount?")
        
        # Second attempt: If it fails or is unconfident, ask in Malay
        if not result or result[0].get('score') < 0.5:
            result = nlp_pipeline(image_path, "What is the Jumlah?")
            
        if result and len(result) > 0:
            # Clean up the string (remove currency symbols like RM or $)
            clean_answer = result[0].get('answer').replace('RM', '').replace('$', '').strip()
            return clean_answer
            
    except Exception as e:
        print(f"Error during LayoutLM extraction: {e}")
        
    return None

def extract_merchant_and_date(image_path):
    # Extracting other highly useful fields for your classification step later
    merchant = nlp_pipeline(image_path, "What is the name of the store or merchant?")
    date = nlp_pipeline(image_path, "What is the date of the receipt?")
    
    return {
        "merchant": merchant[0].get('answer') if merchant else None,
        "date": date[0].get('answer') if date else None
    }