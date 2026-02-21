from transformers import pipeline

# Initialize the zero-shot classifier
nlp_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the categories for your budget predictor
CATEGORIES = [
    "Groceries and Supermarket", 
    "Food and Restaurant", 
    "Fuel and Gas Station", 
    "Utilities and Bills", 
    "Hardware and DIY", 
    "Stationery and Office Supplies", 
    "Pharmacy and Medical", 
    "Transportation and Parking", 
    "Electronics"
]

def categorize_receipt(merchant, raw_text_data):
    """
    Combines the merchant name and raw text, then uses NLP to predict the expense category.
    """

    merchant_name = merchant if merchant else "Unknown Store"
    raw_words = " ".join([item['text'] for item in raw_text_data[10:70]])

    text_to_classify = f"A receipt from {merchant_name}. The items purchased include: {raw_words}"
    
    try:

        result = nlp_classifier(text_to_classify, CATEGORIES)
        
        # The result returns lists sorted by highest probability
        top_category = result['labels'][0]
        confidence = result['scores'][0]
        
        return top_category, float(confidence)
        
    except Exception as e:
        print(f"Classification error: {e}")
        return "Uncategorized", 0.0