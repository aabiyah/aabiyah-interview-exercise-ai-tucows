from typing import List, Dict
import logging
import json
from config import DATA_DIR


def load_all_faqs() -> List[Dict]:
    # Combining FAQs from multiple JSON files into a single list.
    faq_files = {
        "domain_management": DATA_DIR / "domain_management.json",
        "renewals_and_redemptions": DATA_DIR / "renewals_and_redemptions.json",
        "transfers": DATA_DIR / "transfers.json",
        "data_use": DATA_DIR / "data_use_information.json",
        "top_questions": DATA_DIR / "top_questions.json"
    }

    all_faqs = []

    for category, file_path in faq_files.items():
        if not file_path.exists():
            logging.warning(f"{file_path} does not exist. Skipping.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        faqs = data if isinstance(data, list) else data.get('faqs', [])

        for faq in faqs:
            all_faqs.append({
                "question": faq.get("question", ""),
                "answer": faq.get("answer", ""),
                # removed "category": category,
                "related_links": faq.get("related_links", [])
            })

    logging.info(f"Loaded {len(all_faqs)} FAQs from {len(faq_files)} files")
    return all_faqs

def prepare_faq_texts(faqs: List[Dict]) -> List[str]:
    # Combining question and answer fields from each FAQ into a single text block for embedding generation.
    texts = []
    for faq in faqs:
        combined = f"Question: {faq['question']}\n\nAnswer: {faq['answer']}"
        texts.append(combined)

    return texts