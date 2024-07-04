import os
import requests

from unstructured.partition.pdf import partition_pdf
from read_config import DATA_FOLDER, LOG_FILE

import logging
from logger_config import setup_logger
logger = setup_logger('ai-docs-docs')

def extract_images_texts_from_pdf(pdf_path):
    # Extract images, tables, and chunk text

    raw_pdf_elements = partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=DATA_FOLDER,
    )

    return raw_pdf_elements

def categorize_elements(raw_pdf_elements):
    """
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  
    logger.debug(f"Found {len(texts)} texts and {len(tables)} tables")
    return texts, tables