import os
import requests

from unstructured.partition.pdf import partition_pdf
from read_config import DATA_FOLDER, LOG_FILE
import logging
from logger_config import setup_logger
logger = setup_logger('ai-docs')


# def download_doc(doc_name,doc_path=None, doc_url=None,):

#     doc_found = False
#     try:
#         if doc_path != None:
#             doc_name = doc_url.split("/")[-1]
#             logger.debug("doc_name:",doc_name)
#             doc_path = os.path.join(DATA_FOLDER, doc_name)       
#             with open(doc_path, "wb") as f:
#                 f.write(requests.get(doc_url).content)
#             doc_found = True
#         else:
#             # look for the file in DATA_FOLDER
#             # if no file in DATA_FOLDER
#             pass

#         if (doc_found == False  and doc_url != None ):
#             doc_name = doc_url.split("/")[-1]
#             doc_path = os.path.join(DATA_FOLDER, doc_name)       
#             with open(doc_path, "wb") as f:
#                 f.write(doc_path)
#             logger.info(f"doc downloaded successfully as {doc_name}")
#         return doc_path   
#     except Exception as e:
#         logger.error(f"Error downloading doc: {e}")

# # Example usage
# doc_url = "https://www.loc.gov/lcm/pdf/LCM_2020_1112.pdf"
# doc_name = "LCM_2020_1112.pdf"
# download_doc(doc_url, doc_name)


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
    # print(f"Found {len(texts)} texts and {len(tables)} tables")
    return texts, tables