import os.path
import base64
import json
import time
from IPython.display import HTML, display
import subprocess
import requests
import shutil
import argparse

from pathlib import Path
from io import BytesIO
from PIL import Image
from langchain_community.vectorstores import VDMS
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores.vdms import VDMS_Client
from unstructured.partition.pdf import partition_pdf
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from docs import extract_images_texts_from_pdf, categorize_elements
from images import vectorize, is_base64, plt_img_base64
from assistent import multi_modal_rag_chain


# from vector_store import vectorize

from read_config import DATA_FOLDER, LOG_FILE

import logging
from logger_config import setup_logger

print("Starting AI-Docs...")
logger = setup_logger('ai-docs')


def process_doc(doc_name: str):
        
    raw_pdf_elements = extract_images_texts_from_pdf(doc_name)
    texts, tables = categorize_elements(raw_pdf_elements)
    logger.info(f"Found {len(texts)} texts and {len(tables)} tables")
    # logger.debug("\n\n Texts:", texts)
    retr = vectorize(DATA_FOLDER, texts)
    
    # TODO loop for questions and answers
    
    # query = "Woman with children"
    query = "viking cat"

    # generate_response()
    docs = retr.invoke(query, k=10)
    logger.debug("query")

    for doc in docs:
        if is_base64(doc.page_content):
            logger.debug("making html image")
            imghtml = plt_img_base64(doc.page_content)
        else:
            print(doc.page_content)

    chain = multi_modal_rag_chain(retr)
    logger.debug("multi_modal_rag_chain(retr)", chain)

    response = chain.invoke(query)

    # Save the response
    logger.debug(f"Writing response: {response}")
    file_text = DATA_FOLDER  + "/response.txt"
    with open(file_text, "w") as file:
        file.write(response)

    print("Response saved successfully.")
    print(response)

    # Display the image by rendering the HTML
    display(HTML(imghtml))

def main():
    parser = argparse.ArgumentParser(description='Process a document.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--doc', help='Path to the local document')
    group.add_argument('--url', help='URL to the document')

    args = parser.parse_args()
    print()
    if args.doc:
        doc_path = args.doc
        print(f"Processing local document: {doc_path}")
    
        dest_path = os.path.join(DATA_FOLDER, os.path.basename(doc_path))
        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}")
        else:
            shutil.copy2(doc_path, dest_path)
            logger.info(f"Copied document to: {dest_path}")

        doc_name = os.path.basename(doc_path)
    elif args.url:
        print(f"Downloading document from URL: {args.url}")
        # Implement downloading logic here if necessary
        doc_name = args.url.split("/")[-1]
        logger.debug("doc_name:",doc_name)
        doc_path = os.path.join(DATA_FOLDER, doc_name)       
        with open(doc_path, "wb") as f:
            f.write(requests.get(args.url).content)
    else:
        print("Please provide either --doc or --url.")
        return

    process_doc(doc_path)



if __name__ == "__main__":
    main()