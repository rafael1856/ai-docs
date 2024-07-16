import os.path
import base64
# import json
import time
from IPython.display import HTML, display
# import subprocess
import requests
import shutil
import argparse

# from pathlib import Path
# from io import BytesIO
# from PIL import Image
# from langchain_community.vectorstores import VDMS
# from langchain_experimental.open_clip import OpenCLIPEmbeddings
# from langchain_community.vectorstores.vdms import VDMS_Client
# from unstructured.partition.pdf import partition_pdf
# from langchain_community.llms.ollama import Ollama
# from langchain_core.messages import HumanMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from docs import extract_images_texts_from_pdf, categorize_elements
from images import vectorize, is_base64, plt_img_base64
from assistent import multi_modal_rag_chain

from read_config import DATA_FOLDER

# import logging
from logger_config import setup_logger


logger = setup_logger('ai-docs-main')
logger.info("Starting main")

def generate_response(retr, query):
        # generate_response()
    docs = retr.invoke(query, k=10)
    logger.debug(f"query: {query}")

    for doc in docs:
        if is_base64(doc.page_content):
            logger.debug("Building html image")
            imghtml = plt_img_base64(doc.page_content)
        else:
            print(doc.page_content)

    chain = multi_modal_rag_chain(retr)
    logger.debug("multi_modal_rag_chain(retr)", chain)

    response = chain.invoke(query)

    return response, imghtml

def process_doc(folder: str, doc_name: str):
    
    logger.debug(f"process_doc --- folder: {folder} doc: {doc_name}")
    raw_pdf_elements = extract_images_texts_from_pdf(folder, doc_name)
    
    texts, tables = categorize_elements(raw_pdf_elements)
    logger.info(f"Found {len(texts)} texts and {len(tables)} tables")

    retr = vectorize(folder, texts) 
    query = "viking cat"

    # loop for questions and answers
    while True:
        respo, imgh = generate_response(retr, query)
        file_name = query.replace(" ", "-")

        # Save the response
        logger.debug(f"Writing response: {respo}")
        file_text = f"{folder}/{file_name}.txt"
        with open(file_text, "w") as file:
            file.write(respo)

        logger.info("Response saved successfully.")
        logger.debug(f"Response saved successfully: {respo}")

        # Save the image as an HTML file
        file_image = folder + "/image.html"
        with open(file_image, "w") as file:
            file.write(imgh)

        #TODO: review? Display the image by rendering the HTML
        # the html was generated at images.py
        display(HTML(imgh))
        
        query_input = input("Next quetion ? (type 'end' or empty to stop): ")
        if query_input.lower() in ["end", ""]:
            break
        query = query_input  # Update the query variable

def main():
    parser = argparse.ArgumentParser(description='Process a document.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--doc', help='Path to the local document')
    group.add_argument('--url', help='URL to the document')

    args = parser.parse_args()
    print()

    doc_path = args.doc
    print(f"Processing local document: {doc_path}")
    # create a folder for processing the document
    folder_name= os.path.splitext(os.path.basename(doc_path))[0]
    # print("folder_name:",folder_name)
    new_folder = os.path.join(DATA_FOLDER,folder_name)
    # print("new_folder:",new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        logger.info(f"Created folder: {new_folder}")

    if args.doc:
        # Copy the document to the new folder
        dest_path = os.path.join(new_folder, os.path.basename(doc_path))
        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}")
        else:
            shutil.copy2(doc_path, dest_path)
            logger.info(f"Copied document to: {dest_path}")

        doc_name = os.path.basename(doc_path)
    elif args.url:
        logger.info(f"Downloading document from URL: {args.url}")
        doc_name = args.url.split("/")[-1]
        logger.debug("doc_name:",doc_name)


        doc_path = os.path.join(new_folder, doc_name)       
        with open(doc_path, "wb") as f:
            f.write(requests.get(args.url).content)
    else:
        print("Please provide either --doc <path_to_the_doc> or --url <url>")
        return

    process_doc(new_folder, doc_path)

    print(f"\n Results are saved at folder:, {new_folder}\n")

if __name__ == "__main__":
    main()
