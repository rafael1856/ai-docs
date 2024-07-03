
# Connect to VDMS Vector Store
import os
import os.path
import base64
import json
import time
from IPython.display import display
import subprocess
import requests

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

from read_config import DATA_FOLDER, LOG_FILE


def vectorize(datapath, texts):

    # subprocess.run(["docker", "run", "--rm", "-d", "-p", "55559:55555", "--name", "vdms_rag_nb", "intellabs/vdms:latest"])

    try:
        vdms_client = VDMS_Client(port=55559)
        print("\nConnected to VDMS\n")
    except Exception as e:
        print(f"Error connecting to VDMS: {e}")
        exit()

    # datapath = str(DATA_FOLDER)


    # Create VDMS
    vectorstore = VDMS(
        client=vdms_client,
        collection_name="mm_rag_clip_photos",
        embedding=OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k"),
    )

    # Get image URIs with .jpg extension only
    image_uris = sorted(
        [
            os.path.join(datapath, image_name)
            for image_name in os.listdir(datapath)
            if image_name.endswith(".jpg")
        ]
    )

    # Add images
    if image_uris:
        vectorstore.add_images(uris=image_uris)

    # Add documents
    if texts:
        vectorstore.add_texts(texts=texts)

    # Make retriever
    retriever = vectorstore.as_retriever()

    return retriever




def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string.

    Args:
    base64_string (str): Base64 string of the original image.
    size (tuple): Desired size of the image as (width, height).

    Returns:
    str: Base64 string of the resized image.
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

def split_image_text_types(docs):
    """Split numpy array images and texts"""
    images = []
    text = []
    for doc in docs:
        # print("\nDOC=",doc)

        # print("\ntype of DOC=", type(doc).__name__)

        doc = doc.page_content  # Extract Document contents
        if is_base64(doc):
            # Resize image to avoid OAI server error
            images.append(
                resize_base64_image(doc, size=(250, 250))
            )  # base64 encoded str
        else:
            text.append(doc)
            
    # print({"images": images, "texts": text})
    return {"images": images, "texts": text}


from IPython.display import HTML, display

def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Save the image as an HTML file
    file_image = DATA_FOLDER + "/image.html"
    with open(file_image, "w") as file:
        file.write(image_html)

    return image_html

