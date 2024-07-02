
# Connect to VDMS Vector Store
import os
import os.path
import base64
import json
import time
from IPython.display import HTML, display
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


# ## Start VDMS Server
# 
# Let's start a VDMS docker using port 55559 instead of default 55555. 
# Keep note of the port and hostname as this is needed for the vector store as it uses the VDMS Python client to connect to the server.

# subprocess.run(["docker", "run", "--rm", "-d", "-p", "55559:55555", "--name", "vdms_rag_nb", "intellabs/vdms:latest"])

try:
    vdms_client = VDMS_Client(port=55559)
except Exception as e:
    print(f"Error connecting to VDMS: {e}")
    exit()

# # TODO pass the pdf to analize via parameters
# try:
#     pdf_url = "https://www.loc.gov/lcm/pdf/LCM_2020_1112.pdf"
#     pdf_name = pdf_url.split("/")[-1]
#     pdf_path = os.path.join(DATA_FOLDER, pdf_name)
#     # pdf_path = "/home/rafael/dev/projects/data-samples/make-dome-house-spiritual-retriet.pdf"
#     with open(pdf_path, "wb") as f:
#         f.write(requests.get(pdf_url).content)
# except Exception as e:
#     print(f"Error downloading PDF: {e}")


# # Extract images, tables, and chunk text

# raw_pdf_elements = partition_pdf(
#     filename=pdf_path,
#     extract_images_in_pdf=True,
#     infer_table_structure=True,
#     chunking_strategy="by_title",
#     max_characters=4000,
#     new_after_n_chars=3800,
#     combine_text_under_n_chars=2000,
#     image_output_dir_path=DATA_FOLDER,
# )

datapath = str(DATA_FOLDER)

# # Categorize elements by type
# def categorize_elements(raw_pdf_elements):
#     """
#     Categorize extracted elements from a PDF into tables and texts.
#     raw_pdf_elements: List of unstructured.documents.elements
#     """
#     tables = []
#     texts = []
#     for element in raw_pdf_elements:
#         if "unstructured.documents.elements.Table" in str(type(element)):
#             tables.append(str(element))
#         elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
#             texts.append(str(element))  
#     # print(f"Found {len(texts)} texts and {len(tables)} tables")
#     return texts, tables

# pdf_path = download_doc(doc_name,doc_path=None, doc_url=None,):


# raw_pdf_elements = extract_images_texts_from_pdf(pdf_path)

# texts, tables = categorize_elements(raw_pdf_elements)
# print(f"Found {len(texts)} texts and {len(tables)} tables")

# print("Texts:",texts)


# vectorstore.set_embedding_function(OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k"))

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

exit()

# ## RAG
# `vectorstore.add_images` will store / retrieve images as base64 encoded string


# def resize_base64_image(base64_string, size=(128, 128)):
#     """
#     Resize an image encoded as a Base64 string.

#     Args:
#     base64_string (str): Base64 string of the original image.
#     size (tuple): Desired size of the image as (width, height).

#     Returns:
#     str: Base64 string of the resized image.
#     """
#     # Decode the Base64 string
#     img_data = base64.b64decode(base64_string)
#     img = Image.open(BytesIO(img_data))

#     # Resize the image
#     resized_img = img.resize(size, Image.LANCZOS)

#     # Save the resized image to a bytes buffer
#     buffered = BytesIO()
#     resized_img.save(buffered, format=img.format)

#     # Encode the resized image to Base64
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")


# def is_base64(s):
#     """Check if a string is Base64 encoded"""
#     try:
#         return base64.b64encode(base64.b64decode(s)) == s.encode()
#     except Exception:
#         return False


# def split_image_text_types(docs):
#     """Split numpy array images and texts"""
#     images = []
#     text = []
#     for doc in docs:
#         doc = doc.page_content  # Extract Document contents
#         if is_base64(doc):
#             # Resize image to avoid OAI server error
#             images.append(
#                 resize_base64_image(doc, size=(250, 250))
#             )  # base64 encoded str
#         else:
#             text.append(doc)
            
#     print({"images": images, "texts": text})
#     return {"images": images, "texts": text}




# # Currently, we format the inputs using a `RunnableLambda` while we add image support to `ChatPromptTemplates`.
# # 
# # Our runnable follows the classic RAG flow - 
# # * We first compute the context (both "texts" and "images" in this case) and the question (just a RunnablePassthrough here) 
# # * Then we pass this into our prompt template, which is a custom function that formats the message for the llava model. 
# # * And finally we parse the output as a string.
# # 
# # Here we are using Ollama to serve the Llava model.
# # Please see [Ollama](https://python.langchain.com/docs/integrations/llms/ollama) for setup instructions.





# def prompt_func(data_dict):
#     # Joining the context texts into a single string
#     formatted_texts = "\n".join(data_dict["context"]["texts"])
#     messages = []

#     # Adding image(s) to the messages if present
#     if data_dict["context"]["images"]:
#         image_message = {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
#             },
#         }
#         messages.append(image_message)

#     # Adding the text message for analysis
#     text_message = {
#         "type": "text",
#         "text": (
#             "As an expert art critic and historian, your task is to analyze and interpret images, "
#             "considering their historical and cultural significance. Alongside the images, you will be "
#             "provided with related text to offer context. Both will be retrieved from a vectorstore based "
#             "on user-input keywords. Please convert answers to english and use your extensive knowledge "
#             "and analytical skills to provide a comprehensive summary that includes:\n"
#             "- A detailed description of the visual elements in the image.\n"
#             "- The historical and cultural context of the image.\n"
#             "- An interpretation of the image's symbolism and meaning.\n"
#             "- Connections between the image and the related text.\n\n"
#             f"User-provided keywords: {data_dict['question']}\n\n"
#             "Text and / or tables:\n"
#             f"{formatted_texts}"
#         ),
#     }
#     messages.append(text_message)
#     return [HumanMessage(content=messages)]


# def multi_modal_rag_chain(retriever):
#     """Multi-modal RAG chain"""

#     # Multi-modal LLM
#     llm_model = Ollama(
#         verbose=True, temperature=0.5, model="llava", base_url="http://localhost:11434"
#     )

#     # RAG pipeline
#     chain = (
#         {
#             "context": retriever | RunnableLambda(split_image_text_types),
#             "question": RunnablePassthrough(),
#         }
#         | RunnableLambda(prompt_func)
#         | llm_model
#         | StrOutputParser()
#     )

#     return chain


# ## Test retrieval and run RAG






def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Save the image as an HTML file
    with open("image.html", "w") as file:
        file.write(image_html)
    # Display the image by rendering the HTML
    display(HTML(image_html))


# query = "Woman with children"
query = "viking cat"

docs = retriever.invoke(query, k=10)

for doc in docs:
    if is_base64(doc.page_content):
        plt_img_base64(doc.page_content)
    else:
        print(doc.page_content)

chain = multi_modal_rag_chain(retriever)
response = chain.invoke(query)
# Save the response
with open("response.txt", "w") as file:
    file.write(response)

print("Response saved successfully.")
print(response)




