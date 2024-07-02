# Currently, we format the inputs using a `RunnableLambda` while we add image support to `ChatPromptTemplates`.
# 
# Our runnable follows the classic RAG flow - 
# * We first compute the context (both "texts" and "images" in this case) and the question (just a RunnablePassthrough here) 
# * Then we pass this into our prompt template, which is a custom function that formats the message for the llava model. 
# * And finally we parse the output as a string.
# 
# Here we are using Ollama to serve the Llava model.
# Please see [Ollama](https://python.langchain.com/docs/integrations/llms/ollama) for setup instructions.

import os
import os.path
# import base64
# import json
# import time
# from IPython.display import HTML, display
# import subprocess
# import requests

# from pathlib import Path
# from io import BytesIO
# from PIL import Image

# from langchain_community.vectorstores import VDMS
# from langchain_experimental.open_clip import OpenCLIPEmbeddings
# from langchain_community.vectorstores.vdms import VDMS_Client
# from unstructured.partition.pdf import partition_pdf
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from images import split_image_text_types

def prompt_func(data_dict):
    # Joining the context texts into a single string
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{data_dict['context']['images'][0]}"
            },
        }
        messages.append(image_message)

    # Adding the text message for analysis
    text_message = {
        "type": "text",
        "text": (
            "As an expert art critic and historian, your task is to analyze and interpret images, "
            "considering their historical and cultural significance. Alongside the images, you will be "
            "provided with related text to offer context. Both will be retrieved from a vectorstore based "
            "on user-input keywords. Please convert answers to english and use your extensive knowledge "
            "and analytical skills to provide a comprehensive summary that includes:\n"
            "- A detailed description of the visual elements in the image.\n"
            "- The historical and cultural context of the image.\n"
            "- An interpretation of the image's symbolism and meaning.\n"
            "- Connections between the image and the related text.\n\n"
            f"User-provided keywords: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """Multi-modal RAG chain"""

    # Multi-modal LLM
    llm_model = Ollama(
        verbose=True, temperature=0.5, model="llava", base_url="http://localhost:11434"
    )

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
        | llm_model
        | StrOutputParser()
    )

    return chain

