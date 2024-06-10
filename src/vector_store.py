import os

from langchain_community.vectorstores import VDMS
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores.vdms import VDMS_Client

# - running from the script for now....try again to run from here
# subprocess.run(["docker", "run", "--rm", "-d", "-p", "55559:55555", "--name", "vdms_rag_nb", "intellabs/vdms:latest"])
try:
    vdms_client = VDMS_Client(port=55559)
except Exception as e:
    print(f"Error connecting to VDMS: {e}")
    exit()



def vectorize(datapath: str, texts: list[str] = None):

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