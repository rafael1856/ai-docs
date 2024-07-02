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



