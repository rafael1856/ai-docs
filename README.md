# Read PDF with images, analisys and decomposition



Multi-modal embeddings with PDF documents

Using [OpenClip multimodal embeddings](https://python.langchain.com/docs/integrations/text_embedding/open_clip).

It uses a larger model for better performance (set in `langchain_experimental.open_clip.py`).

el_name = "ViT-g-14"
ckpoint = "laion2b_s34b_b88k"



## Features

* Read local or online pdf files
* Extract images and texts
* Queries about images, answers with images and texts


## Quick start

1. 

## Requirements

It needs docker to start a vsdm vector database running in a docker
It consume CPU and memory for the vector database

## Upcoming Features

* Query loop
* Web interface, show the selected PDF, a query box and open result in a new windows.


### For more information

* [GitHub] (https://github.com/rafael1856/ai-docs)

### How to setup

1. Setup conda (or mamba) enviroment running: bin/setup.sh 
        or running 
        mamba create --name your_enviroment_name --file ../conf/c-requirements.txt -y

2. Add non conda libraries running: 
        pip install -r conf/p-requirements.txt

3. Create a config file in folder conf/system_config.json
```
{
    "db_name": "ai-docs",
    "db_user": "your_user",
    "db_password": "your_password",
    "windows_data_folder": "..\\data\\",
    "windows_log_file": "..\\app.log",
    "linux_data_folder": "/your_path/data/",
    "linux_log_file": "/your_path/logs/app.log",
    "schema": "ai-docs",
    "log_level": "INFO",
    "list_models": ["llama3", "mistral"]
}
```
pip install -U vdms langchain-experimental
pip install pdf2image "unstructured[all-docs]==0.10.19" pillow pydantic lxml open_clip_torch
pip uninstall torch torchvision
pip install torch==1.8.1 torchvision==0.9.1
pip install tesseract
sudo apt-get install tesseract-ocr

# folder structure
```
.
├── bin
│   └── setup.sh            # shell script, to install dependencies, configure environment variables
├── conf
│   ├── c-requirements.txt      # dependencies to install via mamba  
│   ├── p-requirements.txt      # dependencies to install via pip
│   └── system_config.json      # config file for app, db credentials and paths to data folder
├── data                        # data folder and results
│   ├── LCM_2020_1112
│       ├── LCM_2020_1112.pdf
│       ├── image.html
│       └── response.txt
├── docs                        # documentation, contains diagrams and other files for the project
│   ├── main-diagram2.dot
│   └── main-diagram.dot
├── LICENSE
├── logs
│   └── app.log
├── README.md
├── src
│   ├── assistent.py
│   ├── docs.py
│   ├── images.py
│   ├── logger_config.py
│   ├── main.py
│   ├── read_config.py
│   ├── run-docker.sh
│   └── stop-docker.sh
├── start.sh
├── TODO.md
└── utils
    └──db_utils.py
```


# Database
It is a basic sqLite database with one table about the documents, queries and results

### How to run

run: conda activate ai-docs
run : ./start.sh 


### For more information

* [GitHub](https://github.com/rafael1856/ai-docs

## Upcoming Features

- [ ] Save results in a DB   
- [ ] Add chatbot to talk about the pictures and text in the document

**Enjoy!**