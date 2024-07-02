# AI Youtube summarizer - Ollama models

## Features

* Summarize a youtube video given a url via streamlit interface
* Summarize a youtube video given a url via command line
* Store records in a DB (sqlite), and retrive previous summaries

### How to setup

1. Setup the Sqlite database running: 
        python src/db_functions.py

2. Setup conda (or mamba) enviroment running: bin/setup.sh 
        or running 
        mamba create --name your_enviroment_name --file ../conf/c-requirements.txt -y

3. Add non conda libraries running: 
        pip install -r conf/p-requirements.txt

4. Create a config file in folder conf/system_config.json
```
{
    "db_name": "ai-youtube",
    "db_user": "your_user",
    "db_password": "your_password",
    "windows_data_folder": "..\\data\\",
    "windows_log_file": "..\\app.log",
    "linux_data_folder": "/your_path/data/",
    "linux_log_file": "/your_path/logs/app.log",
    "schema": "ai-youtube",
    "log_level": "DEBUG",
    "list_models": ["llama3", "phi3", "qwen2", "mistral"]
}
```

### How to run
```
run : ./start.sh 

or run : 
1. cd src
2. streamlit run app.py
```

# Database
It is a basic Sqlite database with one table about the videos summarized

# folder structure
```
├── bin
│   └── setup.sh           # Shell script for setting up the environment.
├── conf
│   ├── c-requirements.txt  # Conda environment requirements file.
│   ├── p-requirements.txt  # Pip package requirements file.
│   └── system_config.json  # Configuration settings for the application.
├── data
│   ├── ai-youtube.db       # SQLite database file for storing video summaries.
│   └── youtube-list.txt    # Text file listing YouTube video URLs.
├── docs                    # Documentation directory.
├── LICENSE                 # License file for the project.
├── logs
│   └── app.log             # Log file for application activities.
├── README.md               # This file, providing an overview and instructions.
├── src
│   ├── app.py              # Main application file.
│   ├── assistant.py        # File for the AI assistant functionality.
│   ├── db_functions.py     # Functions for interacting with the database.
│   ├── logger_config.py    # Configuration for the logging module.
├── start.sh                # Shell script to start the application.
├── TODO.md                 # List of tasks or features to implement.
```

### For more information

* [GitHub](https://github.com/rafael1856/ai-youtube)

## Upcoming Features

- [ ] Allow to change the AI prompt to summarize the transcript  
- [ ] Add translate Summary to other languages  
- [ ] Add chatbot to talk about the transcript  
- [ ] Summarize local videos  
- [ ] Make an Andrioid App  

**Enjoy!**