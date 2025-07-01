# Get Started
1. Move webshop directory and install requirements
```bash
pip install -r requirements.txt
```

2. According to the [instruction](https://github.com/princeton-nlp/WebShop?tab=readme-ov-file#-setup), download the data, setup the webserver and set environment variables 

* Note: host webserver and all data on localhost as provided web-link (http://3.83.245.205:3000) by Webshop is not valid.

<br>

> Note: Alternative setup procedure for steps 1 and 2:
> * Install full list of requirements, merged with WebShop
>
> ```bash
> pip install -r merged_requirements.txt
> ```
>
> * Execute SpaCy download
> ```bash
> python -m spacy download en_core_web_lg
> ```
>
> * Clone the WebShop repository from [here](https://github.com/princeton-nlp/WebShop)
> * Download all necessary data specified in WebShop 

<br>

3. Run (in a **separate** terminal) the webserver. (Note: Stop/terminate and restart the webserver before each experiment.)
```bash
cd <path to webshop server directory>
./run_dev.sh
```

4. Prepare OpenAI API key and put the key in ```OpenAI_api_key.txt```

5. Run RAP on Webshop
```bash
python main.py
```

Following are the hyper-parametes for RAP.

* num_trials: Number of recursive trials. Default is 3.
* num_steps: Maximum steps in one task. Default is 50.
* model: Model to be used in evaluation. Default is "gpt-3.5-turbo-instruct".
* output: Folder path to output logs and memory.
* emb_model: Embedding model to be used in evaluation. Default is "sentence-transformers/all-MiniLM-L6-v2".

Also, Python 3.8.10 is recommended for evaluation. (Note: Current hard requirement of Python 3.8 due to incompatibility issues of pyserini/nmslib with python >= 3.9)