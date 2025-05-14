# RAG Web+PDF QA

A Question Answering (QA) system in Italian based on Retrieval-Augmented Generation (RAG). It can answer user queries by leveraging content from **PDF documents** and/or **web URLs**.

## Features

- 📄 Advanced PDF extraction (uses 3 loaders: `PyPDFium2`, `PDFMiner`, `PyPDF`)
- 🌐 Content loading from URLs (`WebBaseLoader`)
- ✂️ Automatic text splitting using `RecursiveCharacterTextSplitter`
- 🧠 Semantic embeddings via `OllamaEmbeddings` (`nomic-embed-text` model)
- 🛢️ Indexing and retrieval with `Chroma` vector store
- 💬 Answer generation with a local LLM (`gemma3`)
- 🗂️ Metadata enhancement by extracting section titles from documents
- 🌐 User interface built with [Gradio](https://gradio.app/)

## How It Works

- **Input**: The user provides a URL and/or a PDF file  
- **Extraction**: The content is cleaned, split, and converted into embeddings  
- **Indexing**: The text chunks are stored in a Chroma vectorstore  
- **Retrieval**: The system selects the most relevant documents for the question  
- **LLM**: A contextual answer is generated using the `gemma3` model

## Notes

- Only supports `.pdf` files smaller than 50MB  
- Answers are based solely on the provided content (Web + PDF)  
- Questions must be asked in **Italian**  
- The system clearly states when no relevant information is found in the context

## Project 
The project has the following structure:
```plaintext
.
├── rag_llm.py                # Code
├── requirements.txt          # Requirements 
└── README.md             

```

## Environment
### 1. Conda environment
To run the script, it is first necessary to create a Conda environment with Python 3.12 and a virtual environment to install the required libraries.
The Conda environment is used to set up the Python version.
To create a Conda environment, run the following command:
```
sudo apt apt-get install tesseract-ocr tesseract-ocr-ita libmagic-dev poppler-utils
conda create --name conda_python_3_12 python=3.12
```
To activate the Conda environment:
```
conda activate conda_python_3_12
```

### 2. Virtual environment
After activating the Conda environment, it is necessary to create a virtual environment to install the required libraries.
```
python3 -m venv venv
```
Now, activate the virtual environment:
```
source venv/bin/activate
```
Finally, install the necessary libraries using pip.
```
pip install -r requirements.txt
```

### 3. Ollama 
Download and install Ollama on Linux
```
curl -fsSL https://ollama.com/install.sh | sh
```

Check the service status:
```
systemctl status ollama
```

Pull models:
```
ollama pull nomic-embed-text
ollama pull gemma3:1b
```

## Run script
```
python rag_llm.py
```

## Example of Visualization 

This figure shows an example of the second pipeline is when object detection was performed only for the classes: book, potted plant, and vase.

<img src="Example.png" >
