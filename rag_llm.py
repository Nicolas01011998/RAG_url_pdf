from PIL import Image
import base64
import requests
import gradio as gr
import logging
import os
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Tuple, Optional, List, Dict, Any
from langchain.schema import Document
import requests

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen3:0.6b"
VLM_MODEL = "gemma3:4b"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Text splitter initialization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
    keep_separator=True
)

# Embeddings initialization
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

#
# FUNCTIONS FOR RAG PDF/URL
#

def clean_text(text: str) -> str:
    """
    Cleans the text extracted from the PDF by removing special characters, multiple spaces, and other elements that might interfere.
    """
    if not text:
        return ""
    
    # Removes control characters (non-printable ASCII, except newline)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Reduces multiple spaces to a single space
    text = re.sub(r' +', ' ', text)
    
    # Removes multiple consecutive line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()


def extract_section_titles(text: str) -> List[str]:
    """
    Extracts potential section titles from the text to improve context.
    Works with various document formats, not just scientific papers.
    """
    # Defines different patterns to detect titles in documents
    patterns = [
        r'^(?:\d+\.){1,3}\s*([A-Z][^.!?]*)',             # Format like "1.2.3 Title"
        r'^(?:[A-Z][A-Z\s]+)(?:\:|\s*\n)',               # Titles in uppercase
        r'^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})(?:\:|\s*\n)',  # Title Case
        r'(?:^|\n)(?:[A-Z][a-z]+\s?){1,7}(?:\:|\n)',      # Multiline capitalized titles
        r'(?:^|\n)(?:\*\*|__)(?:[^\*\n_]+)(?:\*\*|__)(?:\:|\n)',  # Markdown titles
    ]
    
    titles = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        # Filters out results that are too short
        titles.extend([m.strip() for m in matches if len(m.strip()) > 5])
    
    # Removes duplicates
    return list(set(titles))


def enhance_document_metadata(docs: List[Document]) -> List[Document]:
    """
    Enhances the metadata of documents extracted from the PDF to improve search.
    """
    enhanced_docs = []
    
    for i, doc in enumerate(docs):
        # Cleans the textual content of the page
        text = clean_text(doc.page_content)
        if not text:
            continue
        
        # Extracts section titles from the cleaned text
        section_titles = extract_section_titles(text)
        
        # Creates a new Document object with additional metadata
        enhanced_doc = Document(
            page_content=text,
            metadata={
                **doc.metadata,
                "section_titles": " | ".join(section_titles),
                "doc_id": i,
                "content_length": len(text)
            }
        )
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Loads a PDF using different loaders for better text extraction.
    Tries different extraction methods to ensure the highest text quality.
    """
    docs = []
    errors = []
    
    # First attempt with PyPDFium2Loader
    try:
        logging.info(f"Load PDF with PyPDFium2Loader: {pdf_path}")
        docs = PyPDFium2Loader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PyPDFium2Loader: {str(e)}")
    
    # Second attempt with PDFMinerLoader
    try:
        logging.info(f"Load PDF with PDFMinerLoader: {pdf_path}")
        docs = PDFMinerLoader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PDFMinerLoader: {str(e)}")
    
    # Last attempt with PyPDFLoader
    try:
        logging.info(f"Load PDF with PyPDFLoader: {pdf_path}")
        docs = PyPDFLoader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PyPDFLoader: {str(e)}")
    
    # If all loaders fail, raise an error with details
    raise ValueError(f"Unable to extract text from PDF.\nErrors:\n{'\n'.join(errors)}")


def load_url(url: str) -> List[Document]:
    """
    Loads content from a URL.
    """
    logging.info(f"Load URL: {url}")
    try:
        docs = WebBaseLoader(url).load()
        # Checks that there is valid content
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("Empty or invalid URL content")
        return docs
    except Exception as e:
        raise ValueError(f"Unable to load URL: {e}")


def create_vectorstore(url: Optional[str] = None, pdf_path: Optional[str] = None) -> Chroma:
    """
    Creates a vectorstore from documents loaded from URL and/or PDF.
    """
    docs = []
    error_messages = []
    
    # Load from URL, if provided
    if url:
        try:
            url_docs = load_url(url)
            docs += url_docs
        except Exception as e:
            error_messages.append(f"URL Error: {str(e)}")
    
    # Load from PDF, if provided
    if pdf_path:
        try:
            pdf_docs = load_pdf(pdf_path)
            docs += pdf_docs
        except Exception as e:
            error_messages.append(f"PDF Error: {str(e)}")
    
    # If no documents were loaded, raise an error
    if not docs:
        raise ValueError(f"Unable to create vectorstore:\n{'\n'.join(error_messages)}")
    
    # Splits the documents into chunks
    splits = text_splitter.split_documents(docs)
    
    # Adds extra metadata to chunks to improve traceability
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
        split.metadata["content_preview"] = split.page_content[:100] + "..."
    
    # Creates the Chroma vectorstore using embeddings
    return Chroma.from_documents(documents=splits, embedding=embeddings)

def call_llm(question: str, context: str) -> str:
    """
    Calls a local LLM model using the Ollama REST API with the given question and context.
    Returns the generated response or an error message.
    """
    url = "http://localhost:11434/api/generate"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    prompt = f"""
You are an expert assistant specializing in document analysis of all types.
Answer the following question, using only the information in the provided context.
If the answer cannot be found in the context, clearly state this limitation.
Organize information clearly and appropriately based on the document type and question.
If the question concerns:
- data or statistics: report the exact values cited in the document
- concepts or descriptive topics: organize your response logically
- procedures or instructions: list the steps clearly

Question: {question}

Context:
{context}
    """

    data = {
        "model": LLM_MODEL,  
        "prompt": prompt,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": 2048
        },
        "stream": False
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get("response", "No response content found.")
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error in response generation: {e}"



def save_uploaded_file(file_obj) -> str:
    """
    Saves the uploaded file to a temporary location and returns the path.
    """
    if not file_obj:
        return None
    
    # Returns the file path
    return file_obj.name


def rag_answer(
    url: str,
    pdf_file: Optional[gr.File],
    question: str,
    vectorstore: Optional[Dict[str, Any]] = None,
    history_id: str = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        # Checks if at least one valid input was provided
        has_url = bool(url and url.strip())
        has_pdf = bool(pdf_file and pdf_file.name)
        
        if not has_url and not has_pdf:
            return "Error: Provide at least one valid URL or PDF file.", vectorstore
        
        # Temporarily saves the PDF and checks if it is valid
        pdf_path = None
        if has_pdf:
            if not pdf_file.name.lower().endswith('.pdf'):
                return "Error: The uploaded file is not a valid PDF.", vectorstore
            
            pdf_path = save_uploaded_file(pdf_file)
            if os.path.getsize(pdf_path) > 50 * 1024 * 1024:
                return "Error: The PDF exceeds the maximum size of 50MB.", vectorstore
        
        # Determines if the vectorstore needs to be recreated
        current_id = f"{url}_{pdf_path}"
        if not vectorstore or vectorstore.get("id") != current_id:
            vs = create_vectorstore(url=url, pdf_path=pdf_path)
            vectorstore = {"store": vs, "id": current_id}
        
        # Prepares the query for semantic search
        search_query = question
        
        # Retrieves relevant documents using MMR
        retriever = vectorstore["store"].as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.7}
        )
        hits = retriever.invoke(search_query)
        
        if not hits:
            return "I found no relevant information to answer your question.", vectorstore
        
        # Builds the context to pass to the LLM model
        context_parts = []
        for i, doc in enumerate(hits):
            metadata = doc.metadata
            page_info = f"[Page {metadata.get('page', 'N/A')}]" if "page" in metadata else ""
            context_parts.append(f"\n--- EXCERPT {i+1} {page_info} ---\n{doc.page_content}")
        
        context = "\n".join(context_parts)
        
        # Calls the LLM model and returns the response
        answer = call_llm(question, context)
        
        return answer, vectorstore
    
    except Exception as e:
        return f"Error during processing: {str(e)}", vectorstore

#
# FUNCTIONS FOR IMAGE CAPTIONING
#

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image in base64 for sending to the model.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path) -> Tuple[str, str]:
    """
    Processes the uploaded image, resizing it if necessary,
    and converts it to base64.
    
    Note: image_path is already a file path when it comes from gr.Image(type="filepath")
    """
    if not image_path or not os.path.exists(image_path):
        raise ValueError("No image loaded or invalid path")
    
    # Checks the file size
    file_size = os.path.getsize(image_path)
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(f"The image is too big: {file_size/1024/1024:.1f}MB. Max allowed: 10MB")
    
    # Verifies that it is a valid image
    try:
        img = Image.open(image_path)
        img_format = img.format.lower() if img.format else "unknown"
        return image_path, img_format
    except Exception as e:
        raise ValueError(f"Invalid file or unsupported format: {e}")

def generate_caption(
    image_path: str,
    prompt_template: str,
    temperature: float = 0.3
) -> str:
    """
    Generates a caption for the uploaded image using the Ollama REST API.
    Uses the Gemma3 model and a base64-encoded image in the request payload.
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return "Error: Upload an image to generate a caption."
        
        # Process and validate image
        processed_path, img_format = process_image(image_path)
        
        # Encode image to base64
        base64_image = encode_image_to_base64(processed_path)
        
        # Default prompt if none provided
        if not prompt_template or prompt_template.strip() == "":
            prompt_template = "Describe in detail what you see in this image."
        
        # Prepare JSON payload
        url = "http://localhost:11434/api/chat"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = {
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_template,
                    "images": [base64_image]
                }
            ],
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": 2048
            },
            "stream": False
        }

        # Send the request
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", "No response content found.")
        else:
            return f"Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"Error during image processing or API call: {e}"
#
# GRADIO INTERFACE WITH MULTIPLE TABS
#

# Creation of the main interface with tabs
with gr.Blocks(theme=gr.themes.Soft(), title="Gemma3 Document & Image AI") as app:
    gr.Markdown("# 🧠 Gemma3 Document & Image AI")
    
    with gr.Tabs():
        # First tab: RAG Web + PDF
        with gr.TabItem("🔍 Document QA"):
            gr.Markdown("First, upload a URL or a PDF file.")
            gr.Markdown("Next, ask a question.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="URL", 
                        placeholder="https://..."
                    )
                    pdf_input = gr.File(
                        label="Upload a PDF", 
                        file_types=[".pdf"]
                    )
                    question_input = gr.Textbox(
                        label="Question", 
                        placeholder="Question...",
                        lines=2
                    )
                    submit_btn_rag = gr.Button("Ask", variant="primary")
                    
                with gr.Column(scale=3):
                    output_rag = gr.Textbox(
                        label="Answer", 
                        interactive=False, 
                        show_copy_button=True, 
                        lines=10
                    )
                    status_rag = gr.Textbox(
                        label="Status", 
                        interactive=False, 
                        visible=False
                    )
            
            gr.Examples(
                examples=[
                    ["https://it.wikipedia.org/wiki/Campionato_mondiale_di_calcio_2006", None, "Who won the World Cup in 2006?"],
                    ["https://aws.amazon.com/it/what-is/anomaly-detection/", None, "What is Anomaly Detection?"],
                    ["https://it.wikipedia.org/wiki/Leonardo_da_Vinci", None, "What are Leonardo's most famous works?"]
                ],
                inputs=[url_input, pdf_input, question_input],
                label="Examples of use"
            )
            
            # State for the vectorstore
            vectorstore_state = gr.State()
            history_id = gr.State("")
            
            submit_btn_rag.click(
                fn=rag_answer,
                inputs=[url_input, pdf_input, question_input, vectorstore_state, history_id],
                outputs=[output_rag, vectorstore_state],
                api_name="rag-qa"
            )
            
            pdf_input.upload(
                lambda: "PDF uploaded. Ready for processing.",
                outputs=status_rag
            )
            
        # Second tab: Image Captioning
        with gr.TabItem("🖼️ Image Captioning"):
            gr.Markdown('<h1 style="color: red">⚠️ ATTENTION: GEMMA3:4B REQUIRED ⚠️</h1>')
            gr.Markdown("---")
            gr.Markdown("Upload an image and get a detailed description generated by the Gemma3 model.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="filepath",
                        label="Upload an image",
                    )
                    prompt_input = gr.Textbox(
                        label="Prompt (optional)",
                        placeholder="Default: 'Describe in detail what you see in this image.'",
                        lines=2
                    )
                    submit_btn_img = gr.Button("Image Captioning", variant="primary")
                    
                with gr.Column(scale=1):
                    output_img = gr.Textbox(
                        label="Answer", 
                        interactive=False, 
                        show_copy_button=True, 
                        lines=10
                    )

            submit_btn_img.click(
                fn=generate_caption,
                inputs=[image_input, prompt_input],
                outputs=[output_img],
                api_name="generate-caption"
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False, show_error=True)