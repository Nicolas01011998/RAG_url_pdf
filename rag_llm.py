'''
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import gradio as gr
import logging
import os
import tempfile
import re
from typing import Tuple, Optional, List, Dict, Any
from langchain.schema import Document


# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Costanti
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:latest"

# Inizializzazione del text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
    keep_separator=True
)

# Inizializzazione degli embeddings
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

def clean_text(text: str) -> str:
    """
    Pulisce il testo estratto dal PDF rimuovendo caratteri speciali, spazi multipli e altri elementi che potrebbero interferire.
    """
    if not text:
        return ""
    
    # Rimuove caratteri di controllo (ASCII non stampabili, eccetto newline)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Riduce gli spazi multipli a un singolo spazio
    text = re.sub(r' +', ' ', text)
    
    # Rimuove interruzioni di riga multiple consecutive
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()


def extract_section_titles(text: str) -> List[str]:
    """
    Estrae potenziali titoli di sezione dal testo per migliorare il contesto.
    Funziona con diversi formati di documenti, non solo paper scientifici.
    """
    # Definisce diversi pattern per intercettare titoli nei documenti
    patterns = [
        r'^(?:\d+\.){1,3}\s*([A-Z][^.!?]*)',             # Formato tipo "1.2.3 Titolo"
        r'^(?:[A-Z][A-Z\s]+)(?:\:|\s*\n)',               # Titoli in maiuscolo
        r'^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})(?:\:|\s*\n)',  # Title Case
        r'(?:^|\n)(?:[A-Z][a-z]+\s?){1,7}(?:\:|\n)',      # Titoli capitalizzati multilinea
        r'(?:^|\n)(?:\*\*|__)(?:[^\*\n_]+)(?:\*\*|__)(?:\:|\n)',  # Titoli markdown
    ]
    
    titles = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        # Filtra i risultati troppo brevi
        titles.extend([m.strip() for m in matches if len(m.strip()) > 5])
    
    # Rimuove duplicati
    return list(set(titles))


def enhance_document_metadata(docs: List[Document]) -> List[Document]:
    """
    Arricchisce i metadati dei documenti estratti dal PDF per migliorare la ricerca.
    """
    enhanced_docs = []
    
    for i, doc in enumerate(docs):
        # Pulisce il contenuto testuale della pagina
        text = clean_text(doc.page_content)
        if not text:
            continue
        
        # Estrae titoli di sezione dal testo pulito
        section_titles = extract_section_titles(text)
        
        # Crea un nuovo oggetto Document con metadati aggiuntivi
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
    Carica un PDF utilizzando diversi loader per una migliore estrazione del testo.
    Prova con diversi metodi di estrazione per assicurare la massima qualità del testo.
    """
    docs = []
    errors = []
    
    # Primo tentativo con PyPDFium2Loader
    try:
        logging.info(f"Caricamento PDF con PyPDFium2Loader: {pdf_path}")
        docs = PyPDFium2Loader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PyPDFium2Loader: {str(e)}")
    
    # Secondo tentativo con PDFMinerLoader
    try:
        logging.info(f"Caricamento PDF con PDFMinerLoader: {pdf_path}")
        docs = PDFMinerLoader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PDFMinerLoader: {str(e)}")
    
    # Ultimo tentativo con PyPDFLoader
    try:
        logging.info(f"Caricamento PDF con PyPDFLoader: {pdf_path}")
        docs = PyPDFLoader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PyPDFLoader: {str(e)}")
    
    # Se tutti i loader falliscono, solleva un errore con dettagli
    raise ValueError(f"Impossibile estrarre testo dal PDF.\nErrori:\n{'\n'.join(errors)}")


def load_url(url: str) -> List[Document]:
    """
    Carica il contenuto da un URL.
    """
    logging.info(f"Caricamento URL: {url}")
    try:
        docs = WebBaseLoader(url).load()
        # Controlla che ci sia contenuto valido
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("Contenuto URL vuoto o non valido")
        return docs
    except Exception as e:
        raise ValueError(f"Impossibile caricare l'URL: {e}")


def create_vectorstore(url: Optional[str] = None, pdf_path: Optional[str] = None) -> Chroma:
    """
    Crea un vectorstore dai documenti caricati da URL e/o PDF.
    """
    docs = []
    error_messages = []
    
    # Carica da URL, se presente
    if url:
        try:
            url_docs = load_url(url)
            docs += url_docs
        except Exception as e:
            error_messages.append(f"Errore URL: {str(e)}")
    
    # Carica da PDF, se presente
    if pdf_path:
        try:
            pdf_docs = load_pdf(pdf_path)
            docs += pdf_docs
        except Exception as e:
            error_messages.append(f"Errore PDF: {str(e)}")
    
    # Se nessun documento è stato caricato, solleva errore
    if not docs:
        raise ValueError(f"Impossibile creare il vectorstore:\n{'\n'.join(error_messages)}")
    
    # Divide i documenti in chunk
    splits = text_splitter.split_documents(docs)
    
    # Aggiunge metadati extra ai chunk per migliorare la tracciabilità
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
        split.metadata["content_preview"] = split.page_content[:100] + "..."
    
    # Crea il vectorstore Chroma usando gli embeddings
    return Chroma.from_documents(documents=splits, embedding=embeddings)


def call_llm(question: str, context: str) -> str:
    """
    Chiama il modello LLM con la domanda e il contesto.
    """
    prompt = f"""
    Sei un assistente esperto in analisi di documenti di ogni tipo.
    Rispondi in italiano alla seguente domanda basandoti esclusivamente sul contesto fornito.
    Se la risposta non è presente nel contesto, dillo chiaramente.
    
    Organizza le informazioni in modo chiaro e strutturato in base al tipo di documento e domanda.
    Se la domanda riguarda:
    - dati o statistiche: riporta i valori esatti citati nel documento
    - concetti o argomenti descrittivi: organizza la risposta in modo logico
    - procedure o istruzioni: elenca i passaggi in modo chiaro
    
    Domanda: {question}
    
    Contesto:
    {context}
    """
    try:
        # Chiama il modello con parametri ottimizzati per risposte accurate
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                'temperature': 0.2,
                'top_p': 0.9,
                'num_ctx': 8192,
                'num_predict': 2048
            }
        )
        return resp['message']['content']
    except Exception as e:
        return f"Errore nella generazione della risposta: {e}"


def save_uploaded_file(file_obj) -> str:
    """
    Salva il file caricato in una posizione temporanea e restituisce il percorso.
    """
    if not file_obj:
        return None
    
    # Crea una directory temporanea e restituisce il percorso completo del file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, os.path.basename(file_obj.name))
    
    return file_obj.name


def rag_answer(
    url: str,
    pdf_file: Optional[gr.File],
    question: str,
    vectorstore: Optional[Dict[str, Any]] = None,
    history_id: str = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        # Verifica se è stato fornito almeno un input valido
        has_url = bool(url and url.strip())
        has_pdf = bool(pdf_file and pdf_file.name)
        
        if not has_url and not has_pdf:
            return "Errore: Fornisci almeno un URL valido o un file PDF.", vectorstore
        
        # Salva temporaneamente il PDF e verifica che sia valido
        pdf_path = None
        if has_pdf:
            if not pdf_file.name.lower().endswith('.pdf'):
                return "Errore: Il file caricato non è un PDF valido.", vectorstore
            
            pdf_path = save_uploaded_file(pdf_file)
            if os.path.getsize(pdf_path) > 50 * 1024 * 1024:
                return "Errore: Il PDF supera la dimensione massima di 50MB.", vectorstore
        
        # Determina se è necessario ricreare il vectorstore
        current_id = f"{url}_{pdf_path}"
        if not vectorstore or vectorstore.get("id") != current_id:
            vs = create_vectorstore(url=url, pdf_path=pdf_path)
            vectorstore = {"store": vs, "id": current_id}
        
        # Prepara la query per la ricerca semantica
        search_query = question
        
        # Recupera documenti rilevanti usando MMR
        retriever = vectorstore["store"].as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.7}
        )
        hits = retriever.invoke(search_query)
        
        if not hits:
            return "Non ho trovato informazioni rilevanti per rispondere alla tua domanda.", vectorstore
        
        # Costruisce il contesto da passare al modello LLM
        context_parts = []
        for i, doc in enumerate(hits):
            metadata = doc.metadata
            page_info = f"[Pagina {metadata.get('page', 'N/A')}]" if "page" in metadata else ""
            context_parts.append(f"\n--- ESTRATTO {i+1} {page_info} ---\n{doc.page_content}")
        
        context = "\n".join(context_parts)
        
        # Chiama il modello LLM e restituisce la risposta
        answer = call_llm(question, context)
        
        return answer, vectorstore
    
    except Exception as e:
        return f"Errore durante l'elaborazione: {str(e)}", vectorstore

# Interfaccia Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Web+PDF QA") as app:
    gr.Markdown("# 🔍 RAG Web+PDF QA")
    gr.Markdown("Inserisci un URL e/o carica un PDF, poi poni la tua domanda.")
    
    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="URL", 
                placeholder="https://..."
            )
            pdf_input = gr.File(
                label="Carica un PDF", 
                file_types=[".pdf"]
            )
            question_input = gr.Textbox(
                label="Domanda", 
                placeholder="Cosa vorresti sapere?",
                lines=2
            )
            submit_btn = gr.Button("Chiedi", variant="primary")
            
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Risposta", 
                interactive=False, 
                show_copy_button=True, 
                lines=10
            )
            status = gr.Textbox(
                label="Stato", 
                interactive=False, 
                visible=False
            )
    
    gr.Examples(
        examples=[
            ["https://it.wikipedia.org/wiki/Campionato_mondiale_di_calcio_2006", None, "Chi vinse i mondiali di calcio nel 2006?"],
            ["https://it.wikipedia.org/wiki/Divina_Commedia", None, "Quali sono i tre regni visitati da Dante?"],
            ["https://it.wikipedia.org/wiki/Leonardo_da_Vinci", None, "Quali sono le opere più famose di Leonardo?"]
        ],
        inputs=[url_input, pdf_input, question_input],
        label="Esempi di utilizzo"
    )
    
    # Stato per il vectorstore
    vectorstore_state = gr.State()
    history_id = gr.State("")
    
    submit_btn.click(
        fn=rag_answer,
        inputs=[url_input, pdf_input, question_input, vectorstore_state, history_id],
        outputs=[output, vectorstore_state],
        api_name="rag-qa"
    )
    
    pdf_input.upload(
        lambda: "PDF caricato. Pronto per l'elaborazione.",
        outputs=status
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False, show_error=True)
'''

from PIL import Image
import io
import base64
import ollama
import gradio as gr
import logging
import os
import tempfile
import re
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader, PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Tuple, Optional, List, Dict, Any
from langchain.schema import Document

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Costanti
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:latest"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# Inizializzazione del text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""],
    keep_separator=True
)

# Inizializzazione degli embeddings
embeddings = OllamaEmbeddings(model=EMBED_MODEL)

#
# FUNZIONI PER RAG PDF/URL
#

def clean_text(text: str) -> str:
    """
    Pulisce il testo estratto dal PDF rimuovendo caratteri speciali, spazi multipli e altri elementi che potrebbero interferire.
    """
    if not text:
        return ""
    
    # Rimuove caratteri di controllo (ASCII non stampabili, eccetto newline)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Riduce gli spazi multipli a un singolo spazio
    text = re.sub(r' +', ' ', text)
    
    # Rimuove interruzioni di riga multiple consecutive
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()


def extract_section_titles(text: str) -> List[str]:
    """
    Estrae potenziali titoli di sezione dal testo per migliorare il contesto.
    Funziona con diversi formati di documenti, non solo paper scientifici.
    """
    # Definisce diversi pattern per intercettare titoli nei documenti
    patterns = [
        r'^(?:\d+\.){1,3}\s*([A-Z][^.!?]*)',             # Formato tipo "1.2.3 Titolo"
        r'^(?:[A-Z][A-Z\s]+)(?:\:|\s*\n)',               # Titoli in maiuscolo
        r'^(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})(?:\:|\s*\n)',  # Title Case
        r'(?:^|\n)(?:[A-Z][a-z]+\s?){1,7}(?:\:|\n)',      # Titoli capitalizzati multilinea
        r'(?:^|\n)(?:\*\*|__)(?:[^\*\n_]+)(?:\*\*|__)(?:\:|\n)',  # Titoli markdown
    ]
    
    titles = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        # Filtra i risultati troppo brevi
        titles.extend([m.strip() for m in matches if len(m.strip()) > 5])
    
    # Rimuove duplicati
    return list(set(titles))


def enhance_document_metadata(docs: List[Document]) -> List[Document]:
    """
    Arricchisce i metadati dei documenti estratti dal PDF per migliorare la ricerca.
    """
    enhanced_docs = []
    
    for i, doc in enumerate(docs):
        # Pulisce il contenuto testuale della pagina
        text = clean_text(doc.page_content)
        if not text:
            continue
        
        # Estrae titoli di sezione dal testo pulito
        section_titles = extract_section_titles(text)
        
        # Crea un nuovo oggetto Document con metadati aggiuntivi
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
    Carica un PDF utilizzando diversi loader per una migliore estrazione del testo.
    Prova con diversi metodi di estrazione per assicurare la massima qualità del testo.
    """
    docs = []
    errors = []
    
    # Primo tentativo con PyPDFium2Loader
    try:
        logging.info(f"Load PDF with PyPDFium2Loader: {pdf_path}")
        docs = PyPDFium2Loader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PyPDFium2Loader: {str(e)}")
    
    # Secondo tentativo con PDFMinerLoader
    try:
        logging.info(f"Load PDF with PDFMinerLoader: {pdf_path}")
        docs = PDFMinerLoader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PDFMinerLoader: {str(e)}")
    
    # Ultimo tentativo con PyPDFLoader
    try:
        logging.info(f"Load PDF with PyPDFLoader: {pdf_path}")
        docs = PyPDFLoader(pdf_path).load()
        if docs and any(doc.page_content.strip() for doc in docs):
            return enhance_document_metadata(docs)
    except Exception as e:
        errors.append(f"PyPDFLoader: {str(e)}")
    
    # Se tutti i loader falliscono, solleva un errore con dettagli
    raise ValueError(f"Unable to extract text from PDF.\nErrors:\n{'\n'.join(errors)}")


def load_url(url: str) -> List[Document]:
    """
    Carica il contenuto da un URL.
    """
    logging.info(f"Load URL: {url}")
    try:
        docs = WebBaseLoader(url).load()
        # Controlla che ci sia contenuto valido
        if not docs or not any(doc.page_content.strip() for doc in docs):
            raise ValueError("Empty or invalid URL content")
        return docs
    except Exception as e:
        raise ValueError(f"Unable to load URL: {e}")


def create_vectorstore(url: Optional[str] = None, pdf_path: Optional[str] = None) -> Chroma:
    """
    Crea un vectorstore dai documenti caricati da URL e/o PDF.
    """
    docs = []
    error_messages = []
    
    # Carica da URL, se presente
    if url:
        try:
            url_docs = load_url(url)
            docs += url_docs
        except Exception as e:
            error_messages.append(f"URL Error: {str(e)}")
    
    # Carica da PDF, se presente
    if pdf_path:
        try:
            pdf_docs = load_pdf(pdf_path)
            docs += pdf_docs
        except Exception as e:
            error_messages.append(f"PDF Error: {str(e)}")
    
    # Se nessun documento è stato caricato, solleva errore
    if not docs:
        raise ValueError(f"Unable to create vectorstore:\n{'\n'.join(error_messages)}")
    
    # Divide i documenti in chunk
    splits = text_splitter.split_documents(docs)
    
    # Aggiunge metadati extra ai chunk per migliorare la tracciabilità
    for i, split in enumerate(splits):
        split.metadata["chunk_id"] = i
        split.metadata["content_preview"] = split.page_content[:100] + "..."
    
    # Crea il vectorstore Chroma usando gli embeddings
    return Chroma.from_documents(documents=splits, embedding=embeddings)


def call_llm(question: str, context: str) -> str:
    """
    Chiama il modello LLM con la domanda e il contesto.
    """
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
    try:
        # Chiama il modello con parametri ottimizzati per risposte accurate
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                'temperature': 0.3,
                'top_p': 0.9,
                'num_ctx': 8192,
                'num_predict': 2048
            }
        )
        return resp['message']['content']
    except Exception as e:
        return f"Error in response generation: {e}"


def save_uploaded_file(file_obj) -> str:
    """
    Salva il file caricato in una posizione temporanea e restituisce il percorso.
    """
    if not file_obj:
        return None
    
    # Restituisce il percorso del file
    return file_obj.name


def rag_answer(
    url: str,
    pdf_file: Optional[gr.File],
    question: str,
    vectorstore: Optional[Dict[str, Any]] = None,
    history_id: str = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        # Verifica se è stato fornito almeno un input valido
        has_url = bool(url and url.strip())
        has_pdf = bool(pdf_file and pdf_file.name)
        
        if not has_url and not has_pdf:
            return "Error: Provide at least one valid URL or PDF file.", vectorstore
        
        # Salva temporaneamente il PDF e verifica che sia valido
        pdf_path = None
        if has_pdf:
            if not pdf_file.name.lower().endswith('.pdf'):
                return "Error: The uploaded file is not a valid PDF.", vectorstore
            
            pdf_path = save_uploaded_file(pdf_file)
            if os.path.getsize(pdf_path) > 50 * 1024 * 1024:
                return "Error: The PDF exceeds the maximum size of 50MB.", vectorstore
        
        # Determina se è necessario ricreare il vectorstore
        current_id = f"{url}_{pdf_path}"
        if not vectorstore or vectorstore.get("id") != current_id:
            vs = create_vectorstore(url=url, pdf_path=pdf_path)
            vectorstore = {"store": vs, "id": current_id}
        
        # Prepara la query per la ricerca semantica
        search_query = question
        
        # Recupera documenti rilevanti usando MMR
        retriever = vectorstore["store"].as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.7}
        )
        hits = retriever.invoke(search_query)
        
        if not hits:
            return "I found no relevant information to answer your question.", vectorstore
        
        # Costruisce il contesto da passare al modello LLM
        context_parts = []
        for i, doc in enumerate(hits):
            metadata = doc.metadata
            page_info = f"[Pagina {metadata.get('page', 'N/A')}]" if "page" in metadata else ""
            context_parts.append(f"\n--- ESTRATTO {i+1} {page_info} ---\n{doc.page_content}")
        
        context = "\n".join(context_parts)
        
        # Chiama il modello LLM e restituisce la risposta
        answer = call_llm(question, context)
        
        return answer, vectorstore
    
    except Exception as e:
        return f"Error during processing: {str(e)}", vectorstore

#
# FUNZIONI PER IMAGE CAPTIONING
#

def encode_image_to_base64(image_path: str) -> str:
    """
    Codifica un'immagine in base64 per l'invio al modello.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path) -> Tuple[str, str]:
    """
    Processa l'immagine caricata, ridimensionandola se necessario,
    e la converte in base64.
    
    Nota: image_path è già un percorso di file quando arriva da gr.Image(type="filepath")
    """
    if not image_path or not os.path.exists(image_path):
        raise ValueError("No image loaded or invalid path")
    
    # Controlla la dimensione del file
    file_size = os.path.getsize(image_path)
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(f"The image is too big: {file_size/1024/1024:.1f}MB. Max allowed: 10MB")
    
    # Verifica che sia un'immagine valida
    try:
        img = Image.open(image_path)
        img_format = img.format.lower() if img.format else "unknown"
        return image_path, img_format
    except Exception as e:
        raise ValueError(f"Invalided file or not supported format: {e}")

def generate_caption(
    image_path: str,
    prompt_template: str,
    temperature: float = 0.3
) -> str:
    """
    Genera una didascalia per l'immagine caricata utilizzando il modello Gemma3.
    
    Nota: image_path è già un percorso di file quando arriva da gr.Image(type="filepath")
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return "Error: Upload an image to generate a caption."
        
        # Processa l'immagine
        processed_path, img_format = process_image(image_path)
        
        # Codifica l'immagine in base64
        base64_image = encode_image_to_base64(processed_path)
        
        # Prepara il prompt finale
        if not prompt_template or prompt_template.strip() == "":
            prompt_template = "Describe in detail what you see in this image."
            
        # Chiamata alla API di Ollama con l'immagine
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": prompt_template,
                    "images": [base64_image]
                }
            ],
            options={
                'temperature': temperature,
                'top_p': 0.9,
                'num_predict': 2048
            }
        )
            
        return response['message']['content']
        
    except Exception as e:
        return f"Error during image's processing: {str(e)}"

#
# INTERFACCIA GRADIO CON SCHEDE MULTIPLE
#

# Creazione dell'interfaccia principale con schede
with gr.Blocks(theme=gr.themes.Soft(), title="Gemma3 Document & Image AI") as app:
    gr.Markdown("# 🧠 Gemma3 Document & Image AI")
    
    with gr.Tabs():
        # Prima scheda: RAG Web + PDF
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
            
            # Stato per il vectorstore
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
            
        # Seconda scheda: Image Captioning
        with gr.TabItem("🖼️ Image Captioning"):
            gr.Markdown('<h1 style="color: red">⚠️ ATTENTION: GPU AND GEMMA3:4B REQUIRED ⚠️</h1>')
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
                    '''temperature_slider = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.3, 
                        step=0.1, 
                        label="Temperatura"
                    )'''
                    submit_btn_img = gr.Button("Image Captioning", variant="primary")
                    
                with gr.Column(scale=1):
                    output_img = gr.Textbox(
                        label="Answear", 
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