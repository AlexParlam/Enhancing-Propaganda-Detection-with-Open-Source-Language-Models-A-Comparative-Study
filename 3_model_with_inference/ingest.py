# taken from https://github.com/PromptEngineer48/Ollama 

import os
import glob
from typing import List, Dict, Tuple
from multiprocessing import Pool
from tqdm import tqdm
import logging

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', '/3_model_with_inference/db')
SOURCE_DIRECTORY = os.environ.get('SOURCE_DIRECTORY', '/3_model_with_inference /source_documents')
EMBEDDINGS_MODEL_NAME = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 50))

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = super().load()
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = super().load()
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING: Dict[str, Tuple[type, Dict]] = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()
    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """Loads all documents from the source documents directory, ignoring specified files"""
    all_files = [
        file_path
        for ext in LOADER_MAPPING
        for file_path in glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
    ]
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for docs in pool.imap_unordered(load_single_document, filtered_files):
                results.extend(docs)
                pbar.update()
    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """Load documents and split in chunks"""
    logger.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY, ignored_files)
    if not documents:
        logger.info("No new documents to load")
        exit(0)
    logger.info(f"Loaded {len(documents)} new documents from {SOURCE_DIRECTORY}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(texts)} chunks of text (max. {CHUNK_SIZE} tokens each)")
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """Checks if vectorstore exists"""
    index_exists = os.path.exists(os.path.join(persist_directory, 'index'))
    parquet_files_exist = (
        os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and
        os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet'))
    )
    sufficient_index_files = len(glob.glob(os.path.join(persist_directory, 'index/*.bin'))) > 3

    return index_exists and parquet_files_exist and sufficient_index_files

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

    if does_vectorstore_exist(PERSIST_DIRECTORY):
        # Update and store locally vectorstore
        logger.info(f"Appending to existing vectorstore at {PERSIST_DIRECTORY}")
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        logger.info(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        logger.info("Creating new vectorstore")
        texts = process_documents()
        logger.info(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)
    db.persist()
    db = None

    logger.info("Ingestion complete! Now your model has context from the source documents")

if __name__ == "__main__":
    main()
