# the most effective tester for inference
# inspired by https://github.com/PromptEngineer48/Ollama 
import os
import logging
import argparse
import time
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables or set default values
model = os.environ.get("MODEL", "llama3") 
# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
input_directory = os.getenv("INPUT_DIRECTORY", "data/dev-articles")
output_directory = os.getenv("OUTPUT_DIRECTORY", "results_inference")

def main():
    args = parse_arguments()

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    # Initialize the Chroma vector store
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    # Set up the retriever
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    # Activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # Initialize the LLM with Ollama
    llm = Ollama(model=model, callbacks=callbacks)

    # Set up the RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    logging.info("Starting the process...")
    process_directory(input_directory, output_directory, qa, args)
    logging.info("All files have been processed.")

def process_directory(directory_path, output_directory, qa, args):
    os.makedirs(output_directory, exist_ok=True)
    files_processed = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            logging.info(f"Processing {file_path}...")
            try:
                content = read_file(file_path)
                query = ("Analyze the article for any of 14 propaganda techniques, if no propaganda detected return 'no propaganda detected':\n\n" + content)
                res = qa(query)
                answer = res['result']
                output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
                save_results(answer, output_path)
                logging.info(f"Results saved to {output_path}")
                files_processed += 1
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

    logging.info(f"Processing completed. {files_processed} files were processed.")

def read_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_results(techniques, output_path):
    with open(output_path, 'w') as file:
        file.write(techniques or "no propaganda detected")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Propaganda Detector: Ask questions to your documents without an internet connection, using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

if __name__ == "__main__":
    main()
