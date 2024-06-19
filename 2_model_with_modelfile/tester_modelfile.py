#testing model which was enhanced with a modelfile

import os
import logging
from langchain.llms import Ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_directory(directory_path, output_directory):
    """Processes each text file in the directory, analyzes it, and saves the results."""
    os.makedirs(output_directory, exist_ok=True)
    files_processed = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            logging.info(f"Processing {file_path}...")
            try:
                content = read_file(file_path)
                techniques = detect_propaganda(content)
                output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
                save_results(techniques, output_path)
                logging.info(f"Results saved to {output_path}")
                files_processed += 1
            except Exception as e:
                logging.error(f"Failed to process {filename}: {e}")

    logging.info(f"Processing completed. {files_processed} files were processed.")

def read_file(file_path):
    """Reads and returns the content of the file."""
    with open(file_path, 'r') as file:
        return file.read()

def detect_propaganda(content):
    """Uses the Ollama model to detect propaganda techniques in the given content."""
    model_name = os.getenv("MODEL_NAME", "llama3-modelfile") #model was enhanced with ollama's modelfile
    llm = Ollama(model=model_name)
    prompt = ("Analyze the article for 14 propaganda techniques. If one or more propaganda techniques "
              "are identified, list each technique on a new line. If no techniques are identified, "
              "return with only the phrase 'no propaganda detected', here is the article:\n\n" + content)
    try:
        response = llm.predict(prompt)
        if response and "no propaganda detected" not in response:
            techniques = response.split('\n')
            return '\n'.join(techniques)
        return "no propaganda detected"
    except Exception as e:
        logging.error(f"Error in model prediction: {e}")
        return "no propaganda detected"

def save_results(techniques, output_path):
    """Saves the detected propaganda techniques to a file."""
    with open(output_path, 'w') as file:
        file.write(techniques or "no propaganda detected")

def main():
    input_directory = os.getenv("INPUT_DIRECTORY", "/data/dev-articles")
    output_directory = os.getenv("OUTPUT_DIRECTORY", "/results_modelfile")
    logging.info("Starting the process...")
    process_directory(input_directory, output_directory)
    logging.info("All files have been processed.")

if __name__ == "__main__":
    main()
