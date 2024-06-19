# automatic tester out of the box
# opens folder with articles, reads it, collects responses in output folder

import os
from langchain.llms import Ollama

def process_directory(directory_path, output_directory, llm):
    """Processes each text file in the directory, analyzes it, and saves the results."""
    os.makedirs(output_directory, exist_ok=True)
    files_processed = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")  # Progress indicator for processing a file
            try:
                content = read_file(file_path)
                techniques = detect_propaganda(content, llm)
                output_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.txt")
                save_results(techniques, output_path)
                print(f"Results saved to {output_path}")  # Indicator of saving results
                files_processed += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Processing completed. {files_processed} files were processed.")

def read_file(file_path):
    """Reads and returns the content of the file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        raise IOError(f"Could not read file {file_path}: {e}")

def detect_propaganda(content, llm):
    """Uses the Ollama model to detect propaganda techniques in the given content."""
    prompt = ("Analyze the article for propaganda techniques. If one or more propaganda techniques "
              "are identified, list each technique on a new line. If no techniques are identified, "
              "return with only the phrase 'no propaganda detected', here is the article:\n\n" + content)
    try:
        response = llm.predict(prompt)
        if response and "no propaganda detected" not in response:
            techniques = response.split('\n')
            return '\n'.join(techniques)
        return "no propaganda detected"
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

def save_results(techniques, output_path):
    """Saves the detected propaganda techniques to a file."""
    try:
        with open(output_path, 'w') as file:
            file.write(techniques)
    except Exception as e:
        raise IOError(f"Could not write to file {output_path}: {e}")

def main():
    input_directory = os.getenv("INPUT_DIRECTORY", "data/dev-articles")
    output_directory = os.getenv("OUTPUT_DIRECTORY", "fieldtesting/results_oob")
    model_name = os.getenv("MODEL", "llama3")

    print("Initializing the language model...")
    llm = Ollama(model=model_name)
    print("Starting the process...")
    process_directory(input_directory, output_directory, llm)
    print("All files have been processed.")

if __name__ == "__main__":
    main()
