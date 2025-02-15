import os
import fitz  
import pandas as pd
import google.generativeai as genai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure Gemini API Keys
API_KEY_1 = "AIzaSyB9IxOYFT52yols8lo09edHIQniMUWw7c8"
API_KEY_2 = "AIzaSyD32EYWRxvgYoAg8m9td2js_Xn7RDfmapI"  
current_api_key = API_KEY_1

def configure_api(api_key):
    """Configures Google Gemini API with the given key."""
    genai.configure(api_key=api_key)

configure_api(current_api_key)

# Define annotation labels
categories = [
    "Deep Learning", "Computer Vision", "Reinforcement Learning", "NLP", "Optimization",
    "Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning",
    "Generative Models (GANs, VAEs)", "Graph Neural Networks (GNNs)", "Transfer Learning",
    "Self-Supervised Learning", "Few-Shot and Zero-Shot Learning", "Bayesian Methods",
    "Fairness, Accountability, and Transparency in AI", "Robotics and Embodied AI",
    "Neuroscience and Brain-Inspired AI", "Optimization Algorithms (Gradient Descent, SGD, etc.)",
    "Federated Learning and Privacy-Preserving AI", "Meta-Learning (Learning to Learn)"
]

# Folder containing research papers
papers_folder = r"C:\Users\Ahmad\Documents\PDFScrapper\downloads\2017"

def extract_text_from_pdf(pdf_path):
    """Extracts text from the first page of a PDF."""
    try:
        with fitz.open(pdf_path) as doc:
            if len(doc) == 0:
                print(f"Skipping {pdf_path}: No pages found.")
                return None

            text = doc[0].get_text()

            if not text.strip():
                print(f"Skipping {pdf_path}: No extractable text found.")
                return None

            return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def classify_paper(text, classification_type):
    """
    Classifies a paper based on its Abstract or Title.
    classification_type = 'abstract' or 'title'.
    """
    global current_api_key

    if not text:
        return "Unknown"

    prompt = f"""
    Given the following research paper {classification_type}, classify it into one of these categories: {', '.join(categories)}.

    {classification_type.capitalize()}: {text}

    Return only the category name from the given list.
    """

    retries = 3  # Maximum retries per API key
    for attempt in range(retries):
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)

            if response and hasattr(response, "text"):
                return response.text.strip()
            else:
                return "Unknown"

        except Exception as e:
            print(f"Error classifying paper ({classification_type}): {e}")

            # If we get a rate limit or quota exceeded error, switch API keys
            if "429" in str(e) or "quota exceeded" in str(e):
                if current_api_key == API_KEY_1:
                    print("⚠️ Switching to secondary API key...")
                    current_api_key = API_KEY_2
                else:
                    print("⚠️ Switching back to primary API key...")
                    current_api_key = API_KEY_1

                configure_api(current_api_key)  # Apply new API key
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "Unknown"

    return "Unknown"

def process_pdf(filename):
    """Processes a single PDF file: extracts text, classifies, and returns results."""
    pdf_path = os.path.join(papers_folder, filename)
    text = extract_text_from_pdf(pdf_path)

    if not text:
        return None

    # Extract title and abstract
    title = text.split("\n")[0].strip() if text.split("\n") else "Untitled"
    abstract = "\n".join(text.split("\n")[1:]).strip() if len(text.split("\n")) > 1 else "No abstract"

    # Step 1: Classify based on abstract
    category = classify_paper(abstract, classification_type="abstract")

    # Step 2: If abstract classification fails, classify based on title
    if category == "Unknown":
        print(f"🔄 Re-classifying using Title: {title}")
        category = classify_paper(title, classification_type="title")

    return [filename, title, abstract, category]

# Get list of PDFs
pdf_files = [f for f in os.listdir(papers_folder) if f.endswith(".pdf")]
total_pdfs = len(pdf_files)

print(f"Found {total_pdfs} PDFs. Starting processing...\n")

annotated_papers = []
num_threads = min(2, total_pdfs)  # Limit to 5 threads or number of PDFs (whichever is lower)

# Parallel Processing with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    future_to_file = {executor.submit(process_pdf, filename): filename for filename in pdf_files}

    for i, future in enumerate(as_completed(future_to_file), 1):
        result = future.result()
        if result:
            annotated_papers.append(result)
            print(f"Processed {i}/{total_pdfs}: {result[0]} → Category: {result[3]}")

# Convert to DataFrame
df = pd.DataFrame(annotated_papers, columns=["Filename", "Title", "Abstract", "Category"])

# Save annotated dataset
output_file = "neurips_papers_annotated.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Annotation complete! The dataset is saved as '{output_file}'.")
