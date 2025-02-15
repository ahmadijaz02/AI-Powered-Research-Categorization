# NeurIPS Paper Classifier: AI-Powered Research Categorization  

## 📌 Overview  
This project automates the classification of NeurIPS research papers using **Google Gemini AI**. It extracts titles and abstracts from PDFs and assigns them to predefined AI/ML categories.  

## 🚀 Features  
- 📄 **PDF Text Extraction**: Extracts title and abstract from the first page of research papers.  
- 🤖 **AI-Based Categorization**: Uses Gemini AI to classify papers into 20+ categories.  
- 🔄 **Automatic Fallback**: If classification fails, retries with a second API key.  
- ⚡ **Multithreading Support**: Processes multiple PDFs in parallel for faster execution.  
- 🛠️ **Error Handling & Debugging**: Handles API failures, empty PDFs, and missing abstracts.  

## 🏗️ Implementation  
1. **Extract Text from PDFs**: Uses PyMuPDF (`fitz`) to extract the first-page content.  
2. **Classify Papers**: Sends extracted text to Gemini AI to determine the category.  
3. **Parallel Processing**: Uses `ThreadPoolExecutor` for efficiency.  
4. **Fallback API Handling**: Switches API keys in case of rate limits or failures.  

## 🛠️ Installation  
### Prerequisites  
- Python 3.x  
- Required Libraries:  
  ```sh
  pip install pymupdf pandas google-generativeai
## ▶️ Usage
Run the script to process all PDFs in a specified folder:
python classify_papers.py

## 📊 Output
The classified results are saved as:
neurips_papers_annotated.csv

##🔥 Challenges & Solutions
Slow Processing? → Used multithreading to parallelize execution.
Rate Limits? → Implemented an API key fallback mechanism.
Empty Abstracts? → If no abstract, classification is done based on title.
Incorrect Classifications? → Logs unknown classifications for manual review.
##📜 License
This project is open-source under the MIT License.

##🤝 Contributions
Contributions are welcome! Feel free to fork the repo and submit a pull request.

##✨ Author
Ahmad Ijaz – [Your Medium Blog Link : https://medium.com/@mahmadijaz192/automating-research-paper-classification-using-google-gemini-api-and-python-679478503adf ]

