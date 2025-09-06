import streamlit as st
try:
    import google.generativeai as genai
except ImportError:
    st.error("Missing 'google-generativeai'. Install it using: `pip install google-generativeai`")
    st.stop()
try:
    from PyPDF2 import PdfReader
except ImportError:
    st.error("Missing 'PyPDF2'. Install it using: `pip install PyPDF2`")
    st.stop()
try:
    import pytesseract
except ImportError:
    st.error("Missing 'pytesseract'. Install it using: `pip install pytesseract`. Also ensure Tesseract OCR is installed and added to PATH.")
    st.stop()
try:
    from PIL import Image
except ImportError:
    st.error("Missing 'Pillow'. Install it using: `pip install Pillow`")
    st.stop()
try:
    import numpy as np
except ImportError:
    st.error("Missing 'numpy'. Install it using: `pip install numpy`")
    st.stop()
try:
    import faiss
except ImportError:
    st.error("Missing 'faiss'. Install it using: `pip install faiss-cpu`. Ensure Microsoft Visual C++ Build Tools are installed.")
    st.stop()
try:
    import docx
except ImportError:
    st.error("Missing 'python-docx'. Install it using: `pip install python-docx`")
    st.stop()
try:
    from dotenv import load_dotenv
except ImportError:
    st.error("Missing 'python-dotenv'. Install it using: `pip install python-dotenv`")
    st.stop()
import os
from datetime import datetime
import io

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("GEMINI_API_KEY not found in .env file. Please create a .env file in `C:\\Users\\bharg\\OneDrive\\Desktop\\ces_main` with: `GEMINI_API_KEY=your-api-key`")
    st.stop()

genai.configure(api_key=API_KEY)

# Set Streamlit layout to wide for desktop, but allow responsive CSS for mobile
st.set_page_config(layout="wide")

# Sidebar Instructions
st.sidebar.title("Setup Instructions")
st.sidebar.markdown("""
**Required Setup**:
1. Save the following as `requirements.txt` in your project folder and run `pip install -r requirements.txt`:
```
streamlit==1.38.0
google-generativeai==0.8.3
PyPDF2==3.0.1
pytesseract==0.3.13
Pillow==10.4.0
python-docx==1.1.2
numpy==2.1.1
faiss-cpu==1.8.0
python-dotenv==1.0.1
```
2. Create a `.env` file in `C:\\Users\\bharg\\OneDrive\\Desktop\\ces_main` with:
```
GEMINI_API_KEY=your-api-key
```
3. Install Tesseract OCR: [Download](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH (e.g., `C:\\Program Files\\Tesseract-OCR`).
4. Install Microsoft Visual C++ Build Tools: [Download](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select 'Desktop development with C++'.
5. Place your logo at `C:\\Users\\bharg\\OneDrive\\Desktop\\ces_main\\logo.png`.
6. Run the app: `python -m streamlit run app.py`
""")

QNA_PROMPT = """
You are an advanced Retrieval-Augmented Generation (RAG) assistant for enterprise document analysis. 
Your goal is to deliver only the most relevant information in a highly compressed, easy-to-read format.
also strictly folow the output response as given below in response rules the foramt should be follwing as given in below.

Response Rules:
1. **Answer** – start with brief description of question and continue the following answering in it , Present the key information in 1–2 short, direct sentences (≤2 lines). Compress all retrieved content into minimal words while retaining essential details. Cite document sections if available.  
2. **Recommendations and Insights** – Provide exactly 1–2 concise bullet points. Each should be ≤1 line where possible, strictly ≤2 lines, combining all insights into clear, executive-style statements. Distinguish between:
   - Facts directly from the documents.
   - Inferences or best practices (labeled clearly).

Guidelines:
- Always prioritize brevity and clarity: answers must be instantly scannable.  
- No filler, repetition, or verbose explanations.  
- Use simple, structured language for readability.  
- If multiple documents overlap, merge into a single compact statement.  
- If the document lacks the information, state clearly: "The document does not provide this information."
"""






# Generated Analysis Report Prompt (Big 4 Showcase)
REPORT_PROMPT = """
You are an elite consulting assistant producing Big 4–style executive reports (KPMG, Deloitte, PwC, EY). 
Transform the provided documents into a professional, fact-based, and insight-driven analysis that is concise, structured, and decision-focused.

Structure (adapt dynamically based on content relevance):
1. **Executive Summary** – Provide a high-level overview in the fewest possible lines, capturing only the most critical takeaways and implications.  
2. **Key Insights and Findings** – Present condensed, data-driven highlights. Use short bullets or tables. Compress all relevant details into minimal lines without losing meaning.  
3. **Risk Assessment & Opportunities** – Summarize the main risks, compliance issues, and growth opportunities in the most compressed form possible (preferably 1–3 short lines).  
4. **Strategic Recommendations** – Provide 2–3 actionable recommendations, each expressed in a single concise statement (≤2 lines). Clearly mark what is fact-based vs inference/best practice.  
5. **Supporting Evidence** – Cite relevant document excerpts in a compressed manner (1 line per evidence point).  

Guidelines:
- Prioritize brevity and density: compress multi-line content into minimal statements while preserving key meaning.  
- No filler or repetition. Every line should deliver decision-making value.  
- Maintain a professional, neutral, and authoritative consulting tone.  
- Use bullets/tables instead of long paragraphs wherever possible.  
- Explicitly state if information is missing and suggest clear next steps.  
- The final output must be executive-ready: short, fact-based, and strategically insightful.
"""




# Function to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    try:
        if "pdf" in file_type:
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        elif "image" in file_type:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)
            return text
        elif "text" in file_type:
            return uploaded_file.read().decode("utf-8")
        else:
            return "Unsupported file type."
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return ""

# Function to chunk text for embedding
def chunk_text(text, chunk_size=500):
    if not text or text == "Unsupported file type.":
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Function to generate embeddings using Gemini
def get_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            st.error(f"Error generating embedding for chunk: {str(e)}")
            return None
    return np.array(embeddings, dtype=np.float32)

# Function to initialize FAISS index
def create_faiss_index(embeddings):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        st.error(f"Error creating FAISS index: {str(e)}")
        return None

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, chunk_metadata, k=5):
    try:
        query_embedding = get_embeddings([query])[0]
        distances, indices = index.search(np.array([query_embedding]), k)
        relevant = []
        for idx in indices[0]:
            if idx < len(chunks):
                relevant.append((chunks[idx], chunk_metadata[idx]))
        return relevant
    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {str(e)}")
        return []

# Function to save report as Word document
def save_report_to_word(report_text, filename_prefix="Generated_Analysis_Report"):
    try:
        doc = docx.Document()
        doc.add_heading("Generated Analysis Report", 0)
        for line in report_text.split("\n"):
            if line.startswith("1) ") or line.startswith("2) ") or line.startswith("3) ") or line.startswith("4) ") or line.startswith("5) "):
                doc.add_heading(line, level=1)
            elif line.startswith("- "):
                doc.add_paragraph(line.replace("- ", ""), style="ListBullet")
            else:
                doc.add_paragraph(line)
        filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
        doc.save(filename)
        return filename
    except Exception as e:
        st.error(f"Error saving docx: {str(e)}")
        return None

# Streamlit App Layout
# Custom Logo and Title with Responsive Layout
LOGO_PATH = "logo.png"
with st.container():
    # Responsive columns: more balanced for desktop, stacked for mobile
    col1, col2 = st.columns([1, 6])
    with col1:
        if os.path.exists(LOGO_PATH):
            st.markdown(
                """
                <style>
                @media (max-width: 600px) {
                    .logo-img {max-width: 48px !important; height: auto !important;}
                }
                @media (min-width: 601px) {
                    .logo-img {max-width: 72px !important; height: auto !important;}
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.image(LOGO_PATH, output_format="auto", caption=None, use_container_width=False, clamp=False, channels="RGB", width=72)
        else:
            st.warning("Logo file not found at 'logo.png'. Please place your logo in `C:\\Users\\bharg\\OneDrive\\Desktop\\ces_main`.")
    with col2:
        st.markdown(
            """
            <style>
            .responsive-heading {
                font-size: clamp(1.1rem, 4vw, 2.2rem);
                text-align: left;
                margin-top: 0;
                margin-bottom: 0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                font-weight: 700;
                letter-spacing: -1px;
            }
            @media (max-width: 600px) {
                .responsive-heading {
                    font-size: 1.1rem !important;
                }
            }
            </style>
            <h1 class="responsive-heading">PDF Analyst Chatbot and Report Generator</h1>
            """,
            unsafe_allow_html=True
        )

# Adjusted subheader size for responsiveness
st.markdown(
    """
    <style>
    .responsive-subheader {
        font-size: clamp(0.9rem, 2.5vw, 1.3rem);
        margin-top: 10px;
        margin-bottom: 0;
    }
    </style>
    <div class="responsive-subheader">Upload PDF for Q&amp;A and Generate Analysis Reports</div>
    """,
    unsafe_allow_html=True
)

# File Uploader
uploaded_files = st.file_uploader("Upload PDF", accept_multiple_files=True)

# Store extracted document contents and chunks
documents_content = []
chunks = []
chunk_metadata = []
faiss_index = None
if uploaded_files:
    for uploaded_file in uploaded_files:
        content = extract_text_from_file(uploaded_file)
        if content and content != "Unsupported file type.":
            documents_content.append(f"--- Document: {uploaded_file.name} ---\n{content}")
            doc_chunks = chunk_text(content)
            chunks.extend(doc_chunks)
            chunk_metadata.extend([(uploaded_file.name, i) for i in range(len(doc_chunks))])

    # Initialize Vector DB
    if chunks:
        embeddings = get_embeddings(chunks)
        if embeddings is not None:
            faiss_index = create_faiss_index(embeddings)
            if faiss_index:
                st.success("PDf indexed successfully in Vector DB.")
            else:
                st.error("Failed to create Vector DB index. Check error messages above.")
                st.stop()
        else:
            st.error("Failed to generate embeddings. Check error messages above.")
            st.stop()

# Generate Analysis Report Button
if st.button("Generate  Report") and uploaded_files:
    with st.spinner("Generating Report..."):
        # Use a general query for report retrieval
        report_query = "Key insights, risks, and recommendations from the documents"
        relevant = retrieve_relevant_chunks(report_query, faiss_index, chunks, chunk_metadata, k=10)
        relevant_content = "\n\n".join([f"Relevant Section from {meta[0]} (Chunk {meta[1]}): {chunk}" for chunk, meta in relevant])

        # Construct full prompt for report
        full_prompt = f"{REPORT_PROMPT}\n\nDocument Content for Analysis:\n{relevant_content}"

        # Call Gemini API
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(full_prompt)
            report_text = response.text

            # Display the Report
            st.markdown("### Generated Analysis Report")
            st.markdown(report_text)

            # Save to Word Document
            word_filename = save_report_to_word(report_text)
            if word_filename:
                with open(word_filename, "rb") as file:
                    st.download_button(
                        label="Download Generated Analysis Report as Word Document",
                        data=file,
                        file_name=word_filename,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
else:
    if not uploaded_files:
        st.info("Please upload at least one PDF to generate the report.")

# Chat Input for Q&A
st.markdown("### Ask RE Analyst  ")
# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=50)
        st.markdown(message["content"])

# Chat Input
user_query = st.chat_input("Ask a question about the PDF:")
if user_query and faiss_index:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.spinner("Generating answer..."):
        # Retrieve relevant chunks
        relevant = retrieve_relevant_chunks(user_query, faiss_index, chunks, chunk_metadata)
        relevant_content = "\n\n".join([f"Relevant Section from {meta[0]} (Chunk {meta[1]}): {chunk}" for chunk, meta in relevant])

        # Construct full prompt for Q&A
        full_prompt = f"{QNA_PROMPT}\n\nRelevant Document Content:\n{relevant_content}\n\nUser Question: {user_query}"

        # Call Gemini API
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(full_prompt)
            answer_text = response.text

            # Display assistant message with logo
            with st.chat_message("assistant"):
                if os.path.exists(LOGO_PATH):
                    st.image(LOGO_PATH, width=50)
                st.markdown(answer_text)
            st.session_state.chat_history.append({"role": "assistant", "content": answer_text})
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
else:
    if not uploaded_files:
        st.info("Please upload at least one PDF to begin Q&A.")
    if not faiss_index and uploaded_files:
        st.info("Document indexing failed. Check error messages above.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("This app uses Google Gemini 2.0 Flash and FAISS for document analysis. For support, check the Setup Instructions above.")
