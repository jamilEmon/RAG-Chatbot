# Company Policy RAG Chatbot (Streamlit)

This project is a self-contained Streamlit app that demonstrates a Retrieval-Augmented Generation (RAG) chatbot for company policies.
The app ingests policy documents (PDF or URL), splits them into chunks, embeds them with SentenceTransformers, stores vectors in Chroma, and answers user questions with a local HuggingFace model (FLAN-T5).

**Project Overview:**

The application allows users to upload company policy documents (in PDF format) or provide a URL to a policy webpage. It then processes the document, splits it into smaller chunks, and creates vector embeddings of these chunks using SentenceTransformers. These embeddings are stored in a Chroma vector store. When a user asks a question, the application uses the vector store to find the most relevant chunks of text from the policy documents and feeds them to a HuggingFace language model (FLAN-T5) to generate an answer. The application also displays the source references used to generate the answer.

## Contents
- `app.py` - Streamlit application
- `requirements.txt` - Python dependencies
- `policies/` - sample policy text files (placeholders)

**Project Diagram:**

A professional diagram illustrating the project's architecture would be highly beneficial. Consider creating a **component diagram** or a **flowchart** to visualize the system. Here's a guide to creating a more detailed and professional diagram:

1.  **Choose a Diagram Type:**
    *   **Component Diagram:** This type of diagram emphasizes the different components of the system and their relationships. It's good for showing the overall architecture.
    *   **Flowchart:** This type of diagram emphasizes the flow of data and control through the system. It's good for showing the sequence of operations.

2.  **Select a Diagramming Tool:**
    *   **draw.io:** A free, online diagramming tool that offers a wide range of shapes and connectors.
    *   **Lucidchart:** A web-based diagramming tool with collaboration features and a professional look and feel (paid plans available).
    *   **Microsoft Visio:** A desktop diagramming tool with advanced features (paid).

3.  **Identify the Main Components:** The main components are:
    *   Streamlit UI: The user interface built with Streamlit.
    *   PDF/URL Input: The component that handles user input (either a PDF file or a URL).
    *   Text Extractor (PyPDF2, BeautifulSoup): The components responsible for extracting text from PDF files and web pages.
    *   Text Splitter (RecursiveCharacterTextSplitter): The component that splits the extracted text into smaller chunks.
    *   Embedding Model (SentenceTransformers): The component that generates vector embeddings of the text chunks.
    *   Vector Store (Chroma): The component that stores the vector embeddings.
    *   Language Model (FLAN-T5): The HuggingFace language model used to generate answers.

4.  **Represent the Components:**
    *   Use rectangles or rounded rectangles to represent each component.
    *   Use different colors to distinguish between different types of components (e.g., input components, processing components, storage components).
    *   Add labels to each component to clearly identify it.

5.  **Draw the Connections:**
    *   Use arrows to show the flow of data between the components.
    *   Use different types of arrows to indicate different types of relationships (e.g., data flow, control flow).
    *   Label the arrows to describe the data or control that is being passed between the components.

    For example:
    *   User Input -> PDF/URL Input: User provides a PDF file or URL.
    *   PDF/URL Input -> Text Extractor: The input is passed to the text extractor.
    *   Text Extractor -> Text Splitter: Extracted text is split into chunks.
    *   Text Splitter -> Embedding Model: Text chunks are converted into embeddings.
    *   Embedding Model -> Vector Store: Embeddings are stored in the vector store.
    *   User Question -> Vector Store: User's question is used to search the vector store.
    *   Vector Store -> Language Model: Relevant text chunks are retrieved from the vector store and passed to the language model.
    *   Language Model -> Streamlit UI: The language model generates an answer, which is displayed in the Streamlit UI.

6.  **Add a Legend:** Include a legend to explain the different shapes, colors, and arrow types used in the diagram.

Here's a simplified text-based diagram of the system:

```
+-----------------+    +-----------------+    +-----------------+
|    Streamlit UI   |    |   PDF/URL Input  |    |  Text Extractor |
+-----------------+    +-----------------+    +-----------------+
        ^                     |                     |
        |                     |                     |
        |                     v                     v
+-----------------+    +-----------------+    +-----------------+
| Language Model  |    |   Text Splitter  |    | Embedding Model |
+-----------------+    +-----------------+    +-----------------+
        ^                     |                     |
        |                     |                     |
        |                     v                     v
+-----------------+    +-----------------+
|   Vector Store    |    |  User Question  |
+-----------------+    +-----------------+
```

By following these guidelines, you can create a professional-looking diagram that effectively communicates the architecture of the project.

## Setup (local)
1. Create and activate a virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Run the app
```
streamlit run app.py
```
4. Open http://localhost:8501 in your browser

**Running the Project for the First Time:**

A new user can run this project by following these steps:

1.  **Clone the repository:** Clone the project repository to your local machine using Git.
2.  **Install Python:** Ensure that you have Python 3.7 or higher installed on your system.
3.  **Create a virtual environment:** Create a virtual environment to isolate the project dependencies.
4.  **Activate the virtual environment:** Activate the virtual environment.
5.  **Install dependencies:** Install the required Python packages using `pip install -r requirements.txt`.
6.  **Run the Streamlit app:** Execute the command `streamlit run app.py` in your terminal.
7.  **Access the application:** Open your web browser and navigate to the address displayed in the terminal (usually `http://localhost:8501`).

## Notes
- The sample policy files in `policies/` are provided as plaintext placeholders. For best results, upload real PDF policy documents via the app UI.
- The app uses a local HuggingFace model; ensure you have sufficient disk space and a compatible PyTorch setup.
- If you prefer using a cloud LLM (OpenAI or Gemini), modify `load_chain` in `app.py` accordingly and set the appropriate API keys.
