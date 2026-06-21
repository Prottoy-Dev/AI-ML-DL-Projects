# Bangla RAG System with Metadata Filtering

This notebook implements a simple Retrieval Augmented Generation (RAG) system for answering questions in Bengali (Bangla). It demonstrates how to build a custom knowledge base, filter information based on metadata (category and difficulty), and use a Large Language Model (LLM) to generate answers.

## Features

- **Custom Knowledge Base:** Uses pre-defined chunks of text in Bengali related to various topics (Education, Health, Travel, Technology, Sports).
- **Metadata Filtering:** Filters relevant information based on the detected category and difficulty level of the user's question.
- **Hugging Face Embeddings:** Utilizes a Bengali sentence similarity model for creating vector representations of the text chunks.
- **FAISS Vector Store:** Employs FAISS for efficient similarity search within the filtered knowledge base.
- **LLM Integration:** Uses an LLM (hosted on GitHub) to generate answers based on the retrieved context.
- **Category and Difficulty Detection:** Leverages the LLM to determine the category and difficulty of user queries.

## Setup

1.  **Install Dependencies:**
    Run the first code cell to install the necessary libraries.

2.  **Import Modules:**
    Run the second code cell to import all required modules.

3.  **Setup Embeddings:**
    Run the third code cell to load the Hugging Face embedding model.

4.  **Prepare Chunks with Metadata:**
    Run the fourth code cell to define the knowledge base chunks with their respective categories and difficulty levels.

5.  **Create FAISS Vector Store:**
    Run the fifth code cell to create a FAISS vector store from the prepared chunks using the embedding model.

6.  **Setup GitHub-hosted OpenAI LLM:**
    Run the sixth code cell to set up the connection to the GitHub-hosted OpenAI LLM. **Note:** You will need to replace key with your actual GitHub token or use Colab secrets to store it securely. The code includes commented-out lines for using Colab secrets.

7.  **Define Detection Functions:**
    Run the seventh and eighth code cells to define the `detect_category_llm` and `detect_difficulty_llm` functions.

8.  **Define FAQ Bot Function:**
    Run the ninth code cell to define the main `ask_faq_bot` function.

9.  **Ask a Question:**
    Modify the `user_question` variable in the subsequent cells and run them to test the RAG system with your own questions.
