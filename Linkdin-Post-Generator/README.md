# Linkdin-Post-Generator
```
+-----------------+         HTTP POST         +-----------------+       LLM Call       +-----------------------------+
|                 |  -------------------->   |                 |  ----------------->  |                             |
|  Streamlit UI   |  User inputs: topic,     |  FastAPI Backend |  generate_post()     |  LangChain Agent Function   |
|                 |  language, length, tone  |                 |                      |                             |
+-----------------+                           +-----------------+                      +-----------------------------+
        ^                                             |                                             |
        |                                             |                                             |
        |                                             | Prompt with example + user inputs           |
        |                                             v                                             |
        |                                 +----------------------------+                           |
        |                                 |    GroqLLM class wraps     |                           |
        |                                 |   API call to Groq LLM    |                           |
        |                                 +----------------------------+                           |
        |                                             |                                             |
        |                                             | payload:                                   |
        |                                             | model="openai/gpt-oss-120b"               |
        |                                             | prompt=prompt                               |
        |                                             v                                             |
        |                                 +----------------------------+                           |
        |                                 |  Groq API Endpoint         |                           |
        |                                 | https://api.groq.com/v1/   |                           |
        |                                 | generate                   |                           |
        |                                 +----------------------------+                           |
        |                                             |                                             |
        |                                             | Returns generated text (LinkedIn post)     |
        |                                             v                                             |
        |                                 +----------------------------+                           |
        |                                 |  LangChain Agent returns   |                           |
        |                                 |     post_text to FastAPI   |                           |
        |                                             |                                             |
        |                                             v                                             |
        |                                 +----------------------------+                           |
        |                                 |  FastAPI returns JSON:     |                           |
        |                                 |  {"post": post_text}       |                           |
        |                                             |                                             |
        +---------------------------------------------v---------------------------------------------+
                                                      |
                                                      v
                                           +-----------------+
                                           | Streamlit UI    |
                                           | Displays LinkedIn|
                                           | Post to user     |
                                           +-----------------+
```

Generate professional LinkedIn posts instantly using AI! This project combines FastAPI, Streamlit, LangChain, and Groq LLM to create a seamless content creation experience.

## Features
- FastAPI backend for post generation
- Streamlit web interface for easy user input
- LangChain wrapper for Groq LLM integration
- Docker & docker-compose for deployment
- Secure environment variable management

## Tech Stack
- FastAPI
- Streamlit
- LangChain
- Groq
- Docker
- Render (deployment)

## How It Works
1. **User Input:** Enter your topic, tone, language, and preferences in the Streamlit app.
2. **API Request:** Streamlit sends a request to the FastAPI backend.
3. **AI Generation:** FastAPI uses a custom LangChain wrapper to interact with Groq LLM and generate the post.
4. **Result:** The generated LinkedIn post is displayed in the Streamlit app.

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for deployment)

### Local Development
1. Clone the repo:
   ```
   git clone https://github.com/Prottoy-Dev/Linkdin-Post-Generator.git
   cd Linkdin-Post-Generator
   ```
2. Install Dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your Groq API key to a .env file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
4. Start FastAPI:
   ```
   uvicorn backend.main:app --reload
   ```
5. Start Streamlit:
   ```
   streamlit run streamlit_app.py
   ```
## Docker Compose
Build and run both services:
```
docker-compose up --build
```
## Deployment
Deployed on Render(https://linkdin-post-generator.onrender.com/docs)

Deployed on Streamlit(https://linkdinpostgenerator.streamlit.app/)


