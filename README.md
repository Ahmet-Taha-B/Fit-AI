# 💪 Fit AI

A bilingual (Turkish/English) AI-powered fitness and nutrition assistant built with LangGraph, LangChain, and Groq's GPT-OSS model. Features intelligent RAG (Retrieval-Augmented Generation) for fitness knowledge and built-in conversational memory.

🌐 **Live Demo**: [atb-fit-ai.streamlit.app](https://atb-fit-ai.streamlit.app/)

## ✨ Features

- 🤖 **GPT-OSS Powered**: Uses Groq's powerful `openai/gpt-oss-120b` model for fast, intelligent responses
- 🧠 **LangGraph Memory**: Automatically remembers conversation context for natural follow-up questions
- 📚 **RAG Integration**: Retrieves relevant information from fitness and nutrition PDFs
- 🌍 **Bilingual Support**: Full Turkish and English interface
- 💬 **Context-Aware**: Understands references like "it", "that", "them" from previous messages
- 🎯 **Personalized Advice**: Tailored fitness and nutrition recommendations
- 📊 **Session Statistics**: Track your conversation metrics

## 🏗️ Architecture

```
User Question
     ↓
LangGraph Agent (with Memory)
     ↓
GPT-OSS Model (Groq)
     ↓
RAG Retriever Tool → ChromaDB Vector Store → Fitness PDFs
     ↓
Intelligent Response
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/keys))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ahmet-Taha-B/Fit-AI
   cd Fit-AI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your knowledge base**
   ```bash
   mkdir -p data/fitness_pdfs
   ```
   
   Add your fitness and nutrition PDF files to `data/fitness_pdfs/` directory.

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Enter your Groq API key** in the sidebar when the app opens

## 📁 Project Structure

```
fitness-ai-coach/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                 # Git ignore rules
├── data/
│   └── fitness_pdfs/          # Your PDF knowledge base (create this)
└── README.md                  # This file
```

## 🔑 Getting Your Groq API Key

1. Visit [Groq Console](https://console.groq.com/keys)
2. Sign up or log in to your account
3. Navigate to the "API Keys" section
4. Click "Create API Key"
5. Copy your key (it starts with `gsk_...`)
6. Paste it into the sidebar of the application

**Note**: Keep your API key secure! Never commit it to version control.

## 💡 How It Works

### 1. **RAG (Retrieval-Augmented Generation)**
   - PDFs in `data/fitness_pdfs/` are loaded and split into chunks
   - Embeddings are created using `sentence-transformers/all-MiniLM-L6-v2`
   - Stored in a ChromaDB vector database
   - Relevant context is retrieved when you ask questions

### 2. **LangGraph Memory**
   - Conversations are tracked with unique thread IDs
   - MemorySaver checkpoint system stores conversation history
   - Enables natural follow-up questions without repeating context

### 3. **Agent Architecture**
   - Uses `create_react_agent` from LangGraph
   - Equipped with a retriever tool for knowledge base access
   - GPT-OSS model processes queries with retrieved context

## 🎯 Example Usage

**First Question:**
> "What's a good home arm workout?"

**Follow-up (memory-aware):**
> "How many sets should I do for each exercise?"
> (Agent remembers we're talking about arm workouts!)

**Reference-based:**
> "Which muscles does it target?"
> (Agent understands "it" refers to the workout discussed)

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web interface
- **[LangChain](https://python.langchain.com/)**: LLM orchestration framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: Agent workflow and memory management
- **[Groq](https://groq.com/)**: Fast LLM inference (GPT-OSS model)
- **[ChromaDB](https://www.trychroma.com/)**: Vector database for embeddings
- **[HuggingFace Transformers](https://huggingface.co/)**: Sentence embeddings
- **[PyPDF](https://pypdf.readthedocs.io/)**: PDF document processing

## ⚙️ Configuration

### Customizing the System Prompt

Edit the `system_prompt` in the `TRANSLATIONS` dictionary in `app.py`:

```python
"system_prompt": """Your custom instructions here..."""
```

### Adjusting RAG Parameters

In the `load_vectorstore()` function:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `k`: Number of retrieved documents (default: 3)

### Model Settings

In the `create_agent()` function:
- `model`: Groq model name (default: "openai/gpt-oss-120b")
- `temperature`: Response randomness (default: 0.3)

**Made by Ahmet Taha Berberoglu**

*Powered by Groq, LangChain, and LangGraph*
