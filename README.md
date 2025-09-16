# 🧠 RAG-Medibot

A Retrieval-Augmented Generation (RAG) chatbot.  
It combines **vector stores** (FAISS) with **LLMs** (OpenAI, HuggingFace, etc.) to provide context-aware answers from medical knowledge sources.

---

## ✨ Features

- 🗂️ **Document Retrieval**: Uses FAISS to fetch relevant context.  
- 🧠 **LLM + Memory**: Maintains conversational memory with RAG.  
- ⚡ **Extensible**: Plug in any reference documents.  
- 🔌 **Modular Scripts**: Separate scripts for memory creation and chatbot execution.  

---
## 🛠️ Requirements

- Python 3.x (recommend 3.8 or newer)  
- [FAISS](https://github.com/facebookresearch/faiss) for vector store  
- [Pipenv](https://pipenv.pypa.io/) (or virtualenv + pip)  
- Access to an LLM (OpenAI API, HuggingFace Hub, etc.)  
- (Optional) Medical documents to populate the vectorstore  

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/sairitvikb/RAG-medibot.git
cd RAG-medibot
```

# Using pipenv
```
pipenv install
```

# Or using pip + virtualenv
```
python3 -m venv venv
source venv/bin/activate
```

# Install dependencies
```
pip install -r requirements.txt   # (create one if missing)
```



