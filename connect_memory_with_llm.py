import os
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# ----------------------------
# 1️⃣ Load PDFs
# ----------------------------
DATA_PATH = "data/"  # folder with PDFs

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ----------------------------
# 2️⃣ FAISS vectorstore
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(DB_FAISS_PATH):
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS database from {DB_FAISS_PATH}")
else:
    print("📂 Creating FAISS database...")
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Saved FAISS database at {DB_FAISS_PATH}")

# ----------------------------
# 3️⃣ Public HuggingFace model (CPU/GPU)
# ----------------------------
# Flan-T5 works for text2text-generation and is public
flan_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1  # use 0 for GPU
)

llm = HuggingFacePipeline(pipeline=flan_pipeline)

# ----------------------------
# 4️⃣ RetrievalQA
# ----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

# ----------------------------
# 5️⃣ Query loop
# ----------------------------
while True:
    user_query = input("\nWrite Query Here (or type 'exit' to quit): ").strip()
    if user_query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    if not user_query:
        continue

    try:
        response = qa_chain.invoke({"query": user_query})
        print("\n📝 RESULT:\n", response.get("result", "No result found"))

        source_docs = response.get("source_documents", [])
        print(f"\n📄 SOURCE DOCUMENTS ({len(source_docs)} chunks retrieved):")
        for i, doc in enumerate(source_docs, start=1):
            print(f"\n--- Chunk {i} ---\n{doc.page_content[:500]}...")
    except Exception as e:
        print("⚠️ Error during query:", e)




















