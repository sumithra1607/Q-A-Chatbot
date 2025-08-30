# rag_with_groq_flask.py
import os
import faiss
import json
import requests
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import pdfplumber
from bs4 import BeautifulSoup
from datetime import datetime

from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from flask import session, redirect
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from flask_cors import CORS

# ========= CONFIG =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

FAISS_DIR   = os.path.join(BASE_DIR, "..", "vectorstores", "faiss_index")
FAISS_INDEX = os.path.join(FAISS_DIR, "faiss.index")
META_FILE   = os.path.join(FAISS_DIR, "metadata.jsonl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
TOP_K = 3
NUM_GOOGLE_RESULTS = 3

TRUSTED_DOMAINS = [
    "fda.gov", "ema.europa.eu", "rxabbvie.com",
    "medlineplus.gov", "drugs.com", "humira.com", "skyrizi.com", "abbvie.com"
]

# ========= LOAD ENV =========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY missing in .env")

# ========= INIT FAISS =========
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

if os.path.exists(FAISS_INDEX) and os.path.exists(META_FILE):
    index = faiss.read_index(FAISS_INDEX)
    metadata_rows = [json.loads(line) for line in open(META_FILE, "r", encoding="utf-8")]
    mutable_docs = {
        str(i): Document(
            page_content=m["text"],
            metadata={"source": m.get("file"), "page": m.get("page"), "chunk_id": m.get("chunk_id")}
        )
        for i, m in enumerate(metadata_rows)
    }
else:
    index = faiss.IndexFlatL2(384)  # empty
    mutable_docs = {}

index_to_docstore_id = {int(i): str(i) for i in range(len(mutable_docs))}
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(mutable_docs),
    index_to_docstore_id=index_to_docstore_id
)

# ========= LLM =========
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="gemma2-9b-it",
    temperature=0.2,
    max_tokens=512
)

# ========= PROMPT TEMPLATE =========
prompt_template = """
You are a knowledgeable and helpful medical assistant. 
Answer the user's question clearly and concisely using the provided context. 
Structure it in a user-friendly way.
Don't give answers in large blocks of text. Use bullet points or numbered lists where appropriate.
Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Guidelines:
- If you don’t know, say so honestly.
- If the context is from a PDF, highlight the page number in sources.
- If the context is from a website, show only the URL (no page).
- Keep tone friendly and professional.
- structure answers for easy reading.

Answer:
"""

CUSTOM_PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

# ========= MEMORY =========
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
    memory=memory,
    return_source_documents=True,
    output_key="answer",
    combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
)

# ========= HELPERS =========
def pdf_to_documents(pdf_path, source_name):
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks = TEXT_SPLITTER.split_text(text)
                for j, chunk in enumerate(chunks):
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": source_name, "page": i, "chunk_id": j+1}
                    ))
    return docs

def persist_new_docs(docs):
    global vectorstore, mutable_docs, index_to_docstore_id
    if not docs:
        return

    start_idx = vectorstore.index.ntotal
    vectors = [embeddings.embed_query(doc.page_content) for doc in docs]
    vectors = np.array(vectors, dtype="float32")
    vectorstore.index.add(vectors)

    for i, doc in enumerate(docs):
        new_id = str(start_idx + i)
        mutable_docs[new_id] = doc
        index_to_docstore_id[start_idx + i] = new_id

    vectorstore.docstore = InMemoryDocstore(mutable_docs)
    vectorstore.index_to_docstore_id = index_to_docstore_id

    faiss.write_index(vectorstore.index, FAISS_INDEX)
    with open(META_FILE, "a", encoding="utf-8") as f:
        for doc in docs:
            meta = {
                "text": doc.page_content,
                "file": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id")
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

# ========= FIXED GOOGLE SEARCH =========
def fetch_google_results(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": NUM_GOOGLE_RESULTS}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    items = resp.json().get("items", [])

    # First filter for trusted domains
    trusted = [i["link"] for i in items if any(domain in i["link"] for domain in TRUSTED_DOMAINS)]

    # If nothing matches trusted domains, return general results
    return trusted if trusted else [i["link"] for i in items]

def fetch_and_add_url(url):
    try:
        if url.endswith(".pdf"):
            tmp_path = os.path.join(UPLOAD_DIR, "temp.pdf")
            with open(tmp_path, "wb") as f:
                f.write(requests.get(url).content)
            docs = pdf_to_documents(tmp_path, source_name=url)
        else:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            for s in soup(["script", "style"]): s.extract()
            text = soup.get_text(" ")
            docs = [Document(page_content=c, metadata={"source": url}) for c in TEXT_SPLITTER.split_text(text)]
        persist_new_docs(docs)
    except Exception as e:
        print(f"⚠ Failed to fetch {url}: {e}")

# ========= FLASK =========
app = Flask(__name__, template_folder="templates")
CORS(app, supports_credentials=True)
app.secret_key = os.getenv("SECRET_KEY", "super-secret-key-change-this")

# ========= AUTH DB =========
DB_FILE = os.path.join(BASE_DIR, "users.db")

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fullname TEXT,
                        email TEXT UNIQUE,
                        password TEXT,
                        age INTEGER,
                        height REAL,
                        weight REAL
                    )""")
        conn.commit()
init_db()

# ========= AUTH =========

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF allowed"}), 400

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(pdf_path)

    # Convert PDF → docs → FAISS
    docs = pdf_to_documents(pdf_path, source_name=file.filename)
    persist_new_docs(docs)

    # Add memory marker: last uploaded file
    memory.chat_memory.add_user_message(f"I just uploaded {file.filename}")
    memory.chat_memory.add_ai_message(
        f"Got it! I’ve added **{file.filename}** to the knowledge base. "
        f"You can now ask me questions about it immediately."
    )

    return jsonify({"message": f"✅ PDF '{file.filename}' uploaded and ready for Q&A."})

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"answer": "❌ Please enter a question.", "sources": []})

    # Handle chit-chat separately
    chit_chat_keywords = ["hi", "hello", "hey", "how are you", "good morning", "good evening", "what's up"]
    if query.lower() in chit_chat_keywords or len(query.split()) <= 2:
        raw_answer = llm.invoke(f"Respond to this greeting in a friendly way: '{query}'").content
        return jsonify({"answer": raw_answer, "sources": []})

    # 🔹 Personalization step
    user_context = ""
    if "user_id" in session:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("SELECT age, weight FROM users WHERE id=?", (session["user_id"],))
            row = c.fetchone()
            if row:
                age, weight = row
                if any(keyword in query.lower() for keyword in ["dosage", "dose", "how much", "prescribe", "mg"]):
                    user_context = f"Make it personalised by using the available user details. The patient is {age} years old and weighs {weight} kg. Consider this when answering."

    personalized_query = query + user_context

    # Step 1: Check FAISS (local PDFs + stored docs)
    results = vectorstore.similarity_search(personalized_query, k=TOP_K)

    # Step 2: Fallback to Google if no relevant results
    if not results:
        try:
            print(f"🔎 No FAISS results for: {personalized_query}")
            urls = fetch_google_results(personalized_query)
            for u in urls:
                fetch_and_add_url(u)
            results = vectorstore.similarity_search(personalized_query, k=TOP_K)
        except Exception as e:
            print(f"⚠ Google fetch failed: {e}")

    # Step 3: Query with conversational memory
    result = qa_chain.invoke({"question": personalized_query})
    raw_answer = result.get("answer", "").strip()
    sources = result.get("source_documents", results)

    # Step 4: Deduplicate + format sources
    seen, unique_sources = set(), []
    for doc in sources:
        meta = doc.metadata or {}
        src = meta.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            if src.endswith(".pdf") or src.lower().endswith(".pdf"):
                display_src = f"📄 {src}"
                page_num = meta.get("page")
            else:
                display_src = src
                page_num = None
            unique_sources.append({
                "source": display_src,
                "page": page_num,
                "snippet": (doc.page_content or "").replace("\n", " ")[:200]
            })

    formatted_answer = raw_answer if raw_answer else "🤖 Sorry, I couldn’t find an answer."
    return jsonify({
        "answer": formatted_answer.strip(),
        "sources": unique_sources[:2]
    })

# ========= AUTOCOMPLETE =========
@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"suggestions": []})

    try:
        prompt = (
            f"User is typing: '{query}'. Suggest 5 likely next words or short phrases. "
            "Return them as a plain list, one per line, no numbering or extra symbols."
        )
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        suggestions = [s.strip() for s in text.split("\n") if s.strip()]
        return jsonify({"suggestions": suggestions[:5]})
    except Exception as e:
        print(f"⚠ Suggestion error: {e}")
        return jsonify({"suggestions": []})
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/auth")
def auth_page():
    mode = request.args.get("mode", "signin")
    return render_template("sign.html", mode=mode)

@app.route("/details")
def details_page():
    return render_template("detail.html")

@app.route("/chat")
def chat_page():
    if "user_id" not in session:
        return redirect("/auth")
    return render_template("index.html")

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    fullname = data.get("fullname")
    email = data.get("email")
    password = generate_password_hash(data.get("password"))
    try:
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (fullname, email, password, created_at) VALUES (?, ?, ?, datetime('now'))",
                      (fullname, email, password))
            user_id = c.lastrowid
            conn.commit()
        session["user_id"] = user_id
        return jsonify({"success": True, "message": "Account created"})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email already exists"}), 400

@app.route("/signin", methods=["POST"])
def signin():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE email=?", (email,))
        row = c.fetchone()
        if row and check_password_hash(row[1], password):
            session["user_id"] = row[0]
            print("Login success, session user_id:", session["user_id"])
            return jsonify({"success": True, "message": "Login successful"})
        return jsonify({"success": False, "message": "Invalid credentials"}), 401


# Support both GET (fetch profile) and POST (update profile)
@app.route("/user_details", methods=["GET", "POST"])
def user_details():
    if "user_id" not in session:
        print("❌ No session found")
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    if request.method == "POST":
        data = request.get_json()
        print("👉 Received details:", data, "for user:", session["user_id"])
        with sqlite3.connect(DB_FILE) as conn:
            c = conn.cursor()
            c.execute("""UPDATE users 
                         SET fullname=?, age=?, height=?, weight=? 
                         WHERE id=?""",
                      (data.get("name"), data.get("age"), data.get("height"), data.get("weight"), session["user_id"]))
            conn.commit()
        return jsonify({"success": True, "message": "Details saved successfully"})

    # GET: fetch user profile
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT fullname, email, age, height, weight, created_at FROM users WHERE id=?", (session["user_id"],))
        row = c.fetchone()

        if row:
            # handle missing/null values gracefully
            fullname = row[0] if row[0] else "Not Provided"
            email = row[1] if row[1] else "Not Provided"
            age = row[2] if row[2] else "Not Provided"
            height = row[3] if row[3] else "Not Provided"
            weight = row[4] if row[4] else "Not Provided"
            since = row[5].split(" ")[0]

            return jsonify({
                "success": True,
                "name": fullname,
                "email": email,
                "age": age,
                "height": height,
                "weight": weight,
                "since": since
            })

        return jsonify({"success": False, "message": "User not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 locally
    app.run(host="0.0.0.0", port=port, debug=True)
