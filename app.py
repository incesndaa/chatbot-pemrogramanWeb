import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Paket integrasi terbaru untuk embedding, endpoint, dan model chat
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma

# 1. KONFIGURASI HALAMAN UI
st.set_page_config(
    page_title="Asisten Akademik Pemrograman Web", 
    layout="centered"
)

# Custom CSS 
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stChatInputContainer {padding-bottom: 20px;}
        .reportview-container .main .block-container {padding-top: 2rem;}
        h1 {font-weight: 700; color: #1E293B;}
        p {color: #475569;}
    </style>
""", unsafe_allow_html=True)

st.title("Asisten Akademik: Pemrograman Web")
st.markdown("Sistem ini dikembangkan untuk membantu mahasiswa memahami materi HTML, CSS, dan JavaScript secara spesifik berdasarkan e-book referensi perkuliahan.")
st.markdown("---")


# 2. MANAJEMEN AKSES API (HUGGING FACE TOKEN)
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    st.error("Konfigurasi Gagal: Token Hugging Face Hub tidak ditemukan.")
    st.info("Pastikan token telah dikonfigurasi pada Environment Variables / Secrets dengan nama HF_TOKEN.")
    st.stop()


# 3. PIPELINE RAG: INGESTION 
@st.cache_resource
def inisialisasi_retriever():
    nama_file_pdf = "ebook-ddp.pdf"
    
    if not os.path.exists(nama_file_pdf):
        st.error(f"Berkas Sistem Hilang: Berkas '{nama_file_pdf}' tidak ditemukan di server.")
        st.stop()

    try:
        loader = PyPDFLoader(nama_file_pdf)
        dokumen_mentah = loader.load()

        pemotong_teks = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""]
        )
        potongan_dokumen = pemotong_teks.split_documents(dokumen_mentah)

        # Model embedding 
        model_embedding = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # In-Memory Vector Database
        database_vektor = Chroma.from_documents(
            documents=potongan_dokumen, 
            embedding=model_embedding
        )
        
        # Mengambil 3 potongan dokumen paling relevan untuk dianalisis LLM
        return database_vektor.as_retriever(search_kwargs={"k": 3})
        
    except Exception as e:
        st.error(f"Gagal memproses basis data dokumen: {str(e)}")
        st.stop()

retriever_sistem = inisialisasi_retriever()


# 4. PIPELINE RAG: GENERATION
@st.cache_resource
def inisialisasi_llm():
    # Menggunakan Qwen 2.5 7B Instruct karena akurasi penulisan sintaks kodenya tinggi
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.2,
        huggingfacehub_api_token=HF_TOKEN
    )
    
    # 2. Bungkus endpoint ke dalam ChatHuggingFace agar mendukung task conversational
    llm = ChatHuggingFace(llm=endpoint)
    return llm

llm_sistem = inisialisasi_llm()

# Instruksi ketat: LLM dilarang keras menjawab di luar teks e-book yang diberikan
template_prompt = """Anda adalah asisten akademik yang profesional, objektif, dan ahli dalam Pemrograman Web. 
Tugas Anda adalah menjawab pertanyaan mahasiswa secara terstruktur dan akurat hanya berdasarkan konteks dokumen yang disediakan.

Aturan Kerja:
1. Jawablah menggunakan Bahasa Indonesia yang baku, jelas, dan mudah dipahami.
2. Jika pertanyaan memerlukan contoh sintaks atau instruksi coding, berikan cuplikan kode yang valid menggunakan format code-block markdown (misalnya: ```html ... ```).
3. Jika jawaban dari pertanyaan mahasiswa tidak tercantum atau tidak dapat disimpulkan dari konteks e-book di bawah ini, jawab dengan kalimat: 
   "Maaf, informasi mengenai hal tersebut tidak dibahas di dalam e-book referensi pemrograman web ini." Jangan memberikan informasi di luar dokumen.

KONTEKS E-BOOK:
{context}

PERTANYAAN MAHASISWA:
{question}

JAWABAN OTORETIF:"""

prompt = ChatPromptTemplate.from_template(template_prompt)

# Integrasi RAG Chain
rag_chain = (
    {"context": retriever_sistem, "question": RunnablePassthrough()}
    | prompt
    | llm_sistem
    | StrOutputParser()
)

# =========================================================================
# 5. ANTARMUKA CHAT INTERAKTIF (STREAMLIT SESSION STATE)
# =========================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "Halo. Saya adalah asisten virtual untuk mata kuliah Pemrograman Web. Silakan ajukan pertanyaan mengenai materi HTML, CSS, atau JavaScript yang bersumber dari e-book kuliah."
        }
    ]

# Menampilkan riwayat percakapan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Penanganan input pertanyaan user
if user_query := st.chat_input("Ketik pertanyaan Anda di sini..."):
    
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Menganalisis e-book..."):
            try:
                response = rag_chain.invoke(user_query)
                clean_response = response.strip()
                
                message_placeholder.markdown(clean_response)
                st.session_state.messages.append({"role": "assistant", "content": clean_response})
                
            except Exception as e:
                st.error(f"Sistem mengalami kendala saat memproses jawaban: {str(e)}")