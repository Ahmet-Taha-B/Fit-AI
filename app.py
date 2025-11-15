import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
import os

st.set_page_config(
    page_title="Fitness AI Coach",
    page_icon="💪",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton>button:hover {
        background-color: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>💪 Fitness AI Coach</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.8);'>Kişisel fitness ve beslenme asistanınız (LangGraph Hafızalı! 🧠)</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("🔑 API Ayarları")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Groq API anahtarınızı girin (https://console.groq.com/keys)"
    )
    
    if not groq_api_key:
        st.warning("⚠️ Lütfen API anahtarınızı girin!")
    else:
        st.success("✅ API anahtarı girildi!")
    
    st.markdown("---")
    
    st.header("🎯 Özellikler")
    st.info("""
    ✅ Kişiselleştirilmiş fitness tavsiyeleri
    ✅ Beslenme programları
    ✅ Egzersiz önerileri
    ✅ Bilimsel kaynaklara dayalı
    🧠 **LangGraph Checkpoint Memory**
    """)
    
    st.markdown("---")
    
    st.header("📊 Seans İstatistikleri")
    if "messages" in st.session_state:
        col1, col2 = st.columns(2)
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        col1.metric("Toplam Mesaj", total_messages)
        col2.metric("Sorularınız", user_messages)
    
    st.markdown("---")
    
    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Powered by Groq & LangGraph")

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    with st.spinner("📚 Bilgi tabanı yükleniyor..."):
        try:
            loader = PyPDFDirectoryLoader("data/fitness_pdfs/")
            documents = loader.load()
            
            if not documents:
                st.warning("⚠️ data/fitness_pdfs/ klasöründe PDF bulunamadı!")
                return None
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(texts, embeddings)
            
            st.success(f"✅ {len(documents)} PDF yüklendi!")
            return vectorstore
        except Exception as e:
            st.error(f"❌ Vector store hatası: {e}")
            return None

def create_agent(api_key):
    if not api_key:
        return None
        
    vectorstore = load_vectorstore()
    if not vectorstore:
        return None
    
    from langchain_core.tools import create_retriever_tool
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="fitness_knowledge",
        description="Fitness ve beslenme bilgilerini içeren PDF'lerden arama yapar. Kullan: egzersiz, beslenme, protein, antrenman soruları için."
    )
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.3
    )
    
    memory = MemorySaver()
    
    system_prompt = """Sen profesyonel bir fitness koçu ve beslenme uzmanısın.

KURALLAR:
- Kısa ve öz cevapla (max 150 kelime)
- Yoga, meditasyon veya ruhani pratiklerden BAHSETME
- Sadece klasik fitness egzersizleri öner (şınav, dambıl, squat, vb.)
- ÖNCEKİ KONUŞMAYI HATIRLA ve takip sorularını akıllıca yanıtla
- "onu", "bunu", "bunlar" gibi referansları önceki mesajlardan anla"""
    
    agent = create_react_agent(
        llm,
        tools=[retriever_tool],
        checkpointer=memory,
        prompt=system_prompt
    )
    
    return agent

# Ana uygulama alanı - API key kontrolü
if not groq_api_key:
    st.error("🔑 Lütfen yan menüden Groq API anahtarınızı girin!")
    st.info("""
    **Groq API Anahtarı Nasıl Alınır?**
    
    1. [Groq Console](https://console.groq.com/keys) adresine gidin
    2. Hesap oluşturun veya giriş yapın
    3. "API Keys" bölümünden yeni anahtar oluşturun
    4. Anahtarı kopyalayıp yan menüdeki alana yapıştırın
    """)
    st.stop()

st.subheader("💡 Örnek Sorular")
col1, col2 = st.columns(2)

with col1:
    if st.button("🏋️ Evde yapabileceğim bir kol antrenmanı öner"):
        st.session_state.example_clicked = "Evde yapabileceğim bir kol antrenmanı öner"
    if st.button("🏃 Kardiyo mu ağırlık mı daha etkili?"):
        st.session_state.example_clicked = "Kardiyo mu ağırlık mı daha etkili?"

with col2:
    if st.button("🥗 Kas yapmak için nasıl beslenmem gerekir?"):
        st.session_state.example_clicked = "Kas yapmak için nasıl beslenmem gerekir?"
    if st.button("💪 Günlük kaç protein almalıyım?"):
        st.session_state.example_clicked = "Günlük kaç protein almalıyım?"

st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

if hasattr(st.session_state, 'example_clicked'):
    prompt = st.session_state.example_clicked
    delattr(st.session_state, 'example_clicked')
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("💭 Düşünüyorum..."):
            agent = create_agent(groq_api_key)
            if agent:
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config
                    )
                    response = result["messages"][-1].content
                    st.markdown(response)
                except Exception as e:
                    response = f"❌ Hata: {str(e)}"
                    st.error(response)
            else:
                response = "❌ Agent oluşturulamadı."
                st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    st.info("👋 **Hoş Geldiniz!** Yukarıdaki örnek sorulardan birini seçin veya aşağıya kendi sorunuzu yazın.")
    st.success("""🧠 **LangGraph Checkpoint Memory Aktif!**
    
Otomatik hafıza ile:
- "Peki bunu kaç set yapmalıyım?"
- "Hangi besinlerden alabilirim?"
- "Onu daha detaylı anlat"

gibi takip soruları anlıyorum! 🎯""")

if prompt := st.chat_input("💬 Fitness hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("💭 Düşünüyorum..."):
            agent = create_agent(groq_api_key)
            if agent:
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=prompt)]},
                        config
                    )
                    response = result["messages"][-1].content
                    st.markdown(response)
                except Exception as e:
                    response = f"❌ Hata: {str(e)}"
                    st.error(response)
            else:
                response = "❌ Agent oluşturulamadı."
                st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()