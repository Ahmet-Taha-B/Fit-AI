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

# Çeviriler
TRANSLATIONS = {
    "tr": {
        "page_title": "Fitness AI Coach",
        "page_subtitle": "Kişisel fitness ve beslenme asistanınız (LangGraph Hafızalı! 🧠)",
        "api_settings": "🔑 API Ayarları",
        "api_key_label": "Groq API Key",
        "api_key_help": "Groq API anahtarınızı girin (https://console.groq.com/keys)",
        "api_warning": "⚠️ Lütfen API anahtarınızı girin!",
        "api_success": "✅ API anahtarı girildi!",
        "features": "🎯 Özellikler",
        "features_list": """
    ✅ Kişiselleştirilmiş fitness tavsiyeleri
    ✅ Beslenme programları
    ✅ Egzersiz önerileri
    ✅ Bilimsel kaynaklara dayalı
    🧠 **LangGraph Checkpoint Memory**
    """,
        "stats": "📊 Seans İstatistikleri",
        "total_messages": "Toplam Mesaj",
        "your_questions": "Sorularınız",
        "clear_chat": "🗑️ Sohbeti Temizle",
        "powered_by": "Powered by Groq & LangGraph",
        "loading_kb": "📚 Bilgi tabanı yükleniyor...",
        "no_pdfs": "⚠️ data/fitness_pdfs/ klasöründe PDF bulunamadı!",
        "pdfs_loaded": "✅ {count} PDF yüklendi!",
        "vectorstore_error": "❌ Vector store hatası: {error}",
        "retriever_desc": "Fitness ve beslenme bilgilerini içeren PDF'lerden arama yapar. Kullan: egzersiz, beslenme, protein, antrenman soruları için.",
        "system_prompt": """Sen profesyonel bir fitness koçu ve beslenme uzmanısın.

KURALLAR:
- Kısa ve öz cevapla (max 150 kelime)
- Yoga, meditasyon veya ruhani pratiklerden BAHSETME
- Sadece klasik fitness egzersizleri öner (şınav, dambıl, squat, vb.)
- ÖNCEKİ KONUŞMAYI HATIRLA ve takip sorularını akıllıca yanıtla
- "onu", "bunu", "bunlar" gibi referansları önceki mesajlardan anla
- TÜRKÇE CEVAP VER""",
        "api_error": "🔑 Lütfen yan menüden Groq API anahtarınızı girin!",
        "api_info": """
    **Groq API Anahtarı Nasıl Alınır?**
    
    1. [Groq Console](https://console.groq.com/keys) adresine gidin
    2. Hesap oluşturun veya giriş yapın
    3. "API Keys" bölümünden yeni anahtar oluşturun
    4. Anahtarı kopyalayıp yan menüdeki alana yapıştırın
    """,
        "example_questions": "💡 Örnek Sorular",
        "q1": "🏋️ Evde yapabileceğim bir kol antrenmanı öner",
        "q2": "🏃 Kardiyo mu ağırlık mı daha etkili?",
        "q3": "🥗 Kas yapmak için nasıl beslenmem gerekir?",
        "q4": "💪 Günlük kaç protein almalıyım?",
        "q1_text": "Evde yapabileceğim bir kol antrenmanı öner",
        "q2_text": "Kardiyo mu ağırlık mı daha etkili?",
        "q3_text": "Kas yapmak için nasıl beslenmem gerekir?",
        "q4_text": "Günlük kaç protein almalıyım?",
        "thinking": "💭 Düşünüyorum...",
        "agent_error": "❌ Agent oluşturulamadı.",
        "welcome": "👋 **Hoş Geldiniz!** Yukarıdaki örnek sorulardan birini seçin veya aşağıya kendi sorunuzu yazın.",
        "memory_info": """🧠 **LangGraph Checkpoint Memory Aktif!**
    
Otomatik hafıza ile:
- "Peki bunu kaç set yapmalıyım?"
- "Hangi besinlerden alabilirim?"
- "Onu daha detaylı anlat"

gibi takip soruları anlıyorum! 🎯""",
        "chat_placeholder": "💬 Fitness hakkında bir soru sorun...",
        "language": "🌐 Dil / Language",
        "select_language": "Lütfen dilinizi seçin",
        "welcome_title": "Hoş Geldiniz! 💪"
    },
    "en": {
        "page_title": "Fitness AI Coach",
        "page_subtitle": "Your personal fitness and nutrition assistant (With LangGraph Memory! 🧠)",
        "api_settings": "🔑 API Settings",
        "api_key_label": "Groq API Key",
        "api_key_help": "Enter your Groq API key (https://console.groq.com/keys)",
        "api_warning": "⚠️ Please enter your API key!",
        "api_success": "✅ API key entered!",
        "features": "🎯 Features",
        "features_list": """
    ✅ Personalized fitness advice
    ✅ Nutrition plans
    ✅ Exercise recommendations
    ✅ Science-backed information
    🧠 **LangGraph Checkpoint Memory**
    """,
        "stats": "📊 Session Statistics",
        "total_messages": "Total Messages",
        "your_questions": "Your Questions",
        "clear_chat": "🗑️ Clear Chat",
        "powered_by": "Powered by Groq & LangGraph",
        "loading_kb": "📚 Loading knowledge base...",
        "no_pdfs": "⚠️ No PDFs found in data/fitness_pdfs/ folder!",
        "pdfs_loaded": "✅ {count} PDFs loaded!",
        "vectorstore_error": "❌ Vector store error: {error}",
        "retriever_desc": "Searches fitness and nutrition information from PDFs. Use for: exercise, nutrition, protein, workout questions.",
        "system_prompt": """You are a professional fitness coach and nutrition expert.

RULES:
- Keep answers short and concise (max 150 words)
- DO NOT mention yoga, meditation or spiritual practices
- Only suggest classic fitness exercises (push-ups, dumbbells, squats, etc.)
- REMEMBER PREVIOUS CONVERSATION and answer follow-up questions intelligently
- Understand references like "it", "this", "those" from previous messages
- ANSWER IN ENGLISH""",
        "api_error": "🔑 Please enter your Groq API key from the sidebar!",
        "api_info": """
    **How to Get a Groq API Key?**
    
    1. Go to [Groq Console](https://console.groq.com/keys)
    2. Create an account or sign in
    3. Create a new key from "API Keys" section
    4. Copy and paste the key into the sidebar field
    """,
        "example_questions": "💡 Example Questions",
        "q1": "🏋️ Suggest a home arm workout",
        "q2": "🏃 Is cardio or weights more effective?",
        "q3": "🥗 How should I eat to build muscle?",
        "q4": "💪 How much protein should I consume daily?",
        "q1_text": "Suggest a home arm workout",
        "q2_text": "Is cardio or weights more effective?",
        "q3_text": "How should I eat to build muscle?",
        "q4_text": "How much protein should I consume daily?",
        "thinking": "💭 Thinking...",
        "agent_error": "❌ Agent could not be created.",
        "welcome": "👋 **Welcome!** Select one of the example questions above or type your own question below.",
        "memory_info": """🧠 **LangGraph Checkpoint Memory Active!**
    
With automatic memory:
- "How many sets should I do?"
- "Which foods can I get it from?"
- "Explain that in more detail"

I understand follow-up questions like these! 🎯""",
        "chat_placeholder": "💬 Ask a fitness question...",
        "language": "🌐 Language / Dil",
        "select_language": "Please select your language",
        "welcome_title": "Welcome! 💪"
    }
}

st.set_page_config(
    page_title="Fitness AI Coach",
    page_icon="💪",
    layout="wide"
)

# Dil seçimi kontrolü
if "language" not in st.session_state:
    st.session_state.language = None

# Eğer dil seçilmemişse, dil seçim ekranını göster
if st.session_state.language is None:
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .stButton > button {
            height: 150px !important;
            font-size: 18px !important;
            white-space: pre-line !important;
            line-height: 1.6 !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: white; margin-top: 60px;'>💪 Fitness AI Coach</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: rgba(255,255,255,0.9); margin-bottom: 40px;'>Welcome! / Hoş Geldiniz!</h3>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; color: white; margin-bottom: 50px;'>🌐 Please select your language / Lütfen dilinizi seçin</h2>", unsafe_allow_html=True)
    
    # Dil seçim butonları
    col_space1, col1, col_space2, col2, col_space3 = st.columns([1, 2, 0.5, 2, 1])
    
    with col1:
        if st.button("🇹🇷\n\nTÜRKÇE", key="select_turkish", use_container_width=True, type="primary"):
            st.session_state.language = "tr"
            st.rerun()
        
    with col2:
        if st.button("🇬🇧\n\nENGLISH", key="select_english", use_container_width=True, type="primary"):
            st.session_state.language = "en"
            st.rerun()
    
    st.stop()

# Aktif dil
lang = st.session_state.language
t = TRANSLATIONS[lang]

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

st.markdown(f"<h1 style='text-align: center; color: white;'>💪 {t['page_title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: rgba(255,255,255,0.8);'>{t['page_subtitle']}</p>", unsafe_allow_html=True)

# Dil değiştirme butonları ortada
col1, col2, col3, col4, col5 = st.columns([2, 0.5, 0.3, 0.5, 2])
with col2:
    if st.button("🇹🇷", key="change_turkish", use_container_width=True, help="Türkçe"):
        st.session_state.language = "tr"
        st.rerun()
with col4:
    if st.button("🇬🇧", key="change_english", use_container_width=True, help="English"):
        st.session_state.language = "en"
        st.rerun()

st.markdown("---")

with st.sidebar:
    st.header(t["api_settings"])
    groq_api_key = st.text_input(
        t["api_key_label"],
        type="password",
        placeholder="gsk_...",
        help=t["api_key_help"]
    )
    
    if not groq_api_key:
        st.warning(t["api_warning"])
    else:
        st.success(t["api_success"])
    
    st.markdown("---")
    
    # İstatistikler üstte
    st.header(t["stats"])
    if "messages" in st.session_state:
        col1, col2 = st.columns(2)
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        col1.metric(t["total_messages"], total_messages)
        col2.metric(t["your_questions"], user_messages)
    
    st.markdown("---")
    
    # Sohbeti temizle butonu altta
    if st.button(t["clear_chat"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Özellikler en altta
    st.header(t["features"])
    st.info(t["features_list"])
    
    st.markdown("---")
    st.caption(t["powered_by"])

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    with st.spinner(t["loading_kb"]):
        try:
            loader = PyPDFDirectoryLoader("data/fitness_pdfs/")
            documents = loader.load()
            
            if not documents:
                st.warning(t["no_pdfs"])
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
            
            st.success(t["pdfs_loaded"].format(count=len(documents)))
            return vectorstore
        except Exception as e:
            st.error(t["vectorstore_error"].format(error=e))
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
        description=t["retriever_desc"]
    )
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.3
    )
    
    memory = MemorySaver()
    
    agent = create_react_agent(
        llm,
        tools=[retriever_tool],
        checkpointer=memory,
        prompt=t["system_prompt"]
    )
    
    return agent

# Ana uygulama alanı - API key kontrolü
if not groq_api_key:
    st.error(t["api_error"])
    st.info(t["api_info"])
    st.stop()

st.subheader(t["example_questions"])
col1, col2 = st.columns(2)

with col1:
    if st.button(t["q1"]):
        st.session_state.example_clicked = t["q1_text"]
    if st.button(t["q2"]):
        st.session_state.example_clicked = t["q2_text"]

with col2:
    if st.button(t["q3"]):
        st.session_state.example_clicked = t["q3_text"]
    if st.button(t["q4"]):
        st.session_state.example_clicked = t["q4_text"]

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
        with st.spinner(t["thinking"]):
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
                    response = f"❌ {lang.upper()}: {str(e)}"
                    st.error(response)
            else:
                response = t["agent_error"]
                st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if len(st.session_state.messages) == 0:
    st.info(t["welcome"])
    st.success(t["memory_info"])

if prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(t["thinking"]):
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
                    response = f"❌ {lang.upper()}: {str(e)}"
                    st.error(response)
            else:
                response = t["agent_error"]
                st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()