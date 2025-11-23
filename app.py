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
from dotenv import load_dotenv
from translations import TRANSLATIONS

load_dotenv()

st.set_page_config(
    page_title="Fitness AI Coach",
    page_icon="💪",
    layout="wide"
)

# Initialize session state for language if not exists
if "language" not in st.session_state:
    st.session_state.language = None

def load_custom_css():
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_custom_css()

# Language Selection Logic
if st.session_state.language is None:
    st.markdown("<h1 style='text-align: center;'>Fitness AI Coach 💪</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Please select your language / Lütfen dilinizi seçin</p>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🇹🇷 TÜRKÇE", key="select_tr", use_container_width=True):
            st.session_state.language = "tr"
            st.rerun()
            
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        if st.button("🇬🇧 ENGLISH", key="select_en", use_container_width=True):
            st.session_state.language = "en"
            st.rerun()
            
    st.markdown("---")
    st.caption("Powered by Fitness AI Coach")
    st.stop()

t = TRANSLATIONS[st.session_state.language]

# Header
st.markdown("<h1 style='text-align: center; margin: 0; padding: 0;'>Fitness AI Coach 💪</h1>", unsafe_allow_html=True)

st.markdown("---")

with st.sidebar:
    st.header(t["api_settings"])
    groq_api_key = st.text_input(
        t["api_key_label"],
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
        help=t["api_key_help"]
    )
    st.markdown(t["get_api_key_link"])
    
    if not groq_api_key:
        st.warning(t["api_warning"])
    else:
        st.success(t["api_success"])
    
    st.markdown("---")
    
    st.header(t["stats"])
    if "messages" in st.session_state:
        col1, col2 = st.columns(2)
        total_messages = len(st.session_state.messages)
        user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        
        col1.metric(t["total_messages"], total_messages)
        col2.metric(t["your_questions"], user_messages)
    
    st.markdown("---")
    
    if st.button(t["clear_chat"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")

    # Language Switcher (Expandable)
    with st.expander("🌐 Language / Dil"):
        lang_options = {"tr": "🇹🇷 Türkçe", "en": "🇬🇧 English"}
        selected_lang = st.radio(
            "Select Language / Dil Seçin",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            index=0 if st.session_state.language == "tr" else 1,
            key="sidebar_lang_select"
        )
        
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
    
    st.markdown("---")
    
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

@st.cache_resource(show_spinner=False)
def create_agent(api_key, system_prompt):
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
        prompt=system_prompt
    )
    
    return agent

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
            agent = create_agent(groq_api_key, t["system_prompt"])
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
                    response = f"❌ {st.session_state.language.upper()}: {str(e)}"
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
            agent = create_agent(groq_api_key, t["system_prompt"])
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
                    response = f"❌ {st.session_state.language.upper()}: {str(e)}"
                    st.error(response)
            else:
                response = t["agent_error"]
                st.error(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()