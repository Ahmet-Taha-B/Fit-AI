import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from translations import TRANSLATIONS

load_dotenv()

# --- Configuration and Initialization ---
# Set page config
st.set_page_config(
    page_title="Fitness AI Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "language" not in st.session_state:
    st.session_state.language = None # Default to None to show selection screen

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
                return None, 0
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = Chroma.from_documents(texts, embeddings)
            
            # st.success removed from here to prevent caching issue with language
            return vectorstore, len(documents)
        except Exception as e:
            st.error(t["vectorstore_error"].format(error=e))
            return None, 0

@st.cache_resource(show_spinner=False)
def create_agent(api_key, system_prompt, temperature=0.5):
    if not api_key:
        return None
        
    vectorstore, doc_count = load_vectorstore()
    if not vectorstore:
        return None
    
    # Display success message here. Since create_agent re-runs when system_prompt changes 
    # (which happens when language changes), this ensures the message is in the correct language.
    st.success(t["pdfs_loaded"].format(count=doc_count))
    
    from custom_tools import create_retriever_tool
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    retriever_tool = create_retriever_tool(
        retriever,
        name="fitness_knowledge",
        description=t["retriever_desc"]
    )
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=temperature
    )
    
    memory = MemorySaver()
    
    # Custom ReAct Agent Implementation
    
    # Define the state
    class AgentState(MessagesState):
        pass

    # Define the nodes
    def call_model(state: AgentState):
        messages = state['messages']
        # Prepend system prompt to ensure instructions are followed
        messages_with_prompt = [SystemMessage(content=system_prompt)] + messages
        response = llm.bind_tools([retriever_tool]).invoke(messages_with_prompt)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Define the graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([retriever_tool]))
    
    workflow.add_edge(START, "agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        ["tools", END]
    )
    
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    agent = workflow.compile(checkpointer=memory)
    
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
            # Default style for example questions or use a default
            temperature = 0.5
            agent = create_agent(groq_api_key, t["system_prompt"], temperature)
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
        content = message["content"]
        if "<thinking>" in content and "</thinking>" in content:
            try:
                start_tag = "<thinking>"
                end_tag = "</thinking>"
                start_index = content.find(start_tag) + len(start_tag)
                end_index = content.find(end_tag)
                
                thinking_content = content[start_index:end_index].strip()
                final_answer = content[end_index + len(end_tag):].strip()
                
                with st.expander(f"🧠 {t['thinking_process']}"):
                    st.markdown(thinking_content)
                st.markdown(final_answer)
            except:
                st.markdown(content)
        else:
            st.markdown(content)

if len(st.session_state.messages) == 0:
    st.info(t["welcome"])

# Style Selection (Popover above chat input)
style_options = {
    "concise": t.get("style_concise", "Concise"),
    "normal": t.get("style_normal", "Normal"),
    "creative": t.get("style_creative", "Creative")
}

# Add custom styles to options
if "custom_styles" not in st.session_state:
    st.session_state.custom_styles = {}

for style_id, style_data in st.session_state.custom_styles.items():
    style_options[style_id] = f"✨ {style_data['name']}"

@st.dialog(t["custom_style_title"])
def configure_custom_style(style_id=None):
    st.write(t["custom_style_title"])
    
    # Initialize defaults
    default_name = t.get("default_custom_name", "My Custom Coach")
    default_prompt = t.get("default_custom_prompt", "You are a helpful coach.")
    default_temp = 0.7
    
    if style_id and style_id in st.session_state.custom_styles:
        data = st.session_state.custom_styles[style_id]
        default_name = data["name"]
        default_prompt = data["prompt"]
        default_temp = data["temperature"]
    
    name = st.text_input(t["custom_style_name"], value=default_name)
    prompt = st.text_area(t["custom_style_prompt"], value=default_prompt)
    temperature = st.slider(t["custom_style_temp"], 0.0, 2.0, default_temp)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(t["save"], use_container_width=True):
            import uuid
            new_id = style_id if style_id else str(uuid.uuid4())
            
            st.session_state.custom_styles[new_id] = {
                "name": name,
                "prompt": prompt,
                "temperature": temperature
            }
            st.session_state.style_select_main = new_id
            st.success(t["style_saved"])
            st.rerun()
            
    with col2:
        if style_id and st.button(t["delete_style"], use_container_width=True):
            del st.session_state.custom_styles[style_id]
            st.session_state.style_select_main = "normal"
            st.rerun()

# Create a container for the style selector to keep it close to the input
with st.container():
    # Initialize style selection if not present
    if "style_select_main" not in st.session_state:
        st.session_state.style_select_main = "normal"
        
    # Get current label for the popover button
    current_style_key = st.session_state.style_select_main
    
    # Handle potential key error if state has invalid value (e.g. deleted style)
    if current_style_key not in style_options:
        current_style_key = "normal"
        st.session_state.style_select_main = "normal"
    
    popover_label = style_options[current_style_key]

    # Use columns to position it to the right, similar to the screenshot
    # [Spacer, Thinking Toggle, Style Selector]
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col2:
        st.toggle("🧠", key="thinking_mode", help="Thinking Mode / Düşünme Modu")

    with col3:
        # Using a popover with the current selection as label
        with st.popover(popover_label, use_container_width=True):
            selected = st.radio(
                t.get("style_label", "Style"),
                options=list(style_options.keys()),
                format_func=lambda x: style_options[x],
                key="style_select_main",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            if st.button(t["add_custom_style"], use_container_width=True):
                configure_custom_style()
                
            if selected in st.session_state.custom_styles:
                if st.button(t["edit_custom_style"], use_container_width=True):
                    configure_custom_style(selected)

if prompt := st.chat_input(t["chat_placeholder"]):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(t["thinking"]):
            # Determine style parameters
            temperature = 0.5
            style_prompt = ""
            
            # Re-fetch selected style key because it might have changed
            selected_style_key = st.session_state.get("style_select_main", "normal")
            is_thinking_mode = st.session_state.get("thinking_mode", False)
            
            # Language-specific prompts
            if st.session_state.language == "tr":
                if selected_style_key == "concise":
                    temperature = 0.3
                    style_prompt = "\n\nEK TALİMAT: Cevabını kısa, öz ve doğrudan tut. Maksimum 100 kelime."
                elif selected_style_key == "creative":
                    temperature = 1.0
                    style_prompt = "\n\nEK TALİMAT: Cevabında yaratıcı, hayal gücü yüksek ve ilgi çekici ol. Maksimum 250 kelime."
                elif selected_style_key in st.session_state.custom_styles:
                    config = st.session_state.custom_styles[selected_style_key]
                    temperature = config["temperature"]
                    style_prompt = f"\n\nEK TALİMAT: {config['prompt']}"
                else: # Normal
                    style_prompt = "\n\nEK TALİMAT: Cevabını dengeli tut. Maksimum 180 kelime."
                
                if is_thinking_mode:
                    style_prompt += "\n\nKRİTİK: 'Düşünme Modu'ndasın. Mantık yürütme sürecini açıkça göstermelisin. \n\nCEVABINI TAM OLARAK ŞU FORMATTA VER:\n<thinking>\n[Buraya adım adım düşünme sürecini yaz. Problemi parçalara ayır, farklı açıları değerlendir ve varsayımlarını doğrula.]\n</thinking>\n[Nihai cevabın buraya.]\n\nHATIRLATMA: <thinking> etiketlerini ASLA atlama. Cevap vermeden önce mutlaka düşün."
            else: # English
                if selected_style_key == "concise":
                    temperature = 0.3
                    style_prompt = "\n\nEXTRA INSTRUCTION: Keep your response short, concise and to the point. Maximum 100 words."
                elif selected_style_key == "creative":
                    temperature = 1.0
                    style_prompt = "\n\nEXTRA INSTRUCTION: Be creative, imaginative and engaging in your response. Maximum 250 words."
                elif selected_style_key in st.session_state.custom_styles:
                    config = st.session_state.custom_styles[selected_style_key]
                    temperature = config["temperature"]
                    style_prompt = f"\n\nEXTRA INSTRUCTION: {config['prompt']}"
                else: # Normal
                    style_prompt = "\n\nEXTRA INSTRUCTION: Keep your response balanced. Maximum 180 words."
                
                if is_thinking_mode:
                    style_prompt += "\n\nCRITICAL: You are in 'Thinking Mode'. You MUST explicitly show your reasoning process. \n\nFORMAT YOUR RESPONSE EXACTLY LIKE THIS:\n<thinking>\n[Your step-by-step reasoning goes here. Break down the problem, consider multiple angles, and verify your assumptions.]\n</thinking>\n[Your final answer goes here.]\n\nREMINDER: NEVER skip the <thinking> tags. Always think before answering."
            
            if is_thinking_mode:
                temperature = 0.2 # Force low temp for reasoning

            final_system_prompt = t["system_prompt"] + style_prompt
            
            agent = create_agent(groq_api_key, final_system_prompt, temperature)
            if agent:
                try:
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    if is_thinking_mode:
                        with st.status(t["thinking_process_streaming"], expanded=True) as status:
                            response_placeholder = st.empty()
                            full_response = ""
                            raw_response = ""
                            
                            # Stream the events from the graph
                            events = agent.stream(
                                {"messages": [HumanMessage(content=prompt)]},
                                config,
                                stream_mode="values"
                            )
                            
                            for event in events:
                                if "messages" in event:
                                    last_msg = event["messages"][-1]
                                    if isinstance(last_msg, AIMessage):
                                        content = last_msg.content
                                        
                                        # Handle tool calls
                                        if last_msg.tool_calls:
                                            for tool_call in last_msg.tool_calls:
                                                status.write(t["consulting_tool"].format(tool_name=tool_call['name']))
                                                status.update(label=t["consulting_tool_status"].format(tool_name=tool_call['name']), state="running")
                                        
                                        # Handle structured thinking content
                                        if "<thinking>" in content and "</thinking>" in content:
                                            # Extract thinking part
                                            start_tag = "<thinking>"
                                            end_tag = "</thinking>"
                                            start_index = content.find(start_tag) + len(start_tag)
                                            end_index = content.find(end_tag)
                                            
                                            thinking_content = content[start_index:end_index].strip()
                                            final_answer = content[end_index + len(end_tag):].strip()
                                            
                                            # Update status with the reasoning
                                            status.write(thinking_content)
                                            full_response = final_answer
                                        elif "<thinking>" in content:
                                            # Partial thinking content (streaming)
                                            status.write(t["reasoning_streaming"])
                                            full_response = content # Fallback
                                        else:
                                            # No tags found yet or normal response
                                            full_response = content
                                        
                                        # Keep track of the raw content for saving
                                        raw_response = content
                                            
                            status.update(label=t["thinking_complete"], state="complete", expanded=False)
                            
                            # Fallback if response is empty
                            if not raw_response:
                                full_response = t.get("error_no_response", "⚠️ Bir hata oluştu, cevap üretilemedi.")
                                raw_response = full_response
                                
                            st.markdown(full_response)
                            response = raw_response
                    else:
                        # Standard invoke
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