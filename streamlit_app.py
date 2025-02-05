import streamlit as st
import os
import dotenv
import uuid

# Ensure Streamlit adapts to all devices
st.set_page_config(
    page_title="RAG LLM App",
    page_icon="📚",
    layout="wide",  # Makes it responsive on all screen sizes
    initial_sidebar_state="expanded",
)

# Apply CSS to remove sidebar padding and adjust heading sizes
st.markdown("""
    <style>
        /* Move sidebar content to the very top */
        [data-testid="stSidebar"] {
            padding-top: 0px !important;
            margin-top: -1000px !important; /* Forces content up */
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 0px !important;
            margin-top: -1000px !important; /* Forces content up */
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
dotenv.load_dotenv()

# Header with responsive styling
st.markdown("<h1 style='text-align: center; font-size: 24px;'>🔍 <i>RAG Chatbot</i></h1>", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

# Define available models
if "AZ_OPENAI_API_KEY" not in os.environ:
    MODELS = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet-20240620",
    ]
else:
    MODELS = ["azure-openai/gpt-4o"]

# Sidebar & Main Content layout for responsiveness
sidebar, main_content = st.columns([1, 3])

with sidebar:
    st.markdown("<h2 style='text-align: left; font-size: 24px;'>🔐 API Keys</h2>", unsafe_allow_html=True)
    if "AZ_OPENAI_API_KEY" not in os.environ:
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
        anthropic_api_key = st.text_input("Enter Anthropic API Key", type="password")
    else:
        openai_api_key, anthropic_api_key = None, None
        st.session_state.openai_api_key = None
        az_openai_api_key = os.getenv("AZ_OPENAI_API_KEY")
        st.session_state.az_openai_api_key = az_openai_api_key

    st.markdown("<h2 style='text-align: left; font-size: 24px;'>🤖 Model Selection</h2>", unsafe_allow_html=True)
    st.selectbox("Choose a Model", MODELS, key="model")

    with st.expander("⚙️ Use RAG Mode", expanded=True):
        is_vector_db_loaded = "vector_db" in st.session_state and st.session_state.vector_db is not None
        st.toggle("Enable RAG", value=is_vector_db_loaded, key="use_rag", disabled=not is_vector_db_loaded)

    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    with st.expander("📄 Upload RAG Documents", expanded=True):
        st.file_uploader("Upload a document", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True, key="rag_docs")

    with st.expander("🌐 Add URL Source", expanded=True):
        st.text_input("Enter a URL", placeholder="https://example.com", key="rag_url")

    with st.expander(f"📚 Documents in DB ({0 if not is_vector_db_loaded else len(st.session_state.rag_sources)})"):
        st.write([] if not is_vector_db_loaded else [source for source in st.session_state.rag_sources])

with main_content:
    # Display chat messages inside a scrollable container
    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User Input with scrollable text box
    prompt = st.text_area("Your message", height=50)
    if st.button("Send"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            from langchain.schema import HumanMessage, AIMessage
            from langchain_openai import ChatOpenAI, AzureChatOpenAI
            from langchain_anthropic import ChatAnthropic
            from rag_methods import (
                load_doc_to_db,
                load_url_to_db,
                stream_llm_response,
                stream_llm_rag_response,
            )

            messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in st.session_state.messages
            ]

            model_provider = st.session_state.model.split("/")[0]
            if model_provider == "openai":
                llm_stream = ChatOpenAI(
                    api_key=openai_api_key,
                    model_name=st.session_state.model.split("/")[-1],
                    temperature=0.3,
                    streaming=True,
                )
            elif model_provider == "anthropic":
                llm_stream = ChatAnthropic(
                    api_key=anthropic_api_key,
                    model=st.session_state.model.split("/")[-1],
                    temperature=0.3,
                    streaming=True,
                )
            elif model_provider == "azure-openai":
                llm_stream = AzureChatOpenAI(
                    azure_endpoint=os.getenv("AZ_OPENAI_ENDPOINT"),
                    openai_api_version="2024-02-15-preview",
                    model_name=st.session_state.model.split("/")[-1],
                    openai_api_key=os.getenv("AZ_OPENAI_API_KEY"),
                    openai_api_type="azure",
                    temperature=0.3,
                    streaming=True,
                )

            if not st.session_state.use_rag:
                st.write_stream(stream_llm_response(llm_stream, messages))
            else:
                st.write_stream(stream_llm_rag_response(llm_stream, messages))
