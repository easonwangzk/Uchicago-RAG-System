# app.py - Streamlit Web Interface for UChicago MS Applied Data Science RAG System
import streamlit as st
import os
from dotenv import load_dotenv
from datetime import datetime
import time
from rag import (
    load_or_build_vectordb,
    answer_question,
    getenv_str,
    getenv_int,
    getenv_float,
    ensure_api_key
)

# Page configuration
st.set_page_config(
    page_title="UChicago MS Applied Data Science - AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://datascience.uchicago.edu',
        'Report a bug': None,
        'About': "UChicago MS Applied Data Science AI Assistant - Powered by RAG (Retrieval Augmented Generation)"
    }
)

# Load environment variables
load_dotenv()

# Helper functions to get config from Streamlit secrets or environment variables
def get_config(key: str, default: str = "") -> str:
    """Get configuration from Streamlit secrets first, then fall back to environment variables"""
    # Try Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and key in st.secrets:
        return str(st.secrets[key])
    # Fall back to environment variables (for local development)
    return os.getenv(key, default)

def get_config_int(key: str, default: int) -> int:
    """Get integer configuration"""
    try:
        value = get_config(key, str(default))
        return int(value)
    except (ValueError, TypeError):
        return default

def get_config_float(key: str, default: float) -> float:
    """Get float configuration"""
    try:
        value = get_config(key, str(default))
        return float(value)
    except (ValueError, TypeError):
        return default

# Enhanced CSS with modern design system and larger fonts
def load_custom_css():
    st.markdown("""
    <style>
    /* Modern CSS Reset and Base Styles */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 20px;
    }

    /* Sidebar background color */
    section[data-testid="stSidebar"] {
        background-color: #800000 !important;
        color: white !important;
    }

    /* Adjust all text in sidebar */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Adjust links in sidebar */
    section[data-testid="stSidebar"] a {
        color: white !important;
    }

    /* Enhanced Header with Gradient */
    .main-header {
        background: linear-gradient(135deg, #800000 0%, #5a0000 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(128, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #FFD700, #FF6B35);
    }

    /* Modern Chat Messages - LARGER FONTS */
    .chat-message-container {
        display: flex;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.3s ease-out;
    }

    .user-message-container {
        justify-content: flex-end;
    }

    .assistant-message-container {
        justify-content: flex-start;
    }

    .chat-message {
        max-width: 70%;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        line-height: 1.7;
        position: relative;
        backdrop-filter: blur(10px);
        font-size: 1.6rem !important;
    }

    .user-message {
        background: linear-gradient(135deg, #007399 0%, #005577 100%);
        color: white;
        border-bottom-right-radius: 4px;
    }

    .assistant-message {
        background: white;
        border: 1px solid #e1e5e9;
        border-bottom-left-radius: 4px;
    }

    /* Message Timestamp */
    .message-timestamp {
        font-size: 1.1rem;
        opacity: 0.7;
        margin-top: 0.5rem;
        text-align: right;
    }

    /* Enhanced Source Cards */
    .source-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 1.2rem;
        margin-top: 0.75rem;
        transition: all 0.2s ease;
        border-left: 4px solid #800000;
        font-size: 1.4rem;
    }

    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #800000 0%, #5a0000 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.2rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 14px rgba(128, 0, 0, 0.2);
        font-size: 1.4rem !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(128, 0, 0, 0.3);
        background: linear-gradient(135deg, #5a0000 0%, #400000 100%);
    }

    /* Suggested Questions with Cards */
    .suggestion-card {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: left;
        width: 100%;
        font-size: 1.4rem;
    }

    .suggestion-card:hover {
        background: #f8f9fa;
        border-color: #800000;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }

    /* Larger font for category expanders */
    .stExpander summary {
        font-size: 1.7rem !important;
        font-weight: bold;
    }

    /* Loading Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: #800000;
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #5a0000;
    }

    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: 500;
    }

    .status-online {
        background: #d4edda;
        color: #155724;
    }

    .status-offline {
        background: #f8d7da;
        color: #721c24;
    }

    /* Increase font sizes for other elements */
    h1 {
        font-size: 3rem !important;
    }
    h2 {
        font-size: 2.5rem !important;
    }
    h3 {
        font-size: 2rem !important;
    }
    h4, h5, h6 {
        font-size: 1.7rem !important;
    }
    p, div, span, li {
        font-size: 1.4rem !important;
    }
    .stMarkdown {
        font-size: 1.4rem !important;
    }
    .stInfo {
        font-size: 1.4rem !important;
    }

    /* MUCH LARGER chat input */
    div[data-testid="stChatInput"] {
        font-size: 2rem !important;
    }

    div[data-testid="stChatInput"] input {
        font-size: 2rem !important;
        padding: 2rem 1.8rem !important;
        border-radius: 16px !important;
        border: 3px solid #800000 !important;
        min-height: 65px !important;
    }

    div[data-testid="stChatInput"] input::placeholder {
        color: #666 !important;
        font-weight: 500 !important;
        font-size: 1.8rem !important;
    }

    /* Clear chat button styling */
    .clear-chat-button {
        margin: 1rem 0;
    }

    /* Enhanced Thinking Animation - Assistant Style */
    .thinking-message-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.5rem;
        animation: fadeInUp 0.3s ease-out;
    }

    .thinking-message {
        max-width: 200px;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        border-bottom-left-radius: 4px;
        background: white;
        border: 1px solid #e1e5e9;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Floating dots animation */
    .thinking-dots {
        display: flex;
        gap: 6px;
        padding: 0 8px;
    }

    .thinking-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #800000;
        animation: floatingDots 1.4s infinite ease-in-out;
    }

    .thinking-dot:nth-child(1) {
        animation-delay: 0s;
    }

    .thinking-dot:nth-child(2) {
        animation-delay: 0.2s;
    }

    .thinking-dot:nth-child(3) {
        animation-delay: 0.4s;
    }

    @keyframes floatingDots {
        0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
        }
        30% {
            transform: translateY(-15px);
            opacity: 1;
        }
    }

    /* Robot icon animation */
    .thinking-icon {
        font-size: 1.5rem;
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
            opacity: 0.8;
        }
        50% {
            transform: scale(1.1);
            opacity: 1;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def init_session_state():
    """Initialize session state with default values"""
    defaults = {
        'messages': [],
        'vectordb': None,
        'db_loaded': False,
        'processing': False,          # Whether currently calling the model
        'feedback': {},
        'query_count': 0,
        'session_start': datetime.now(),
        'temperature': 0.2,
        'top_k': 5,
        'min_sim': 0.20,
        'last_interaction': datetime.now(),
        'pending_query': None,        # Question waiting to be sent to model
        'waiting_for_response': False # Control frontend: disable input & show "thinking"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def load_vectordb_cached():
    """Load or build vector database with enhanced error handling"""
    try:
        # Set API key from config (supports both Streamlit secrets and env vars)
        api_key = get_config("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        ensure_api_key()

        # Get configuration from Streamlit secrets or environment variables
        pdf_path = get_config("PDF_PATH", "data/mastersprograminanalytics.pdf")
        html_dir = get_config("HTML_DIR", "data")
        db_dir = get_config("CHROMA_DIR", ".chroma")
        embed_model = get_config("EMBED_MODEL", "text-embedding-3-large")
        chunk_tokens = get_config_int("CHUNK_TOKENS", 600)
        overlap_tokens = get_config_int("OVERLAP_TOKENS", 150)

        vectordb = load_or_build_vectordb(
            pdf_path=pdf_path,
            persist_dir=db_dir,
            embed_model=embed_model,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            html_dir=html_dir
        )

        return vectordb, True
    except Exception as e:
        st.error(f"Failed to load vector database: {str(e)}")
        return None, False

def render_message(message):
    """Render a single chat message with enhanced styling"""
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message-container user-message-container">
            <div class="chat-message user-message">
                {message["content"]}
                <div class="message-timestamp">
                    {message["timestamp"].strftime("%H:%M")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Format assistant message with sources if available
        content = message["content"]

        # Add similarity score if available
        if "similarity" in message and message.get("grounded", False):
            similarity_pct = int(message["similarity"] * 100)
            content += f'<div style="margin-top: 1rem; font-size: 1.2rem; opacity: 0.7;">Relevance: {similarity_pct}%</div>'

        st.markdown(f"""
        <div class="chat-message-container assistant-message-container">
            <div class="chat-message assistant-message">
                {content}
                <div class="message-timestamp">
                    {message["timestamp"].strftime("%H:%M")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def get_ai_response(question):
    """Get AI response from RAG system with query expansion and smart retrieval"""
    try:
        # Get response from RAG system
        response = answer_question(
            vdb=st.session_state.vectordb,
            question=question,
            top_k=st.session_state.top_k,
            min_sim=st.session_state.min_sim,
            chat_model=get_config("CHAT_MODEL", "gpt-4o-mini"),
            temperature=st.session_state.temperature,
            max_tokens=get_config_int("MAX_TOKENS", 800),
            verbose=False
        )

        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "similarity": response.get("best_similarity", 0),
            "grounded": response.get("grounded", False),
            "timestamp": datetime.now()
        }
        return assistant_message

    except Exception as e:
        error_message = {
            "role": "assistant",
            "content": f"I encountered an error while processing your request: `{str(e)}`. Please try again or rephrase your question.",
            "error": True,
            "timestamp": datetime.now()
        }
        return error_message

def add_user_message(question):
    """Add user message to chat history and prepare for AI response"""
    user_message = {
        "role": "user",
        "content": question,
        "timestamp": datetime.now()
    }
    st.session_state.messages.append(user_message)
    st.session_state.query_count += 1
    st.session_state.last_interaction = datetime.now()
    # Set state to wait for response - actual model call handled in main()
    st.session_state.waiting_for_response = True
    st.session_state.pending_query = question

def render_suggested_questions():
    """Render suggested questions as interactive cards organized by category"""
    st.markdown("### Quick Questions")

    categories = {
        "Program Structure": [
            "What are the core courses?",
            "What elective courses are available?",
            "How long does the program take to complete?"
        ],
        "Admissions": [
            "What are the admission requirements?",
            "What is the application deadline?",
            "Is GRE required for admission?"
        ],
        "Capstone & Career": [
            "Tell me about the capstone project",
            "What are the career outcomes?",
            "What companies hire graduates?"
        ]
    }

    for category, questions in categories.items():
        with st.expander(category, expanded=True):
            for question in questions:
                if st.button(
                    question,
                    key=f"suggest_{hash(question)}",
                    use_container_width=True,
                    disabled=st.session_state.waiting_for_response
                ):
                    # Add user message and trigger rerun
                    add_user_message(question)
                    st.rerun()

def render_sidebar():
    """Enhanced sidebar with system info and resources"""
    with st.sidebar:
        # Logo and Header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <img src="https://datascience.uchicago.edu/wp-content/uploads/2023/01/DSI_Logo_Color@1x.svg"
                 width="180" style="margin-bottom: 1rem;">
        </div>
        """, unsafe_allow_html=True)

        # System Status
        st.markdown("### System Status")
        if st.session_state.db_loaded:
            st.success("RAG System: Online")
            st.info(f"Vector DB: 207 chunks loaded")
            st.info(f"Model: {get_config('CHAT_MODEL', 'gpt-4o-mini')}")
        else:
            st.error("RAG System: Offline")

        st.markdown("---")

        # Session Statistics
        st.markdown("### Session Stats")
        session_duration = datetime.now() - st.session_state.session_start
        st.metric("Queries", st.session_state.query_count)
        st.metric("Duration", f"{session_duration.seconds // 60}m {session_duration.seconds % 60}s")

        st.markdown("---")

        # Clear Chat Button
        if st.button("Clear Chat History", use_container_width=True, key="clear_chat"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.session_start = datetime.now()
            st.session_state.waiting_for_response = False
            st.session_state.pending_query = None
            st.session_state.processing = False
            st.rerun()

        st.markdown("---")

        # Advanced Settings
        with st.expander("Advanced Settings"):
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Lower values make responses more focused and deterministic"
            )
            st.session_state.top_k = st.slider(
                "Top K Chunks",
                min_value=1,
                max_value=10,
                value=st.session_state.top_k,
                help="Number of document chunks to retrieve"
            )
            st.session_state.min_sim = st.slider(
                "Min Similarity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.min_sim,
                step=0.05,
                help="Minimum similarity threshold for retrieval"
            )

        st.markdown("---")

        # Resources
        st.markdown("### Resources")
        resources = {
            "Program Website": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/",
            "Apply Now": "https://apply-psd.uchicago.edu/apply/",
            "FAQs": "https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/",
            "Contact": "https://datascience.uchicago.edu/contact/"
        }

        for text, url in resources.items():
            st.markdown(
                f'<a href="{url}" target="_blank" '
                f'style="text-decoration: none; color: white; display: block; '
                f'padding: 0.5rem; border-radius: 8px; margin: 0.25rem 0; '
                f'border: 1px solid rgba(255,255,255,0.2);">{text}</a>',
                unsafe_allow_html=True
            )

def main():
    """Main application with enhanced layout and RAG integration"""
    # Initialize
    init_session_state()
    load_custom_css()

    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">UChicago MS in Applied Data Science</h1>
        <h3 style="margin: 0.5rem 0 0 0; font-weight: 400; opacity: 0.9;">AI-Powered Program Information Assistant</h3>
        <p style="margin: 1rem 0 0 0; font-size: 1.1rem; opacity: 0.8;">
            Intelligent Q&A powered by RAG (Retrieval Augmented Generation) with query expansion and smart deduplication
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load Vector Database
    if not st.session_state.db_loaded:
        with st.container():
            with st.spinner("Loading knowledge base..."):
                vectordb, success = load_vectordb_cached()
            if success:
                st.session_state.vectordb = vectordb
                st.session_state.db_loaded = True
                st.success("Knowledge base loaded successfully! Ready to answer your questions.")
            else:
                st.error("""
                **Unable to initialize AI Assistant**

                Please check:
                - Your OpenAI API key is configured in .env file
                - The required data files exist in the data/ directory
                - Your internet connection is stable
                """)
                return

    # Main Layout
    col1, col2 = st.columns([7, 3])

    with col1:
        # Chat Container
        st.markdown("### Conversation")

        chat_container = st.container()
        with chat_container:
            # Display message history
            for message in st.session_state.messages:
                render_message(message)

            # Show "thinking" animation while waiting for response
            if st.session_state.waiting_for_response and st.session_state.pending_query is not None:
                st.markdown("""
                <div class="thinking-message-container">
                    <div class="thinking-message">
                        <span class="thinking-icon">🤖</span>
                        <div class="thinking-dots">
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                            <div class="thinking-dot"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Input area
        user_input = st.chat_input(
            "Ask about admissions, courses, careers, or anything else...",
            key="chat_input",
            disabled=st.session_state.waiting_for_response
        )
        if user_input:
            # Add user message and trigger rerun
            add_user_message(user_input)
            st.rerun()

    with col2:
        # Suggested Questions
        render_suggested_questions()

    # Render sidebar
    render_sidebar()

    # Process pending query (unified handling to avoid duplicate responses)
    if (
        st.session_state.waiting_for_response
        and st.session_state.pending_query is not None
        and not st.session_state.processing
    ):
        st.session_state.processing = True
        # Get AI response using RAG system
        with st.spinner("Processing with RAG system..."):
            assistant_message = get_ai_response(st.session_state.pending_query)
        st.session_state.messages.append(assistant_message)
        st.session_state.waiting_for_response = False
        st.session_state.pending_query = None
        st.session_state.processing = False
        st.rerun()

if __name__ == "__main__":
    main()
