from pathlib import Path
import streamlit as st
import time

from rag import create_app_services, OllamaHealth

# ---------- Setup ----------
PROJECT_ROOT = Path(__file__).resolve().parent
svc = create_app_services(PROJECT_ROOT)
ollama = OllamaHealth()

st.set_page_config(
    page_title="RAG Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- TIGHT & CUSTOM CSS ----------
st.markdown(
    """
    <style>
        /* ... existing sidebar title styles ... */
        [data-testid="stSidebar"] h1 {
            margin-top: -10px !important;
            margin-bottom: 15px !important;
            font-size: 1.8rem;
            color: #4F8BF9;
        }

        /* --- 2. UPDATED OLLAMA BUTTON STYLES (Paste this here) --- */
        
        /* Green "Active" Button (We use type="secondary" for this) */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: #d1e7dd !important;
            color: #0f5132 !important;
            border: 1px solid #badbcc !important;
            height: 40px;
        }
        /* Green Hover */
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-color: #0f5132 !important;
            color: #0f5132 !important;
        }

        /* Red "Down" Button (We use type="primary" for this) */
        [data-testid="stSidebar"] button[kind="primary"] {
            background-color: #f8d7da !important;
            color: #842029 !important;
            border: 1px solid #f5c2c7 !important;
             height: 40px;
        }
        /* Red Hover */
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            border-color: #842029 !important;
            color: #842029 !important;
        }

        /* ... continue with your file upload/chat styles ... */
        div[data-testid="stFileUploader"] {
            margin-bottom: -1rem !important; 
            padding-bottom: 0.5rem !important;
        }
        /* ... etc ... */
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🤖 BuildRAG")
    
    # --- Status Section (Merged) ---
    st.markdown('<div style="margin-top: -10px;"></div>', unsafe_allow_html=True) # Fine-tune top margin
    
    is_ready = ollama.is_ready(timeout_s=0.25)
    
    if is_ready:
        # Show GREEN button if active
        if st.button("✅ Ollama Active", type="secondary", use_container_width=True, help="System is online. Click to check again."):
            st.rerun()
    else:
        # Show RED button if down
        if st.button("❌ Ollama Down", type="primary", use_container_width=True, help="System is offline. Click to retry."):
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True) # Hafif bir boşluk bırak

    # --- File Upload Section ---
    with st.expander("📂 Files & Upload", expanded=False):
        uploaded = st.file_uploader(
            "Upload Docs",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"uploader_{st.session_state.uploader_key}",
        )
        upload_notice = st.empty()
        if uploaded:
            for uf in uploaded:
                svc.doc_manager.save_upload_bytes(uf.name, uf.getbuffer().tobytes())

            # Show a short-lived notice inside the expander
            upload_notice.success(f"Uploaded {len(uploaded)} file(s) ✅")

            # Reset uploader so we don't re-process on rerun
            st.session_state.uploader_key += 1

            # Keep the notice visible briefly, then rerun (will clear it)
            time.sleep(4)
            st.rerun()

        # Dosya Listesi (Boşluk azaltıldı)
        files = svc.doc_manager.list_files()
        if files:
            options = [str(p.name) for p in files]
            selected_files = st.multiselect("Manage Files", options=options, placeholder="Select to delete...")
            
            if selected_files:
                if st.button(f"🗑️ Delete ({len(selected_files)})", type="secondary"):
                    full_paths = [f for f in files if f.name in selected_files]
                    svc.doc_manager.delete_files(full_paths)
                    st.rerun()
        else:
             st.caption("No files yet.")

    # --- Knowledge Base Section ---
    with st.expander("🧠 Knowledge Base", expanded=False):
        # Info row
        if svc.vector_db.exists():
            cnt = svc.vector_db.count_chunks()
            st.caption(f"Status: **Ready** | Chunks: **{cnt}**")
        else:
            st.caption("Status: **Not Indexed**")

        # Buttons Stacked Vertically (Full Width)
        if st.button("🚀 Build / Update Index", type="primary", use_container_width=True):
            with st.spinner("Indexing..."):
                msg, n = svc.index_manager.build_or_update()
            st.toast(f"Done: {n} chunks", icon="🎉")
            time.sleep(0.5)
            st.rerun()
            
        if st.button("🧹 Reset Database", type="secondary", use_container_width=True):
            svc.index_manager.reset()
            st.toast("DB Reset", icon="🗑️")
            time.sleep(0.5)
            st.rerun()

    # --- Settings Section ---
    with st.expander("⚙️ Config", expanded=False):
        # Custom HTML box to replace st.info for a tighter look
        st.markdown(f"""
        <div style="
            background-color: rgba(28, 131, 225, 0.1); 
            border: 1px solid rgba(28, 131, 225, 0.25);
            padding: 8px 10px;
            border-radius: 5px;
            font-size: 12px;
            color: inherit;
            margin-bottom: 10px;
        ">
            <div style="margin-bottom: 4px;"><strong>LLM:</strong> <code style="font-size:11px;">{svc.cfg.llm_model}</code></div>
            <div><strong>Indexer:</strong> <code style="font-size:11px;">{svc.cfg.embed_model}</code></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Slider max value 12
        k_val = st.slider("Context (Top-k)", 1, 12, svc.cfg.default_k)


# ---------- Main Chat Area ----------
st.subheader("💬 Chat Assistant")

# Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 Sources"):
                for idx, s in enumerate(m["sources"], 1):
                    st.markdown(f"**{idx}.** {s}")

# Input
user_q = st.chat_input("Ask about your documents...")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ans, sources = svc.chat.answer(user_q, k=k_val)
            st.markdown(ans)
            if sources:
                with st.expander("📚 Sources"):
                    for idx, s in enumerate(sources, 1):
                        st.write(s)
            
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": sources})