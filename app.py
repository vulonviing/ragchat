from pathlib import Path
import streamlit as st
import time

from rag.retrieval import RetrievalParams
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

# ---------- CSS & Styling ----------
st.markdown(
    """
    <style>
        /* Sidebar Title */
        [data-testid="stSidebar"] h1 {
            margin-top: -10px !important;
            margin-bottom: 15px !important;
            font-size: 1.8rem;
            color: #4F8BF9;
        }

        /* --- Button Styling --- */
        
        /* General transition and spacing */
        [data-testid="stSidebar"] button {
            transition: all 0.2s ease-in-out !important;
            margin-top: 1px !important;
            margin-bottom: 1px !important;
        }
        
        /* Hover lift effect */
        [data-testid="stSidebar"] button:hover {
            transform: translateY(-3px);
        }

        /* Secondary Buttons (Glass/Grey) */
        [data-testid="stSidebar"] button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: rgba(235, 245, 255, 0.9) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            height: 40px;
        }
        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-color: rgba(106, 169, 255, 0.6) !important;
            color: #fff !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        /* Primary Buttons (Purple Gradient) */
        [data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.2), rgba(106, 169, 255, 0.2)) !important;
            color: #a78bfa !important;
            border: 1px solid rgba(139, 92, 246, 0.5) !important;
            height: 40px;
        }
        [data-testid="stSidebar"] button[kind="primary"]:hover {
            border-color: #8b5cf6 !important;
            color: #fff !important;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        }

        div[data-testid="stFileUploader"] {
            margin-bottom: -1rem !important; 
            padding-bottom: 0.5rem !important;
        }

        /* --- Ollama Status Badge --- */
        .ollama-status {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 12px 14px;
            border-radius: 14px;
            font-weight: 800;
            font-size: 15px;
            border: 1px solid transparent;
            transition: all 0.3s ease;
        }

        /* Active: Slow Pulse Animation (4s) */
        .ollama-status.active {
            background: rgba(25, 135, 84, 0.15);
            border-color: #198754;
            color: #2ea043;
            animation: ollamaPulse 4s ease-in-out infinite;
        }

        @keyframes ollamaPulse {
            0% {
                box-shadow: 0 0 5px rgba(46, 160, 67, 0.2);
                border-color: rgba(46, 160, 67, 0.3);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 0 20px rgba(46, 160, 67, 0.6);
                border-color: #4ade80;
                color: #b6f2d6;
                transform: scale(1.02);
            }
            100% {
                box-shadow: 0 0 5px rgba(46, 160, 67, 0.2);
                border-color: rgba(46, 160, 67, 0.3);
                transform: scale(1);
            }
        }

        /* Down: Static Purple */
        .ollama-status.down {
            background: rgba(139, 92, 246, 0.15);
            border-color: rgba(139, 92, 246, 0.6);
            color: #d8b4fe;
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.25);
        }

        /* --- Branding --- */
        .brand-title {
            font-size: 44px;
            font-weight: 800;
            letter-spacing: -0.8px;
            line-height: 1.05;
            margin: 6px 0 6px 0;
            background: linear-gradient(90deg, #6aa9ff 0%, #8b5cf6 45%, #22d3ee 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
        }
        .brand-subtitle {
            margin-top: 2px;
            margin-bottom: 12px;
            font-size: 12px;
            opacity: 0.78;
        }
        .brand-divider {
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, transparent, rgba(139,92,246,0.5), transparent);
            margin: 10px 0 14px 0;
        }

        /* Sidebar Expander */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid rgba(255,255,255,0.10) !important;
            background: rgba(255,255,255,0.03) !important;
            border-radius: 16px !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            color: #e0e7ff !important;
        }

        /* Chat Avatars */
        span[data-testid="stChatMessageAvatar"] {
            background: linear-gradient(135deg, #6aa9ff 0%, #8b5cf6 100%) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.2);
        }

        /* Chat Header */
        .chat-title {
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -0.6px;
            margin: 2px 0 6px 0;
            background: linear-gradient(90deg, #6aa9ff 0%, #7b7cff 30%, #8b5cf6 55%, #22d3ee 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .chat-divider {
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
            margin: 10px 0 18px 0;
        }
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
    st.markdown(
        """
        <div class="brand-title">BuildRAG</div>
        <div class="brand-subtitle">Local RAG workspace • Ollama + Chroma</div>
        <div class="brand-divider"></div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div style="margin-top: -10px;"></div>', unsafe_allow_html=True)
    
    is_ready = ollama.is_ready(timeout_s=0.25)
    
    if is_ready:
        st.markdown('<div class="ollama-status active">Ollama Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ollama-status down">Ollama Disconnected</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
            upload_notice.success(f"Uploaded {len(uploaded)} file(s) ✅")
            st.session_state.uploader_key += 1
            time.sleep(2)
            st.rerun()

        files = svc.doc_manager.list_files()
        if files:
            rel_map = {str(p.relative_to(svc.cfg.docs_dir)): p for p in files}
            options = list(rel_map.keys())

            selected = st.multiselect(
                "Manage Files",
                options=options,
                placeholder="Select files..."
            )

            if selected:
                if st.button(
                    f"🗑️ Delete Files ({len(selected)})",
                    type="secondary", 
                    use_container_width=True,
                    key="btn_delete_files",
                ):
                    paths = [rel_map[s] for s in selected]
                    deleted = svc.doc_manager.delete_files(paths)
                    st.toast(f"Deleted {deleted} file(s) from disk.", icon="🗑️")
                    st.rerun()

                if st.button(
                    f"🧽 Remove from Index ({len(selected)})",
                    type="secondary",
                    use_container_width=True,
                    key="btn_remove_from_index",
                ):
                    for s in selected:
                        svc.index_manager.remove_from_index(rel_map[s])
                    st.toast("Removed selected files from index.", icon="🧹")
                    st.rerun()
        else:
            st.caption("No files yet.")

    # --- Knowledge Base Section ---
    with st.expander("🧠 Knowledge Base", expanded=False):
        if svc.vector_db.exists():
            cnt = svc.vector_db.count_chunks()
            st.caption(f"Status: **Ready** | Chunks: **{cnt}**")
        else:
            st.caption("Status: **Not Indexed**")

        # Only Build/Update is Primary (Purple)
        if st.button("🚀 Build / Update Index", type="primary", use_container_width=True):
            with st.spinner("Indexing..."):
                msg, stats = svc.index_manager.build_or_update()
            st.toast(f"Indexed: {stats['chunks']} chunks", icon="🎉")
            time.sleep(0.5)
            st.rerun()
            
        if st.button("🧹 Reset Database", type="secondary", use_container_width=True):
            svc.index_manager.reset()
            st.toast("DB Reset", icon="🗑️")
            time.sleep(0.5)
            st.rerun()

    # --- Settings Section ---
    with st.expander("⚙️ Config", expanded=False):
        st.markdown(f"""
        <div style="
            background-color: rgba(139, 92, 246, 0.1); 
            border: 1px solid rgba(139, 92, 246, 0.25);
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
        
        retrieval_mode = st.selectbox("Retrieval Mode", ["similarity", "mmr", "threshold"])
        k_val = st.slider("Context (Top-k)", 1, 12, svc.cfg.default_k)

        fetch_k = 20
        score_threshold = 0.35
        if retrieval_mode == "mmr":
            fetch_k = st.slider("MMR fetch_k", 10, 60, 20)
        if retrieval_mode == "threshold":
            score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.35, 0.01)

        show_retrieved = st.checkbox("Show retrieved chunks (debug)", value=False)


# ---------- Main Chat Area ----------
st.markdown(
    """
    <div class="chat-title">Chat Assistant</div>
    <div class="chat-subtitle">Ask questions and get grounded answers with sources.</div>
    <div class="chat-divider"></div>
    """,
    unsafe_allow_html=True,
)

# Chat History
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="👤" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 Sources"):
                for idx, s in enumerate(m["sources"], 1):
                    st.markdown(f"**{idx}.** {s}")

# Input
user_q = st.chat_input("Ask about your documents...")

if user_q:
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_q)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            params = RetrievalParams(
                mode=retrieval_mode,
                k=k_val,
                fetch_k=fetch_k,
                score_threshold=score_threshold,
            )
            
            ans, sources, debug_chunks = svc.chat.answer(user_q, params=params)
            st.markdown(ans)
            
            if sources:
                with st.expander("📚 Sources"):
                    for idx, s in enumerate(sources, 1):
                        st.write(s)

            if show_retrieved and debug_chunks:
                with st.expander("🔎 Retrieved chunks (debug)"):
                    for i, ch in enumerate(debug_chunks, 1):
                        st.markdown(f"**{i}.** `{ch.source}` | Score: `{ch.score:.3f}`")
                        st.code(ch.text[:300])
            
            st.session_state.messages.append({"role": "assistant", "content": ans, "sources": sources})