"""
MEMO-EXI: Streamlit Interactive Memory Dashboard.
Provides a multi-tab interface for Chat, Memory Explorer, Temporal History,
Conflict Monitor, Retrieval Analytics, and System Configuration.
"""

from datetime import datetime, timezone
import json
import pandas as pd
import streamlit as st
from memo_exi.config import settings
from memo_exi.database.sqlite_manager import SQLiteMemoryManager
from memo_exi.models.memory_models import MemoryStatus, MemoryType, QueryIntent
from memo_exi.service import MEMOEXIEngine
from memo_exi.vector_store.chroma_manager import ChromaMemoryManager

st.set_page_config(
    page_title="MEMO-EXI: Temporal Memory Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished research dashboard
st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 1rem; border: 1px solid #E2E8F0; }
    .badge-current { background-color: #DCFCE7; color: #166534; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
    .badge-superseded { background-color: #F3F4F6; color: #6B7280; padding: 3px 8px; border-radius: 12px; font-size: 0.8rem; }
    .badge-disputed { background-color: #FEE2E2; color: #991B1B; padding: 3px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    """Singleton engine loader for Streamlit runtime."""
    settings.ensure_directories()
    return MEMOEXIEngine()


engine = load_engine()

# Sidebar: Session & Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("### 🧠 MEMO-EXI")
    st.markdown("**Adaptive Temporal Memory Engine**")
    st.divider()

    session_id = st.text_input("Active Session ID", value="default_session")

    st.markdown("#### Real-Time Status")
    stats = engine.sqlite.get_statistics(session_id=session_id)
    st.metric("Total Memories", stats["total_memories"])
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Current", stats["current_memories"])
    with col_s2:
        st.metric("Disputed", stats["disputed_memories"])

    st.divider()
    if st.button("🧹 Clear Session Memories", type="secondary"):
        with engine.sqlite.get_connection() as conn:
            conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
        st.success("Session memory cleared.")
        st.rerun()

# Top Bar
st.markdown('<div class="main-title">MEMO-EXI: Adaptive Temporal Memory Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Version-aware persistent knowledge management, conflict resolution, and context-budget optimization.</div>', unsafe_allow_html=True)

# Tabs
tab_chat, tab_explorer, tab_history, tab_conflict, tab_analytics, tab_config = st.tabs([
    "💬 Chat Interface",
    "🔍 Memory Explorer",
    "⏳ Temporal History",
    "⚠️ Conflict Monitor",
    "📊 Retrieval Analytics",
    "⚙️ System Configuration",
])

# -----------------------------------------------------------------------------
# TAB 1: Chat Interface
# -----------------------------------------------------------------------------
with tab_chat:
    st.markdown("#### Conversational Assistant with Adaptive Memory")
    st.caption("Interact naturally. The agent extracts facts, tracks state evolutions, detects conflicts, and grounds responses.")

    # Initialize session chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "trace" in msg and msg["trace"]:
                with st.expander("🔍 Memory & Controller Execution Trace"):
                    st.json(msg["trace"])

    # User Input
    if user_prompt := st.chat_input("Enter your message or temporal question (e.g., 'My salary is $110k', 'What was my salary before?')"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Process with engine
        with st.spinner("Processing memory lifecycle..."):
            resp = engine.process_conversation(
                session_id=session_id,
                message=user_prompt,
            )

        trace_data = {
            "query_intent": resp.query_intent.value,
            "extracted_count": len(resp.extracted_memories),
            "retrieved_count": len(resp.retrieved_memories),
            "tokens_used": f"{resp.token_usage}/{resp.token_budget}",
            "budget_compliant": resp.budget_compliant,
            "controller_actions": resp.controller_decision.actions_taken,
            "utility_scores": resp.utility_scores,
        }

        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": resp.reply,
            "trace": trace_data,
        })

        with st.chat_message("assistant"):
            st.markdown(resp.reply)
            with st.expander("🔍 Memory & Controller Execution Trace"):
                st.json(trace_data)

# -----------------------------------------------------------------------------
# TAB 2: Memory Explorer
# -----------------------------------------------------------------------------
with tab_explorer:
    st.markdown("#### Structured Temporal Memory Records")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_status = st.selectbox("Status Filter", ["All", "current", "superseded", "disputed"])
    with col_f2:
        filter_type = st.selectbox("Type Filter", ["All", "preference", "factual", "constraint", "episodic"])
    with col_f3:
        search_query = st.text_input("Filter by Subject or Attribute", "")

    status_val = MemoryStatus(filter_status) if filter_status != "All" else None
    type_val = MemoryType(filter_type) if filter_type != "All" else None

    memories = engine.sqlite.query_memories(
        session_id=session_id,
        status=status_val,
        memory_type=type_val,
        limit=200,
    )

    if search_query:
        memories = [m for m in memories if search_query.lower() in m.subject.lower() or search_query.lower() in m.attribute.lower() or search_query.lower() in m.value.lower()]

    if memories:
        table_data = []
        for m in memories:
            table_data.append({
                "Subject": m.subject,
                "Attribute": m.attribute,
                "Value": m.value,
                "Type": m.memory_type.value,
                "Status": m.status.value,
                "Valid From": m.valid_from.strftime("%Y-%m-%d %H:%M"),
                "Valid Until": m.valid_until.strftime("%Y-%m-%d %H:%M") if m.valid_until else "Active",
                "Confidence": f"{m.confidence_score:.2f}",
                "Importance": f"{m.importance_score:.2f}",
                "Memory ID": m.memory_id,
            })
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No memories found matching the current filters.")

# -----------------------------------------------------------------------------
# TAB 3: Temporal History
# -----------------------------------------------------------------------------
with tab_history:
    st.markdown("#### Chronological Evolution and Lineage Tracking")
    st.caption("Inspect how specific facts evolve over time without destructive overwriting.")

    with engine.sqlite.get_connection() as conn:
        sub_attrs = conn.execute(
            "SELECT DISTINCT subject, attribute FROM memories WHERE session_id = ?",
            (session_id,),
        ).fetchall()

    if sub_attrs:
        options = [f"{row['subject']} - {row['attribute']}" for row in sub_attrs]
        selected_option = st.selectbox("Select Fact to Visualize Evolution:", options)
        if selected_option:
            sel_sub, sel_attr = selected_option.split(" - ")
            history = engine.temporal_mgr.get_history(sel_sub, sel_attr, session_id=session_id)

            st.markdown(f"**Version Timeline for `{sel_sub}.{sel_attr}` ({len(history)} versions):**")
            for i, ver in enumerate(history):
                badge_class = f"badge-{ver.status.value}"
                badge_html = f'<span class="{badge_class}">{ver.status.value.upper()}</span>'
                v_until = ver.valid_until.strftime("%Y-%m-%d %H:%M") if ver.valid_until else "Present (Active)"

                with st.container():
                    st.markdown(
                        f"""
                        <div style="border-left: 4px solid #3B82F6; padding-left: 1rem; margin-bottom: 1rem;">
                            <strong>Version {i+1}: {ver.value}</strong> &nbsp; {badge_html}<br/>
                            <small>⏱️ Valid: {ver.valid_from.strftime('%Y-%m-%d %H:%M')} &rarr; {v_until} | Supersedes: <code>{ver.supersedes_id or 'None (Root)'}</code></small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.info("No fact versions recorded yet for this session. Add facts in the Chat tab.")

# -----------------------------------------------------------------------------
# TAB 4: Conflict Monitor
# -----------------------------------------------------------------------------
with tab_conflict:
    st.markdown("#### Disputed Memory Resolution Queue")
    st.caption("Review conflicting claims flagged by Module E and resolve them with a single click.")

    disputed = engine.sqlite.query_memories(
        session_id=session_id,
        status=MemoryStatus.DISPUTED,
    )

    if disputed:
        st.warning(f"Found {len(disputed)} disputed record(s) requiring clarification.")
        for d in disputed:
            col_d1, col_d2, col_d3 = st.columns([3, 1, 1])
            with col_d1:
                st.markdown(f"**{d.subject} &middot; {d.attribute}**: `{d.value}`")
                st.caption(f"Recorded: {d.valid_from.strftime('%Y-%m-%d %H:%M')} | Context: {d.raw_context or 'N/A'}")
            with col_d2:
                if st.button(f"✅ Confirm Active", key=f"conf_{d.memory_id}"):
                    engine.temporal_mgr.resolve_dispute(d.memory_id, MemoryStatus.CURRENT)
                    st.success("Resolved to CURRENT")
                    st.rerun()
            with col_d3:
                if st.button(f"❌ Mark Superseded", key=f"sup_{d.memory_id}"):
                    engine.temporal_mgr.resolve_dispute(d.memory_id, MemoryStatus.SUPERSEDED)
                    st.info("Resolved to SUPERSEDED")
                    st.rerun()
            st.divider()
    else:
        st.success("✅ Zero active conflicts. All memories are currently consistent.")

# -----------------------------------------------------------------------------
# TAB 5: Retrieval Analytics
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown("#### Memory Distribution & Retrieval Telemetry")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("**Distribution by Memory Type**")
        type_dist = stats.get("type_distribution", {})
        if type_dist:
            df_type = pd.DataFrame(list(type_dist.items()), columns=["Type", "Count"])
            st.bar_chart(df_type.set_index("Type"))
        else:
            st.info("No memories available for chart.")

    with col_a2:
        st.markdown("**Status Breakdown**")
        status_data = {
            "Current": stats.get("current_memories", 0),
            "Superseded": stats.get("superseded_memories", 0),
            "Disputed": stats.get("disputed_memories", 0),
        }
        df_status = pd.DataFrame(list(status_data.items()), columns=["Status", "Count"])
        st.bar_chart(df_status.set_index("Status"))

# -----------------------------------------------------------------------------
# TAB 6: System Configuration
# -----------------------------------------------------------------------------
with tab_config:
    st.markdown("#### Dynamic Memory Controller & Budget Parameters")
    st.caption("Adjust real-time retrieval parameters, context budgets, and utility function weights.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.markdown("**Context Budget & Retrieval Bounds**")
        new_budget = st.slider("Token Budget (B_tokens)", min_value=128, max_value=4096, value=settings.default_token_budget, step=128)
        new_top_k = st.slider("Default Top-K", min_value=1, max_value=20, value=settings.default_top_k)

    with col_cfg2:
        st.markdown("**Utility Function Weights**")
        st.latex(r"U(m, q) = w_{rel} S_{rel} + w_{temp} S_{temp} + w_{imp} I(m) - w_{red} R(m)")
        w_rel = st.slider("w_relevance", 0.0, 1.0, settings.weight_relevance, step=0.05)
        w_temp = st.slider("w_temporal", 0.0, 1.0, settings.weight_temporal, step=0.05)
        w_imp = st.slider("w_importance", 0.0, 1.0, settings.weight_importance, step=0.05)
        w_red = st.slider("w_redundancy", 0.0, 1.0, settings.weight_redundancy, step=0.05)

    if st.button("Save Configuration", type="primary"):
        settings.default_token_budget = new_budget
        settings.default_top_k = new_top_k
        settings.weight_relevance = w_rel
        settings.weight_temporal = w_temp
        settings.weight_importance = w_imp
        settings.weight_redundancy = w_red
        st.success("Configuration successfully updated for active session.")
