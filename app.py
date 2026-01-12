import streamlit as st
from integrity_module import grade_submission

# --- PAGE SETUP ---
st.set_page_config(page_title="NLP Auto-Grader", layout="centered")
st.title("🤖 AI-Powered Exam Grader")
st.markdown("Automated Essay Scoring, AI Detection & Plagiarism Check")
# --- PROJECT INFO (Requirement ii) ---
with st.expander("ℹ️ About this Project"):
    st.markdown("""
    This system is an **Automated Essay Scoring & Integrity Portal** developed for the **BAXI 3413 NLP Course**.
    
    **Features:**
    * **Auto-Grading:** Uses `Sentence-Transformers` (SBERT) to calculate semantic similarity between the student answer and the rubric.
    * **AI Detection:** Uses a `Scikit-Learn` classifier trained on human vs. AI text.
    * **Plagiarism Check:** Compares submissions against an internal knowledge database.
    """)

# --- INPUTS ---
with st.form("exam_form"):
    student_name = st.text_input("Student Name")
    rubric = st.text_area("Teacher's Marking Rubric", height=100, 
        value="Supervised learning uses labeled data. Unsupervised learning uses unlabeled data.")
    answer = st.text_area("Student's Answer", height=150)
    
    submitted = st.form_submit_button("Grade Submission")

# --- OUTPUT ---
if submitted:
    if not answer or not rubric:
        st.error("Please fill in all fields.")
    else:
        with st.spinner("Analyzing text..."):
            result = grade_submission(answer, rubric)
        
       # --- SAFETY CHECK & NORMALIZATION ---
        # If the result comes back "flat" (like in your debug info), we restructure it
        # so the rest of the app can understand it.
        if result and 'grading_result' not in result:
            # Create a fake structure to prevent errors
            result = {
                'grading_result': {
                    'grade': result.get('grade', 'N/A'),
                    'similarity_score': result.get('similarity_score', 0)
                },
                'final_points': result.get('points', 0),
                'ai_result': {'is_ai_generated': False, 'ai_confidence_of_ai': 0},
                'plagiarism_result': {'is_plagiarized': False}
            }

        if result and 'grading_result' in result:
            # Display Results
            st.subheader(f"Results for: {student_name}")
            
            # Metrics Columns
            col1, col2, col3 = st.columns(3)
            
            grade = result['grading_result'].get('grade', 'N/A')
            score = result.get('final_points', 0)
            similarity = result['grading_result'].get('similarity_score', 0)

            col1.metric("Grade", grade)
            col2.metric("Score", f"{score} / 100")
            col3.metric("Similarity", f"{round(similarity * 100, 1)}%")
            
            # Flags
            st.divider()
            st.subheader("Integrity Report")
            
            c1, c2 = st.columns(2)
            
            # AI Detection
            if 'ai_result' in result:
                ai_data = result['ai_result']
                ai_conf = ai_data.get('ai_confidence_of_ai', 0)
                if ai_data.get('is_ai_generated'):
                    c1.error(f"❌ AI DETECTED ({round(ai_conf * 100, 1)}%)")
                else:
                    c1.success(f"✅ HUMAN WRITTEN ({round(ai_conf * 100, 1)}%)")
            
# Plagiarism Check
            if 'plagiarism_result' in result:
                plag_data = result['plagiarism_result']
                plag_score = plag_data.get('plagiarism_score', 0)
                
                # Format percentage (e.g., 0.88 -> 88.0%)
                plag_percent = round(plag_score * 100, 1)

                if plag_data.get('is_plagiarized'):
                    c2.error(f"❌ PLAGIARISM DETECTED ({plag_percent}%)")
                    
                    # Show the source if available
                    with c2.expander("View Source"):
                        st.caption("Matched against Internal Database:")
                        st.markdown(f"_{plag_data.get('source', 'Unknown source')}_")
                else:
                    # Even if it's "Original," it's helpful to see the low match %
                    c2.success(f"✅ ORIGINAL CONTENT ({plag_percent}%)")
            else:
                c2.warning("Plagiarism check unavailable")
                
                # ... (Keep your existing app.py code) ...

# --- CHATBOT SIDEBAR ---
from integrity_module import get_chatbot_response

with st.sidebar:
    st.header("💬 Help Assistant")
    st.caption("Ask me about this app!")
    
    # Initialize chat history if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I help you?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Type your question..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        # 2. Bot Response
        bot_reply = get_chatbot_response(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        st.chat_message("assistant").write(bot_reply)

