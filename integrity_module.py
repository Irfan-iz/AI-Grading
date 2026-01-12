import joblib
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
AI_MODEL_PATH = "ai_detector_classifier.joblib"
VECTORIZER_PATH = "ai_detector_vectorizer.joblib"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# --- GLOBAL VARIABLES ---
AI_CLF = None
AI_VECT = None
SBERT_MODEL = None

# --- 🚀 FAST MANUAL DATABASE (No text file reading) ---
# Add your specific sentences here. This runs instantly.
PLAGIARISM_DB = [
    # 1. The Definitions you uploaded
    "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
    "The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    
    # 2. General Knowledge (Safe to keep)
    "Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn.",
    "Supervised learning algorithms are designed to learn from labeled training data to predict outcomes or classify data.",
    
    # 3. Add more sentences here if you want...
    "Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers."
]

# We will store the pre-calculated math here
DB_EMBEDDINGS = None

# --- GRADE THRESHOLDS ---
GRADE_A_THRESHOLD = 0.75
GRADE_B_THRESHOLD = 0.65
GRADE_C_THRESHOLD = 0.50
GRADE_D_THRESHOLD = 0.35
GRADE_E_THRESHOLD = 0.20

# --- CHATBOT KNOWLEDGE BASE ---
FAQ_DATA = {
    "What is this website?": "This is an AI Grading & Integrity Portal for BAXI 3413.",
    "How do I use this system?": "Enter the student's name, rubric, and answer, then click 'Grade Submission'.",
    "What does the grading score mean?": "The score (0-100) represents the semantic similarity to the rubric.",
    "What AI models are used?": "We use 'all-MiniLM-L6-v2' (SBERT) and Scikit-Learn.",
    "How does plagiarism work?": "It scans text against an internal database of known academic sources.",
    "How do you detect AI?": "We use a classifier trained on human vs. AI writing patterns.",
    "Why did I get N/A?": "The input might be too short. Please write at least one full sentence.",
    "Is my data safe?": "Yes, all processing is done locally on this server.",
    "Who developed this?": "Developed by the BAXI 3413 NLP Group (Session 2025/2026)."
}

def load_models():
    """Loads models and pre-calculates the manual database."""
    global AI_CLF, AI_VECT, SBERT_MODEL, DB_EMBEDDINGS
    
    # 1. Load AI Detection Models
    if AI_CLF is None:
        try:
            AI_CLF = joblib.load(AI_MODEL_PATH)
            AI_VECT = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            print(f"Warning: AI models missing. {e}")

    # 2. Load SBERT Model
    if SBERT_MODEL is None:
        print("Loading SBERT Model...")
        try:
            SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
            
            # 3. Pre-Calculate the Manual List (Instant)
            print(f"Optimizing: Calculating vectors for {len(PLAGIARISM_DB)} sentences...")
            DB_EMBEDDINGS = SBERT_MODEL.encode(PLAGIARISM_DB, convert_to_tensor=True)
        except Exception as e:
            print(f"Error loading SBERT: {e}")

def normalize_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def check_ai_generated(text):
    if AI_CLF is None: load_models()
    if not text or AI_CLF is None: return {"is_ai_generated": False, "ai_confidence_of_ai": 0.0}
    try:
        vec = AI_VECT.transform([text])
        prob = AI_CLF.predict_proba(vec)[0][1]
        return {"is_ai_generated": prob > 0.5, "ai_confidence_of_ai": float(prob)}
    except:
        return {"is_ai_generated": False, "ai_confidence_of_ai": 0.0}

def check_plagiarism(text):
    """Checks against the manual PLAGIARISM_DB list."""
    if SBERT_MODEL is None or DB_EMBEDDINGS is None: load_models()
    
    if not text:
        return {"is_plagiarized": False, "plagiarism_score": 0.0, "source": None}

    try:
        # Encode the student's text
        student_emb = SBERT_MODEL.encode(text, convert_to_tensor=True)
        
        # Compare against the pre-calculated list
        cosine_scores = util.pytorch_cos_sim(student_emb, DB_EMBEDDINGS)
        
        # Find best match
        best_match_idx = cosine_scores.argmax().item()
        max_score = cosine_scores[0][best_match_idx].item()
        best_source = PLAGIARISM_DB[best_match_idx]

        # Threshold: 0.85 (85%)
        is_plag = True if max_score > 0.85 else False

        return {
            "is_plagiarized": bool(is_plag),
            "plagiarism_score": float(max_score),
            "source": best_source if is_plag else None
        }
    except Exception as e:
        print(f"Plagiarism Check Error: {e}")
        return {"is_plagiarized": False, "plagiarism_score": 0.0, "source": None}

def grade_submission(student_text, teacher_rubric):
    load_models()
    if not student_text or not teacher_rubric: return {}

    try:
        # Grading
        embeddings = SBERT_MODEL.encode([student_text, teacher_rubric])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        dynamic_score = min(int(similarity * 100), 100)
        
        if dynamic_score >= 80: grade = "A (Distinction)"
        elif dynamic_score >= 60: grade = "B (Credit)"
        elif dynamic_score >= 40: grade = "C (Pass)"
        elif dynamic_score >= 20: grade = "D (Weak)"
        else: grade = "F (Fail)"

        grading_result = {"grade": grade, "similarity_score": float(similarity)}

        # Checks
        ai_result = check_ai_generated(student_text)
        plagiarism_result = check_plagiarism(student_text)

        return {
            "grading_result": grading_result,
            "final_points": dynamic_score,
            "ai_result": ai_result,
            "plagiarism_result": plagiarism_result
        }
    except Exception as e:
        print(f"Grading Error: {e}")
        return {}

def get_chatbot_response(user_query):
    if SBERT_MODEL is None: load_models()
    questions = list(FAQ_DATA.keys())
    q_embs = SBERT_MODEL.encode(questions, convert_to_tensor=True)
    u_emb = SBERT_MODEL.encode(user_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(u_emb, q_embs)[0]
    best_idx = np.argmax(scores.cpu().numpy())
    if scores[best_idx] < 0.4: return "I'm sorry, I don't understand."
    return FAQ_DATA[questions[best_idx]]

