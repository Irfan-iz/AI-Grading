import joblib
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
AI_MODEL_PATH = "ai_detector_classifier.joblib"
VECTORIZER_PATH = "ai_detector_vectorizer.joblib"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

# Global variables to hold loaded models
AI_CLF = None
AI_VECT = None
SBERT_MODEL = None

# --- GRADE THRESHOLDS ---
GRADE_A_THRESHOLD = 0.75
GRADE_B_THRESHOLD = 0.65
GRADE_C_THRESHOLD = 0.50
GRADE_D_THRESHOLD = 0.35
GRADE_E_THRESHOLD = 0.20

def load_models():
    """Loads the AI and Plagiarism models into memory."""
    global AI_CLF, AI_VECT, SBERT_MODEL
    
    # 1. Load AI Detection Models
    if AI_CLF is None or AI_VECT is None:
        try:
            # We try to load files, if they don't exist, we skip gracefully
            AI_CLF = joblib.load(AI_MODEL_PATH)
            AI_VECT = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            # Print warning but don't crash (allows app to run without AI features)
            print(f"Warning: Could not load AI models. Make sure .joblib files exist. {e}")

    # 2. Load Plagiarism Model (SBERT)
    if SBERT_MODEL is None:
        print("Loading SBERT Plagiarism model...")
        try:
            SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Could not load SBERT model: {e}")

def normalize_text(text):
    """Cleans text for better comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def check_ai_generated(text):
    """Detects if text is AI-generated."""
    # Ensure models are loaded before checking
    if AI_CLF is None: load_models()
    
    if not text or AI_CLF is None or AI_VECT is None:
        return {"is_ai_generated": False, "ai_confidence_of_ai": 0.0}

    try:
        vectorized_text = AI_VECT.transform([text])
        prediction = AI_CLF.predict(vectorized_text)[0]
        probabilities = AI_CLF.predict_proba(vectorized_text)
        confidence = probabilities[0][1]

        is_ai = True if confidence > 0.8 else False
        
        return {
            "is_ai_generated": bool(is_ai),
            "ai_confidence_of_ai": float(confidence)
        }
    except Exception as e:
        print(f"AI Check Error: {e}")
        return {"is_ai_generated": False, "ai_confidence_of_ai": 0.0}

def check_plagiarism(text):
    """Checks for plagiarism and returns the MATCHED source text."""
    # Ensure models are loaded
    if SBERT_MODEL is None: load_models()

    internet_database = [
        "Supervised learning algorithms are designed to learn from labeled training data.",
        "Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets.",
        "Machine learning is a branch of artificial intelligence (AI) and computer science."
    ]
    
    if not text or SBERT_MODEL is None:
        return {"is_plagiarized": False, "plagiarism_score": 0.0, "source": None}

    try:
        student_emb = SBERT_MODEL.encode(text, convert_to_tensor=True)
        max_score = 0.0
        best_match_source = None  # <--- New variable to track the culprit
        
        for source in internet_database:
            source_emb = SBERT_MODEL.encode(source, convert_to_tensor=True)
            score = util.pytorch_cos_sim(student_emb, source_emb).item()
            
            # If this sentence is a better match than the last one, remember it
            if score > max_score:
                max_score = score
                best_match_source = source 
        
        is_plag = True if max_score > 0.85 else False

        return {
            "is_plagiarized": bool(is_plag),
            "plagiarism_score": float(max_score),
            "source": best_match_source if is_plag else None # <--- Return the source text
        }
    except Exception as e:
        print(f"Plagiarism Check Error: {e}")
        return {"is_plagiarized": False, "plagiarism_score": 0.0, "source": None}

def grade_submission(student_text, teacher_rubric):
    """
    Main function called by App.py.
    Orchestrates Grading, AI Detection, and Plagiarism Checking.
    """
    # 1. ENSURE MODELS ARE LOADED
    load_models()

    # 2. Safety Check
    if not student_text or not teacher_rubric or SBERT_MODEL is None:
        return {
            "grading_result": {"grade": "N/A", "similarity_score": 0.0},
            "final_points": 0,
            "ai_result": {"is_ai_generated": False, "ai_confidence_of_ai": 0.0},
            "plagiarism_result": {"is_plagiarized": False, "plagiarism_score": 0.0}
        }

    try:
        # --- A. GRADING LOGIC ---
        embeddings = SBERT_MODEL.encode([student_text, teacher_rubric])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Convert similarity directly to a score (e.g., 0.88 -> 88 points)
        # We limit it to max 100 and ensure it's an integer
        dynamic_score = min(int(similarity * 100), 100)
        
        # Assign Grade Label based on the dynamic score
        if dynamic_score >= 85:
            grade = "A (Distinction)"
        elif dynamic_score >= 70:
            grade = "B (Credit)"
        elif dynamic_score >= 55:
            grade = "C (Pass)"
        elif dynamic_score >= 40:
            grade = "D (Weak)"
        else:
            grade = "F (Fail)"
            
        grading_result = {
            "grade": grade, 
            "similarity_score": float(similarity)
        }

        # --- B. RUN OTHER CHECKS ---
        ai_result = check_ai_generated(student_text)
        plagiarism_result = check_plagiarism(student_text)

        # --- C. RETURN COMBINED RESULT ---
        return {
            "grading_result": grading_result,
            "final_points": dynamic_score,  # <--- Now uses exact score!
            "ai_result": ai_result,
            "plagiarism_result": plagiarism_result
        }

    except Exception as e:
        print(f"Grading Error: {e}")
        # Return fallback if crash
        return {
            "grading_result": {"grade": "Error", "similarity_score": 0.0},
            "final_points": 0,
            "ai_result": {"is_ai_generated": False},
            "plagiarism_result": {"is_plagiarized": False}
        }
        
        # ... (Keep all your existing imports and functions) ...

# --- CHATBOT KNOWLEDGE BASE ---
FAQ_DATA = {
    "What is this website?": 
        "This is an Integrity Checker system. It compares your uploaded documents against a database to detect potential plagiarism.",
    
    "How do I use this?": 
        "Simply type a student's answer and the teacher's rubric into the text boxes, then click 'Grade Submission'.",
    
    "What does the score mean?": 
        "The similarity score shows how closely the student's answer matches the teacher's rubric. Higher is better.",
    
    "How does plagiarism check work?": 
        "We compare the answer against an internal database of known texts using AI to find semantic matches.",
    
    "Is my data safe?": 
        "Yes, this runs locally on your machine. No data is sent to the cloud."
}

def get_chatbot_response(user_query):
    """
    Answers user questions using the existing SBERT model.
    """
    # 1. Ensure model is loaded (reuses the one from grading!)
    if SBERT_MODEL is None: load_models()
    
    # 2. Prepare the Knowledge Base (Questions)
    questions = list(FAQ_DATA.keys())
    
    # 3. Encode User Query & Questions
    # (In a real app, you would pre-compute question embeddings to be faster, 
    # but for a small list, doing it here is fine)
    query_emb = SBERT_MODEL.encode(user_query, convert_to_tensor=True)
    question_embs = SBERT_MODEL.encode(questions, convert_to_tensor=True)
    
    # 4. Find best match
    # We use cosine similarity (simpler than util.semantic_search for simple tensors)
    scores = util.pytorch_cos_sim(query_emb, question_embs)[0]
    
    # Find the highest score
    best_score_index = np.argmax(scores.cpu().numpy())
    best_score = scores[best_score_index].item()
    
    # 5. Threshold Check (If match is too weak, say "I don't know")
    if best_score < 0.4:
        return "I'm sorry, I don't understand. Try asking about grading, plagiarism, or how to use the app."
    
    # Return the matching answer
    matching_question = questions[best_score_index]
    return FAQ_DATA[matching_question]