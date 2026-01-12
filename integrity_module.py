import joblib
import numpy as np
import re
import os  # <--- Added for file path handling
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
AI_MODEL_PATH = "ai_detector_classifier.joblib"
VECTORIZER_PATH = "ai_detector_vectorizer.joblib"
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
PLAGIARISM_FILE = "plagiarism_data.txt" # <--- Name of your text file

# Global variables to hold loaded models
AI_CLF = None
AI_VECT = None
SBERT_MODEL = None
PLAGIARISM_DB = [] # <--- New global to hold the sentences

# --- GRADE THRESHOLDS ---
GRADE_A_THRESHOLD = 0.75
GRADE_B_THRESHOLD = 0.65
GRADE_C_THRESHOLD = 0.50
GRADE_D_THRESHOLD = 0.35
GRADE_E_THRESHOLD = 0.20

# --- CHATBOT KNOWLEDGE BASE ---
FAQ_DATA = {
    "What is this website?": 
        "This is an AI Grading & Integrity Portal for BAXI 3413. It detects plagiarism, checks for AI-generated text, and grades essays automatically.",
    "How do I use this system?": 
        "Enter the student's name, the grading rubric, and the answer in the text boxes on the main page, then click the 'Grade Submission' button.",
    "What does the grading score mean?": 
        "The score (0-100) represents the semantic similarity between the student's answer and the teacher's rubric. A higher score means a better match.",
    "What AI models are used here?": 
        "We use 'all-MiniLM-L6-v2' (SBERT) for semantic similarity and plagiarism checks, and a custom Scikit-Learn classifier for AI detection.",
    "How does the plagiarism check work?": 
        "The system scans the student's text against our internal database of known sources. If a high similarity is found, the specific source is flagged.",
    "How do you detect AI-generated text?": 
        "We use a Machine Learning classifier trained on patterns common in AI writing, such as low perplexity and specific sentence structures.",
    "Why did I get a 'N/A' or error result?": 
        "This usually happens if the input text is too short or empty. Please ensure the answer is at least one full sentence.",
    "Is the submitted data safe?": 
        "Yes, all processing is done locally on the server. We do not store essays or rubrics in any external cloud database.",
    "Can I use this for coding or math questions?": 
        "This system is optimized for natural language essays and text explanations. It may not grade code snippets or mathematical formulas accurately.",
    "Who developed this project?": 
        "This system was developed by our group for the BAXI 3413 Natural Language Processing course (Semester 1, 2025/2026)."
}

def load_models():
    """Loads the AI models, SBERT, and the Plagiarism Database into memory."""
    global AI_CLF, AI_VECT, SBERT_MODEL, PLAGIARISM_DB
    
    # 1. Load AI Detection Models
    if AI_CLF is None or AI_VECT is None:
        try:
            AI_CLF = joblib.load(AI_MODEL_PATH)
            AI_VECT = joblib.load(VECTORIZER_PATH)
        except Exception as e:
            print(f"Warning: Could not load AI models. Make sure .joblib files exist. {e}")

    # 2. Load SBERT Model
    if SBERT_MODEL is None:
        print("Loading SBERT Model...")
        try:
            SBERT_MODEL = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception as e:
            print(f"Warning: Could not load SBERT model: {e}")

    # 3. Load Plagiarism Database (The Fix!)
    # We load this once so we don't have to read the file every time a user clicks "Grade"
    if not PLAGIARISM_DB:
        print("Loading Plagiarism Database...")
        # Check if file exists in the current directory
        txt_path = os.path.join(os.path.dirname(__file__), PLAGIARISM_FILE)
        
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                
                # SPLIT LOGIC: 
                # We split by periods (.) to get individual sentences.
                # This ensures we match exact sentences (100% match) rather than big paragraphs (40% match).
                sentences = raw_text.split('.')
                
                # Clean up and remove short/empty lines
                PLAGIARISM_DB = [s.strip() for s in sentences if len(s.strip()) > 20]
                print(f"Success: Loaded {len(PLAGIARISM_DB)} sentences into Plagiarism DB.")
            except Exception as e:
                print(f"Error reading plagiarism file: {e}")
        else:
            # Fallback if file is missing
            print("Warning: plagiarism_data.txt not found. Using default list.")
            PLAGIARISM_DB = [
                "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
                "Machine learning is a branch of artificial intelligence (AI) and computer science.",
                "Supervised learning algorithms are designed to learn from labeled training data."
            ]

def normalize_text(text):
    """Cleans text for better comparison."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def check_ai_generated(text):
    """Detects if text is AI-generated."""
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
    """Checks for plagiarism against the loaded database."""
    if SBERT_MODEL is None: load_models()

    if not text or not PLAGIARISM_DB:
        return {"is_plagiarized": False, "plagiarism_score": 0.0, "source": None}

    try:
        # Encode student text
        student_emb = SBERT_MODEL.encode(text, convert_to_tensor=True)
        
        # Encode database (Batch processing is faster)
        db_embeddings = SBERT_MODEL.encode(PLAGIARISM_DB, convert_to_tensor=True)
        
        # Calculate similarity (Student vs All DB Sentences)
        cosine_scores = util.pytorch_cos_sim(student_emb, db_embeddings)
        
        # Find the single highest match (Best matching sentence)
        max_score_tensor = cosine_scores.max()
        max_score = max_score_tensor.item()
        
        # Find which sentence it matched
        best_match_idx = cosine_scores.argmax().item()
        best_source = PLAGIARISM_DB[best_match_idx]

        # Threshold: 0.85 (85%) is a good cutoff for "stolen" text
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
    """
    Main function called by App.py.
    Orchestrates Grading, AI Detection, and Plagiarism Checking.
    """
    load_models()

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
        
        # Convert similarity to score
        dynamic_score = min(int(similarity * 100), 100)
        
        if dynamic_score >= 85: grade = "A (Distinction)"
        elif dynamic_score >= 70: grade = "B (Credit)"
        elif dynamic_score >= 55: grade = "C (Pass)"
        elif dynamic_score >= 40: grade = "D (Weak)"
        else: grade = "F (Fail)"
            
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
            "final_points": dynamic_score,
            "ai_result": ai_result,
            "plagiarism_result": plagiarism_result
        }

    except Exception as e:
        print(f"Grading Error: {e}")
        return {
            "grading_result": {"grade": "Error", "similarity_score": 0.0},
            "final_points": 0,
            "ai_result": {"is_ai_generated": False},
            "plagiarism_result": {"is_plagiarized": False}
        }

def get_chatbot_response(user_query):
    """Answers user questions using the existing SBERT model."""
    if SBERT_MODEL is None: load_models()
    
    questions = list(FAQ_DATA.keys())
    
    # Simple semantic search
    query_emb = SBERT_MODEL.encode(user_query, convert_to_tensor=True)
    question_embs = SBERT_MODEL.encode(questions, convert_to_tensor=True)
    
    scores = util.pytorch_cos_sim(query_emb, question_embs)[0]
    best_idx = np.argmax(scores.cpu().numpy())
    best_score = scores[best_idx].item()
    
    if best_score < 0.4:
        return "I'm sorry, I don't understand. Try asking about grading, plagiarism, or how to use the app."
    
    return FAQ_DATA[questions[best_idx]]
