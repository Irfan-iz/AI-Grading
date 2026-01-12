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
"""Checks for plagiarism against a TEXT file database."""
# Ensure models are loaded
if SBERT_MODEL is None: load_models()

# --- NEW: LOAD REAL DATABASE FROM TEXT FILE ---
internet_database = []

# 1. Define the path to your .txt file
    txt_path = os.path.join(os.path.dirname(__file__), 'AllCombined.txt')
    txt_path = os.path.join(os.path.dirname(__file__), 'plagiarism_data.txt')

if os.path.exists(txt_path):
try:
# 2. Open the text file safely
with open(txt_path, 'r', encoding='utf-8') as f:
# Read all lines and strip whitespace
lines = f.readlines()

# Filter out empty lines or very short lines (less than 10 chars)
internet_database = [line.strip() for line in lines if len(line.strip()) > 10]

# print(f"DEBUG: Loaded {len(internet_database)} sentences from text file.")
except Exception as e:
print(f"Error reading text file: {e}")
else:
print("Warning: plagiarism_data.txt not found. Using fallback.")
internet_database = [
"Supervised learning algorithms are designed to learn from labeled training data.",
"Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled datasets."
]

# If database is empty or text is empty, return Safe
if not text or not internet_database:
return {"is_plagiarized": False, "plagiarism_score": 0.0, "source": None}

try:
student_emb = SBERT_MODEL.encode(text, convert_to_tensor=True)

# Calculate similarity against ALL database lines
source_embs = SBERT_MODEL.encode(internet_database, convert_to_tensor=True)
scores = util.pytorch_cos_sim(student_emb, source_embs)[0]

# Find the single highest score
max_score_idx = np.argmax(scores.cpu().numpy())
max_score = scores[max_score_idx].item()
best_match_source = internet_database[max_score_idx]

# Threshold: 0.85 means 85% similar
is_plag = True if max_score > 0.85 else False

return {
"is_plagiarized": bool(is_plag),
"plagiarism_score": float(max_score),
"source": best_match_source if is_plag else None
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

# --- CHATBOT KNOWLEDGE BASE (Requirement iv & v) ---
FAQ_DATA = {
# 1. Identity & Purpose
"What is this website?": 
"This is an AI Grading & Integrity Portal for BAXI 3413. It detects plagiarism, checks for AI-generated text, and grades essays automatically.",

# 2. Usage Guide (Guides user to input)
"How do I use this system?": 
"Enter the student's name, the grading rubric, and the answer in the text boxes on the main page, then click the 'Grade Submission' button.",

# 3. Grading Logic
"What does the grading score mean?": 
"The score (0-100) represents the semantic similarity between the student's answer and the teacher's rubric. A higher score means a better match.",

# 4. Technical Details (NLP Models)
"What AI models are used here?": 
"We use 'all-MiniLM-L6-v2' (SBERT) for semantic similarity and plagiarism checks, and a custom Scikit-Learn classifier for AI detection.",

# 5. Plagiarism Explanation
"How does the plagiarism check work?": 
"The system scans the student's text against our internal database of known sources. If a high similarity is found, the specific source is flagged.",

# 6. AI Detection Logic
"How do you detect AI-generated text?": 
"We use a Machine Learning classifier trained on patterns common in AI writing, such as low perplexity and specific sentence structures.",

# 7. Troubleshooting (Guides user to fix input)
"Why did I get a 'N/A' or error result?": 
"This usually happens if the input text is too short or empty. Please ensure the answer is at least one full sentence.",

# 8. Privacy & Security
"Is the submitted data safe?": 
"Yes, all processing is done locally on the server. We do not store essays or rubrics in any external cloud database.",

# 9. Scope of the System
"Can I use this for coding or math questions?": 
"This system is optimized for natural language essays and text explanations. It may not grade code snippets or mathematical formulas accurately.",

# 10. Authorship
"Who developed this project?": 
"This system was developed by our group for the BAXI 3413 Natural Language Processing course (Semester 1, 2025/2026)."
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


