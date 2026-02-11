import spacy
from sklearn.feature_extraction.text import TfidfVectorizer # Changed to TF-IDF
from sklearn.metrics.pairwise import cosine_similarity

# Load the English language model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    """
    Extracts potential skills (Nouns and Proper Nouns) from the text.
    """
    doc = nlp(text)
    skills = []
    
    # Loop through every word
    for token in doc:
        # We only keep Nouns (e.g. "Data") and Proper Nouns (e.g. "Python")
        # We also remove "stop words" (is, and, the)
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            skills.append(token.text)
            
    # Return unique skills (using set to remove duplicates)
    return list(set(skills))

def calculate_similarity(resume_text, jd_text):
    """
    Calculates the match percentage using TF-IDF and Cosine Similarity.
    """
    # 1. Create a list of the texts
    text_list = [resume_text, jd_text]
    
    # 2. Convert text to numbers (Vectors) using TF-IDF
    # TF-IDF automatically lowers the weight of common words 
    # and increases the weight of unique technical terms.
    tfidf = TfidfVectorizer()
    
    # This creates a matrix of numbers
    tfidf_matrix = tfidf.fit_transform(text_list)
    
    # 3. Calculate Cosine Similarity (The angle between the vectors)
    match_score = cosine_similarity(tfidf_matrix)[0][1]
    
    # 4. Convert to percentage (0-100) and round to 2 decimal places
    return round(match_score * 100, 2)