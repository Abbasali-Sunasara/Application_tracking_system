import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the English language model
# This is the "brain" that identifies words, nouns, and entities
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback if model isn't found
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_skills(text):
    """
    Extracts potential skills (Nouns and Proper Nouns) from the text.
    In a real massive project, we would use a predefined list of 10,000 skills.
    For this mini-project, we use NLP to find 'important words'.
    """
    doc = nlp(text)
    skills = []
    
    # Loop through every word in the document
    for token in doc:
        # If the word is a Noun (e.g., "Python", "Management") or Proper Noun
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop:
            skills.append(token.text)
            
    # Return unique skills only (remove duplicates)
    return list(set(skills))

def calculate_similarity(resume_text, jd_text):
    """
    Calculates the match percentage using Cosine Similarity.
    """
    # 1. Create a list of the two texts
    text_list = [resume_text, jd_text]
    
    # 2. Convert text to numbers (Vectors)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text_list)
    
    # 3. Calculate similarity (0 to 1)
    match_score = cosine_similarity(count_matrix)[0][1]
    
    # 4. Convert to percentage (0 to 100)
    return round(match_score * 100, 2)