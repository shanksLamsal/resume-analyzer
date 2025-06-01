import re
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Function to extract text from PDF
def extract_resume_text(pdf_path):
    return extract_text(pdf_path)

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Compare resume vs job description
def compare_texts(resume_text, jd_text):
    vectorizer = CountVectorizer().fit([resume_text, jd_text])
    vectors = vectorizer.transform([resume_text, jd_text])
    vocab = vectorizer.get_feature_names_out()

    resume_vec = vectors[0].toarray()[0]
    jd_vec = vectors[1].toarray()[0]

    score = 0
    total = sum(jd_vec)

    matched_keywords = []
    missing_keywords = []

    for i, val in enumerate(jd_vec):
        if val > 0:
            if resume_vec[i] > 0:
                score += 1
                matched_keywords.append(vocab[i])
            else:
                missing_keywords.append(vocab[i])

    match_percent = round((score / len(matched_keywords + missing_keywords)) * 100, 2)
    return match_percent, matched_keywords, missing_keywords

# Main driver
def main():
    resume_path = 'sample_resume.pdf'
    job_description = input("Paste the job description:\n")

    resume_raw = extract_resume_text(resume_path)
    resume_clean = preprocess(resume_raw)
    jd_clean = preprocess(job_description)

    score, matched, missing = compare_texts(resume_clean, jd_clean)

    print(f"\nMatch Score: {score}%")
    print("\n✅ Keywords in Resume:", ', '.join(matched))
    print("\n❌ Missing Important Keywords:", ', '.join(missing))

if __name__ == "__main__":
    main()
