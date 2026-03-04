from flask import Flask, render_template, request
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def calculate_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(similarity[0][0] * 100, 2)

@app.route('/', methods=['GET', 'POST'])
def index():
    similarity_score = None
    if request.method == 'POST':
        file = request.files['resume']
        job_desc = request.form['job_description']
        resume_text = extract_text_from_pdf(file)
        similarity_score = calculate_similarity(resume_text, job_desc)

    return render_template('index.html', score=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)