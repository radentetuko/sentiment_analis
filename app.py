from flask import Flask, render_template, request
import joblib
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import time
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from collections import Counter
from datetime import datetime, timedelta

app = Flask(__name__)

# Load model dan vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Inisialisasi stemmer untuk Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Menggunakan regular expression untuk menghapus karakter non-alfanumerik dan mengubah teks menjadi lowercase
    cleaned_text = re.sub(r'\W', ' ', text.lower())
    return cleaned_text

# Fungsi untuk pra-pemrosesan teks
def preprocess_text(text):
    # Regular Expression: Menghapus karakter non-alfanumerik
    cleaned_text = re.sub(r'\W', ' ', text)
    
    # Case Folding: Mengubah teks menjadi lowercase
    cleaned_text = cleaned_text.lower()
    
    # Tokenizing: Memecah teks menjadi token
    tokens = word_tokenize(cleaned_text)
    
    # Filtering: Menghapus stopwords
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming: Melakukan stemming pada setiap token
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # Formalization: Menggabungkan kembali token menjadi teks
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

# Fungsi untuk mendapatkan ulasan dari URL
def scrape_reviews(url):
    if url:
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(options=options)
        driver.get(url)

        time.sleep(2)

        # Scroll ke bawah dan mengklik ulasan
        driver.execute_script("window.scrollTo(0, window.scrollY + 300);")
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='review']"))).click()
        
        data = []
        for i in range(0, 10):  # Ganti angka 10 menjadi 100 untuk mendapatkan 100 data
            soup = BeautifulSoup(driver.page_source, "html.parser")
            containers = soup.findAll('article', attrs={'class': 'css-72zbc4'})

            for container in containers:
                try:
                    review = container.find('span', attrs={'data-testid': 'lblItemUlasan'}).text
                    rating = container.find('div', attrs={'data-testid': 'icnStarRating'}).get('aria-label')
                    date_string = container.find('p', attrs={'data-unify': 'Typography'}).text
                    date = parse_date(date_string)
                    data.append({"Review": review, "Rating": rating, "Date": date})
                except AttributeError:
                    continue

            time.sleep(2)
            try:
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']"))).click()
            except:
                break
            time.sleep(3)

        driver.quit()

        # Membuat DataFrame dari data
        df = pd.DataFrame(data)

        return df
    
# Fungsi untuk mendapatkan kata-kata yang paling sering muncul dalam ulasan
def get_common_words(reviews):
    # Menggabungkan semua ulasan menjadi satu string
    all_reviews = ' '.join(reviews)
    # Tokenisasi ulasan
    tokens = all_reviews.split()
    # Menghitung frekuensi kemunculan kata-kata
    word_counts = Counter(tokens)
    # Mengambil 10 kata yang paling sering muncul
    common_words = word_counts.most_common(10)
    return common_words

# Fungsi untuk menganalisis sentimen ulasan
def analyze_sentiment(reviews):
    reviews_vec = vectorizer.transform(reviews)
    predictions = model.predict(reviews_vec)
    return predictions

# Fungsi untuk mengonversi format tanggal "X [satuan waktu] lalu" menjadi tanggal yang sesuai
def parse_date(date_string):
    if 'minggu' in date_string:
        weeks_ago = int(date_string.split()[0])
        return datetime.now() - timedelta(weeks=weeks_ago)
    elif 'bulan' in date_string:
        months_ago = int(date_string.split()[0])
        return datetime.now() - timedelta(days=months_ago*30)
    elif 'tahun' in date_string:
        years_ago = int(date_string.split()[0])
        return datetime.now() - timedelta(days=years_ago*365)
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        reviews_df = scrape_reviews(url)
        
        # Membersihkan teks, melakukan tokenisasi, menghapus stopwords, dan menggabungkan kembali teks
        cleaned_reviews = reviews_df['Review'].apply(lambda x: preprocess_text(clean_text(x)))
        
        predictions = analyze_sentiment(cleaned_reviews)
        sentiment = round(sum(predictions) / len(predictions), 1)

        # Menghitung jumlah rating
        rating_counts = reviews_df['Rating'].value_counts().to_dict()

        # Mendapatkan kata-kata yang sering muncul
        common_words = get_common_words(cleaned_reviews)

        return render_template('result.html', sentiment=sentiment, rating_counts=rating_counts, common_words=common_words)

if __name__ == '__main__':
    app.run(debug=True)
