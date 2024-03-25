import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

# Fungsi untuk tokenisasi
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# Fungsi untuk menghapus stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# Fungsi untuk menggabungkan kembali teks
def join_text(tokens):
    clean_text = ' '.join(tokens)
    return clean_text

# Load data ulasan dari file CSV
data = pd.read_csv('D:/kuliah/project_folder/reviews.csv')

# Pra-pemrosesan data
data['Review'] = data['Review'].apply(clean_text)
data['Review'] = data['Review'].apply(tokenize_text)
data['Review'] = data['Review'].apply(remove_stopwords)
data['Review'] = data['Review'].apply(join_text)

# Membersihkan kolom Rating
data['Rating'] = data['Rating'].str.extract('(\d+)')

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Rating'], test_size=0.2, random_state=42)

# Ekstraksi fitur menggunakan TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Ubah tipe data kolom Rating menjadi numerik
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Latih model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Latih model SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

# Memilih model terbaik
best_model = nb_model if nb_model.score(X_train_vec, y_train) > svm_model.score(X_train_vec, y_train) else svm_model

# Simpan model ke dalam file
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')