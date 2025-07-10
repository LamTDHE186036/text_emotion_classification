import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Tải stopwords nếu chưa có
nltk.download('punkt')
nltk.download('stopwords')

# Load model và tokenizer
model = load_model("emotions_model.h5")

# Giả sử bạn đã lưu tokenizer từ trước, nếu chưa:
# tokenizer = Tokenizer(...) rồi fit lại trên toàn bộ tập
# Còn nếu đã lưu thì load bằng pickle:
import pickle
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Các tham số cố định
max_len = 100  # Hoặc lấy lại đúng giá trị bạn dùng lúc training
stop_words = set(stopwords.words('english'))
important_words = {'no', 'not', 'nor', 'never'}
label_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Hàm tiền xử lý + dự đoán
def predict_emotion(text):
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.lower() not in stop_words or word.lower() in important_words]
    text_processed = ' '.join(filtered)
    seq = tokenizer.texts_to_sequences([text_processed])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    y_pred = model.predict(padded)
    predicted_class = int(np.argmax(y_pred, axis=1)[0])
    confidence = float(np.max(y_pred))

    return f"{label_map[predicted_class]} (Confidence: {confidence:.2f})"

# Giao diện Gradio
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=3, placeholder="Enter a sentence..."),
    outputs="text",
    title="Emotion Classifier",
    description="Enter an English sentence and get the predicted emotion."
)

demo.launch()
