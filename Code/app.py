import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pickle

try : 
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    raise Exception(f"Error loading NLTK resources: {str(e)}")

stop_words = set(stopwords.words('english'))
important_words = {'no', 'not', 'nor', 'never'}


def load_resources():
    model = load_model("emotions_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer


label_map = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

def predict_emotion(text):
    

    
    model, tokenizer = load_resources()

    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word.lower() not in stop_words or word.lower() in important_words]
    text_processed = ' '.join(filtered)
        
    seq = tokenizer.texts_to_sequences([text_processed])
    max_len = 100 
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    y_pred = model.predict(padded, verbose=0)
    predicted_class = int(np.argmax(y_pred, axis=1)[0])
    confidence = float(np.max(y_pred))

    return f"Emotional sentences: {label_map[predicted_class]}-{predicted_class} (Confidence: {confidence:.0%})"
  
demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=3, placeholder="Enter an English sentence..."),
    outputs=gr.Textbox(),
    title="Emotion Classifier",
    description="Enter an English sentence or text to predict the emotion (Sadness, Joy, Love, Anger, Fear, Surprise).",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()