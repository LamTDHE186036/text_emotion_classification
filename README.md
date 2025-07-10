# text_emotion_classification

# Phân loại cảm xúc trên dữ liệu bằng LSTM

Dự án sử dụng mạng nơ-ron hồi tiếp **LSTM (Long Short-Term Memory)** để phân loại cảm xúc các đoạn văn trong tập dữ liệu **emotion** thành các nhãn : sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)

  
# data
- giải nén file .rar để sử dụng data

# Train_model

- Tiền xử lý dữ liệu văn bản: làm sạch, mã hóa, padding
- Xây dựng mô hình Deep Learning (LSTM) để phân loại cảm xúc văn bản
- Huấn luyện và đánh giá mô hình với tập IMDB
- Lưu mô hình file .h5 sau khi train

# Test_model
- Sử dụng model đã lưu (.h5) đưa vào 1 câu nhận xét bất kì
- Tiền xử lý dữ liệu đưa vào: làm sạch, mã hóa, padding
- Đưa ra nhận xét cảm xúc
# Demo app
 - Link demo : https://ba3d8bf8f6aa106064.gradio.live/
 - nhập 1 câu bất kì đưa ra dự đoán các nhãn : sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)
# Liên hệ 
Bạn có thể liên hệ qua email : Anhlam2k44@gmail.com
