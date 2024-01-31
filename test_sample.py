import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('brand_logo_classifier.h5')

# 테스트할 이미지 불러오기 및 전처리
img_path = 'train/tropicapple/custom_0_22.jpg'  # 테스트할 이미지 파일 경로
img = image.load_img(img_path, target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.

# 예측 수행
prediction = model.predict(img)
formatted_prediction = [f'{prob:.4f}' for prob in prediction[0]]
print(formatted_prediction)
predicted_class = np.argmax(prediction)

# 클래스 레이블로 변환
if predicted_class == 0:
    label = '코카콜라'
elif(predicted_class == 1):
    label = '환타'
elif(predicted_class == 2):
    label = '펩시'
else:
    label = '스프라이트'

print(f"예측된 클래스: {label}")