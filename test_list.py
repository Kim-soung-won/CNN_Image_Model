import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# 모델 불러오기
model = load_model('brand_logo_classifier.h5')

test_datagen = ImageDataGenerator(rescale=1./255)  # 테스트 데이터셋 전처리

test_generator = test_datagen.flow_from_directory(
    directory='test',  # 테스트 데이터셋 경로
    target_size=(150, 150),  # 이미지 크기 조정
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"테스트 데이터셋 정확도: {test_accuracy*100:.2f}%")