import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지 데이터 불러오기 및 전처리
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# rescale = 픽셀값을 0~255사이 정수에서 0 ~ 1 사이의 값으로 변환

train_generator = train_datagen.flow_from_directory(
    directory='train',  # 학습 데이터셋 경로
    target_size=(150, 150),  # 이미지 크기 조정
    batch_size=32,
    class_mode='categorical',  # 다중 클래스 분류
    subset='training'
)


validation_generator = train_datagen.flow_from_directory(
    directory='train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN 모델 생성
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(12, activation='softmax')  # 클래스 수에 따라 변경
])

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=3,  # 학습 반복 횟수
    validation_data=validation_generator,
    validation_steps=validation_generator.n // validation_generator.batch_size
)

# 학습된 모델 저장
model.save('brand_logo_classifier.h5')
