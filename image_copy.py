from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os

# 데이터 증식 옵션 설정
datagen = ImageDataGenerator(
    rotation_range=80,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=(0.01, 1)
)

# 이미지 경로
class1_dir = 'test/wellchesGraph'

# 클래스 1 이미지 증식
for filename in os.listdir(class1_dir):
    img_path = os.path.join(class1_dir, filename)
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)  # 배치 차원 추가

    # 이미지 증식 및 저장
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='test/wellchesGraph', save_prefix='custom', save_format='jpg'):
        i += 1
        if i >= 9:  # 적절한 수준의 데이터 증식을 위해 5장으로 제한
            break
