from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 인수 파서와 변환할 인수를 분석
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
   help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
   help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
   default="mask_detector.model",
   help="path to output face mask detector model")
args = vars(ap.parse_args())

# 초기 학습률, 학습 할 Epoch 수 및 배치 크기 초기화
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# 데이터 세트 디렉토리에서 이미지 목록을 가져온 다음 데이터 목록(예: 이미지) 및 클래스 이미지를 초기화
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# 이미지 경로를 반복
for imagePath in imagePaths:
   # 파일 이름에서 클래스 레이블 추출
   label = imagePath.split(os.path.sep)[-2]

   # 입력 이미지 (224x224)를로드하고 전처리
   image = load_img(imagePath, target_size=(224, 224))
   image = img_to_array(image)
   image = preprocess_input(image)

   # 데이터 및 레이블 목록을 각각 업데이트
   data.append(image)
   labels.append(label)

# 데이터와 레이블을 NumPy 배열로 변환
data = np.array(data, dtype="float32")
labels = np.array(labels)

# 라벨에 one-hot 인코딩 수행
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#학습용 데이터 75퍼센트 25퍼센트 테스트용데이터로 나눔
(trainX, testX, trainY, testY) = train_test_split(data, labels,
   test_size=0.20, stratify=labels, random_state=42)

# 데이터 증대를 위한 훈련 이미지 생성기 구성
aug = ImageDataGenerator(
   rotation_range=20,
   zoom_range=0.15,
   width_shift_range=0.2,
   height_shift_range=0.2,
   shear_range=0.15,
   horizontal_flip=True,
   fill_mode="nearest")

# MobileNetV2 네트워크를로드하여 헤드 FC 레이어 세트가 꺼져 있는지 확인합니다.
baseModel = MobileNetV2(weights="imagenet", include_top=False,
   input_tensor=Input(shape=(224, 224, 3)))

# 기본 모델 위에 배치 될 모델의 헤드 구성
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#헤드 FC 모델을 기본 모델위에 배치합니다. (이것은 우리가 훈련 할 실제 모델)
model = Model(inputs=baseModel.input, outputs=headModel)

# 기본 모델의 모든 레이어를 반복하고 고정하여 첫 번째 학습 프로세스 중에 업데이트되지 않도록합니다.
for layer in baseModel.layers:
   layer.trainable = False

# 모델 컴파일
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
   metrics=["accuracy"])

# 네트워크의 head 훈련
print("[INFO] training head...")
H = model.fit(
   aug.flow(trainX, trainY, batch_size=BS),
   steps_per_epoch=len(trainX) // BS,
   validation_data=(testX, testY),
   validation_steps=len(testX) // BS,
   epochs=EPOCHS)

# 테스트 세트에 대한 예측
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# 테스트 세트의 각 이미지에 대해 가장 큰 예측 확률을 가진 레이블의 인덱스를 찾아야한다.
predIdxs = np.argmax(predIdxs, axis=1)

# 형식화 된 분류 보고서 표시
print(classification_report(testY.argmax(axis=1), predIdxs,
   target_names=lb.classes_))

# 모델을 디스크에 직렬화
print("[INFO] saving mask detector model…")
model.save(args["model"], save_format="h5")

# 훈련 손실과 정확도를 플로팅
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])