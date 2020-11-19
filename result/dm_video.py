from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
   # frame의 수치를 잡고 blob을 생성합니다.
   # 그값을 넣어줌
   (h, w) = frame.shape[:2]
   blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
      (104.0, 177.0, 123.0))

   # 네트워크를 통해 blob을 전달하고 얼굴 감지를 얻습니다.
   faceNet.setInput(blob)
   detections = faceNet.forward()

   # 얼굴 목록, 해당위치를 초기화
   # 안면 마스크 네트워크의 예측 목록
   faces = []
   locs = []
   preds = []

   # 탐지를 반복
   for i in range(0, detections.shape[2]):

      #탐지와 관련된 신뢰도 (i.e., 확률) 추출
      confidence = detections[0, 0, i, 2]

      # 신뢰도가 최소 신뢰도보다 큰지 확인하여 약한 감지를 필터링
      if confidence > args["confidence"]:
         # 객체에 대한 경계 상자의 (x,y)좌표를 계산합니다.
         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
         (startX, startY, endX, endY) = box.astype("int")

         # 경계 상자가 프레임 크기 내에 있는지 확인
         (startX, startY) = (max(0, startX), max(0, startY))
         (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

         # 얼굴 ROI를 추출하고 BGR에서 RGB 채널로 변환
         # 주문하고 224x224로 크기를 조정하고 전처리
         try:
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
         except Exception as e:
            print(e)


         # 얼굴과 경계 boxes를 각각 추가
         # 목록들
         faces.append(face)
         locs.append((startX, startY, endX, endY))

   # 얼굴이 하나 이상 감지 된 경우에만 예측
   if len(faces) > 0:
      # 더 빠른 추론을 위해 일대일 예측이 아닌 모든 얼굴에 대한 일괄 예측을 동시에 수행합니다.
      # for문안에서 루프도는동안
      try:
         faces = np.array(faces, dtype="float32")
         preds = maskNet.predict(faces, batch_size=32)
      except Exception as e:
         print(e)


   # 얼굴 위치와 해당 위치의 2- 튜플을 반환합니다.
   return (locs, preds)

# 인수 파서를 구성하고있는 녀석을 딴인수로 바꿔준다.
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
   default="face_detector",
   help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
   #default="mask_detector.model",
   default="mask_detector_overLearning.model",
   help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
   help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# 직렬화 된 얼굴 감지기 모델을 디스크에서 로드
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
   "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# 아면 마스크 감지기 모델 로드
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# 비디오 스트림을 초기화하고 카메라 센서 작동 대.
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 비디오 스트림의 프레임을 반복
while True:
   #스레드 된 비디오 스트림에서 프레임을 가져와 사이즈 지정.
   frame = vs.read()
   frame = imutils.resize(frame, width=800,height=450)
   # 프레임에서 얼굴을 감지하고 얼굴 마스크를 착용하고 있는지 확인
   try:
      (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
   except Exception as e:
      print(str(e))


   # 감지 된 얼굴 위치와 해당 위치를 반복합니다.
   for (box, pred) in zip(locs, preds):
      # 경계 상자 및 예측 압축 풀기
      (startX, startY, endX, endY) = box
      (mask, withoutMask) = pred

      # 경계 상자와 텍스트를 그리는 데 사용할 클레스 레이블과 색상 결정
      label = "Mask" if mask > 0.8 else "No Mask"
      color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

      # 라벨에 확률도 넣어줌
      label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

      # 출력 프레임에 레이블 및 경계 상자 직사각형 표시
      cv2.putText(frame, label, (startX, startY - 10),
         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
      cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

   #출력 프레임 표시
   cv2.imshow("Frame", frame)
   key = cv2.waitKey(1) & 0xFF

   # q키를 누를시 종료
   if key == ord("q"):
      break
vs.stop()
cv2.destroyAllWindows()