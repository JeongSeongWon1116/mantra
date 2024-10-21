import cv2

# 얼굴 검출기 로드 (OpenCV에서 제공하는 pre-trained 얼굴 검출기)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 해상도 설정 (4096x2160)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

while True:
    ret, frame = cap.read()

    if ret:
        # 흑백 이미지로 변환 (얼굴 검출을 위해)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # 인식된 얼굴마다 사각형 그리기
        for (x, y, w, h) in faces:
            # 얼굴 영역에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 얼굴이 인식되고 있는지 메시지 출력
        if len(faces) > 0:
            print(f"얼굴 {len(faces)}개 인식됨.")
        else:
            print("얼굴이 인식되지 않음.")

        # 실시간 화면 출력
        cv2.imshow('Camera', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("카메라에서 프레임을 읽지 못했습니다.")

# 웹캠 해제
cap.release()
cv2.destroyAllWindows()
