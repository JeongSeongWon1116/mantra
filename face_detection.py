import cv2

# 얼굴 검출기 로드 (OpenCV에서 제공하는 pre-trained 얼굴 검출기)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 해상도 설정 (4096x2160)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4096)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# 적절한 얼굴 크기 범위 설정 (전체 프레임의 비율)
min_face_ratio = 0.2  # 얼굴이 프레임의 최소 20% 차지
max_face_ratio = 0.5  # 얼굴이 프레임의 최대 50% 차지

while True:
    ret, frame = cap.read()

    if ret:
        # 흑백 이미지로 변환 (얼굴 검출을 위해)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            # 첫 번째 얼굴을 가져옴
            (x, y, w, h) = faces[0]

            # 얼굴이 전체 프레임에서 차지하는 비율 계산
            face_area = w * h
            frame_area = frame.shape[0] * frame.shape[1]
            face_ratio = face_area / frame_area

            # 얼굴이 적절한 크기일 때만 사진 촬영
            if min_face_ratio <= face_ratio <= max_face_ratio:
                # 얼굴이 적절한 크기 -> 사진 촬영
                cv2.imwrite('perfect_face_photo.jpg', frame)
                print("사진 촬영 완료. 얼굴이 적절한 크기입니다.")
                break
            elif face_ratio < min_face_ratio:
                # 얼굴이 너무 작음 -> 가까이 오라고 유도
                cv2.putText(frame, "Too far! Please come closer.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("너무 멀리 있습니다. 가까이 오세요.")
            elif face_ratio > max_face_ratio:
                # 얼굴이 너무 큼 -> 멀어지라고 유도
                cv2.putText(frame, "Too close! Please move back.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                print("너무 가까이 있습니다. 뒤로 물러나세요.")

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
