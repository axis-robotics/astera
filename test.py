import cv2
cam = cv2.VideoCapture()
cam.open("rtsp://192.168.42.129:8080/h264_pcm.sdp")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    
