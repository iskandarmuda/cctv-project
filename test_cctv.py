import cv2

RTSP_URL = "rtsp://USERNAME:PASSWORD@IP_CCTV:554/STREAM"

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("GAGAL: CCTV tidak dapat dibuka")
    exit()

print("CCTV BERHASIL TERHUBUNG")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Gagal mengambil frame")
        break

    cv2.imshow("CCTV ONLINE", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()