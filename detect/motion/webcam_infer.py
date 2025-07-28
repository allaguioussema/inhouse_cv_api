import cv2
from batch_detect import process_image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    result = process_image(frame)
    vis = result["annotated_image"]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(vis), np.uint8), cv2.IMREAD_COLOR)

    cv2.imshow("Live Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
