import cv2
import numpy as np
import base64
import time
from batch_detect import process_image  # from detect/
from utils import encode_base64         # from detect/

def webcam_detection():
    print("📸 Starting live detection. Press 'q' to quit, 's' to save frame.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not available.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_process = time.time()
    interval = 1.0  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_process > interval:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = process_image(frame_rgb)
            last_process = now

            # Decode base64 image
            try:
                decoded = base64.b64decode(result["annotated_image"])
                image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
            except:
                image = frame.copy()

            # Add OCR text summary
            ocr_text = result.get("ocr_result", {}).get("text", "")[:80]
            lang = result.get("ocr_result", {}).get("language", "en")
            cv2.putText(image, f"OCR ({lang}): {ocr_text}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Display box count
            ingr_count = len(result.get("ingredient_blocks", []))
            nutr_count = len(result.get("nutrition_tables", []))
            cv2.putText(image, f"Ingredients: {ingr_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(image, f"Nutrition: {nutr_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            image = frame

        cv2.imshow("🧠 Real-Time Detection", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved image as {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam detection stopped.")

if __name__ == "__main__":
    webcam_detection()
