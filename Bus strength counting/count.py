# model with yolo model 
import cv2
from ultralytics import YOLO
model = YOLO('/Users/sahnoontm/Desktop/Yolo-Project/crowd-detection/best.pt')
cap = cv2.VideoCapture(0)
minium_crowd = 0
above_avg = 1
over_crowd = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame, conf=.7, cls=0)
    for r in result:
        head_count = 0
        for box in r.boxes:
            x1, y1, x2, y2 = map(int,box.xyxy[0])
            conf = float(box.conf[0])
            roi = frame[y1:y2, x1:x2]
            roi_blur = cv2.GaussianBlur(roi, (25, 25), 30)
            frame[y1:y2, x1:x2] = roi_blur
            cv2.putText(frame,f"Head {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            head_count += 1
            def condition(count):
                if count > 10:
                    return 'Normal Crowds'
                elif count > 15:
                    return 'Average Crowds'
                elif count > 20:
                    return 'Over Crowd'
                elif count == 0:
                    return 'No people out there'

        cv2.putText(frame, f"Current NO: Poeple : {head_count}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, f'Status :{condition(head_count)}', (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)             

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
