import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetector(maxHands=2, detectionCon=.8)

    while True:
        success, frame = cap.read()
        if not success:
            break

        hands, fram = detector.findHands(frame)
        if hands:
            for hand in hands:
                lmList = hand["lmList"]
                bbox = hand["bbox"]
                center = hand["center"]
                handType = hand["type"]
                
                for lm in lmList:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 
                             5, (255, 0, 0), cv2.FILLED)
                    
        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()