# Installing libraries and packages.....
import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# Assigning different needed modules.....
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4,2)
detector = HandDetector(maxHands=1, detectionCon=0.8)

#For webcam input.....
cap = cv2.VideoCapture(0)

# Initializing the modules.....
with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.8) as hands,mp_face.FaceDetection(min_detection_confidence=0.8) as face:
  while cap.isOpened() & True:
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve the performances of the frames.....
    image.flags.writeable = False 
    results1 = hands.process(image)
    results2 = face.process(image)

    # Draws the hand and face annotations on the image.....
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #Right hand finger counting.....
    hand = detector.findHands(image, draw=False)
    fing = cv2.imread("Fingers/0.jpg")

    if hand:
	    lmlist = hand[0]
	    if lmlist:
	    	fingerup = detector.fingersUp(lmlist)
	    	if fingerup == [0, 1, 0, 0, 0]:
	    		fing = cv2.imread("Fingers/1.jpg")
	    	if fingerup == [0, 1, 1, 0, 0]:
	    		fing = cv2.imread("Fingers/2.jpg")
	    	if fingerup == [0, 1, 1, 1, 0]:
	    		fing = cv2.imread("Fingers/3.jpg")
	    	if fingerup == [0, 1, 1, 1, 1]:
	    		fing = cv2.imread("Fingers/4.jpg")
	    	if fingerup == [1, 1, 1, 1, 1]:
	    		fing = cv2.imread("Fingers/5.jpg")

    fing = cv2.resize(fing, (220, 280))
    image[0:280, 410:630] = fing
    
    # Draws the hand detection annotations on the image.....
    if results1.multi_hand_landmarks:
      handList = []
      for hand_landmarks in results1.multi_hand_landmarks:
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=2), mp_drawing.DrawingSpec(color=(47,255,255), thickness=3))

        # Left hand finger counting.....
        for idx, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            handList.append((cx, cy))

        upCount = 0

        for coordinate in finger_Coord:
            if handList[coordinate[0]][1] < handList[coordinate[1]][1]:
                upCount += 1

        if handList[thumb_Coord[0]][0] > handList[thumb_Coord[1]][0]:
            upCount += 1

        cv2.putText(image, str(upCount), (150,150), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,0), 12)
        
    # Draws the face detection annotations on the image.....
    if results2.detections:
      for detection in results2.detections:
        mp_drawing.draw_detection(image, detection,mp_drawing.DrawingSpec(color=(7,231,58), thickness=5, circle_radius=2), mp_drawing.DrawingSpec(color=(47,255,255), thickness=3))

    # Printing as output.....
    cv2.imshow('Image', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()  



