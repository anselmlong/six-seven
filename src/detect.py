import cv2
from ultralytics import YOLO
from deepface import DeepFace

model = YOLO("yolov8n.pt")  # lightweight model

cap = cv2.VideoCapture(0)

if not cap.isOpened():
	print("Error: cannot open camera (index 0). Check that /dev/video0 exists and is not in use.")
	cap.release()
	exit(1)

while True:
	ret, frame = cap.read()
	if not ret or frame is None:
		# frame read failed; skip display and try again or break if camera disconnected
		print("Warning: empty frame received from camera; retrying...")
		if cv2.waitKey(100) & 0xFF == ord('q'):
			break
		continue

	# cv2.imshow('Webcam', frame)
	results = model(frame)  # run detection
	detections = results[0].boxes.data

	# Analyze emotions once (avoid duplicate calls)
	result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
	emotions = result[0]['emotion']  # dict of emotion: probability

	# Display percentages
	y_offset = 40
	for emotion, score in emotions.items():
		text = f"{emotion}: {score:.2f}%"
		cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		y_offset += 30

	cv2.imshow('Emotion Percentages', frame)
	# for detection in detections:
	#     # detection usually contains [x1, y1, x2, y2, score, class_id]
	#     x1, y1, x2, y2 = map(int, detection[:4])
	#     score = float(detection[4])
	#     class_id = int(detection[5])
	#     if score > 0.5:  # confidence threshold
	#         cv2.imshow('Detections', frame)
	#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
	#         cv2.putText(frame, f"{model.names[class_id]} {score:.2f}", (x1, y1 - 10),
	#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
