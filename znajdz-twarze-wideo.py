# użycie
# python3 znajdz-twarze-wideo.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import pakietów
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="sciezka do zserializowanego pliku Caffe 'deploy' ")
ap.add_argument("-m", "--model", required=True,
	help="sciezka do wytrenowanego modelu Caffee")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimalne prawdopodobieństwo odfiltrowania słabych detekcji")
args = vars(ap.parse_args())

# wczytaj nasz zserializowany model twarzy z dysku
print("[INFO] wczytywanie wytrenowanego modelu...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Uruchom strumien wideo. Zezwol na dostep do kamery.
print("[INFO] uruchamiania strumienia wideo...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Iteruj po klatka wideo ze stumienia
while True:
	# zlap klatke i zmien jej rozmiar na maxymalny 400 px
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# zlap wymiary ramki i przekonweruj je na bloba
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# wyslij blob i odbierz detekcje oraz przewidywania(prawdopodobienstwo)
	net.setInput(blob)
	detections = net.forward()

	# loopuj przez zbior detekcji
	for i in range(0, detections.shape[2]):
		# wyciagnij procent prawdopodobienstwa polaczony z pewnoscia detekcji
		confidence = detections[0, 0, i, 2]

		# fodfiltruj slabe detekcje przez filtr pewnosci. Upewnij sie, ze poziom filtru jest wiekszy niz minimalna pewnosc
		if confidence < args["confidence"]:
			continue

		# wykryj koordynaty x,y obrysu obiektu
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# narysuj obrys obiektu razem z prawdopodobienstwem
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# wyswietl zmodyfikowany strumien
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# wcisniecie klawisza 'q' przerwie petle
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()