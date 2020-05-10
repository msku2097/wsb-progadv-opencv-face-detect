# użycie
# python3 detect.py --zdjecie rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import pakietów
import numpy as np
import argparse
import cv2

# argumenty do przeparsowani
ap = argparse.ArgumentParser()
ap.add_argument("-z", "--zdjecie", required=True,
	help="sciezka do fotki")
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

# wczytaj zdjecie zrodlowe i zbuduj blob jako plik wejsciowy. 
# zmien jego rozmiar na 300x300 i znormalizuj
image = cv2.imread(args["zdjecie"])
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# wyslij blob i odbierz detekcje oraz przewidywania(prawdopodobienstwo)

print("[INFO] wykrywanie obiektów obliczeniowych...")
net.setInput(blob)
detections = net.forward()

# loopuj przez zbior detekcji
for i in range(0, detections.shape[2]):
	# wyciagnij procent prawdopodobienstwa polaczony z pewnoscia detekcji
	confidence = detections[0, 0, i, 2]

	# odfiltruj slabe detekcje przez filtr pewnosci. Upewnij sie, ze poziom filtru jest wiekszy niz minimalna pewnosc
	if confidence > args["confidence"]:
		# wykryj koordynaty x,y obrysu obiektu
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
 		# narysuj obrys obiektu razem z prawdopodobienstwem
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# wyswietl efekt koncowy
cv2.imshow("Output", image)
cv2.waitKey(0)