import cv2

def detect(filename):
    face_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        './cascades/haarcascade_eye.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5,0, (40,40))
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex+x,ey+y),(ex+ew+x,ey+eh+y),(0,255,0),2)
    cv2.imshow('face detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect('./img/1.jpg')
