# you can tune with calibration.py
import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

cv.namedWindow('Mask')
cv.namedWindow('Image')

UpperH = 158        #0-180
UpperS = 222        #0-255
UpperV = 189        #0-255
LowerH = 120        #0-180
LowerS = 14         #0-255
LowerV = 47         #0-255
MinSize = 15000     #Minimun number of white pixels of the mask
Filter = 9          #Must be odd number

last_five = np.zeros((5,2), dtype=int)
DETECT = False

while(True):
    ret, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_colour = np.array([LowerH,LowerS,LowerV])
    upper_colour = np.array([UpperH,UpperS,UpperV])
    mask = cv.inRange(hsv, lower_colour, upper_colour)
    mask = cv.medianBlur(mask, 9)


    #contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    if sum(sum(mask)) >= MinSize:
        DETECT = True
        M = cv.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
    else:
        DETECT = False
        cX, cY = 0, 0

    #cv.drawContours(frame, mask, -1, (0,255,0), 3)
    cv.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
    cv.putText(frame, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv.imshow('Mask', mask)
    cv.imshow("Image", frame)

    if DETECT:
        current_pos = np.array([[cX, cY]])
        last_five = np.concatenate((last_five[1:5,:],current_pos))
        if np.std(last_five[:,0]) <= 1 and np.std(last_five[:,1]) <= 1:
            print(np.average(last_five, axis=0))


    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
print(UpperH, UpperS, UpperV, LowerH, LowerS, LowerV)
cap.release()
cv.destroyAllWindows()


#img = cv.imread("pick.jpg")
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# convert the grayscale image to binary image

#cv.waitKey(0)