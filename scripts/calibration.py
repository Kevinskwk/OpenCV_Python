import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

cv.namedWindow('Mask')
cv.namedWindow('Frame')

def f(x):
    pass

cv.createTrackbar('UpperH', 'Mask', 0, 180, f)
cv.createTrackbar('UpperS', 'Mask', 0, 255, f)
cv.createTrackbar('UpperV', 'Mask', 0, 255, f)
cv.createTrackbar('LowerH', 'Mask', 0, 180, f)
cv.createTrackbar('LowerS', 'Mask', 0, 255, f)
cv.createTrackbar('LowerV', 'Mask', 0, 255, f)
cv.createTrackbar('MinSize','Frame', 0 , 500, f)
cv.createTrackbar('Filter','Mask', 0 , 10, f)


cv.setTrackbarPos('UpperH', 'Mask', 158)
cv.setTrackbarPos('UpperS', 'Mask', 222)
cv.setTrackbarPos('UpperV', 'Mask', 189)
cv.setTrackbarPos('LowerH', 'Mask', 120)
cv.setTrackbarPos('LowerS', 'Mask', 14)
cv.setTrackbarPos('LowerV', 'Mask', 47)
cv.setTrackbarPos('MinSize','Frame', 150)
cv.setTrackbarPos('Filter', 'Mask', 4)

while(True):
    ret, frame = cap.read()

    UpperH = cv.getTrackbarPos('UpperH', 'Mask')
    UpperS = cv.getTrackbarPos('UpperS', 'Mask')
    UpperV = cv.getTrackbarPos('UpperV', 'Mask')
    LowerH = cv.getTrackbarPos('LowerH', 'Mask')
    LowerS = cv.getTrackbarPos('LowerS', 'Mask')
    LowerV = cv.getTrackbarPos('LowerV', 'Mask')
    MinSize = cv.getTrackbarPos('MinSize', 'Frame')
    Filter = cv.getTrackbarPos('Filter', 'Mask')

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_colour = np.array([LowerH,LowerS,LowerV])
    upper_colour = np.array([UpperH,UpperS,UpperV])
    mask = cv.inRange(hsv, lower_colour, upper_colour)
    mask = cv.medianBlur(mask, Filter*2+1)

    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv.contourArea(c) > MinSize:
            cv.drawContours(frame, [c], -1, (255, 100, 255), 2)

    cv.imshow('Mask', mask)
    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

print(UpperH, UpperS, UpperV, LowerH, LowerS, LowerV, MinSize, Filter)
cap.release()
cv.destroyAllWindows()