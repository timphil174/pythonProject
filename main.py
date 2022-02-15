import cv2
import numpy as np
import Detector_bothV2
import GUI2_both

PUPIL_THRESH = 65

detector = Detector_bothV2.CascadeDetector()
gui = GUI2_both.Gui2()
cap = cv2.VideoCapture(0)
cv2.namedWindow('EyePaint', cv2.WINDOW_FULLSCREEN)
cv2.createTrackbar('Calibration', 'EyePaint', 0, 160, gui.on_trackbar)
cv2.setTrackbarPos('Calibration', 'EyePaint', PUPIL_THRESH)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        detector.find_eyes(frame)
        gui.make_window(frame, detector.get_images(), detector.get_Blinks(), detector.get_phase(), detector.overlap_threshold)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            detector.PhaseChangeTo1()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()