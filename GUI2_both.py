import numpy as np
import cv2
import ctypes
from numpy import zeros


class Gui2:
    def __init__(self):
        self.screensize = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)

    def make_window(self, main_image, images, blink_count, phase, sensibility=0.95):
        ratio = main_image.shape[1] / main_image.shape[0]

        img = zeros([self.screensize[0], self.screensize[1], 3])
        img = np.array(img, dtype=np.uint8)
        left_blink_count = blink_count["left_blink"]
        right_blink_count = blink_count["right_blink"]
        both_blink_count = blink_count["both_blink"]
        window_width = int(self.screensize[0])
        img[0:img.shape[0], 0:img.shape[1]] = (255, 255, 255) #on crée une matrice qu'on interpretera mais également changé certaines valeurs pour afficher les différents images ( face, yeux, ...)

        face_frame = images["face_frame"] # l'image de la tête qui va etre affichée
        if face_frame is not None:
            face_width = int(window_width * 0.25)
            face_eight = int(face_width / ratio)
            face_x_offset = int(window_width * 0.02)
            face_frame = cv2.resize(face_frame, (face_width, face_eight))
            img[40:face_frame.shape[0] + 40, face_x_offset: face_x_offset + face_width] = face_frame
            img = cv2.putText(img, 'Face', (int(window_width/18), 35),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 0, 0))
        if face_frame is None:
            img = cv2.putText(img, 'put down your mask so I can analyse your beautiful face', (int(window_width / 18)-60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 0, 0))
        left_eye_frame = images["left_eye_frame"]
        right_eye_frame = images["right_eye_frame"]

            #left eye image
        if left_eye_frame is not None:
            lefteye_width = int(window_width * 0.08)
            lefteye_eight = lefteye_width
            lefteye_x_offset = 55
            lefteye_y_offset = face_frame.shape[0] + 100
            left_eye_frame = cv2.resize(left_eye_frame, (lefteye_width, lefteye_eight))
            img[lefteye_y_offset:lefteye_width + lefteye_y_offset, lefteye_x_offset: lefteye_x_offset + lefteye_width] = left_eye_frame

            # Right Eye Image
        if right_eye_frame is not None:
            im3_width = int(window_width * 0.08)
            im3_height = im3_width
            im3_x_offset = 55 + int(window_width*0.08) + 30
            im3_y_offset = face_frame.shape[0] + 100
            right_eye_frame = cv2.resize(right_eye_frame, (im3_width, im3_height))
            img[im3_y_offset:im3_width + im3_y_offset, im3_x_offset: im3_x_offset + im3_width] = right_eye_frame

        if left_eye_frame is not None or right_eye_frame is not None:
            img = cv2.putText(img, 'Eyes', (int(window_width/18)+65, face_frame.shape[0] + 90),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=(0, 0, 0))

# Left Pupil Keypoints Image
        lp_frame = images["lp_frame"]
        rp_frame = images["rp_frame"]
        if lp_frame is not None:
            im6_width = int(window_width * 0.075)
            im6_height = im6_width  # int(im6_width / ratio)
            im6_x_offset = int(window_width * 0.29) + 50
            im6_y_offset = 50
            lp_frame = cv2.resize(lp_frame, (im6_width, im6_height))
            img[im6_y_offset:im6_y_offset + im6_height, im6_x_offset: im6_x_offset + im6_width] = lp_frame

            # Right Pupil Keypoints Image
        if rp_frame is not None:
            im7_width = int(window_width * 0.075)
            im7_height = im7_width  # int(im3_width / ratio)
            im7_x_offset = int(window_width*0.37) + 70
            im7_y_offset = 50
            rp_frame = cv2.resize(rp_frame, (im7_width, im7_height))
            img[im7_y_offset:im7_y_offset+im7_height, im7_x_offset: im7_x_offset + im7_width] = rp_frame
        if phase == 2:
            img = cv2.putText(img, 'Pupils detected', (int(window_width / 3) - 25, 60), cv2.FONT_HERSHEY_SIMPLEX,
                              0.70, color=(0, 0, 0))
            img = cv2.putText(img, 'eye Blink Counter :', (int(window_width / 3) - 65, 150),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0, 0, 0))
            img = cv2.putText(img, 'LEFT : ' + str(left_blink_count), (int(window_width / 3) - 65, 185),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(0, 0, 0))
            img = cv2.putText(img, 'RIGHT : ' + str(right_blink_count), (int(window_width / 3) - 65, 220),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(0, 0, 0))
            #img = cv2.putText(img, 'BOTH : ' + str(both_blink_count),
                              #(int(window_width / 3) - 65, 255),
                              #cv2.FONT_HERSHEY_SIMPLEX, 1.2, color=(0, 0, 0))
        if phase == 0:
            img = cv2.putText(img, 'Threshold calibration', (int(window_width / 3) - 25, 60), cv2.FONT_HERSHEY_SIMPLEX,
                              0.75, color=(0, 0, 0))
            img = cv2.putText(img, 'Please stay steady and try', (int(window_width / 3) - 40, 140),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 0))
            img = cv2.putText(img, 'to calibrate so it looks like this :', (int(window_width / 3) - 45, 158),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 0))
            img = cv2.putText(img, 'it is accurate when it can   ', (int(window_width / 3) - 55, 245),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 0))
            img = cv2.putText(img, 'continuously detect your pupils ', (int(window_width / 3) - 55, 263),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 0))
            img = cv2.putText(img, 'If you are ready ', (int(window_width / 3) - 80, 330),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.4, color=(0, 0, 0))
            img = cv2.putText(img, 'PRESS R', (int(window_width / 3) - 60, 370),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.4, color=(0, 0, 0))
            bc_pic = images["bc_example"]
            bc_x_offset = int(window_width * 0.27) + 70
            bc_y_offset = 165
            bc_pic = cv2.resize(bc_pic, (int(bc_pic.shape[1]*0.8), int(bc_pic.shape[0]*0.8)))
            img[bc_y_offset:bc_pic.shape[0] + bc_y_offset, bc_x_offset: bc_x_offset + bc_pic.shape[1]] = bc_pic


        #decompte
        if phase == 1:
            img = cv2.putText(img, 'Start in:', (300, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, color=(0, 0, 0))
            detectCounterl = blink_count["detected_l"]
            detectCounterr = blink_count["detected_r"]
            #print(detectCounterl)
            #print(detectCounterr)
            diff = detectCounterr-detectCounterl
            print(abs(diff))
            if abs(diff) > 40:
                detectCounterr = detectCounterl = 0
            if 7 < detectCounterr <= 20 and 7 < detectCounterl <= 20:
                img = cv2.putText(img, '3', (400, 400),
                                  cv2.FONT_HERSHEY_SIMPLEX, 8, color=(0, 0, 0))
            if 21 < detectCounterr <= 35 and 21 < detectCounterl <= 35:
                img = cv2.putText(img, '2', (400, 400),
                                  cv2.FONT_HERSHEY_SIMPLEX, 8, color=(0, 0, 0))
            if 36 < detectCounterr <= 45 and 36 < detectCounterl <= 45:
                img = cv2.putText(img, '1', (400, 400),
                                  cv2.FONT_HERSHEY_SIMPLEX, 8, color=(0, 0, 0))

        cv2.imshow('EyePaint', img)

    def on_trackbar(self, val):
        pass



