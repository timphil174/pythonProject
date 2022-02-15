import cv2
import numpy as np


class CascadeDetector:

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 500
        self.blobDetector = cv2.SimpleBlobDetector_create(detector_params)
        self.lc_example = cv2.imread('lc_example.jpg', 1)
        self.bc_example = cv2.imread('bc_example.jpg', 1)
        self.face_frame = None
        self.previous_face = [0, 0, 0, 0]
        self.previous_left_eye = [-1, 0, 0, 0]
        self.previous_right_eye = [-1, 0, 0, 0]
        self.left_eye_frame = None
        self.right_eye_frame = None
        self.lp_frame = None
        self.rp_frame = None
        self.lp_thresh_frame = None
        self.rp_thresh_frame = None
        self.move_thresh = 0.4
        self.left_pupil = [0, 0]
        self.right_pupil = [0, 0]
        self.tmp_left_pupil = [0, 0]
        self.tmp_right_pupil = [0, 0]
        self.left_is_visible = False
        self.right_is_visible = False
        self.overlap_threshold = 0.9
        self.consec_l = 0
        self.consec_r = 0
        self.consec_b = 0
        self.detected_count_l = 0
        self.detected_count_r = 0
        self.blink_count_l = 0
        self.blink_count_r = 0
        self.blink_count_b = 0
        self.rp_keypoint = None
        self.lp_keypoint = None
        self.phase = 0

    def detectFace(self,bgr_image):
        self.PUPIL_THRESH = cv2.getTrackbarPos('Calibration', 'EyePaint')
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray_image, 1.3, 5)

    def detectEyes(self, bgr_image):
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        return self.eye_cascade.detectMultiScale(gray_image, 1.3, 5)

    def detectPupils(self, bgr_image, threshold=127):
        img = cv2.copyTo(bgr_image, None)
        img[0:int(img.shape[0] / 4), 0:img.shape[1]] = (255, 255, 255)
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, t_img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
        img = cv2.erode(t_img, None, iterations=2)
        img = cv2.dilate(img, None, iterations=4)
        img = cv2.medianBlur(img, 5)
        return self.blobDetector.detect(img), t_img

    def find_eyes(self, frame):
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]
        frame_ratio = frame_w / frame_h
        frame_original = cv2.copyTo(frame, None)

        faces = self.detectFace(frame)
        for (x, y, w, h) in faces:
            face_w = int(frame_w / 3)
            face_h = int(face_w / frame_ratio)
            face_x = int(x + w / 2 - face_w / 2)
            face_y = int(y + h / 2 - face_h / 2)
            self.face_frame = frame_original[face_y:face_y + face_h, face_x:face_x + face_w]
            x, y, w, h = self.stabilize_face_frame(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            frame[0:y, 0:frame.shape[1]] = cv2.GaussianBlur(frame[0:y, 0:frame.shape[1]], (0, 0), 4)
            frame[y:y + h, 0:x] = cv2.GaussianBlur(frame[y:y + h, 0:x], (0, 0), 4)
            frame[y + h:frame.shape[0], 0:frame.shape[1]] = cv2.GaussianBlur(
                frame[y + h:frame.shape[0], 0:frame.shape[1]], (0, 0), 4)
            frame[y:y + h, x + w:frame.shape[1]] = cv2.GaussianBlur(frame[y:y + h, x + w:frame.shape[1]], (0, 0), 4)

            eyes = self.detectEyes(self.face_frame)
            self.left_is_visible = False
            self.right_is_visible = False
            for (ex, ey, ew, eh) in eyes:
                if ey + eh > face_h / 2:
                    pass
                if ex + ew / 2 < face_w / 2:
                    # Left eye
                    self.left_is_visible = True
                    ex, ey, ew, eh, self.previous_left_eye = self.stabilize_eyes_frame(face_x, face_y, ex, ey, ew,
                                                                                       eh, self.previous_left_eye)

                    cv2.rectangle(frame, (face_x + ex, face_y + ey), (face_x + ex + ew, face_y + ey + eh),
                                  (255, 0, 255), 2)

                    self.left_eye_frame = self.face_frame[ey:ey + eh, ex:ex + ew]
                    self.lp_keypoint, lt_img = self.detectPupils(self.left_eye_frame, self.PUPIL_THRESH)
                    self.lp_thresh_frame = cv2.cvtColor(lt_img, cv2.COLOR_GRAY2BGR)
                    self.lp_frame = cv2.copyTo(self.lp_thresh_frame, None)
                    self.lp_frame = cv2.drawKeypoints(self.lp_frame, self.lp_keypoint,
                                                      self.lp_frame,
                                                      (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #circle on left pupil
## traitement du clignement oeil gauche
                    if len(self.lp_keypoint) == 0 :
                        self.consec_l = self.consec_l + 1
                        self.detected_count_l = 0

                    if len(self.lp_keypoint) > 0:
                        self.detected_count_l = self.detected_count_l + 1
                        self.consec_l = 0
                        self.consec_b = 0
                        self.tmp_left_pupil = [int(self.lp_keypoint[0].pt[0]), int(self.lp_keypoint[0].pt[1])]
                        frame = cv2.circle(frame, (face_x + ex + self.left_pupil[0], face_y + ey + self.left_pupil[1]),
                                           5,
                                           (0, 255, 0), 4)
                        self.face_frame = cv2.circle(self.face_frame,
                                                     (ex + self.tmp_left_pupil[0], ey + self.tmp_left_pupil[1]), 5,
                                                     (0, 255, 0), 4)
                        # self.left_pupil = [face_x + ex + int(ew / 2), face_y + ey + int(eh / 2)]
                    else:
                        self.tmp_left_pupil = [face_x + ex + int(ew / 2), face_y + ey + int(eh / 2)]
                else:
                    # Right
                    self.right_is_visible = True
                    ex, ey, ew, eh, self.previous_right_eye = self.stabilize_eyes_frame(face_x, face_y, ex, ey, ew,
                                                                                        eh, self.previous_right_eye)
                    cv2.rectangle(frame, (face_x + ex, face_y + ey), (face_x + ex + ew, face_y + ey + eh),
                                  (255, 0, 255), 2)
                    self.right_eye_frame = self.face_frame[ey:ey + eh, ex:ex + ew]
                    self.rp_keypoint, rt_img = self.detectPupils(self.right_eye_frame, self.PUPIL_THRESH)
                    self.rp_thresh_frame = cv2.cvtColor(rt_img, cv2.COLOR_GRAY2BGR)
                    self.rp_frame = cv2.copyTo(self.rp_thresh_frame, None)
                    self.rp_frame = cv2.drawKeypoints(self.rp_frame, self.rp_keypoint,
                                                      self.rp_frame,
                                                      (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#traitement du clignement oeil droit
                    if len(self.rp_keypoint) == 0 :
                        self.consec_r = self.consec_r + 1
                        self.detected_count_r = 0

                    if len(self.rp_keypoint) > 0:
                        self.detected_count_r = self.detected_count_r + 1
                        self.consec_r = 0
                        self.consec_b = 0
                        self.tmp_right_pupil = [int(self.rp_keypoint[0].pt[0]), int(self.rp_keypoint[0].pt[1])]
                        frame = cv2.circle(frame,
                                           (face_x + ex + self.right_pupil[0], face_y + ey + self.right_pupil[1]), 5,
                                           (0, 255, 0), 4)
                        self.face_frame = cv2.circle(self.face_frame,
                                                     (ex + self.tmp_right_pupil[0], ey + self.tmp_right_pupil[1]), 5,
                                                     (0, 255, 0), 4)

                        # self.right_pupil = [face_x + ex + int(ew / 2), face_y + ey + int(eh / 2)]
                    else:
                        self.tmp_right_pupil = [face_x + ex + int(ew / 2), face_y + ey + int(eh / 2)]
                if self.phase == 2:
                    self.detectBlinkl()
                    self.detectBlinkr()
                    self.detectBlinkb()
#traitement clignement des deux yeux
                    if len(self.lp_keypoint) == 0 and len(self.rp_keypoint) == 0:
                        self.consec_b = self.consec_b + 1
                if self.phase == 1:
                    self.IsCalibrated()
        return frame

    def stabilize_face_frame(self, x, y, w, h):
        prev_norm = cv2.norm(np.array([x, y, w, h], np.float32), np.array(self.previous_face, np.float32))
        if prev_norm > 60:
            self.previous_face = [x, y, w, h]

        else:
            x = self.previous_face[0]
            y = self.previous_face[1]
            w = self.previous_face[2]
            h = self.previous_face[3]

        return x, y, w, h

    def stabilize_eyes_frame(self, face_x, face_y, x, y, w, h, previous_eyes_coords):
        if self.check_overlap_area(face_x + x, face_y + y, w, h, previous_eyes_coords) or previous_eyes_coords[0] == -1:
            previous_eyes_coords = [face_x + x, face_y + y, w, h]

        else:
            x = previous_eyes_coords[0] - face_x
            y = previous_eyes_coords[1] - face_y
            w = previous_eyes_coords[2]
            h = previous_eyes_coords[3]

        return x, y, w, h, previous_eyes_coords

    def check_overlap_area(self, x, y, w, h, previous_eyes_coords):
        px = previous_eyes_coords[0]
        py = previous_eyes_coords[1]
        pw = previous_eyes_coords[2]
        ph = previous_eyes_coords[3]

        over_x1 = x if x < px else px
        over_y1 = y if y < py else py
        over_x2 = (x + w) if x + w > px + pw else px + pw
        over_y2 = (y + h) if y + h > py + ph else py + ph

        overlap_area = (over_x2 - over_x1) * (over_y2 - over_y1)
        actual_area = w * h

        overlap_rate = actual_area / overlap_area
        return overlap_rate < self.overlap_threshold

    def get_images(self):               ## getter pour envoyer les données traitées à la classe GUI qui va ensuite les afficher
        images = {
            "face_frame": self.face_frame,
            "left_eye_frame": self.left_eye_frame,
            "right_eye_frame": self.right_eye_frame,
            "lp_thresh_frame": self.lp_thresh_frame,
            "rp_thresh_frame": self.rp_thresh_frame,
            "lp_frame": self.lp_frame,
            "rp_frame": self.rp_frame,
            "bc_example": self.bc_example,
        }
        return images

    def detectBlinkl(self):
        if self.consec_l >= 4:  #sensibilité du detecteur
            self.blink_count_l = self.blink_count_l + 1
            self.consec_l = 0

    def detectBlinkr(self):
        if self.consec_r >= 4:
            self.blink_count_r = self.blink_count_r + 1
            self.consec_r = 0

    def detectBlinkb(self):
        if self.consec_b >= 4:
            self.blink_count_b = self.blink_count_b + 1
            self.consec_b = 0

    def get_Blinks(self):
        blinks = {"left_blink": self.blink_count_l,
                  "right_blink": self.blink_count_r,
                  "both_blink": self.blink_count_b,
                  "detected_l": self.detected_count_l,
                  "detected_r": self.detected_count_r

        }
        return blinks

    def get_phase(self):
        return self.phase

    def PhaseChangeTo1(self):
        self.phase = 1

    def IsCalibrated(self):
        if self.detected_count_r > 30 and self.detected_count_l > 30:
            self.phase = 2
            self.blink_count_b = self.blink_count_r = self.blink_count_l = 0