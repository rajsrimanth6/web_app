from inspect import getframeinfo
import math
import mediapipe as mp
import numpy as np
import cv2 as cv
import time

# variables
global both_counter
global both_final_counter
global counter
global final_counter
global right_counter
global right_final_counter
global blink_time
global blink_list
global count
both_counter = 0
both_final_counter = 0
counter = 0
final_counter = 0
right_counter = 0
right_final_counter = 0
blink_time = [0, 0]
blink_list = []
count = 0

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]

L_H_LEFT = [33]  # right eye right most landmark
L_H_RIGHT = [133]  # right eye left most landmark
R_H_LEFT = 362  # left eye right most landmark
R_H_RIGHT = 263  # left eye left most landmark

L_UP = 386  # left eye uppermost  coordinate
L_DOWN = 374  # left eye downmost  coordinate


R_UP = 159  # right eye uppermost  coordinate
R_DOWN = 145  # right eye downmost  coordinate
landmarks = []


class Videocamera(object):
    def __init__(self) -> None:
        self.vedio = cv.VideoCapture(0)

    def release(self):
        self.vedio.release()

    def get_frame(self):

        with mp_face_mesh.FaceMesh(max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5
                                   ) as face_mesh:

            ret, self.frame = self.vedio.read()
            self.frame = cv.flip(self.frame, 1)
            if not ret:
                return
            rgb_frame = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
            img_h, img_w = self.frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points0 = np.array([tuple(np.multiply([p.x, p.y], [img_w, img_h]).astype(
                    int).ravel()) for p in results.multi_face_landmarks[0].landmark])
                mesh_points = [tuple(np.multiply([p.x, p.y], [img_w, img_h]).astype(
                    int).ravel()) for p in results.multi_face_landmarks[0].landmark]
                # print(results.multi_face_landmarks)
                # print(mesh_points)

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(
                    mesh_points0[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                    mesh_points0[RIGHT_IRIS])

                center_left = tuple(np.array([l_cx, l_cy], dtype=np.int32))
                center_right = tuple(
                    np.array([r_cx, r_cy], dtype=np.int32))

                cv.circle(self.frame, center_left, int(l_radius),
                          (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(self.frame, center_right, int(r_radius),
                          (255, 0, 255), 1, cv.LINE_AA)

                iris_pos, ratio = self.iris_position(
                    center_right, mesh_points[R_UP], mesh_points[R_DOWN], mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT])

                L_blink, L_no_blinks = self.right_blink_status(
                    mesh_points[L_UP], mesh_points[L_DOWN])

                R_blink, R_no_blinks = self.blink_status(
                    mesh_points[R_UP], mesh_points[R_DOWN])

                no_L_blinks = "no.L.blinks: "+str(L_no_blinks)
                blink_stat, no_both_blinks = self.both_blinks(
                    L_blink, R_blink)
                cv.putText(self.frame, f"Iris pos: {iris_pos}", (
                    30, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 0), 1, cv.LINE_AA)

        ret, jpeg = cv.imencode('.jpeg', self.frame)
        return jpeg.tobytes()

    def iris_position(self, iris_center, up_point, down_point, right_point, left_point):
        center_to_right_dist = self.euclidean_distance(
            iris_center, right_point)
        total_distance = self.euclidean_distance(right_point, left_point)

        ratio = center_to_right_dist/total_distance

        diff = down_point[1]-up_point[1]
        if diff < 9:
            iris_position = "down"
        elif diff > 12:
            iris_position = "up"
        elif ratio > 0.42 and ratio <= 0.57:
            iris_position = "center"
        elif ratio <= 0.42:
            iris_position = "right"
        else:
            iris_position = "left"
        ratio_vertical = diff
        return iris_position, ratio_vertical

    def right_blink_status(self, down_point, up_point):
        global right_counter
        global right_final_counter
        total_vert_distance = self.euclidean_distance(down_point, up_point)

        if (total_vert_distance < 6):
            right_counter += 1
            return "blink", right_final_counter
        else:
            if right_counter > 0:
                right_final_counter += 1
                right_counter = 0
            return "no blink", right_final_counter

    def both_blinks(self, left, right):
        global both_counter
        global both_final_counter
        if (left == "blink" and right == "blink"):
            both_counter += 1
            return "blink", both_final_counter
        else:
            if both_counter > 0:
                both_final_counter += 1
                both_counter = 0
            return "no blink", both_final_counter

    def double_blink(self, blink_status, interval=1):
        global blink_time
        global blink_list
        global count

        current_status = blink_status

        if current_status == "blink":
            blink_time.append(time.time())
            count += 1

            if count == 2:
                current_time = time.time()
                print('time: ', current_time - blink_time[-1])

                if time.time() - blink_time[-1] <= 0.5:
                    print("double blink")
                    blink_list.append("blink")
                    blink_time.append(current_time)
                    count = 0
                    return "double blink"
                else:
                    return blink_status
            else:
                return blink_status
        else:
            return blink_status

    def blink_status(self, down_point, up_point):
        global counter
        global final_counter
        total_vert_distance = self.euclidean_distance(down_point, up_point)

        if (total_vert_distance < 6):
            counter += 1
            return "blink", final_counter
        else:
            if counter > 0:
                final_counter += 1
                counter = 0
            return "no blink", final_counter

    def euclidean_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        self.distance = math.sqrt((x2 - x1)**2 + (y2-y1)**2)
        return self.distance


# ved = cv.VideoCapture(0)
# while True:
#     ret, frame = ved.read()
#     if not ret:
#         break
#     frame = cv.flip(frame, 1)
#     cv.imshow('img', frame)
#     key = cv.waitKey(1)
#     if key == ord('q'):
#         break
# ved.release()
# cv.destroyAllWindows()
