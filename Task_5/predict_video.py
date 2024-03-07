# import os
# from ultralytics import YOLO
# import cv2


# VIDEOS_DIR = os.path.join('.', 'Task_5','videos')

# video_path = os.path.join(VIDEOS_DIR, 'test.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# model_path = os.path.join('.','Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')

# model = YOLO(model_path) 

# threshold = 0.5

# while ret:

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     out.write(frame)
#     ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()
# ------------------------------------------------------------------------------
import os
from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', 'Task_5','videos')

video_path = os.path.join(VIDEOS_DIR, 'test.mp4')

cap = cv2.VideoCapture(video_path)

model_path = os.path.join('.','Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')

model = YOLO(model_path) 

threshold = 0.5

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# import os
# from ultralytics import YOLO
# import cv2

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)  # Draw center point
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ------------------------------------------------------------------------------

# import os
# from ultralytics import YOLO
# import cv2
# import numpy as np

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# # Function to calculate points around the circumference of the tire
# def calculate_tire_points(x_center, y_center, radius, num_points):
#     points = []
#     for i in range(num_points):
#         angle = 2 * np.pi * i / num_points
#         x = int(x_center + radius * np.cos(angle))
#         y = int(y_center + radius * np.sin(angle))
#         points.append((x, y))
#     return points

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)

#             # Calculate additional points on the tire
#             tire_radius = int((x2 - x1) / 2)
#             num_points = 4  # You can adjust this number based on your requirement
#             tire_points = calculate_tire_points(center_x, center_y, tire_radius, num_points)
            
#             # Draw center point and additional points
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
#             for point in tire_points:
#                 cv2.circle(frame, point, 3, (255, 0, 0), -1)  # Additional points
            
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# ------------------------------------------------------
# import os
# from ultralytics import YOLO
# import cv2
# import math

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             radius = max((x2 - x1) / 2, (y2 - y1) / 2)

#             # Calculate two additional points on the circumference of the tyre
#             point1_x = int(center_x + radius * math.cos(0))
#             point1_y = int(center_y + radius * math.sin(0))
#             point2_x = int(center_x + radius * math.cos(math.pi / 2))
#             point2_y = int(center_y + radius * math.sin(math.pi / 2))

#             # Draw center point and additional points
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
#             cv2.circle(frame, (point1_x, point1_y), 5, (255, 0, 0), -1)  # Additional point 1
#             cv2.circle(frame, (point2_x, point2_y), 5, (255, 0, 0), -1)  # Additional point 2

#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#             # Calculate angle between the two additional points and the center point
#             angle_rad = math.atan2(point1_y - center_y, point1_x - center_x) - math.atan2(point2_y - center_y,
#                                                                                           point2_x - center_x)
#             angle_deg = math.degrees(angle_rad)
#             if angle_deg < 0:
#                 angle_deg += 360

#             cv2.putText(frame, f"Angle: {angle_deg:.2f} degrees", (int(x1), int(y2 + 30)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# ----------
# -------------------------------------------------------------
# import os
# from ultralytics import YOLO
# import cv2
# import numpy as np

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# # Function to calculate points around the circumference of the tire
# def calculate_tire_points(x_center, y_center, radius, num_points):
#     points = []
#     for i in range(num_points):
#         angle = 2 * np.pi * i / num_points
#         x = int(x_center + radius * np.cos(angle))
#         y = int(y_center + radius * np.sin(angle))
#         points.append((x, y))
#     return points

# # Function to track points across frames and estimate rotation angle
# def track_rotation_angle(prev_frame, curr_frame, prev_points):
#     curr_points = []
#     for point in prev_points:
#         search_area = curr_frame[point[1]-5:point[1]+5, point[0]-5:point[0]+5]
#         result = cv2.matchTemplate(curr_frame, search_area, cv2.TM_CCOEFF_NORMED)
#         _, _, _, max_loc = cv2.minMaxLoc(result)
#         curr_points.append((max_loc[0] + point[0] - 5, max_loc[1] + point[1] - 5))
#     # Estimate rotation angle using the tracked points
#     angle = np.arctan2(curr_points[1][1] - curr_points[0][1], curr_points[1][0] - curr_points[0][0])
#     return np.degrees(angle)

# # Initial tracking points
# prev_points = []

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)

#             # Calculate additional points on the tire
#             tire_radius = int((x2 - x1) / 2)
#             num_points = 4  # You can adjust this number based on your requirement
#             tire_points = calculate_tire_points(center_x, center_y, tire_radius, num_points)
            
#             # Draw center point and additional points
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
#             for point in tire_points:
#                 cv2.circle(frame, point, 3, (255, 0, 0), -1)  # Additional points
            
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
#             # Track rotation angle
#             if len(prev_points) == 2:
#                 rotation_angle = track_rotation_angle(prev_frame, frame, prev_points)
#                 print("Rotation angle:", rotation_angle)
            
#             # Update previous frame and points
#             prev_frame = frame.copy()
#             prev_points = [tire_points[0], tire_points[2]]  # Select two points for rotation tracking

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# --------------------

# import os
# from ultralytics import YOLO
# import cv2
# import math

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# common_point = None  # Initialize the common point

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             radius = max((x2 - x1) / 2, (y2 - y1) / 2)

#             # Calculate two additional points on the circumference of the tyre
#             point1_x = int(center_x + radius * math.cos(0))
#             point1_y = int(center_y + radius * math.sin(0))

#             # Assign the first detected point as the common point
#             if common_point is None:
#                 common_point = (point1_x, point1_y)

#             # Calculate angle between the two additional points and the common point
#             angle_rad = math.atan2(point1_y - common_point[1], point1_x - common_point[0])
#             angle_deg = math.degrees(angle_rad)
#             if angle_deg < 0:
#                 angle_deg += 360

#             # Rotate the second point around the common point
#             rotate_angle_rad = math.radians(angle_deg)
#             rotate_distance = 50  # Adjust the distance between the common point and the rotating point
#             point2_x = int(common_point[0] + rotate_distance * math.cos(rotate_angle_rad))
#             point2_y = int(common_point[1] + rotate_distance * math.sin(rotate_angle_rad))

#             # Draw center point, common point, and rotating point
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
#             cv2.circle(frame, common_point, 5, (255, 0, 0), -1)  # Common point
#             cv2.circle(frame, (point2_x, point2_y), 5, (255, 0, 0), -1)  # Rotating point

#             # Draw lines between common point and rotating point
#             cv2.line(frame,(center_x, center_y),common_point,(255,0,0),2)
#             cv2.line(frame, (center_x, center_y), (point2_x, point2_y), (255, 0, 0), 2)


#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#             cv2.putText(frame, f"Angle: {angle_deg:.2f} degrees", (int(x1), int(y2 + 30)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# ----------

# import os
# from ultralytics import YOLO
# import cv2
# import math

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# common_point = None  # Initialize the common point

# # Radius of the tire in meters
# tire_radius = 0.3  # Assuming the tire radius is 0.3 meters

# # Speed of rotation in km/hr
# rotation_speed_km_hr = 10

# # Convert rotation speed to radians per second
# rotation_speed_rad_sec = (rotation_speed_km_hr * 1000) / (3600 * tire_radius)

# # Time elapsed between frames (in seconds)
# frame_rate = cap.get(cv2.CAP_PROP_FPS)
# time_elapsed_per_frame = 1 / frame_rate

# # Initialize angle of rotation
# angle_rad = 0

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)

#             # Assign the first detected point as the common point
#             if common_point is None:
#                 common_point = (center_x, center_y)

#             # Rotate the second point around the common point
#             rotate_angle_deg = math.degrees(angle_rad)
#             rotate_distance = 50  # Adjust the distance between the common point and the rotating point
#             point2_x = int(common_point[0] + rotate_distance * math.cos(math.radians(rotate_angle_deg)))
#             point2_y = int(common_point[1] + rotate_distance * math.sin(math.radians(rotate_angle_deg)))

#             # Draw center point, common point, and rotating point
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
#             cv2.circle(frame, common_point, 5, (255, 0, 0), -1)  # Common point
#             cv2.circle(frame, (point2_x, point2_y), 5, (255, 0, 0), -1)  # Rotating point

#             # Draw lines between common point and rotating point
#             cv2.line(frame, common_point, (point2_x, point2_y), (255, 0, 0), 2)

#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#             cv2.putText(frame, f"Angle: {rotate_angle_deg:.2f} degrees", (int(x1), int(y2 + 30)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('Object Detection', frame)

#     # Predict the angle of rotation
#     angle_rad += rotation_speed_rad_sec * time_elapsed_per_frame

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# ----------------------------------

# import os
# from ultralytics import YOLO
# import cv2
# import math

# VIDEOS_DIR = os.path.join('.', 'Task_5', 'videos')
# video_path = os.path.join(VIDEOS_DIR, 'tyre.mp4')

# cap = cv2.VideoCapture(video_path)

# model_path = os.path.join('.', 'Task_5', 'runs', 'detect', 'train', 'weights', 'last.pt')
# model = YOLO(model_path)
# threshold = 0.5

# common_point = None  # Initialize the common point

# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             center_x = int((x1 + x2) / 2)
#             center_y = int((y1 + y2) / 2)
#             radius = max((x2 - x1) / 2, (y2 - y1) / 2)

#             # Calculate two additional points on the circumference of the tyre
#             point1_x = int(center_x + radius * math.cos(0))
#             point1_y = int(center_y + radius * math.sin(0))

#             # Assign the first detected point as the common point
#             if common_point is None:
#                 common_point = (point1_x, point1_y)

#             # Calculate angle between the two additional points and the common point
#             angle_rad = math.atan2(point1_y - common_point[1], point1_x - common_point[0])
#             angle_deg = math.degrees(angle_rad)
#             if angle_deg < 0:
#                 angle_deg += 360

#             # Rotate the second point around the common point
#             rotate_angle_rad = math.radians(angle_deg)
#             rotate_distance = 50  # Adjust the distance between the common point and the rotating point
#             point2_x = int(common_point[0] + rotate_distance * math.cos(rotate_angle_rad))
#             point2_y = int(common_point[1] + rotate_distance * math.sin(rotate_angle_rad))

#             # Draw center point, common point, and rotating point
#             cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # Center point
#             cv2.circle(frame, common_point, 5, (255, 0, 0), -1)  # Common point
#             cv2.circle(frame, (point2_x, point2_y), 5, (255, 0, 0), -1)  # Rotating point

#             # Draw lines between common point and rotating point
#             cv2.line(frame,(center_x, center_y),common_point,(255,0,0),2)
#             cv2.line(frame, (center_x, center_y), (point2_x, point2_y), (255, 0, 0), 2)
            
            

#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#             cv2.putText(frame, f"Angle: {angle_deg:.2f} degrees", (int(x1), int(y2 + 30)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     cv2.imshow('Object Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

