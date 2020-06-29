import numpy as np
import os
import cv2
import dlib
import time

# enter name of the video file in which you want to swap the face
ip_video_file = "input_video.avi"

# enter name of the video file in which you want to get the processed face swapping
filename = 'output_video.avi'

# enter name of the image whose face is to be swapped in the video
ip_image_name = "input_image.jpg"
frames_per_second = 24.0
res = '720p'

face_flag = 0
loading_count = 0

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


img = cv2.imread(ip_image_name)

cap = cv2.VideoCapture(ip_video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (int(cap.get(3)), int(cap.get(4))))
totalframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

# ready deep learning library
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor2 = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

# Face in image
faces = detector(img_gray)
for face in faces:
    face_flag = 1
    landmarks = predictor(img_gray, face)
    landmarks2 = predictor2(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    for n in range(68, 81):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmarks_points.append((x, y))

        # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)

    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

# Face processing in video
if face_flag == 1:
    print("Processing.....\nplease wait....")
    #print(totalframes)
    while True:
        face2_flag = 0
        ret, img2 = cap.read()


        if ret:
            loading_count = loading_count + 1
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            img2_new_face = np.zeros_like(img2)
            faces2 = detector(img2_gray)
            for face in faces2:
                face2_flag = 1
                landmarks = predictor(img2_gray, face)
                landmarks2 = predictor2(img_gray, face)

                landmarks_points2 = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    landmarks_points2.append((x, y))

                for n in range(68, 81):
                    x = landmarks2.part(n).x
                    y = landmarks2.part(n).y
                    landmarks_points2.append((x, y))
                # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
                points2 = np.array(landmarks_points2, np.int32)
                convexhull2 = cv2.convexHull(points2)

            lines_space_mask = np.zeros_like(img_gray)
            lines_space_new_face = np.zeros_like(img2)

            # Triangulation of both faces
            if face2_flag == 1:
                for triangle_index in indexes_triangles:
                    # Triangulation of the first face
                    tr1_pt1 = landmarks_points[triangle_index[0]]
                    tr1_pt2 = landmarks_points[triangle_index[1]]
                    tr1_pt3 = landmarks_points[triangle_index[2]]
                    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

                    rect1 = cv2.boundingRect(triangle1)
                    (x, y, w, h) = rect1
                    cropped_triangle = img[y: y + h, x: x + w]
                    cropped_tr1_mask = np.zeros((h, w), np.uint8)

                    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                    # cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask=cropped_tr1_mask)

                    # cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
                    # cv2.line(img, tr1_pt3, tr1_pt2, (0, 0, 255), 2)
                    # cv2.line(img, tr1_pt1, tr1_pt3, (0, 0, 255), 2)

                    # Triangulation of second face
                    tr2_pt1 = landmarks_points2[triangle_index[0]]
                    tr2_pt2 = landmarks_points2[triangle_index[1]]
                    tr2_pt3 = landmarks_points2[triangle_index[2]]
                    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

                    rect2 = cv2.boundingRect(triangle2)
                    (x, y, w, h) = rect2
                    # cropped_triangle2 = img2[y: y + h, x: x + w]
                    cropped_tr2_mask = np.zeros((h, w), np.uint8)

                    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

                    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
                    # cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask=cropped_tr2_mask)

                    # cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
                    # cv2.line(img2, tr2_pt3, tr2_pt2, (0, 0, 255), 2)
                    # cv2.line(img2, tr2_pt1, tr2_pt3, (0, 0, 255), 2)

                    # Warp triangles
                    points = np.float32(points)
                    points2 = np.float32(points2)
                    M = cv2.getAffineTransform(points, points2)
                    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                    # Reconstructing destination face
                    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255,
                                                               cv2.THRESH_BINARY_INV)
                    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

                    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

            else:
                print("")

            if face2_flag == 1:
                # Face swapped (putting 1st face into 2nd face)
                img2_face_mask = np.zeros_like(img2_gray)
                img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
                img2_face_mask = cv2.bitwise_not(img2_head_mask)

                # img2_head_noface = np.zeros_like(img)

                img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
                result = cv2.add(img2_head_noface, img2_new_face)
                (x, y, w, h) = cv2.boundingRect(convexhull2)
                center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

                seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.MIXED_CLONE)

                seamlessclone = cv2.flip(seamlessclone, 180)
                out.write(seamlessclone)
                # cv2.imshow("result", result)
                # cv2.imshow("result", seamlessclone)
            else:
                # cv2.imshow("result", img2)
                print("")


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

        _ = os.system('cls')
        print("progress = ", (loading_count*100)//totalframes, "%")

    # cv2.imshow("Img", img)
    # cv2.imshow("img2", img2)
    # cv2.imshow("result", result)
else:
    print("no face found in image")

print("processing done!")
print("please check ",filename)
cap.release()
cv2.destroyAllWindows()
