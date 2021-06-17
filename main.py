import cv2
import mtcnn
import dlib
import facial_feat
import keras

tool = mtcnn.FaceAlignmentTools()
predictor = dlib.shape_predictor(
    "shape_predictor_68_face_landmarks.dat"
)
cap = cv2.VideoCapture(0)
model = keras.models.load_model("fatigue_detect_model.h5")

while(True):
    ret , frame = cap.read()
    img = frame.copy()
    dets = tool.detect(img)
    if dets is None:
        continue
    for face in dets:
        # print(face)
        corpbbox = [int(face[0]), int(face[1]), int(face[2]), int(face[3])]
        rect = dlib.rectangle(corpbbox[0], corpbbox[1], corpbbox[2], corpbbox[3])
        shape = predictor(img, rect)
        land_pts = []
        i = 0
        for pt in shape.parts():
            i = i + 1
            land_pts.append([pt.x, pt.y])
            if i in range(37, 43) or i in range(43, 49) or i in range(49, 69):
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 1, (0, 255, 0), 0)
        rate = facial_feat.judge(land_pts, model)
        # print(rate)
        if flag:
            text = "Warning"
        else:
            text = "Nice"
        cv2.putText(img, text, (60, 60), cv2.FONT_HERSHEY_PLAIN, 5.0, (0, 0, 255), 2)
    cv2.imshow("image", img)
    if cv2.waitKey(1) &0xFF ==ord('q'):
    	break
cap.release()
cv2.destroyAllWindows()