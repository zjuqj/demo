import numpy as np
import cv2
import keras
NORM_FAC = 200
def get_distance(pt1, pt2):
    assert len(pt1) == 2 and len(pt2) == 2, "Assert pt1 and pt2 to be point"
    return np.math.sqrt(np.power(pt1[0]-pt2[0], 2) + np.power(pt1[1]-pt2[1], 2))

def get_eye_aspect_ratio(pt1, pt2,pt3,pt4,pt5,pt6):
    """
    EAR
    pt1---> landmark point 37
    pt2---> landmark point 38
    pt3---> landmark point 39
    pt4---> landmark point 40
    pt5---> landmark point 41
    pt6---> landmark point 42
    """
    return (get_distance(pt2, pt6)+get_distance(pt3, pt5))/(2 * get_distance(pt1, pt4))

def get_mouth_aspect_ratio_over_eye(pt1,pt2,pt3,pt4):
    """
    MAR
    pt1---> landmark point 52
    pt2---> landmark point 58
    pt3---> landmark point 49
    pt4---> landmark point 55
    """
    return get_distance(pt1, pt2) / get_distance(pt3, pt4)

def get_puc(pt1, pt2,pt3,pt4,pt5,pt6):
    """
    PUC
    pt1---> landmark point 37
    pt2---> landmark point 38
    pt3---> landmark point 39
    pt4---> landmark point 40
    pt5---> landmark point 41
    pt6---> landmark point 42
    same map for landmark point as func `get_eye_aspect_ratio`
    """
    area = np.power(get_distance(pt2, pt5)/2, 2) * np.math.pi
    perimeter = get_distance(pt1, pt2) + get_distance(pt2, pt3) +get_distance(pt3, pt4) +get_distance(pt4, pt5) + get_distance(pt5, pt6) + get_distance(pt6, pt1)
    circularity = 4*np.math.pi*area/np.power(perimeter, 2)
    return circularity

def get_moe(mar, ear):
    """
    MOE
    """
    return mar/ear

def extract_facial_features(ldmk_pts):
    feat_eye_aspect = get_eye_aspect_ratio(ldmk_pts[36], ldmk_pts[37], ldmk_pts[38], ldmk_pts[39], ldmk_pts[40], ldmk_pts[41])
    feat_mouth_aspect = get_mouth_aspect_ratio_over_eye(ldmk_pts[51], ldmk_pts[57], ldmk_pts[48], ldmk_pts[54])
    feat_puc = get_puc(ldmk_pts[36], ldmk_pts[37], ldmk_pts[38], ldmk_pts[39], ldmk_pts[40], ldmk_pts[41])
    feat_moe = get_moe(feat_mouth_aspect, feat_eye_aspect)
    return feat_eye_aspect, feat_mouth_aspect, feat_puc, feat_moe

def judge(ldm, model):
    # model = keras.models.load_model("fatigue_detect_model.h5")
    res = False
    rets = extract_facial_features(
            ldm
        )

    if rets[0]:
        rets = list(rets)
        rets[0] = int(rets[0] * 150) / NORM_FAC
        rets[1] = int(rets[1] * 100) / NORM_FAC
        rets[2] = int(rets[2] * 50) / NORM_FAC
        rets[3] = int(rets[3] * 25) / NORM_FAC
        # rets[0] = int(rets[0])
        # rets[1] = int(rets[1])
        # rets[2] = int(rets[2])
        # rets[3] = int(rets[3])
        pred_res = model.predict([[rets[0], rets[1], rets[2], rets[3]]])
        # print("pred_res={}".format(pred_res))
        # print(pred_res[0,1])
        # if pred_res[0,1] > 0.63:
        #     print("Warning")
        #     res = True
        if pred_res[0, 1] > pred_res[0, 0]:
            print("Warning")
            res = True
    # print(res)
    return res

# if rets[0]:
#     rets = list(rets)
#     rets[0] = int(rets[0] * 150) / NORM_FAC
#     rets[1] = int(rets[1] * 100) / NORM_FAC
#     rets[2] = int(rets[2] * 50) / NORM_FAC
#     rets[3] = int(rets[3] * 25) / NORM_FAC
#     pred_res = model.predict([[rets[0], rets[1], rets[2], rets[3]]])
#     print("pred_res={}".format(pred_res))
#     if pred_res[0, 1] > pred_res[0, 0]:
#         cv2.putText(rets[-1], "WARNING!!!!", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 5, cv2.LINE_AA)
            
#     cv2.imshow("", rets[-1])