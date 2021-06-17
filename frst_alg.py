# -*- coding:utf-8 -*-
# Fast radial symmetry transform using OpenCV
# This is an implementation of the Fast radial symmetry transform using OpenCV.

# See details: Loy, G., & Zelinsky, A. (2002). A fast radial symmetry transform for detecting points of interest. Computer Vision, ECCV 2002.

# The code was ported from a MATLAB implementation and tested with OpenCV 3.0.0 (x64).
import numpy as np
import cv2

FRST_MODE_BRIGHT = 0
FRST_MODE_DARK = 1
FRST_MODE_BOTH = 2


def grady(input):
    output = np.zeros_like(input, dtype="float64")
    assert(len(input.shape) == 2), "Assert input to be a 2-dim numpy array!"
    h, w = input.shape
    for i in range(h):
        for j in range(1, w-1):
            output[i, j] = (float(input[i, j+1])-float(input[i, j-1]))/2

    return output


def gradx(input):
    output = np.zeros_like(input, dtype="float64")
    assert(len(input.shape) == 2), "Assert input to be a 2-dim numpy array!"
    h, w = input.shape
    for i in range(1, h-1):
        for j in range(w):
            output[i, j] = (float(input[i+1, j])-float(input[i-1, j]))/2

    return output


def frst2d(input_img, radii_gaussk, alpha, std_factor, mode):
    assert(len(input_img.shape) == 2), "Assert input_img to be 2-dim numpy array!"
    h, w = input_img.shape
    gx = gradx(input_img)
    gy = grady(input_img)
    dark, bright = False, False

    if mode == FRST_MODE_BRIGHT or mode == FRST_MODE_BOTH:
        bright = True
    if mode == FRST_MODE_DARK or mode == FRST_MODE_BOTH:
        dark = True

    output_img = np.zeros_like(input_img, dtype="float64")
    S = np.zeros((h+2*radii_gaussk, w+2*radii_gaussk), dtype="float64")
    O_n = np.zeros_like(S)
    M_n = np.zeros_like(S)

    for i in range(h):
        for j in range(w):
            gpt = (gy[i, j], gx[i, j])
            gnorm = np.math.sqrt(gpt[0] * gpt[0] + gpt[1] * gpt[1])
            if gnorm > 0:
                gvec = [0, 0]
                gvec[0] = np.round((gpt[0]/gnorm)*radii_gaussk)
                gvec[1] = np.round((gpt[1]/gnorm)*radii_gaussk)

                if bright:
                    ppvec = [int(i+gvec[0]+radii_gaussk),
                             int(j+gvec[1]+radii_gaussk)]
                    O_n[ppvec[0], ppvec[1]] = O_n[ppvec[0], ppvec[1]]+1
                    M_n[ppvec[0], ppvec[1]] = M_n[ppvec[0], ppvec[1]]+gnorm

                if dark:
                    pnve = [int(i-gvec[0]+radii_gaussk),
                            int(j-gvec[1]+radii_gaussk)]
                    O_n[pnve[0], pnve[1]] = O_n[pnve[0], pnve[1]]-1
                    M_n[pnve[0], pnve[1]] = M_n[pnve[0], pnve[1]]-gnorm

    O_n = np.abs(O_n)
    O_n = O_n/np.max(O_n)

    M_n = np.abs(M_n)
    M_n = M_n/np.max(M_n)

    S = np.power(O_n, alpha)
    S = np.matmul(S, M_n.T)

    k_size = int(radii_gaussk/2)
    if k_size % 2 == 0:
        k_size += 1

    S = cv2.GaussianBlur(S, (k_size, k_size), sigmaX=radii_gaussk*std_factor, sigmaY=radii_gaussk*std_factor)
    output_img = S[radii_gaussk: h, radii_gaussk: w]
    return output_img


def bw_morph(input_img, operation, mshape=cv2.MORPH_RECT, msize=3, iterations=1):
    m_size = msize if (msize % 2) else (msize+1)
    element = cv2.getStructuringElement(mshape, (m_size, m_size))
    output_img = np.zeros_like(input_img)
    cv2.morphologyEx(input_img, operation, element,
                     output_img, (-1, -1), iterations)
    return output_img

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print("-----------------")
        output_img = frst2d(frame, 12,2,0.1, FRST_MODE_DARK)
        cv2.normalize(output_img, output_img, 0, 1.0, cv2.NORM_MINMAX)
        print("normalized....")
        output_img = np.array(output_img, dtype='uint8')

        _, markers = cv2.threshold(output_img, 0, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
        bw_morph(output_img, cv2.MORPH_CLOSE, cv2.MORPH_ELLIPSE, 5)
        print("morph process....")
        contours, hierarchy = cv2.findContours(markers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("find contours....")
        mu=[]
        for i in range(len(contours)):
            mu.append(cv2.moments(contours[i], False))
        
        mc=[]
        for i in range(len(contours)):
            mc.append((int((mu[i]['m10']+0.0001)/ (mu[i]['m00']+0.0001)), int((mu[i]['m01']+0.0001)/(mu[i]['m00']+0.0001))))
        
        for i in range(len(contours)):
            cv2.circle(frame, mc[i], 2, (0, 255, 0), -1, 8, 0)
        frame = np.array(frame, dtype='uint8')
        cv2.imshow("", frame)
        if cv2.waitKey() == ord('q'):
            break

    cap.release()