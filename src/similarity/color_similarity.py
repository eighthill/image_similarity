"""
import cv2

def color_similarity(
    image1,
    image2
):

    hist1 = cv2.calcHist(
        [image1],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    hist2 = cv2.calcHist(
        [image2],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    corr = cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)

    return (corr + 1) / 2
"""

import cv2

def color_similarity(image1, image2):

    hist1 = cv2.calcHist(
        [image1],
        [0,1,2],
        None,
        [8,8,8],
        [0,256,0,256,0,256]
    )

    hist2 = cv2.calcHist(
        [image2],
        [0,1,2],
        None,
        [8,8,8],
        [0,256,0,256,0,256]
    )

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    distance = cv2.compareHist(
        hist1,
        hist2,
        cv2.HISTCMP_BHATTACHARYYA
    )

    return 1 - distance