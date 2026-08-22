import cv2
import numpy as np

def calculate_color_histogram(image):

    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    hist = cv2.calcHist(
        [image],
        [0, 1, 2],
        None,
        [8, 8, 8],
        [0, 256, 0, 256, 0, 256]
    )

    cv2.normalize(hist, hist)

    return hist.astype(np.float32).flatten()


def color_similarity_from_histograms(hist1, hist2):

    hist1 = np.asarray(hist1, dtype=np.float32)
    hist2 = np.asarray(hist2, dtype=np.float32)

    return float(
        1 - cv2.compareHist(
            hist1.reshape(-1, 1),
            hist2.reshape(-1, 1),
            cv2.HISTCMP_BHATTACHARYYA
        )
    )


def color_similarity(image1, image2):

    hist1 = calculate_color_histogram(image1)
    hist2 = calculate_color_histogram(image2)

    return color_similarity_from_histograms(
        hist1,
        hist2
    )