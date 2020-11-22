import numpy as np
import cv2


class Kmean():

    def k_mean(imagem, K:int,  K_iter=2, criteria_iter=5, criteria_eps=1.0):
        # imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)

        criteria = (cv2.TERM_CRITERIA_MAX_ITER, criteria_iter, criteria_eps)

        Z = imagem.reshape((-1, 3))
        Z = np.float32(Z)

        _, label, center = cv2.kmeans(Z, K, None, criteria, K_iter, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        img_kmean = res.reshape((imagem.shape))

        color_list = Kmean.get_color_value(label, center)

        return img_kmean, color_list

    def get_color_value(Kmean_label, Kmean_center):
        unique_label = len(np.unique(Kmean_label))

        num_labels = np.arange(0, unique_label + 1)
        hist_label, _ = np.histogram(Kmean_label, bins=num_labels)
        hist_label = hist_label.astype("float")
        hist_label /= hist_label.sum()

        centroid_values = {}
        for centroid_creator in range(0, (len(num_labels) - 1)):
            centroid_values[centroid_creator] = Kmean_center[centroid_creator]

        color_list = {}
        for color_creator in range(0, (len(num_labels) - 1)):
            color_list[hist_label[color_creator]] = centroid_values[color_creator]

        color_list_sort = sorted(color_list.items(), reverse=True)

        return color_list_sort

class Pixel():

    def pixeleted(img, size:int):
        height, width = img.shape[:2]

        w, h = (size, size)

        temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        return output