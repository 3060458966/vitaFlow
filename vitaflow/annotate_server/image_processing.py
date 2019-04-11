import cv2


def show_img(data):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(data, 'gray')


def get_threshold_image(image):
    image = (image * 1.0) / image.max()
    THRESHOLD_LIMIT = 0.40
    image[image <= THRESHOLD_LIMIT] = 0.0
    image[image > THRESHOLD_LIMIT] = 1.0
    return image


def get_threshold_image2(image):
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                  cv2.THRESH_BINARY, 11, 2)
    image = (image * 1.0) / image.max()
    THRESHOLD_LIMIT = 0.40
    image[image <= THRESHOLD_LIMIT] = 0.0
    image[image > THRESHOLD_LIMIT] = 1.0
    return image


def get_threshold_image3(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


_avail_thershold_fns = {
    'binary': get_threshold_image,
    'adaptgaussian': get_threshold_image2,
    'gaussian_otsu': get_threshold_image3
}
_selected_threshold_fns = 'binary'


def get_line_segments(image):
    # threshold
    image = _avail_thershold_fns[_selected_threshold_fns](image)
    plt_data = image.min(axis=1)
    # plt.figure(figsize=(15, 2))
    # plt.plot(range(len(plt_data)), plt_data, '*')
    plt_data_index = np.arange(len(plt_data))
    data = plt_data_index[plt_data == 0]
    i = 0
    start = i
    memory = data[i]

    line_segments = []

    while i < len(data) - 1:
        i += 1
        if data[i] == memory + 1:
            memory += 1
        else:
            line_segments.append(
                (data[start], data[i])
            )
            # print(data[start], data[i])
            start = i
            memory = data[i]
    line_segments.append((data[start], data[i]))
    return line_segments
