import numpy as np


def img_norm(img):
    """
    Normalize an input image.

    Parameters:
    img (numpy.ndarray): The input image to be normalized.

    Returns:
    numpy.ndarray: The normalized image.
    """
    if len(img.shape) == 2:
        channel = (img[:, :, np.newaxis] - 0.485) / 0.229
        img = np.concatenate([channel, channel, channel], axis=2)
    else:
        img = (
            img - np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 3))
        ) / np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 3))
    return img


def custom_print(context, log_file, mode):
    """
    Custom print and log function.

    Parameters:
    - context (str): The text to be printed and logged.
    - log_file (str): The path to the log file where the text will be logged.
    - mode (str): The mode of file operation ('w' for write, 'a+' for append and read).

    Raises:
    - Exception: If an unsupported file operation mode is provided.

    Usage:
    - This function can be used to print text to the console and log it to a file simultaneously.

    Example:
    custom_print("Hello, world!", "log.txt", "a+")
    """
    if mode == "w":
        fp = open(log_file, mode)
        fp.write(context + "\n")
        fp.close()
    elif mode == "a+":
        print(context)
        fp = open(log_file, mode)
        print(context, file=fp)
        fp.close()
    else:
        raise Exception(
            'Unsupported file operation mode! Use "w" for write or "a+" for append and read.'
        )


import numpy as np


def generate_binary_map(pred, type, th=0.5):
    """
    Generate a binary map based on a prediction array.

    Parameters:
    - pred (numpy.ndarray): The prediction array to be thresholded.
    - type (str): The type of thresholding to use ('2mean' or 'mean+std').
    - th (float): The maximum threshold value (default is 0.5).

    Returns:
    - numpy.ndarray: The binary map generated based on the specified type and threshold.

    Raises:
    - ValueError: If an unsupported type is provided.

    Usage:
    - This function takes a prediction array and generates a binary map by applying a threshold.
    - '2mean' type uses twice the mean value of the prediction array as the threshold.
    - 'mean+std' type uses the mean plus the standard deviation of the prediction array as the threshold.

    Example:
    binary_map = generate_binary_map(prediction_array, '2mean', th=0.6)
    """
    if type == "2mean":
        threshold = np.mean(pred) * 2
        if threshold > th:
            threshold = th
        binary_map = pred > threshold
        return binary_map.astype(np.float32)

    elif type == "mean+std":
        threshold = np.mean(pred) + np.std(pred)
        if threshold > th:
            threshold = th
        binary_map = pred > threshold
        return binary_map.astype(np.float32)

    else:
        raise ValueError(
            'Unsupported type! Use "2mean" or "mean+std" for thresholding.'
        )


def calc_precision_and_jaccard(pred, gt, th=0.5):
    """
    Calculate precision and Jaccard similarity between a predicted binary map and a ground truth binary map.

    Parameters:
    - pred (numpy.ndarray): The predicted binary map.
    - gt (numpy.ndarray): The ground truth binary map.
    - th (float): The threshold used for generating the binary maps (default is 0.5).

    Returns:
    - float: Precision value.
    - float: Jaccard similarity value.

    Usage:
    - This function calculates the precision and Jaccard similarity between a predicted binary map and a ground truth binary map.
    - Precision is the ratio of true positives to the total number of pixels.
    - Jaccard similarity is a measure of the intersection over the union between the two binary maps.

    Example:
    precision, jaccard = calc_precision_and_jaccard(predicted_map, ground_truth_map, th=0.6)
    """
    bin_pred = generate_binary_map(pred, "mean+std", th)
    tp = (bin_pred == gt).sum()
    precision = tp / pred.size

    i = (bin_pred * gt).sum()
    u = bin_pred.sum() + gt.sum() - i
    jaccard = i / (u + 1e-10)

    return precision, jaccard


def calc_dice_and_iou(pred, gt, th=0.5):
    """
    Calculate Dice coefficient and Intersection over Union (IoU) between a predicted binary map and a ground truth binary map.

    Parameters:
    - pred (numpy.ndarray): The predicted binary map.
    - gt (numpy.ndarray): The ground truth binary map.
    - th (float): The threshold used for generating the binary maps (default is 0.5).

    Returns:
    - float: Dice coefficient value.
    - float: IoU (Intersection over Union) value.

    Usage:
    - This function calculates the Dice coefficient and IoU (Intersection over Union) between a predicted binary map and a ground truth binary map.
    - Dice coefficient is a measure of overlap between the two binary maps.
    - IoU (Intersection over Union) is a measure of the intersection divided by the union of the two binary maps.

    Example:
    dice, iou = calc_dice_and_iou(predicted_map, ground_truth_map, th=0.6)
    """
    bin_pred = generate_binary_map(pred, "mean+std", th)

    i = np.sum(bin_pred * gt)
    u = np.sum(bin_pred) + np.sum(gt) - i

    dice = (2.0 * i) / (np.sum(bin_pred) + np.sum(gt) + 1e-10)
    iou = i / (u + 1e-10)

    return dice, iou
