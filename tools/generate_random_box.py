from numpy import random
import numpy as np


def random_rpn(gts, need_same=False, number=100, scale_random=[0.8, 1.8], offset=[0.03, 0.03]):
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    num_ = number
    if need_same:
        gts=list(gts)
        num_ //= len(gts)
    for index, gt in enumerate(gts):
        if need_same and index == (len(gts) - 1):
            num_ = number - num_ * (len(gts) - 1)
        for i in range(num_):
            x_ = random.uniform(scale_random[0], scale_random[1])
            y_ = random.uniform(scale_random[0], scale_random[1])
            y_add = (gt[2] - gt[0]) * y_ / 2
            x_add = (gt[3] - gt[1]) * x_ / 2

            y_offset_random = (gt[2] - gt[0]) * (y_ - 1) / 2
            x_offset_random = (gt[3] - gt[1]) * (x_ - 1) / 2

            y_offset_random += offset[1] if y_offset_random > 0 else -1 * offset[1]
            x_offset_random += offset[0] if x_offset_random > 0 else -1 * offset[0]

            offset_y = random.uniform(-y_offset_random, y_offset_random)
            offset_x = random.uniform(-x_offset_random, x_offset_random)

            center_y = (gt[2] - gt[0]) / 2 + gt[0] + offset_y
            center_x = (gt[3] - gt[1]) / 2 + gt[1] + offset_x
            x_min.append(max(center_x - x_add, 0))
            y_min.append(max(center_y - y_add, 0))
            x_max.append(min(center_x + x_add, 1))
            y_max.append(min(center_y + y_add, 1))
    return [y_min, x_min, y_max, x_max]


def random_test(gts, scale_random=[0.8, 1.8], offset=[0.03, 0.03]):
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    scales_ = np.array(
        [[scale_random[0], scale_random[0]], [scale_random[0], scale_random[1]],
         [scale_random[1], scale_random[0]], [scale_random[1], scale_random[1]]])
    for gt in gts:
        for scale_ in scales_:
            x_ = scale_[0]
            y_ = scale_[1]
            y_add = (gt[2] - gt[0]) * y_ / 2
            x_add = (gt[3] - gt[1]) * x_ / 2

            y_offset_random = (gt[2] - gt[0]) * (y_ - 1) / 2
            x_offset_random = (gt[3] - gt[1]) * (x_ - 1) / 2
            offset_ys = np.array([1, 1, 1, -1, -1, -1, 0, 0, 0]) * y_offset_random
            offset_xs = np.array([1, -1, 0, 1, -1, 0, 1, -1, 0]) * x_offset_random
            for offset_y, offset_x in zip(offset_ys, offset_xs):
                offset_x += offset[0] if offset_x > 0 else -1 * offset[0]
                offset_y += offset[1] if offset_y > 0 else -1 * offset[1]

                center_y = (gt[2] - gt[0]) / 2 + gt[0] + offset_y
                center_x = (gt[3] - gt[1]) / 2 + gt[1] + offset_x
                x_min.append(max(center_x - x_add, 0))
                y_min.append(max(center_y - y_add, 0))
                x_max.append(min(center_x + x_add, 1))
                y_max.append(min(center_y + y_add, 1))
    return [y_min, x_min, y_max, x_max]


if __name__ == '__main__':
    rpn = random_rpn(zip([16 / 153], [27 / 334], [91 / 153], [82 / 334]))
    rpn = [[ymin * 469, xmin * 1024, ymax * 469, xmax * 1024] for (ymin, xmin, ymax, xmax) in zip(*rpn)]
    rpn = np.array(rpn)
    rpn = rpn.reshape([1, -1, 4])
    print(rpn)
