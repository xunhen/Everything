from matplotlib import pyplot as plt


def draw_line():
    x = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    y = [0.599, 0.619, 0.621, 0.624, 0.622, 0.618, 0.609]
    plt.xlabel('scale')
    plt.ylabel('mAP')
    plt.plot(x, y)
    plt.draw()
    plt.show()


if __name__ == '__main__':
    draw_line()
    pass
