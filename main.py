import time

import cv2
import numpy as np
import random
import math
import time

WIDTH = 640
HEIGHT = 480


def init_random():
    np.random.seed(int(time.time()))


def random_number():
    return np.random.uniform(0, 1)


def random_noise():
    return np.random.normal(0, 10)


def generate_data(N, p, dir, inlier_ratio):
    points = []
    N_inlier = int(N * inlier_ratio)
    N_outlier = N - N_inlier

    for _ in range(N_inlier):
        diag = math.sqrt(WIDTH ** 2 + HEIGHT ** 2)
        t = random_number() * diag - diag / 2
        q = p + t * dir
        q[0] += random_noise()
        q[1] += random_noise()
        points.append(q)

    for _ in range(N_outlier):
        q = np.array([random_number() * WIDTH, random_number() * HEIGHT])
        points.append(q)

    return np.array(points)


def fit_line(points):
    X = np.column_stack((points, np.ones(len(points))))
    XtX = X.T @ X
    evals, evecs = np.linalg.eig(XtX)
    l = evecs[:, np.argmin(evals)]
    l /= np.linalg.norm(l[:2])
    return l


def fit_line_two_points(p1, p2):
    dir = p2 - p1
    dir /= np.linalg.norm(dir)
    l = np.array([-dir[1], dir[0], -(-dir[1] * p1[0] - dir[0] * p1[1])])
    return l


def distance(l, p):
    return abs(l[0] * p[0] + l[1] * p[1] + l[2])


def iteration_number(confidence, sample_size, inliers, N):
    a = math.log(1.0 - confidence)
    ratio = inliers / N
    b = math.log(1.0 - ratio ** sample_size)
    return math.ceil(a / b)


def ransac(points, max_iterations, sigma):
    best_inliers_nr = 0
    best_model = None
    best_inliers = None

    for _ in range(max_iterations):
        idx1, idx2 = random.sample(range(len(points)), 2)
        current_model = fit_line_two_points(points[idx1], points[idx2])

        current_inliers = [p for p in points if distance(current_model, p) < sigma]
        inliers_nr = len(current_inliers)

        if inliers_nr > best_inliers_nr:
            best_inliers_nr = inliers_nr
            best_model = current_model
            best_inliers = current_inliers
            max_iterations = min(max_iterations, iteration_number(0.95, 2, best_inliers_nr, len(points)))

    best_model = fit_line(best_inliers)
    return best_model


def main():
    init_random()
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    p = np.array([HEIGHT / 2, WIDTH / 2])
    alpha = random_number() * 2 * math.pi
    dir = np.array([math.cos(alpha), math.sin(alpha)])
    points = generate_data(1000, p, dir, 0.5)

    for point in points:
        cv2.circle(img, tuple(point.astype(int)), 2, (255, 255, 255), -1)
    cv2.imshow("points", img)
    p1 = p - 1000 * dir
    p2 = p + 1000 * dir

    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 0, 0), 2)
    
    l = ransac(points, 2000, 10.0)
    p1 = np.array([0, -l[2] / l[1]])
    p2 = np.array([WIDTH, (-l[2] - l[0] * WIDTH) / l[1]])
    # l = fit_line(points)

    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 2)

    cv2.imshow("RANSAC", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
