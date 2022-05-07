import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from random import random


class Target:
    def __init__(self, image):
        self.radius = random() * 50 + 50
        self.x = random() * image.shape[1]
        self.y = image.shape[0] + self.radius
        self.xVel = random() * 12 - 6
        self.yVel = random() * -30 - 20
        # self.x = random() * image.shape[1]
        # self.y = random() * image.shape[0]

        self.color = (random() * 255, random() * 255, random() * 255)

        value = random()
        if value < 0.4:
            # self.required_landmark = 19
            self.color = "red"
        elif value < 0.8:
            # self.required_landmark = 20
            self.color = "blue"
        elif value < 0.9:
            self.color = "darkred"
        else:
            self.color = "darkblue"

    def draw(self, image):
        color = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "darkred": (0, 0, 128),
            "darkblue": (128, 0, 0)
        }[self.color]

        cv2.circle(image, (int(self.x), int(self.y)),
                   int(self.radius), color, -1)

    def move(self):
        self.x += self.xVel
        self.y += self.yVel
        self.yVel += 1

    def is_off_screen(self, image):
        return self.y > image.shape[0] + self.radius

    def point_is_within_me(self, x, y, padding_radius=0):
        # Use the distance formula to determine if the point is within the circle
        return (x - self.x) ** 2 + (y - self.y) ** 2 < (self.radius + padding_radius) ** 2


model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)),
                     (int(x2), int(y2)), (0, 0, 255), 4)


def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)


targets = []
redScore = 0
blueScore = 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Resize image
    # input_img = image.copy()
    input_img = tf.image.resize_with_pad(
        tf.expand_dims(image, axis=0), 384, 640)
    input_img = tf.cast(input_img, dtype=tf.int32)

    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[
        :, :, :51].reshape((6, 17, 3))

    keypoint_threshold = 0.1

    # Render keypoints
    loop_through_people(image, keypoints_with_scores,
                        EDGES, keypoint_threshold)

    handpoints_with_scores = keypoints_with_scores[:, 9:11, :].reshape((12, 3))
    for handpoint in handpoints_with_scores:
        if handpoint[2] > keypoint_threshold:
            cv2.circle(image, (int(handpoint[1] * image.shape[1]), int(
                handpoint[0] * image.shape[0])), 12, (255, 0, 0), -1)

    # Process each target
    for target in targets:
        target.move()
        target.draw(image)
        is_hit = False
        for handpoint in handpoints_with_scores:
            if handpoint[2] > keypoint_threshold:
                if target.point_is_within_me(handpoint[1] * image.shape[1], handpoint[0] * image.shape[0], 12):
                    is_hit = True
        if is_hit:
            targets.remove(target)
            if target.color == "red":
                redScore += 1
            elif target.color == "blue":
                blueScore += 1
            elif target.color == "darkred":
                redScore /= 2
            else:
                blueScore /= 2
        elif target.is_off_screen(image):
            targets.remove(target)

    # Create new targets if needed
    while len(targets) < 5:
        targets.append(Target(image))

    # Write the score on the image
    # cv2.putText(image, "Score: {}".format(score), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if redScore + blueScore == 0:
        redPercent = 0
        bluePercent = 0
    else:
        redPercent = redScore / (redScore + blueScore)
        bluePercent = blueScore / (redScore + blueScore)
    cv2.rectangle(
        image, (0, 0), (int(image.shape[1] * redPercent), 40), (255, 0, 0), -1)
    cv2.rectangle(
        image, (int(image.shape[1] * redPercent), 0), (int(image.shape[1] * redPercent + image.shape[1] * bluePercent), 40), (0, 0, 255), -1)

    # Convert image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imshow("Game", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press "q" to quit.
        break

cap.release()
