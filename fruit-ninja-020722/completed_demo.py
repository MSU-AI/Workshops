import cv2
import mediapipe as mp

from random import random

class Target:
    def __init__(self, image):
        # Give this target a randomized x, y, and radius
        self.radius = random() * 50 + 50
        self.x = random() * image.shape[1]
        self.y = random() * image.shape[0]

    def draw(self, image):
        # Draw this target on the screen
        cv2.circle(image, (int(self.x), int(self.y)),
                   int(self.radius), (255, 0, 0), -1)

    def point_is_within_me(self, x, y):
        # Use the distance formula to determine if the point is within the circle
        return (x - self.x) ** 2 + (y - self.y) ** 2 < self.radius ** 2

    def hit_by_points(self, points):
        for point in points:
            if self.point_is_within_me(point[0], point[1]):
                return True
        return False


def toPixelCoordinates(x, y, image):
  
  new_x = x * image.shape[1]
  new_y = y * image.shape[0]
  
  new_x = int(new_x)
  new_y = int(new_y)
  
  return (new_x, new_y)

# Try playing around with the index for the cv2.VideoCapture class,
# As sometimes the camera at index 0 is not valid.

cap = cv2.VideoCapture(0)

targets = []

with mp.solutions.pose.Pose() as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image from BGR to RGB format:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flip the image horizontally:
        # (Without this, the game is very confusing to play.)
        image = cv2.flip(image, 1)

        # Compute the pose results:
        # (Note that the `pose` variable already exists.)
        results = pose.process(image)

        # Convert the image back to BGR format:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        while len(targets) < 5:
            targets.append(Target(image))
            
        for target in targets:
            target.draw(image)

        # Draw the pose annotation on the image.
        mp.solutions.drawing_utils.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS
        )

        if results.pose_landmarks is None:
            
            # No poses detected:
            
            print("No detected poses!")
            
            continue

        # Highlight the wrists
        # Left wrist
        left_wrist = results.pose_landmarks.landmark[15]
        left_point = toPixelCoordinates(left_wrist.x, left_wrist.y, image)
        cv2.circle(image, center=left_point, radius=25, color=(0, 0, 255), thickness=-1)

        # Right wrist
        right_wrist = results.pose_landmarks.landmark[16]
        right_point = toPixelCoordinates(right_wrist.x, right_wrist.y, image)
        cv2.circle(image, center=right_point, radius=25, color=(0,0,255), thickness=-1)

        for target in targets:
            if target.hit_by_points([left_point, right_point]):
                targets.remove(target)

        # Display the image:
        cv2.imshow("Game", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press "q" to quit.
            break

cap.release()
