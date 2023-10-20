import cv2
import time
from geopy.geocoders import Nominatim
from datetime import datetime
import requests

# Initialize the geolocator
geolocator = Nominatim(user_agent="eye_blink_detection")

# Initialize face and eye cascade xml of OpenCV library to detect face and eyes
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

# Function to get the public IP address and location
def get_location():
    response = requests.get("https://ipinfo.io")
    data = response.json()
    loc = data.get("loc").split(",")
    return loc

latitude, longitude = get_location()

first_read = True
blink_detected = False
blink_counter = 0

# Video Capturing by opening the webcam
cap = cv2.VideoCapture(0)

# To check for the first instance of capturing, it will return True and an image
ret, image = cap.read()

while ret:
    # This will keep the webcam running and capturing the image for every loop
    ret, image = cap.read()

    # Convert the recorded image to grayscale
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying filters to remove impurities
    gray_scale = cv2.bilateralFilter(gray_scale, 5, 1, 1)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_scale, 1.3, 5, minSize=(200, 200))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Eye_face var will be input to the eye classifier
            eye_face = gray_scale[y:y + h, x:x + w]

            # Get the eyes
            eyes = eyes_cascade.detectMultiScale(eye_face, 1.3, 5, minSize=(50, 50))

            if len(eyes) >= 2:
                if first_read:
                    cv2.putText(image, "Eyes detected, press s to check blink", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                else:
                    cv2.putText(image, "Eyes Open", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2)
                    blink_detected = False
            else:
                if first_read:
                    cv2.putText(image, "No Eyes detected", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 255, 255), 2)
                else:
                    cv2.putText(image, "Blink Detected.....!!!!", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2)
                    cv2.imshow('image', image)
                    cv2.waitKey(1)
                    print("Blink Detected.....!!!!")
                    blink_detected = True

                 # Capture and save the user's photo after a 1-second delay
                if blink_counter < 1 and blink_detected:
                    time.sleep(1)
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    location = geolocator.reverse(f"{latitude}, {longitude}")
                    
                    # Decrease the font size for time and location text
                    font_scale = 0.5  # Adjust the font size here
                    
                    cv2.putText(image, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)  # Decreased font size
                    cv2.putText(image, f"Location: {location}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 1)  # Decreased font size
                    
                    cv2.imwrite("user_photo.jpg", image)
                    print("User's photo captured and saved as user_photo.jpg")
                    blink_counter += 1
    else:
        cv2.putText(image, "No Face Detected.", (70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow('image', image)
    a = cv2.waitKey(1)

    # Press q to quit and s to start
    if a == ord('q'):
        break
    elif a == ord('s'):
        first_read = False

# Release the webcam
cap.release()
# Close the window
cv2.destroyAllWindows()
