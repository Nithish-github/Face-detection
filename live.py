import cv2
import binascii

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the camera (change the index as per your camera setup, typically 0 or -1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Open a text file to write the hexadecimal data
    with open('live_face_hex_data.txt', 'w') as file:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for i, (x, y, w, h) in enumerate(faces):
                # Crop the face
                face_crop = frame[y:y + h, x:x + w]

                # Encode the cropped face to PNG format
                _, buffer = cv2.imencode('.png', face_crop)

                # Convert the binary data to a hexadecimal string
                face_hex_string = binascii.hexlify(buffer).decode('utf-8')

                # Write the hexadecimal string to the file
                file.write(f"Face {i + 1}: {face_hex_string}\n")

                # Display the cropped face
                cv2.imshow('Cropped Face', face_crop)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

print("Live face hexadecimal data saved.")
