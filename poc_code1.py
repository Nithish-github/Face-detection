import cv2
import binascii

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the image
image_path = 'images.jpg'  # Replace with your image path
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image.")
else:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Open a text file to write the hexadecimal data
    with open('face_hex_data.txt', 'w') as file:
        for i, (x, y, w, h) in enumerate(faces):
            # Crop the face
            face_crop = image[y:y + h, x:x + w]
            
            # Encode the cropped face to PNG format
            _, buffer = cv2.imencode('.png', face_crop)
            
            # Convert the binary data to a hexadecimal string
            face_hex_string = binascii.hexlify(buffer).decode('utf-8')
            
            # Write the hexadecimal string to the file
            file.write(f"Face {i + 1}: {face_hex_string}\n")
    
    print("Face hexadecimal data saved.")
