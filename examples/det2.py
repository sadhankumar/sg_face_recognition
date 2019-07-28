import face_recognition
import cv2
import numpy as np
import urllib
mypic_image = face_recognition.load_image_file("mypic.jpeg")
mypic_face_encoding = face_recognition.face_encodings(mypic_image)[0]
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
sneha_image = face_recognition.load_image_file("sneha.jpg")
sneha_face_encoding = face_recognition.face_encodings(sneha_image)[0]
Dinesh_image = face_recognition.load_image_file("Dinesh.jpeg")
Dinesh_face_encoding = face_recognition.face_encodings(Dinesh_image)[0]
shika_image = face_recognition.load_image_file("shika.jpeg")
shika_face_encoding = face_recognition.face_encodings(shika_image)[0]

known_face_encodings = [
    mypic_face_encoding,
    biden_face_encoding,sneha_face_encoding,Dinesh_face_encoding,shika_face_encoding
]
known_face_names = [
    "sadhan ",
    "Joe Biden","sneha","Dinesh","shika"
]

url='http://10.91.155.104:8080/shot.jpg'
while True:
    # Grab a single frame of video
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgNp,-1)
    ret= frame 

 
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

     
    # Display the resulting image
        cv2.imshow('Frame', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        exit(0)
cv2.destroyAllWindows()
