import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_face_mesh.FaceMesh(refine_landmarks = True).process(frame)
    #ls_single_face=results.multi_face_landmarks[0].landmark
    #for idx in ls_single_face:
        #print(idx.x,idx.y,idx.z, idx.vi)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image = frame,
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec = mp_drawing.DrawingSpec(color = (0,255,0)), 
                connection_drawing_spec = mp_drawing_style.get_default_face_mesh_iris_connections_style()     
            )
        
    cv2.imshow('Face Mesh', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.waitKey(0)

cv2.destroyAllWindows()