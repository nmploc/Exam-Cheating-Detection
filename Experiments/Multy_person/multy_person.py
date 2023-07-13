import cv2
import mediapipe as mp
from getBB import get_bounding_boxes
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def createNN():
    class FFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(24, 120)
            self.fc1 = nn.Linear(120, 240)
            self.fc2 = nn.Linear(240,40)
            self.output = nn.Linear(40, 2)

        def forward(self, x):
            x = f.relu(self.input(x))
            x = f.relu(self.fc1(x))
            x = f.relu(self.fc2(x))
            return torch.log_softmax(self.output(x), axis = 1)
    net = FFN()
    lossfun = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001)

    return net, lossfun, optimizer


cap = cv2.VideoCapture('D:\\Exam-Cheating-Detection\\Experiments\\Multy_person\\istockphoto-1373352218-640_adpp_is.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break
    bounding_boxes = get_bounding_boxes(frame)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox
        # Crop frame using bounding box coordinates
        cropped_frame = frame[y1:y2,x1:x2]

        img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                if(landmark.visibility <= 0.2):
                    landmark.x = 100
                    landmark.y = 100
            nose = results.pose_landmarks.landmark[0]
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_eye = results.pose_landmarks.landmark[1]
            right_eye = results.pose_landmarks.landmark[2]
            left_index = results.pose_landmarks.landmark[24]
            right_index = results.pose_landmarks.landmark[25]

            Net, lossfun, optimizer = createNN()
            Net.load_state_dict(torch.load('model1.pt', map_location=device))
            Net.to(device)
            input = torch.tensor([[nose.x, nose.y, left_shoulder.x, left_shoulder.y, right_shoulder.x, right_shoulder.y, left_elbow.x, left_elbow.y, right_elbow.x, right_elbow.y, left_wrist.x, left_wrist.y, right_wrist.x, right_wrist.y, left_index.x, left_index.y, right_index.x, right_index.y, left_eye.x, left_eye.y, left_eye.z, right_eye.x, right_eye.y, right_eye.z]], device= device)
            print(input.device)
            prediction = Net(input).detach().cpu()
            #prediction.cpu()
            prediction = torch.max(prediction,1)[1]
            pred = prediction[0].tolist()
            print(str(pred))
            if(pred == 1):
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
