from flask import Flask, Response, render_template
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
import time

from model import TemporalLocalizationModel

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TemporalLocalizationModel(in_channels=2048).to(device)
checkpoint_path = 'model_checkpoint.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

resnet = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

temporal_buffer = []
BUFFER_SIZE = 16

def interpret_predictions(boundary_pred, segment_pred):

    boundary_pred = boundary_pred.squeeze(0)  
    segment_pred  = segment_pred.squeeze(0)   
    T = boundary_pred.size(1)

    boundary_abs = boundary_pred.abs().sum(dim=0)  
    segment_abs  = segment_pred.abs().sum(dim=0)   
    total_abs = boundary_abs + segment_abs         

    best_t = torch.argmax(total_abs).item()
    start_off   = boundary_pred[0, best_t].item()
    end_off     = boundary_pred[1, best_t].item()
    center_off  = segment_pred[0, best_t].item()
    dur_off     = segment_pred[1, best_t].item()

    return {
        'best_time': best_t,
        'start_off': start_off,
        'end_off':   end_off,
        'center_off': center_off,
        'duration_off': dur_off
    }

def generate_frames():
    global temporal_buffer
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = feature_extractor(input_tensor)  
            feats = feats.view(2048)                

        temporal_buffer.append(feats.cpu())
        if len(temporal_buffer) > BUFFER_SIZE:
            temporal_buffer.pop(0)

        if len(temporal_buffer) == BUFFER_SIZE:
            # (1, 2048, 16)
            clip_features = torch.stack(temporal_buffer, dim=1).unsqueeze(0).to(device)
            with torch.no_grad():
                boundary_pred, segment_pred = model(clip_features)
            preds = interpret_predictions(boundary_pred, segment_pred)
            output_text = (f"t={preds['best_time']} | "
                           f"Start={preds['start_off']:.2f}, "
                           f"End={preds['end_off']:.2f}, "
                           f"Center={preds['center_off']:.2f}, "
                           f"Dur={preds['duration_off']:.2f}")
        else:
            output_text = "Accumulating frames..."

        cv2.putText(frame, output_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        ret, buffer_img = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer_img.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
