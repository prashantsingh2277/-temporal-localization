import torch
from model import TemporalLocalizationModel


model = TemporalLocalizationModel(in_channels=512)
model.load_state_dict(torch.load('model_checkpoint.pth', map_location=torch.device('cpu')))
model.eval()


scripted_model = torch.jit.script(model)
scripted_model.save("temporal_localization_model.pt")

dummy_input = torch.randn(1, 512, 100)
torch.onnx.export(model, dummy_input, "temporal_localization_model.onnx",
                  input_names=['input'], output_names=['boundary', 'segment'])
