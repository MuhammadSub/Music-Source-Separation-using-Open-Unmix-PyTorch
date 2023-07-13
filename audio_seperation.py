from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import torch
import torchaudio
import subprocess
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def convert_to_wav(mp4_path):
    if mp4_path.endswith(".mp4"):
        wav_path = mp4_path[:-4] + ".wav"
        subprocess.run(["ffmpeg", "-i", mp4_path, wav_path])
        return wav_path
    return mp4_path

def separate_audio(waveform, sample_rate):
    from openunmix import predict

    estimates = predict.separate(
        waveform.unsqueeze(0).to(device),
        rate=sample_rate,
        device=device
    )

    return estimates

def normalize_waveform(waveform):
    min_value = torch.min(waveform)
    max_value = torch.max(waveform)
    normalized_waveform = (waveform - min_value) / (max_value - min_value)
    return normalized_waveform

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Audio Separation</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }

            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }

            .title {
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 20px;
            }

            .upload-form {
                display: flex;
                justify-content: center;
                align-items: center;
                margin-bottom: 20px;
            }

            .upload-input {
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-size: 16px;
                margin-right: 10px;
                flex-grow: 1;
            }

            .upload-button {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                padding: 10px 20px;
                font-size: 16px;
            }

            .audio-container {
                margin-bottom: 20px;
            }

            .audio-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 5px;
            }

            .audio-element {
                width: 100%;
                outline: none;
            }

            .stems-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                grid-gap: 20px;
            }

            .stem-card {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                text-align: center;
                box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
                transition: box-shadow 0.3s ease-in-out;
            }

            .stem-card:hover {
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }

            .stem-name {
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 5px;
            }

            .stem-audio {
                width: 100%;
                outline: none;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">Audio Separation</h1>
            <form class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                <input class="upload-input" type="file" name="audioFile" accept=".mp3, .wav, .mp4">
                <input class="upload-button" type="button" value="Upload" onclick="uploadFile()">
            </form>

            <div class="audio-container">
                <h2 class="audio-title">Original Audio</h2>
                <audio class="audio-element" id="originalAudio" controls></audio>
            </div>

            <h2>Separated Stems</h2>
            <div class="stems-container" id="stems"></div>
        </div>

        <script>
            function uploadFile() {
                var fileInput = document.querySelector('input[type="file"]');
                var file = fileInput.files[0];

                var formData = new FormData();
                formData.append("audioFile", file);

                var xhr = new XMLHttpRequest();
                xhr.open("POST", "/upload", true);
                xhr.onreadystatechange = function () {
                    if (xhr.readyState === 4 && xhr.status === 200) {
                        var response = JSON.parse(xhr.responseText);
                        displayAudio(response);
                        displaySeparatedStems(response);
                    }
                };
                xhr.send(formData);
            }

            function displayAudio(response) {
                var audioElement = document.getElementById("originalAudio");
                audioElement.src = response.originalAudio;
            }

            function displaySeparatedStems(response) {
                var stems = response.stems;
                var stemsContainer = document.getElementById("stems");

                stemsContainer.innerHTML = ""; // Clear the existing content

                for (var i = 0; i < stems.length; i++) {
                    var stem = stems[i];
                    var stemCard = document.createElement("div");
                    stemCard.classList.add("stem-card");

                    var stemName = document.createElement("p");
                    stemName.classList.add("stem-name");
                    stemName.textContent = stem.name;
                    stemCard.appendChild(stemName);

                    var audioElement = document.createElement("audio");
                    audioElement.classList.add("stem-audio");
                    audioElement.controls = true;
                    audioElement.src = stem.audio;
                    stemCard.appendChild(audioElement);

                    stemsContainer.appendChild(stemCard);
                }
            }
        </script>
    </body>
    </html>
    '''


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audioFile']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    wav_path = convert_to_wav(filepath)
    waveform, sample_rate = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    normalized_waveform = normalize_waveform(waveform)
    estimates = separate_audio(normalized_waveform, sample_rate)

    stems_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'stems')
    os.makedirs(stems_folder, exist_ok=True)  # Create the 'stems' folder if it doesn't exist

    stems = []
    for target, estimate in estimates.items():
        audio = estimate.squeeze().detach().cpu().numpy()
        stem_filename = f"{filename}_{target}.wav"
        stem_filepath = os.path.join(stems_folder, stem_filename)
        torchaudio.save(stem_filepath, torch.from_numpy(audio), sample_rate)

        stem = {
            'name': target,
            'audio': f'uploads/stems/{stem_filename}'
        }
        stems.append(stem)

    response = {
        'originalAudio': f'uploads/{filename}',
        'stems': stems
    }

    # Wait for a short period to allow other processes to release the file
    time.sleep(1)

    # Remove the temporary WAV file
    os.remove(wav_path)

    return jsonify(response)

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/uploads/stems/<filename>')
def serve_stem(filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], 'stems'), filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
