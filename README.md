# Music-Source-Separation-using-Open-Unmix-PyTorch

##Methodology:
The methodology hired inside the Music Source Separation task the usage of Open-Unmix PyTorch includes numerous key steps to gain accurate and reliable separation of song sources. The assignment leverages the talents of deep neural networks and pre-skilled fashions to address the intricate task of supply separation. The following Points offer an overview of the methodology concerned with the mission.
1. User Interface: The project utilizes an internet-based interface built with Flask, HTML, and JavaScript. The interface allows users to add their audio documents and examine the separated stems.
2. Pre-processing: The uploaded audio document is converted to the WAV format if it's miles in MP4 layout. Torchaudio library is used to load the waveform and pattern price of the audio.
3. Source Separation: The Open-Unmix model is employed to split the exceptional track resources from the waveform. The model takes the waveform as input, tactics the usage of deep neural networks, and produces estimates for each supply.
4. Normalization: The separated waveforms are normalized to ensure consistency and decorate the listening reveal in. The minimal and most values of the waveforms are determined, and the waveforms are scaled thus.
5. Saving Separated Stems: The separated stems are stored as man or woman audio documents inside the 'stems' folder. Each stem is associated with a name (vocals, drums, bass, or other instruments) and a corresponding audio file path.
6. Output Generation: The challenge generates a response item containing the trails to the authentic audio document and the separated stems. This response is dispatched back to the consumer interface for the show
