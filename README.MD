This project is an automated forest monitoring system designed to identify environmental and anthropogenic sounds
  (such as wildfires, chainsaws, wildlife, and rain) using deep learning.

  Core Functionality
  The system transforms raw audio into Mel-spectrograms and processes them through a CRNN (Convolutional Recurrent
  Neural Network). This hybrid architecture allows the model to extract spatial features from the audio "image" via CNN
  layers while using Recurrent (GRU) layers to understand the temporal sequence and duration of sounds.

  Key Characteristics
   * End-to-End Pipeline: It covers the entire lifecycle of an AI project, including automated data harvesting from
     YouTube, multi-source dataset integration (Fire, FSC22, ESC50), and frame-level audio classification.
   * Adversarial Model Hardening: Beyond standard training, the project uses a "Validator" architecture. This secondary
     model learns to identify when the primary classifier is likely making a mistake, creating a self-improving feedback
     loop to increase reliability in complex forest environments.
   * Production Readiness: The codebase is built for cross-platform deployment (Windows/Linux) and includes robust audio
     I/O handling to manage varying sample rates and file formats.

  In essence, it is a scalable acoustic surveillance tool aimed at conservation and early warning systems for forest
  threats.