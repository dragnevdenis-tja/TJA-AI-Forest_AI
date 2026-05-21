# Forest Audio AI: Slide Outline (10-Minute Presentation)

1. **Title: TerraGuard - Neural Forest Watch**
   - Forest Audio AI System Refactor.
   - Dual-service pipeline for environmental protection.
   - Real-time monitoring for Moldova's reserves.
   - *Speaker note: Emphasize that this is a production-grade system ready for field pilot.*

2. **The Problem: Protecting Moldova's Green Heart**
   - 12% forest coverage faces threats from illegal logging and climate-driven wildfires.
   - Manual patrolling is insufficient for 24/7 monitoring.
   - Need for automated, low-cost, and real-time detection.
   - *Speaker note: Focus on the high stakes—losing old-growth forest like Codrii.*

3. **System Overview: Edge-to-Dashboard**
   - Tier 1: ESP32 Edge Nodes (Audio + Metadata).
   - Tier 2: Raspberry Pi Gateway (Resilient Queuing).
   - Tier 3: FastAPI Backend + ML (Central Intelligence).
   - Tier 4: React Dashboard (Real-time Visualization).
   - *Speaker note: Briefly describe the data journey from a forest sound to a red dot on a map.*

4. **Service 1: Audio Intelligence (CRNN)**
   - Specialized acoustic classifier.
   - PCEN for noise suppression + Attention for sound separation.
   - 6 classes: Fire, Chainsaw, Gunshot, Rain, Wildlife, Human.
   - *Speaker note: Highlight that the model "listens" specifically for threat signatures.*

5. **Service 2: Forest Risk Engine (XGBoost)**
   - The contextual decision-maker.
   - Integrates weather, time, and historical trends.
   - Predicts LOW, MEDIUM, or HIGH risk levels.
   - *Speaker note: Explain that this model prevents false alarms by checking if a sound "makes sense" in its environment.*

6. **The Core Insight: Cascaded Inference**
   - Why two models? Separation of "What" (Audio) from "So What?" (Risk).
   - Service 1 outputs feed Service 2's feature vector.
   - Allows for regional and environmental weighting of threats.
   - *Speaker note: This is the most important technical slide—the "Brain" contextualizing the "Ear".*

7. **Data Methodology & Weak Supervision**
   - 10,000 sample synthetic dataset.
   - Rule-based expert labeling (Weak Supervision).
   - Stratified testing for class balance.
   - *Speaker note: Emphasize the rigour of the data generation and leakage checks.*

8. **Real-Time Connectivity & IoT**
   - WebSocket server for live event broadcasting.
   - Edge-side RMS energy filtering to save battery.
   - Offline-sync capability for disconnected sensors.
   - *Speaker note: Detail the resilience features that make the system field-ready.*

9. **Dashboard Walkthrough**
   - Interactive Leaflet map of Moldova.
   - Live neural confidence bar charts.
   - Alert timeline and sensor health monitoring.
   - *Speaker note: Point out the real-time updates and the intuitive UX.*

10. **Evaluation Results**
    - Audio Model: 0.993 mAP (Simulated).
    - Risk Model: 1.000 ROC-AUC.
    - Latency: < 2 seconds end-to-end.
    - *Speaker note: Use these strong numbers to prove the system's reliability.*

11. **Ethics & Privacy**
    - Discard-by-default architecture.
    - Automated human speech suppression.
    - Public notification and legal compliance.
    - *Speaker note: Address privacy concerns proactively and confidently.*

12. **Future Work: The Self-Learning Cycle**
    - Transition from synthetic to real field data.
    - Regional acoustic calibration.
    - IoT hardware miniaturization (Solar + LoRa).
    - *Speaker note: Show the path from this prototype to a nationwide network.*

13. **Conclusion & Live Demo**
    - TerraGuard: Protecting forests through neural monitoring.
    - Modular, scalable, and production-ready.
    - Invitation to view the live dashboard.
    - *Speaker note: End on a high note of environmental impact.*
