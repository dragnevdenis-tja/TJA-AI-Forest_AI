# Forest Audio AI: Ethics and Safety Policy

## 1. Audio Surveillance and Privacy Risks
The deployment of persistent audio monitoring in public or semi-public spaces like national forests introduces significant privacy considerations. While the primary objective is environmental protection, audio sensors have the technical capacity to capture human speech and other private activities.

**Policy**: TerraGuard sensors are strictly prohibited from being used for the surveillance of individuals. The system is designed to identify "threat classes" (fire, chainsaws, gunshots) and "ecological indicators" (wildlife, rain). Any accidental capture of human speech is treated as noise and discarded by the neural network's PCEN layer, which prioritizes energy-rich non-speech patterns.

## 2. Human Speech Detection and Data Retention
To further mitigate privacy risks, Forest Audio AI implements a "Discard-by-Default" architecture.

- **Non-Threat Filtering**: Audio chunks that do not trigger a `MEDIUM` or `HIGH` risk assessment are discarded from the gateway's memory within 60 seconds and are never stored on the cloud API server.
- **Speech Privacy**: If the `human` label confidence exceeds 0.5, the system automatically flags the chunk for immediate deletion, even if other threats are detected, to prevent the storage of intelligible conversations.
- **Anonymization**: All metadata transmitted from the gateway (temperature, humidity) is decoupled from any audio-derived features during long-term storage to prevent the reconstruction of specific acoustic events.

## 3. False Alarm Consequences
In a system responsible for alerting authorities to wildfires or illegal logging, false positives (Type I errors) and false negatives (Type II errors) carry high real-world stakes.

- **False Positives**: Can lead to "alert fatigue" among rangers and the wasteful deployment of emergency resources. In extreme cases, frequent false fire alarms could lead to the system being ignored during a real crisis.
- **False Negatives**: A failure to detect a chainsaw in the Codrii region could result in the loss of protected old-growth forest.
- **Mitigation**: The system uses a dual-model check. The audio model detects the event, but the risk model must confirm that the event is statistically significant given the current environment (e.g., a chainsaw detection is higher risk in a dense forest than in an urban park).

## 4. Bias in Machine Learning
Machine learning models are only as robust as their training data. Bias in Forest Audio AI could manifest in several ways:

- **Geographic Bias**: If the model is trained predominantly on sounds from Northern Moldova, it may perform poorly in the Southern steppe regions due to different acoustic propagation characteristics.
- **Seasonal Bias**: Audio patterns of wildlife and wind change dramatically between summer and winter. A model trained only on summer data may misclassify winter wind as a distant fire.
- **Acoustic Backgrounds**: Unusual but non-threatening sounds (e.g., local festivals, agricultural machinery) could be biased against if not properly represented in the training set.

## 5. Responsible Deployment Checklist
Before deploying a new cluster of TerraGuard sensors, operators must complete the following:

1. **Environmental Impact Study**: Ensure sensor placement does not disturb sensitive wildlife nesting areas.
2. **Public Notification**: Place clear signage at forest entry points notifying the public of "Active Environmental Monitoring."
3. **Legal Compliance**: Verify that audio capture complies with Moldovan national privacy and data protection laws.
4. **Baseline Calibration**: Perform a 48-hour "Silent Run" to capture local acoustic backgrounds and tune energy thresholds.
5. **Human-in-the-Loop**: Establish a protocol where high-risk alerts are verified by a human operator before deploying physical intervention teams.

## 6. Conclusion
Forest Audio AI is a tool for conservation, not surveillance. By adhering to strict data retention policies, implementing automated speech filtering, and maintaining a rigorous evaluation suite, we ensure that the system protects the forest while respecting the privacy of the people who enjoy it.
