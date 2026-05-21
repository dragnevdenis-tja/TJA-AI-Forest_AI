from typing import Dict, Any

def assign_risk_label(row: Dict[str, Any]) -> str:
    """
    Applies weak supervision rules to assign risk labels (LOW / MEDIUM / HIGH).
    
    Rules:
    - fire_confidence > 0.7 AND humidity < 35 → HIGH
    - chainsaw_confidence > 0.6 AND region == Codrii → HIGH (illegal logging)
    - gunshot_confidence > 0.65 AND region != Chisinau → MEDIUM
    - anomaly_score > 0.8 → MEDIUM minimum
    - rolling_fire_30m > 0.4 → HIGH
    - default → LOW
    """
    
    # Check HIGH risk first (most severe)
    if row["fire_confidence"] > 0.7 and row["humidity"] < 35:
        return "HIGH"
    
    if row["chainsaw_confidence"] > 0.6 and row["region"] == "Codrii":
        return "HIGH"
    
    if row["rolling_fire_30m"] > 0.4:
        return "HIGH"
    
    # Check MEDIUM risk
    is_medium = False
    if row["gunshot_confidence"] > 0.65 and row["region"] != "Chisinau":
        is_medium = True
        
    if row["anomaly_score"] > 0.8:
        is_medium = True
        
    if is_medium:
        return "MEDIUM"
        
    return "LOW"
