def calculate_risk_level(stroke_pred: str, stroke_type: str, num_clots: int, lesion_area: int) -> str:
    \"\"\"
    Rule-based engine for determining stroke risk level.
    \"\"\"
    if stroke_pred == "Normal":
        return "Low"
    
    # Example logic (to be refined based on medical domain knowledge or project specific requirements)
    if stroke_type == "Hemorrhagic" or num_clots > 1 or lesion_area > 2000:
         return "High"
    elif stroke_type == "Ischemic" and num_clots > 0:
        return "Medium"
    else:
        return "Low"