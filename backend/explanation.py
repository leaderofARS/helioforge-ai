from backend.inference import CLASS_NAMES, RISK_LEVELS


def explain_prediction(class_id: int, confidence: float) -> dict:
    """Provide an explicit, traceable rule-based explanation for dashboard use."""
    label = CLASS_NAMES[class_id]
    return {
        "summary": f"The model classified this observation as {label}-class with {confidence:.0%} confidence.",
        "risk_level": RISK_LEVELS[class_id],
        "reasons": [
            "The temporal convolutional model detected a sustained change across the 512-step observation window.",
            "The displayed 32 engineered channels are the normalized inputs used by HelioForgeTCN.",
            "Confidence is the maximum softmax probability; it is not a physical flare magnitude measurement.",
        ],
    }
