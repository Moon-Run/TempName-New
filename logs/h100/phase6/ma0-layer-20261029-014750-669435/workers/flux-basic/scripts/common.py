def require(condition, message):
    if not condition:
        raise ValueError(message)
