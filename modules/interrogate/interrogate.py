def interrogate(image):
    from modules.interrogate import legacy
    prompt = legacy.interrogator.interrogate(image)
    return prompt
