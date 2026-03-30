from src.core.inference import SafetyInspector


inspector = SafetyInspector(
    weights = 'data/model.pt'   
    )


result= inspector.detect_image('data/test_images/image_7.jpg')

print(result)