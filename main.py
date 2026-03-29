from src.core.inference import SafetyInspector


inspector = SafetyInspector(
    weights = 'data/model.pt'   
    )


result= inspector.detect_image('data/test_images/image_1.jpg')

print(result)