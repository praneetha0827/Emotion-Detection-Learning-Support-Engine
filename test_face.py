from deepface import DeepFace

result = DeepFace.analyze(img_path="face.jpg", actions=['emotion'], enforce_detection=False)
print(result[0]['dominant_emotion'])