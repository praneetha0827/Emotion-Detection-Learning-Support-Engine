from transformers import pipeline

classifier = pipeline("text-classification", 
                       model="j-hartmann/emotion-english-distilroberta-base", 
                       top_k=1)
result = classifier("I don't understand this topic at all, it's so frustrating")
print(result)