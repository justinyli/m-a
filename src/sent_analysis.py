from transformers import pipeline

def sentiment_analysis(text):
    # https://huggingface.co/docs/transformers/main/main_classes/pipelines
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # You can change the model if needed
    classifier = pipeline('text-classification', model=model_name)

    # https://huggingface.co/docs/transformers/tasks/sequence_classification
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']

    return label, score

if __name__ == "__main__":
    # written myself
    headlines = [
        "Microsoft and Activision in advanced talks",
        "Sony and Activision not seeing eye to eye",
        "Acquisition of XYZ Corp by ABC Inc finalized",
        "Merger between Company A and Company B rumored",
        "Tech giants explore potential merger",
    ]
    
    for i, headline in enumerate(headlines, 1):
        prediction, confidence = sentiment_analysis(headline)
        print(f"Headline {i}: Predicted Label - {prediction}, Confidence Score - {confidence:.2f} - {headline}")