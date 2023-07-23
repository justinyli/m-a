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
        "Microsoft-Activision deal back in hands of UK regulator after court pauses appeal",
        "SBB, Brookfield end talks on EduCo stake sale, shares tumble",
        "Safran agrees to buy Collins flight controls business",
        "FTC to pause Microsoft-Activision merger trial, Bloomberg reports",
        "Bain Capital raises SoftwareOne takeover offer to $3.7 billion, Bloomberg reports",
        "Ocado not pursuing takeover offers, says boss",
        "Investors in VinFast's SPAC cash out most shares",
        "Chindata says largest shareholder Bain will not sell stake after rival bid",
        "KKR boosts chemicals portfolio with $1.3 billion deal for Chase",
        "Investor group nears $125 million deal for CoinDesk, Wall Street Journal reports",
        "Japan's Sosei buys Idorsia Pharma's Japan, Korea units for $466 mln"
    ]
    
    for i, headline in enumerate(headlines, 1):
        prediction, confidence = sentiment_analysis(headline)
        print(f"Headline {i}: Predicted Label - {prediction}, Confidence Score - {confidence:.2f} - {headline}")