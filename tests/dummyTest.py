from SentiNet.SentiLiteralNet import SentiLiteralNet

sentinet = SentiLiteralNet()

word = "güzel"

senti = sentinet.getSentiLiteral(word)

print("Word:", word)
print("Positive:", senti.getPositiveScore())
print("Negative:", senti.getNegativeScore())
print("Polarity:", senti.getPolarity())