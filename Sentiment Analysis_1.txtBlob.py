from textblob import TextBlob

output=TextBlob("Great! This has been really a grat day for me.")
print(output.sentiment)



print(output.sentences)
print(output.words)



for np in output.noun_phrases:
 print (np)

#for words, tag in output.tags:
# print (words, tag)


print ("")
print ("")
#print(output.correct())
print(output.words[6].spellcheck())

output.detect_language()

print(output.translate(to= 'ar'))
print(output.translate(to= 'fr'))
print(output.translate(to= 'es'))
print(output.translate(to= 'hi'))
print(output.translate(to= 'de'))

print ("")
print ("Checkpoint1")
print ("")


###########  Learning Approach

training = [
('Tom Holland is a terrible spiderman.','pos'),
('a terrible Javert (Russell Crowe) ruined Les Miserables for me...','pos'),
('The Dark Knight Rises is the greatest superhero movie ever!','neg'),
('Fantastic Four should have never been made.','pos'),
('Wes Anderson is my favorite director!','neg'),
('Captain America 2 is pretty awesome.','neg'),
('Let\s pretend "Batman and Robin" never happened..','pos'),
]
testing = [
('Superman was never an interesting character.','pos'),
('Fantastic Mr Fox is an awesome film!','neg'),
('Dragonball Evolution is simply terrible!!','pos')
]

from textblob import classifiers
classifier = classifiers.NaiveBayesClassifier(training)

print(classifier.accuracy(testing))


blob1 = TextBlob('Great! This has been really a great day for me.', classifier=classifier)
#blob1 = TextBlob('This is the worst day of my life.', classifier=classifier)

print (blob1)
print(blob1.sentiment)
print (blob1.classify())


