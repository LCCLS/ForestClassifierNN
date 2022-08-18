# ForestClassifierNN

**Data** -->
the data contains information of different kinds of forest covers based on geographical information in california 
the goal is to correctly classify forest covers to be able to further investigate the landscape for the spread of 
forest fires and possible "danger hotspots"

**Model ** -->
the model used here is a keras.Sequential neural net with a resulting accuracy of 0.89 and a weighted f1 score 
of 0.87 

**Further improvements ** -->
given the output and performance (f1 metrics, accuracy, etc.) underrepresented classes are often classified more 
incorrectly than majority classes. Therefore, it would be interesting to see how the performance would change if 
something like SMOTE undersampling would be applied 

