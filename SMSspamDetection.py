import pandas as pd              #this library is used for data manipulation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import tkinter as tk
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/Acer/Downloads/spamraw.csv")

# print(data.info())

X = data['text'].values
y = data['type'].values

label_encoder = LabelEncoder()    #label encoder is used to convert string into numerical data so that we can use it in machine learning
y = label_encoder.fit_transform(y) #for predicting the output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  #in this function we are splitting the X and y data into training and testing part
                                   #in testing test_size=0.2 we are taking 20 percent of whole data and 80 percent is used to train model by default
                                   #random_state is used to make the result reproducible
                                   #if we don't give any value then it will be random

#convert ing text data into numerical data because later when we run fit function to calculate mean and S.D so it have numerical data.
cv = CountVectorizer()     #it will transform the text data into numerical data by frequency of each word in the text data
X_train = cv.fit_transform(X_train)  #since we also need to train data that's why we are using fit_transform function
X_test = cv.transform(X_test)        #since we also need to test data but do not train data that's why we are using only transform function.

classifier = SVC(kernel='rbf',random_state = 10)
classifier.fit(X_train, y_train)

def check_spam():
    input_message = entry.get()
    input_vector = cv.transform([input_message])
    prediction = classifier.predict(input_vector)[0]
    result_str = "SPAM" if prediction == 1 else "NOT SPAM"
    messagebox.showinfo("Result", f"The SMS is {result_str}!")

window = tk.Tk()
window.title("SMS Spam Detection")

entry = tk.Entry(window, width=50)
entry.pack(padx=10, pady=10)

check_button = tk.Button(window, text="Check Spam", command=check_spam)
check_button.pack(pady=5)

window.mainloop()

print(classifier.score(X_test,y_test))
