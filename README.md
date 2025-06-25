# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*:Marem Hemalatha

*INTERN ID*:CT04DG1515

*DOMAIN*:Machine Learning

*DURATION*:4 WEEKS

*MENTOR*:NEELA SANTOSH

*DESCRIPTION*:

*Task 2: Sentiment Analysis With NLP*:

*INTRODUCTION*:

In Task 2 of the internship, the objective was to perform Sentiment Analysis using Natural Language Processing (NLP) techniques. Sentiment Analysis is a process where we classify a piece of text as positive, negative, or neutral based on its emotional tone. In this task, we applied sentiment analysis on product reviews taken from the Amazon Reviews Dataset. These reviews reflect the opinions and feedback of customers regarding products they have purchased and used. The main aim was to train a machine learning model that can predict the sentiment of customer reviews using supervised learning methods.

*TASK DESCRIPTION*:

The task required building a complete NLP pipeline, which includes the following steps:
1.	Data Preprocessing
2.	Text Cleaning and Normalization
3.	Vectorization using TF-IDF
4.	Training a Classification Model
5.	Model Evaluation using Accuracy and Confusion Matrix
The project involved collecting and cleaning the data, transforming it into a machine-readable format, training a machine learning model, and evaluating the results.

*ABOUT THE DATASET*:

The dataset used for this task is titled AmazonReviews.csv. It contains multiple columns that describe customer feedback data. Some of the important fields are:
•	Serial Number – a unique identifier for each review
•	ReviewerName – name of the customer who wrote the review
•	Overall – numerical rating (1 to 5) given by the reviewer
•	ReviewText – the actual text of the customer's review
•	ReviewTime – date when the review was posted
•	Helpful Yes, Helpful No, Total Vote – information on how helpful other customers found the review
•	Score_pos_neg_diff, Score_Average_Rating, Willson_Lower_bound – calculated metrics for evaluating the helpfulness and fairness of the review
From this dataset, the ReviewText column was used as the main input for the sentiment analysis model, and the Overall rating was used to derive the sentiment label. The ratings were converted into sentiment categories as follows:
•	Ratings 4 and 5 → Positive sentiment (1)
•	Ratings 1 and 2 → Negative sentiment (0)
•	Rating 3 → considered neutral and excluded from training
This approach allowed us to perform binary classification, where the model learns to distinguish between only two categories: positive and negative reviews.

*TOOLS AND LIBRARIES USED*:

To perform this task efficiently, several Python libraries and tools were used. These include:
•	Pandas – for loading, handling, and analyzing tabular data
•	NumPy – for numerical computations
•	Regular Expressions (re) – for removing unwanted characters, URLs, and symbols from text
•	NLTK (Natural Language Toolkit) – used for text preprocessing, stop word removal, and lemmatization
•	Scikit-learn (sklearn) – the primary machine learning library used for:
o	TF-IDF vectorization
o	Splitting the data into training and testing sets
o	Training a Multinomial Naive Bayes model
o	Evaluating the model with accuracy_score, confusion_matrix, and classification_report
•	Matplotlib and Seaborn – for visualization of the confusion matrix
The TF-IDF (Term Frequency–Inverse Document Frequency) method was used to convert the textual reviews into numerical vectors. This method helps highlight important words in a review while reducing the impact of commonly used words.

*PLATFORM USED*:

This entire task was implemented using Jupyter Notebook, a popular open-source web-based IDE that allows combining live code, visualizations, and narrative text. I launched Jupyter Notebook using the command prompt, and the entire model pipeline was built and executed within this environment. Jupyter is widely used for machine learning and data science projects because it allows step-by-step execution and easy debugging.

*APPLICATION OF THE TASK*:

This type of sentiment analysis has real-world applications in various domains, including:
•	E-commerce platforms like Amazon, Flipkart, and Myntra, where companies use sentiment analysis to understand customer satisfaction and feedback
•	Customer Service – to automatically detect negative reviews or complaints and respond accordingly
•	Brand Monitoring – to track public sentiment toward a product, service, or company on platforms like Twitter and Facebook
•	Product Development – helping companies make improvements based on common positive or negative feedback
•	Movie and Book Reviews – analyzing sentiments in reviews to predict success or failure
By building a sentiment classifier, we are automating the process of understanding customer opinions, which is very useful for business analytics and decision-making.

*OUTCOME*:

After completing all preprocessing and balancing steps, the Multinomial Naive Bayes model was trained and tested. Initially, class imbalance led to poor performance, but after balancing the dataset and applying advanced text cleaning (lemmatization and stop word removal), the model achieved better accuracy. The final performance was evaluated using a confusion matrix, accuracy score, precision, recall, and F1-score.

![Image](https://github.com/user-attachments/assets/4b7fe14b-6b76-4452-a4ff-7a2186c44c06)

![Image](https://github.com/user-attachments/assets/dc386ae5-fe1d-4186-9eb4-acd061ca77fc)
