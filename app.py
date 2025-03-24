import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Load and preprocess the dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df.dropna(inplace=True)  # Remove any missing values

# Define label_counts for the pie chart using the "Category" column
label_counts = df['Category'].value_counts()

# Split the dataset into training and testing sets using the original column names
X = df['Message']
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the Naive Bayes model using a pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB(alpha=0.1))
])
model_pipeline.fit(X_train, y_train)

# ---------------------------
# Custom CSS for Background and Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E9F3E7 !important;
    }
    textarea {
        background-color: #D4C8C8 !important;
    }
    .title {
        text-align: center;
        font-family: 'Cooper Black', serif;
        font-size: 48px;
        color: #333333;
        margin-bottom: 20px;
    }
    .intro {
        text-align: justify;
        font-family: 'Emblema One', cursive;
        font-size: 18px;
        color: #333333;
        margin-bottom: 30px;
    }
    .summary {
        text-align: justify;
        font-family: 'Emblema One', cursive;
        font-size: 18px;
        color: #333333;
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------------
# Streamlit App Interface

# Center-aligned title with custom styling
st.markdown('<div class="title">Email Spam Detection System</div>', unsafe_allow_html=True)

# Extended introduction message describing the site's purpose and functionality
st.markdown(
    """
    <div class="intro">
    Welcome to the Email Spam Detection System. Simply enter your email message in the box below and click 
    <strong>Check</strong> to determine if the message is spam or legitimate. The system uses advanced machine learning 
    techniques to analyze the text and also provides a visual breakdown of the overall dataset distribution for spam 
    and non-spam content.
    </div>
    """, unsafe_allow_html=True
)

# Text input area for user message
user_message = st.text_area("Enter your message below:")



if st.button("Check"):
    if user_message.strip() == "":
        st.error("Please enter a message to analyze.")
    else:
        # Get the predicted class
        prediction = model_pipeline.predict([user_message])[0]
        
        # Get prediction probabilities
        proba = model_pipeline.predict_proba([user_message])[0]
        # Get the order of classes from the model
        classes = model_pipeline.classes_
        
        # Map probabilities to custom labels:
        # Assuming one class is "spam" and the other is something like "ham" or "non-spam"
        labels = []
        values = []
        spam_percentage = None
        non_spam_percentage = None
        for cls, p in zip(classes, proba):
            if cls.lower() == "spam":
                labels.append("Spam content")
                spam_percentage = p * 100
            else:
                labels.append("Non-spam content")
                non_spam_percentage = p * 100
            values.append(p)
        
        # Display prediction text
        if prediction.lower() == "spam":
            st.error("The message is **spam**.")
        else:
            st.success("The message is **legitimate (not spam)**.")
        
        # Create a pie chart based on the prediction probabilities
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#D4C8C8")
        ax.set_facecolor("#D4C8C8")
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=['#d3ed5f', '#627a35'])
        ax.set_title("Message Spam Component Distribution", fontfamily='Cooper Black')
        for text in ax.texts:
            text.set_fontfamily('Cooper Black')
        st.pyplot(fig)

        # ---------------------------
        # A word cloud from the dataset based on the predicted category
        if prediction.lower() == "spam":
            subset = df[df['Category'].str.lower() == 'spam']['Message']
        else:
            subset = df[df['Category'].str.lower() != 'spam']['Message']
            
        # Combine the messages into one large text
        text = " ".join(subset)
        
        # Create the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
        
        # Plot the word cloud using matplotlib
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        fig_wc.patch.set_facecolor("#D4C8C8")
        ax_wc.set_facecolor("#D4C8C8")
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title("Word Cloud for " + ("Spam Messages\n" if prediction.lower() == "spam" else "Non-spam Messages\n"),
                        fontfamily='Cooper Black', fontsize=20)
        
        st.pyplot(fig_wc)

        st.divider()
        # st.markdown("---")  # Creates a full-width horizontal line

        # Summary Section: Explaining Visualizations
        st.markdown("### Summary")
        st.markdown(
            f""" 
            <div style="text-align: justify; font-family: 'Emblema One', cursive; font-size: 18px; color: #333333;">
            In this analysis, the system evaluated your email message and determined that it is <b>{'spam' if prediction.lower()=='spam' else 'legitimate (not spam)'}</b>. <br><br>  
            The <b>pie chart</b> above represents the model's confidence levels: it indicates that there is a <b>{spam_percentage:.1f}%</b> chance of spam content and a <b>{non_spam_percentage:.1f}%</b> chance of non-spam content in the message. <br><br>  
            The <b>word cloud</b> visualizes the most common words corresponding to the predicted category (Spam or legitimate).  
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown("<hr style='border:2px solid black'>", unsafe_allow_html=True)

