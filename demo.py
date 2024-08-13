import pickle
import streamlit as st
import torch
import re
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BartForConditionalGeneration, Trainer, TrainingArguments



# Streamlit interface
st.title("News Article Classification")

# User input
text = st.text_area("Enter news article text:")

####################################################################################################
# Load the model and tokenizer for classification task

with open('D:\\news_project\\Project\\model_and_tokenizer.pkl', 'rb') as file:
    data = pickle.load(file)
    model_class = data['model']
    tokenizer_class = data['tokenizer']
    
    
    
# Load the model and tokenizer for summerization task

#with open('model_bart_sum.pkl', 'rb') as file:
    #model_sum = pickle.load(file)
with open('tokenizer.pkl', 'rb') as file:
    tokenizer_sum = pickle.load(file)
# Load the fine-tuned model and tokenizer
#model_dir = "D:/final_demo/bart_fine_tuned_w"
#tokenizer = BartTokenizer.from_pretrained(model_dir)
model_sum = BartForConditionalGeneration.from_pretrained("D:\\news_project\\Project\\bart_fine_tuned_w")
#####################################################################################################
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

#######################################################################################################

# Define a function to make predictions
# Use CPU device
device = torch.device('cpu')
model_class.to(device)

#######################################classification function#########################################
def predict_class(text):
    # Tokenize the input text
    inputs = tokenizer_class(
    text,
    padding=True,
    truncation=True,
    return_tensors='pt',
    max_length=512
)
    # Move tensors to the CPU device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make predictions
    model_class.eval()  # Set model to evaluation mode


    # Get the prediction
    #prediction = model.predict(inputs)
    with torch.no_grad():  # Disable gradient calculation
        outputs = model_class(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    
    # Define the class mapping
    class_mapping = {0: 'center', 1: 'left', 2: 'right'}

    # Convert predictions to class names
    predicted_labels = [class_mapping[prediction.item()] for prediction in predictions]
    
    return predicted_labels

###################################################################################################


# Tokenize the input for summerization task
input_art = tokenizer_sum(text, max_length=1024, return_tensors="pt", truncation=True)

def gen_summary(inputs):
    # Generate summary
    summary_ids = model_sum.generate(
    inputs['input_ids'], 
    max_length=150, 
    min_length=90, 
    length_penalty=2.0, 
    num_beams=4, 
    #early_stopping=True
)
    # Decode the summary into a human-readable string
    summary = tokenizer_sum.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

###################################################################################################

###################################################################################################

#clean text for classification
c_text = clean_text(text)

if st.button("Evaluate"):
    if text:
        result1 = predict_class(c_text)
        
        st.markdown(
            f"""
            <style>
            .prediction-text {{
                display: flex;
                border: 2px solid;
                height: 5vh;
                border-radius: 10px;
                justify-content: center;
                padding: 10px;
                width: 100%;
                backdrop-filter: blur(1px);
                color: rgb(255 99 99);
                background-color: black;
                font-weight: bold;
            }}
            </style>
            <div class="prediction-text">
                Political Bias: {result1}
            </div>
            """,
            unsafe_allow_html=True
        )
        summary = gen_summary(input_art)
        
        st.markdown(
            f"""
            <style>
            .generate-summary {{
                border: 2px solid;
                font-size: 17px;
                font-weight: bold;
                color: rgb(255 99 99); /* Change this to your preferred color */
                background-color: #0000; /* black background */
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
                margin: 10px 0;
                line-height: 2rem;
            }}
            .generate-summary:hover {{
                color: rgb(255 172 99);
            }}
            </style>
            <div class="generate-summary">
                Summary: {summary}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        
        

    else:
        st.write("Please enter some text to classify.")       
        
        