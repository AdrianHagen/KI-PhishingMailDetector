import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the dataset
data = pd.read_csv("data/Phishing_Email.csv")

# Load the sentence transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode the text data to get embeddings
embeddings = model.encode(data["Email Text"].tolist(), show_progress_bar=True)

# Convert embeddings to a list of lists with Python floats
embeddings_list = [list(map(float, embedding)) for embedding in embeddings]

# Convert embeddings to a single string representation
embeddings_str_list = [str(embedding) for embedding in embeddings_list]

# Create a new DataFrame with a single column for embeddings
embeddings_df = pd.DataFrame({"embedding": embeddings_str_list})

# Concatenate the label column with the embeddings
final_df = pd.concat([data["Email Type"].reset_index(drop=True), embeddings_df], axis=1)

# Save the final DataFrame to a CSV file
final_df.to_csv("data/embeddings.csv", index=False, sep=";")
