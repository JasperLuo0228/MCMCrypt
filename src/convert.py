import pandas as pd

# Load a subset of the file
df = pd.read_csv("data/cleaned9.csv", encoding="utf-8", engine="python", on_bad_lines="skip", nrows=1000)

# Combine all entries from the first column
text = "\n".join(df.iloc[:, 0].dropna().astype(str))

# Save to a smaller file
with open("train_email_input.txt", "w", encoding="utf-8") as f:
    f.write(text)