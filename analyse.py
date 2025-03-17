import pandas as pd
import json

def dataframe_to_nested_json(df, output_file="data.json"):
    """
    Convert a DataFrame to a nested JSON format based on 'Payment Frequency' column
    and save it as a JSON file.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        output_file (str): The name of the output JSON file.
    """
    # Convert DataFrame to a dictionary grouped by 'Payment Frequency'
    result = df.groupby("Payment Frequency").apply(lambda x: x.drop(columns=["Payment Frequency"]).to_dict(orient="records")).to_dict()
    
    # Save to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"JSON saved successfully to {output_file}")

# Example Usage
# Assuming `df` is your DataFrame read from a CSV or other source
# df = pd.read_csv("your_file.csv")  # Modify as per your data source
dataframe_to_nested_json(df)
