import pandas as pd
import numpy as np

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import evaluate

def main():
    # ------------------------------------------------------------
    # STEP 1: Load the labeled data (1,200 rows)
    # ------------------------------------------------------------
    labeled_data_path = "training_data/A11_tagged_1k.xlsx"
    df_labeled = pd.read_excel(labeled_data_path)
    df_labeled = df_labeled.reset_index(drop=True)

    print(f"Loaded labeled data. Shape = {df_labeled.shape}")

    # ------------------------------------------------------------
    # STEP 2: Encode labels numerically
    # ------------------------------------------------------------
    label_encoder = LabelEncoder()
    df_labeled["Label"] = label_encoder.fit_transform(df_labeled["Tag"])
    # We'll use label_encoder.inverse_transform() to get back original tags

    # ------------------------------------------------------------
    # STEP 3: Split into train/test
    # ------------------------------------------------------------
    train_df, test_df = train_test_split(df_labeled, test_size=0.2, random_state=42)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # ------------------------------------------------------------
    # STEP 4: Convert DataFrames to Hugging Face Datasets
    # ------------------------------------------------------------
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # We'll define a simple function to access text/labels
    def get_text(example):
        return example["Comment_Cleaned"]

    def get_label(example):
        return example["Label"]

    # ------------------------------------------------------------
    # STEP 5: Load a DistilBERT tokenizer
    # ------------------------------------------------------------
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    # Tokenization function
    def tokenize(batch):
        # Use the correct column name that contains the text to tokenize
        text_column = "Comment_Cleaned"  # Change this to the actual column name
        
        # Convert all inputs to strings and handle None values
        texts = [str(text) if text is not None else "" for text in batch[text_column]]
        
        return tokenizer(
            texts,
            truncation=True, 
            padding="max_length",
            max_length=128
        )

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Rename columns to the format transformers expects
    train_dataset = train_dataset.rename_column("Label", "labels")
    test_dataset = test_dataset.rename_column("Label", "labels")

    # Set the format for PyTorch
    train_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels"]
    )
    test_dataset.set_format(
        type="torch", 
        columns=["input_ids", "attention_mask", "labels"]
    )

    # ------------------------------------------------------------
    # STEP 6: Load DistilBertForSequenceClassification
    # ------------------------------------------------------------
    num_labels = len(label_encoder.classes_)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    )

    # ------------------------------------------------------------
    # STEP 7: Set up TrainingArguments
    # ------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="distilbert_output",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=1e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,  # Increase if you need better performance
        weight_decay=0.01,
        logging_steps=100,
        save_steps=500,
        push_to_hub=False,
        # If you have a GPU, the Trainer will automatically use it.
        # If you only have CPU, it will run but more slowly.
    )

    # ------------------------------------------------------------
    # STEP 8: Define a compute_metrics function
    # ------------------------------------------------------------
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        return {"accuracy": acc, "f1": f1}

    # ------------------------------------------------------------
    # STEP 9: Create the Trainer & train
    # ------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting DistilBERT training...")
    trainer.train()

    # ------------------------------------------------------------
    # STEP 10: Evaluate on the test set
    # ------------------------------------------------------------
    print("Evaluating on test set...")
    metrics = trainer.evaluate(test_dataset)
    print("Evaluation metrics:", metrics)

    # ------------------------------------------------------------
    # STEP 11: Predict tags for unlabeled data with checkpointing
    # ------------------------------------------------------------
    unlabeled_data_path = "training_data/Unlabeled_Cleaned.xlsx"
    df_unlabeled = pd.read_excel(unlabeled_data_path)

    # Ensure Comment_Cleaned column exists
    if "Comment_Cleaned" not in df_unlabeled.columns:
        raise ValueError("Column 'Comment_Cleaned' not found in unlabeled data")

    # Fill NA values in Comment_Cleaned with empty string for checking
    df_unlabeled["Comment_Cleaned"].fillna("", inplace=True)

    # Add prediction columns after Comment_Cleaned
    comment_col_idx = df_unlabeled.columns.get_loc("Comment_Cleaned")
    df_unlabeled.insert(comment_col_idx + 1, "Predicted_Tag", "")
    df_unlabeled.insert(comment_col_idx + 2, "Tag_Confidence", 0.0)

    # Process in batches of 500
    BATCH_SIZE = 500
    total_rows = len(df_unlabeled)

    print(f"Processing {total_rows} unlabeled comments in batches of {BATCH_SIZE}...")

    # Define tokenize_unlabeled inside main where tokenizer is available
    def tokenize_unlabeled(batch):
        # Use the correct column name for unlabeled data
        text_column = "Comment_Cleaned"
        
        # Convert all inputs to strings and handle None values
        texts = [str(text) if text is not None else "" for text in batch[text_column]]
        
        return tokenizer(
            texts,
            truncation=True, 
            padding="max_length",
            max_length=128
        )

    for start_idx in range(0, total_rows, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        print(f"Processing batch {start_idx//BATCH_SIZE + 1}: rows {start_idx} to {end_idx-1}")
        
        # Get current batch
        current_batch = df_unlabeled.iloc[start_idx:end_idx].copy()
        
        # Skip empty comments
        has_comment_mask = current_batch["Comment_Cleaned"].str.strip() != ""
        rows_with_comments = current_batch[has_comment_mask].copy()
        
        if len(rows_with_comments) == 0:
            print("No non-empty comments in this batch. Skipping.")
            continue
        
        # Convert to Dataset
        batch_dataset = Dataset.from_pandas(rows_with_comments)
        batch_dataset = batch_dataset.map(tokenize_unlabeled, batched=True)
        batch_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # Predict
        raw_predictions = trainer.predict(batch_dataset)
        pred_logits = raw_predictions.predictions
        pred_labels = np.argmax(pred_logits, axis=-1)
        
        # Convert to tags and confidence
        predicted_tags = label_encoder.inverse_transform(pred_labels)
        from scipy.special import softmax
        probs = softmax(pred_logits, axis=1)
        confidences = probs.max(axis=1)
        
        # Update only rows with comments in the main dataframe
        # Get the indices of rows with comments in the original batch
        comment_indices = current_batch.index[has_comment_mask]
        
        # Update only those rows
        for i, (idx, tag, conf) in enumerate(zip(comment_indices, predicted_tags, confidences)):
            df_unlabeled.loc[idx, "Predicted_Tag"] = tag
            df_unlabeled.loc[idx, "Tag_Confidence"] = conf
        
        # Save intermediate results
        checkpoint_file = f"UnlabeledData_Checkpoint_{start_idx//BATCH_SIZE + 1}.xlsx"
        df_unlabeled.to_excel(checkpoint_file, index=False)
        print(f"Checkpoint saved to {checkpoint_file}")
        
        try:
            # Allow user to interrupt
            input_text = input("Press Enter to continue to next batch, or type 'stop' to halt: ")
            if input_text.lower().strip() == 'stop':
                print("Halting prediction at user request")
                break
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            break

    # Save final results
    output_file = "training_data/output/UnlabeledData_WithDistilBERT_Tags.xlsx"
    df_unlabeled.to_excel(output_file, index=False)
    print(f"\nPrediction completed up to batch {start_idx//BATCH_SIZE + 1}.")
    print(f"Results saved to '{output_file}'.")

if __name__ == "__main__":
    main()
