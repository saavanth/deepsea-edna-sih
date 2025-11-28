import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path("data/pr2_training_6000_with_seq.csv")

def main():
    df = pd.read_csv(DATA_PATH)

    # Simple random split; later you can make it more clever
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )

    train_df.to_csv("data/pr2_train.csv", index=False)
    val_df.to_csv("data/pr2_val.csv", index=False)

    print("Train:", len(train_df), "Val:", len(val_df))

if __name__ == "__main__":
    main()
