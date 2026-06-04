import pandas as pd


def main():
    data_path = "data/Telco-Customer-Churn.csv"

    df = pd.read_csv(data_path)

    print("Data loaded successfully")
    print("Shape:", df.shape)
    print(df.head())
    print(df.columns)


if __name__ == "__main__":
    main()
