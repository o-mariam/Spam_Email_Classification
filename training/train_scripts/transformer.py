from setfit import SetFitModel, SetFitTrainer
import pandas as pd
import random
from datasets import Dataset
from sklearn.metrics import accuracy_score


class TransformerModel:
    def __init__(self, model_name, dataset_file):
        self.model_name = model_name
        self.dataset_file = dataset_file
        self.df = self.read_dataset_file()
        self.train_df = None
        self.test_df = None

    def read_dataset_file(self):
        df = pd.read_csv(f'./training/dataset/{self.dataset_file}.csv')
        df['status'] = df['label'].apply(lambda x: 'spam' if x == 1 else "not_spam")
        del df['label']
        return df

    def sample_n_per_class(self, n=100, random_seed=42):
        random.seed(random_seed)
        sampled_df = pd.DataFrame(columns=self.df.columns)
        for label in self.df["status"].unique():
            class_samples = self.df[self.df["status"] == label]
            sampled_class = class_samples.sample(min(n, len(class_samples)), random_state=random_seed)
            sampled_df = pd.concat([sampled_df, sampled_class])
        self.df = sampled_df

    def split_train_test(self, test_ratio=0.2, random_seed=42):
        random.seed(random_seed)
        train_df = pd.DataFrame(columns=self.df.columns)
        test_df = pd.DataFrame(columns=self.df.columns)
        for label in self.df["status"].unique():
            class_samples = self.df[self.df["status"] == label]
            class_samples = class_samples.sample(frac=1, random_state=random_seed)
            split_idx = int(len(class_samples) * (1 - test_ratio))
            train_df = pd.concat([train_df, class_samples.iloc[:split_idx]])
            test_df = pd.concat([test_df, class_samples.iloc[split_idx:]])
        self.train_df = train_df
        self.test_df = test_df

    def training(self):
        train_dataset = Dataset.from_pandas(self.train_df.reset_index(drop=True))
        test_dataset = Dataset.from_pandas(self.test_df.reset_index(drop=True))

        model = SetFitModel.from_pretrained(self.model_name)
        trainer = SetFitTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            batch_size=16,
            num_iterations=20,
            num_epochs=1,
            column_mapping={"email": "text", "status": "label"},
            learning_rate=0.01
        )

        trainer.train()
        metrics = trainer.evaluate()
        print("Evaluation metrics:", metrics)

        y_true = self.test_df["status"].tolist()
        y_pred = model.predict(self.test_df["email"].tolist())
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy:", accuracy)

        save_path = f'./models/{self.model_name.replace("/", "_")}'
        trainer.model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")


if __name__ == '__main__':
    pass
