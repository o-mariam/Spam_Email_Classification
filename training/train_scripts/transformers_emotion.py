import transformer as train


class Transformer_E5(train.TransformerModel):
    def __init__(self):
        super().__init__(
                model_name="boltuix/bert-emotion",
                dataset_file="spam_or_not_spam"
        )

if __name__ == "__main__":
    print("Running SetFit model...")
    model = Transformer_E5()
    model.sample_n_per_class(n=50)
    model.split_train_test()
    model.training()