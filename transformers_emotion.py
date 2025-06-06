from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer
import pandas as pd
from collections import defaultdict
import random
from datasets import Dataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


model_name = "boltuix/bert-emotion"

# Example dataset (replace with your actual data)
# data = {
#     "email": ["url URL date not supplied contrary to popular belief a new study shows women receive the same mental health boost from marriage as men ", "hi i was just wondering if anyone experiening difficulty with eircom mail sever i was trying to send mail from mozilla mail but it keeps coming up with this error sorry that domain isnt in my list of allows rcpthosts NUMBER NUMBER NUMBER please check the message recipients and try again what is this all about i am using URL as my smtp and my pop i can recieve mail but i cannot send mail is there any other open relay that i can use nils also i was just browsing the web and came across this URL its a seminar on cough security and reliable cough maybe ilug should go along and hold a seminar on the same day right across the hall from this nils irish linux users group ilug URL URL for un subscription information list maintainer listmaster URL ", " its possibly a different shauna lowery the one i m talking about is about NUMBER foot tall and presents animal hospital or something like that tim h oh and i m less than NUMBER foot tall so it would be her who had to kneel down to unsubscribe from this group send an email to forteana unsubscribe URL your use of yahoo groups is subject to URL ", "wanna see sexually curious teens playing with each other URL click here me and my horny girlfriends are waiting for you we are probably eating each other out on webcam in our dormitory as ur reading this inbetween classes of course wink see you soon baby veronica mcmfkhcpedgetqj ", "request a free no obligation consultation accept credit cards today no set up fees no application fees all credit types accepted retail rates as low as NUMBER NUMBER mail order rates as low as NUMBER NUMBER set up your merchant account within NUMBER hours no cancellation fees no money down no reprogramming fees we will beat anybody s deal by NUMBER we make it easy and affordable to start accepting credit cards today NUMBER of our applicants are approved http NUMBER NUMBER NUMBER NUMBER marketing merchantnet to be removed http NUMBER NUMBER NUMBER NUMBER marketing removeme html ", " worldwide great restaurants shopping activities fema il NUMBER and ada compliant contact riz bhatti NUMBER NUMBER NUMBER ext NUMBER or donald bae NUMBER NUMBER NUMBER ext NUMBER take a virtual tour of our hotel hyperlink URL www radisson chicago com december NUMBER customer satisfaction survey radisson worldwide hotels and resorts embed ms_clipart_gallery NUMBER ", " do you want to teach and grow rich if you are a motivated and qualified communicator i will personally train you to do NUMBER NUMBER minutes presentations per day to qualify prospects that i can provide to you we will demonstrate to you that you can make NUMBER a day part time using this system or if you have NUMBER hours per week as in my case you can make in excess of NUMBER NUMBER per week as i am currently generating verifiable by the way plus i will introduce you to my mentor who makes well in excess of NUMBER NUMBER NUMBER annually many are called few are chosen this opportunity will be limited to one qualified individual per state make the call and call the NUMBER hour pre recorded message number below we will take as much or as little time as you need to see if this program is right for you NUMBER NUMBER NUMBER please do not make this call unless you are genuinely money motivated and qualified i need people who already have people skills in place and have either made large amounts of money in the past or are ready to generate large amounts of money in the future looking forward to your call NUMBER NUMBER NUMBER _______________________________________________________________ to be taken out of this database seccoNUMBER URL NUMBERdmelNUMBER NUMBERbmblNUMBER NUMBERzuadNUMBER NUMBERguslNUMBERwrmhNUMBER NUMBERopgxNUMBERmkeqNUMBER NUMBERquhlNUMBER ", " hyperlink hyperlink hyperlink hyperlink "],
#     "status": ["not_spam", "not_spam", "not_spam", "spam", "spam", "spam", "spam", "spam"]
# }
# df = pd.DataFrame(data)
df = pd.read_csv('spam_or_not_spam.csv', index_col=None)
df['status'] = df['label'].apply(lambda x: 'spam' if x == 1 else "not_spam")
del df['label']



def sample_n_per_class(df, n=100, random_seed=42):
    random.seed(random_seed)
    sampled_df = pd.DataFrame(columns=df.columns)

    # Group by class (spam/not_spam)
    for label in df["status"].unique():
        # Get all samples for this class
        class_samples = df[df["status"] == label]
        # Randomly select N samples (or all if fewer than N)
        sampled_class = class_samples.sample(min(n, len(class_samples)), random_state=random_seed)
        # Add to sampled DataFrame
        sampled_df = pd.concat([sampled_df, sampled_class])

    return sampled_df

sampled_df = sample_n_per_class(df, n=50)
print("Sampled Data:\n", sampled_df)


def split_train_test(sampled_df, test_ratio=0.2, random_seed=42):
    random.seed(random_seed)

    train_df = pd.DataFrame(columns=sampled_df.columns)
    test_df = pd.DataFrame(columns=sampled_df.columns)

    # Split per class to maintain balance
    for label in sampled_df["status"].unique():
        class_samples = sampled_df[sampled_df["status"] == label]
        # Shuffle
        class_samples = class_samples.sample(frac=1, random_state=random_seed)
        # Split index
        split_idx = int(len(class_samples) * (1 - test_ratio))
        # Add to train/test
        train_df = pd.concat([train_df, class_samples.iloc[:split_idx]])
        test_df = pd.concat([test_df, class_samples.iloc[split_idx:]])

    return train_df, test_df


train_df, test_df = split_train_test(sampled_df, test_ratio=0.2)
print("Train Set:\n", train_df)
print("\nTest Set:\n", test_df)

# --------Setfit-----------

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset = train_dataset.remove_columns("__index_level_0__")

model = SetFitModel.from_pretrained(model_name)

# Create trainer
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_class=CosineSimilarityLoss,
    batch_size=16,
    num_iterations=20, # Number of text pairs to generate for contrastive learning
    num_epochs=1,# Number of epochs to use for contrastive learning
    column_mapping = {"email": "text", "status": "label"},
    learning_rate=0.01

)

# Train and evaluate!
trainer.train()
metrics = trainer.evaluate()

y_true = test_dataset["status"]
y_pred = model.predict(test_dataset["email"])

# # Create confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# print(cm)
accuracy=accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)


trainer.model.save_pretrained('./model/'+model_name)
