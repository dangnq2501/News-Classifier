import pandas as pd 
from sklearn.model_selection import train_test_split 

def first_step(df, column_names, dataset_name):
    print(f"Info: ", df.info())
    print(f"Samples of {dataset_name}: ", df.head(5))
    print(f"Labels statistic of {dataset_name}: ", df["Category"].value_counts())

column_names_1 = ["ID", "Title", "URL", "Publisher", "Category", "Story_ID", "Domain", "Timestamp"]
df_1 = pd.read_csv('data/newsCorpora.csv', sep="\t", names=column_names_1, quoting=3, encoding="utf-8")
first_step(df_1, column_names_1, "newsCorpora")

column_names_2 = ["Story", "Hostname", "Category", "URL"]
df_2 = pd.read_csv('data/2pageSessions.csv', sep="\t", names=column_names_2, quoting=3, encoding="utf-8")
first_step(df_2, column_names_2, "2pageSessions")

sub_list =  ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
selected_list = df_1[df_1["Publisher"].isin(sub_list)]
print("Num of selected news: ", len(selected_list))
print(selected_list[:10])

df_train, df_test = train_test_split(selected_list, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

def save_data(df, name):
    df = df[["Title", "Category"]]
    df.to_csv(name, sep="\t", index=False, encoding="utf-8")
save_data(df_train, "data/train.txt")
save_data(df_val, "data/valid.txt")
save_data(df_test, "data/test.txt")