import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def count_word(x):
    return len(x.split())
def visualize_label(name):
    df = pd.read_csv(f"data/{name}.txt", sep="\t", quoting=3, encoding="utf-8")
    label_count = df['Category'].value_counts()
    plt.figure(figsize=(6, 8))
    ax = sns.barplot(x=label_count.index, y=label_count.values)

    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",  
                    (p.get_x() + p.get_width() / 2, p.get_height()),  
                    ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")
    plt.xlabel("Labels", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(f"Label Distribution in {name} Data", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(f"plot/{name}_label_count.png", dpi=300, bbox_inches="tight")
    plt.figure(figsize=(6, 8))

    df['len'] = df['Title'].apply(count_word)
    sns.histplot(df["len"], bins=30, kde=True)

    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Text Lengths in f{name} Data", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(f"plot/{name}_text_length.png", dpi=300, bbox_inches="tight")
    print("Min text length: ", df['len'].min())
    print("Max text length: ", df['len'].max())


for name in ["train", "valid", "test"]:
    visualize_label(name)


