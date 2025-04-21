import re
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, Optional
from gensim.models import Word2Vec
from langdetect import detect, LangDetectException
from wordcloud import WordCloud
from typing import Union, List
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm
from typing import List, Tuple
from lime.lime_text import LimeTextExplainer
from collections import defaultdict

def preview_text(df: pd.DataFrame, n_chars: int = 1000, display_rows: int = 1, position: str = "start") -> None:
    """Display the beginning or end of truncated text for a specified number of rows."""
    if position == "start":
        display(df.head(display_rows).assign(text=df["text"].head(display_rows).apply(lambda x: x[:n_chars])))
    elif position == "end":
        display(df.head(display_rows).assign(text=df["text"].head(display_rows).apply(lambda x: x[-n_chars:])))

def plot_wordclouds(df: pd.DataFrame, column: Literal["text", "title"]) -> None:
    """Generates and displays word clouds for True and Fake news based on the specified column."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    plt.suptitle(f"Most Frequent Words in Article {column.capitalize()}", fontsize=16, y=0.99)
    for ax, label, title in zip(axes, ["True", "Fake"], ["True News", "Fake News"]):
        text = " ".join(df[df["label"] == label][column].astype(str))
        wordcloud = WordCloud(width=500, height=400, background_color="white").generate(text)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def safe_langdetect(text: str) -> Optional[str]:
    """Returns detected language or 'unknown' if detection fails."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def compute_token_lengths(
    df: pd.DataFrame,
    feature: str,
    tokenizer_uncased,
    tokenizer_cased
) -> dict[str, pd.DataFrame]:
    """Compute tokenized sequence lengths for each label using cased and uncased tokenizers."""
    results = {}
    for label in ["True", "Fake"]:
        subset = df[df["label"] == label]
        texts = subset[feature].tolist()
        token_lengths = {
            "uncased": tokenizer_uncased(texts, truncation=False, return_length=True)["length"],
            "cased": tokenizer_cased(texts, truncation=False, return_length=True)["length"],
        }
        results[label] = pd.DataFrame(token_lengths)
    return results

def print_ngrams(label: str, ngrams: list, line_width: int = 15, max_ngrams: int = None) -> None:
    """Prints n-grams in a formatted manner."""
    print(f"\n{label}")
    if max_ngrams is not None:
        ngrams = ngrams[:max_ngrams]
    for i in range(0, len(ngrams), line_width):
        print(", ".join(ngrams[i:i + line_width]))

def pattern_exploration(df: pd.DataFrame, pattern: str, rows: Union[int, List[int]] = 1) -> pd.DataFrame:
    """Returns proportions of a text pattern by label and sample rows where the pattern appears."""
    counts = df.groupby("label")["text"].apply(lambda x: x.str.contains(pattern, case=False, na=False, regex=True).sum())
    proportions = counts / df["label"].value_counts()
    
    print("\n" + "=" * 60)
    print(f"PATTERN ANALYSIS: \"{pattern}\"")
    print("=" * 60)
    print("\nProportion of articles containing the pattern:")
    display(pd.DataFrame({"Count": counts, "Proportion": proportions}))
    
    filtered = df[df["text"].str.contains(pattern, case=False, na=False, regex=True)]
    print("\nExample rows containing the pattern:")
    
    if isinstance(rows, int):
        display(filtered.head(rows))
    else:
        display(filtered.loc[rows])

def show_short_texts(df: pd.DataFrame, max_words: int, display_rows: int) -> None:
    """Displays rows with max_words or fewer words and their label summary."""
    word_counts = df["text"].str.split().str.len()
    short_rows = df[word_counts <= max_words].copy()
    short_rows["word_count"] = word_counts[word_counts <= max_words]
    print(f"\nNumber of rows with {max_words} or fewer words: {len(short_rows)}")
    print(f"Labels present: {short_rows['label'].unique().tolist()}")
    display(short_rows.sort_values("word_count", ascending=False).head(display_rows))

def get_top_ngrams(text_series, ngram_range=(2, 3), top_n=50):
    """Extracts and returns the top n-grams and their frequencies efficiently."""
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    X = vectorizer.fit_transform(text_series)
    ngram_counts = np.array(X.sum(axis=0)).flatten()
    ngram_freq = pd.Series(ngram_counts, index=vectorizer.get_feature_names_out()).sort_values(ascending=False)
    return ngram_freq.head(top_n)

def get_ngram_freq(text_series: pd.Series, ngram_range=(2, 3), top_n=50) -> pd.Series:
    """Returns top n-grams and their frequencies."""
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words="english")
    X = vectorizer.fit_transform(text_series)
    ngram_counts = np.array(X.sum(axis=0)).flatten()
    ngram_freq = pd.Series(ngram_counts, index=vectorizer.get_feature_names_out()).sort_values(ascending=False)
    return ngram_freq.head(top_n)

def create_pattern(ngrams: list[str]) -> str:
    """Builds a regex pattern matching any of the provided n-grams."""
    return r"|".join(re.escape(ngram) for ngram in ngrams)

def true_false_classifier(fake_pattern: str, true_pattern: str):
    """Returns a classifier function that assigns 'Fake' or 'True' based on pattern matches."""
    def classify_text(text: str) -> str:
        has_fake = bool(re.search(fake_pattern, text, re.IGNORECASE))
        has_true = bool(re.search(true_pattern, text, re.IGNORECASE))
        return "Fake" if has_fake or not has_true else "True"
    return classify_text

def plot_confusion_matrix(true_labels, pred_labels, title="Confusion Matrix", ax=None):
    """Plot normalized confusion matrix with optional axis for subplotting."""
    labels = ["Fake", "True"]
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    internal_fig = None
    if ax is None:
        internal_fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        ax=ax,
        zorder=2
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)

    if internal_fig:
        plt.tight_layout()
        plt.show()

def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words."""
    return text.lower().split()

def get_avg_vector(tokens_list: list[list[str]], model: Word2Vec, vector_size: int) -> np.ndarray:
    """Compute average Word2Vec vectors for tokenized texts."""
    features = []
    for tokens in tokens_list:
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            features.append(np.mean(vectors, axis=0))
        else:
            features.append(np.zeros(vector_size))
    return np.array(features)

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> dict:
    """Return classification metrics for a binary model evaluation."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "Model": model_name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1_fake": report["0"]["f1-score"],
        "f1_true": report["1"]["f1-score"],
        "precision_fake": report["0"]["precision"],
        "precision_true": report["1"]["precision"],
        "recall_fake": report["0"]["recall"],
        "recall_true": report["1"]["recall"],
    }


class NewsDataset(Dataset):
    """Dataset for tokenized news text and binary labels."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int = 512):
        """Initialize dataset with texts, labels, tokenizer, and max length."""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize text and return input_ids, attention_mask, and label."""
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.float)
        )


def predict(loader: DataLoader, original_df: pd.DataFrame, model: DistilBertForSequenceClassification) -> pd.DataFrame:
    """Run predictions and return a DataFrame with probabilities and predicted labels."""
    all_probs, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, _ in tqdm(loader):
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    df = original_df.copy()
    df["predicted_label"] = all_preds
    df["predicted_prob_true"] = all_probs
    return df


class LimeGlobalImportanceExplainer:
    """Computes average LIME word importance across multiple samples."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        target_labels: List[str],
        texts: List[str],
    ):
        """Initialize the explainer with model, tokenizer, labels, and texts."""
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.target_labels = target_labels
        self.texts = texts
        self.lime_explainer = LimeTextExplainer(class_names=target_labels)

    def lime_predict(self, texts: List[str]) -> np.ndarray:
        """Tokenize input texts and return model-predicted probabilities."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=400, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )
            probs = torch.sigmoid(outputs.logits)
            probs = torch.cat([1 - probs, probs], dim=1)
        return probs.cpu().numpy()

    def explain_average_importance(self, num_samples: int = 100, num_features: int = 10) -> pd.DataFrame:
        """Compute average word importance over multiple random samples."""
        indices = np.random.choice(len(self.texts), size=num_samples, replace=False)
        word_scores = defaultdict(list)

        for idx in indices:
            text = self.texts[idx]
            exp = self.lime_explainer.explain_instance(
                text,
                self.lime_predict,
                num_features=num_features,
                top_labels=1,
                num_samples=100
            )
            label = exp.available_labels()[0]
            for word, weight in exp.as_list(label=label):
                word_scores[word].append(weight)

        avg_scores = {word: np.mean(weights) for word, weights in word_scores.items()}
        top_words = sorted(avg_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:num_features]
        sorted_words = sorted(top_words, key=lambda x: x[1], reverse=True)
        display(pd.DataFrame(sorted_words, columns=["Word", "Avg Importance"]))


class LimeExplainerMisclassifiedSamples:
    """Explains randomly selected misclassified samples using LIME."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        target_labels: List[str],
        texts: List[str],
        true_labels: List[int],
        predicted_labels: List[int],
    ):
        """Initializes with model, tokenizer, labels, and misclassified texts."""
        self.device = torch.device("cuda")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.target_labels = target_labels
        self.texts = texts
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def lime_predict(self, texts: List[str]) -> np.ndarray:
        """Tokenizes input texts and returns model-predicted probabilities."""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=400, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )
            probs = torch.sigmoid(outputs.logits)
            probs = torch.cat([1 - probs, probs], dim=1)
        return probs.cpu().numpy()

    def explain_random(
        self,
        num_samples: int = 3,
        num_features: int = 10,
    ):
        """Selects random misclassified samples and explains them with LIME."""
        import textwrap

        indices = np.random.choice(len(self.texts), size=num_samples, replace=False)
        explainer = LimeTextExplainer(class_names=self.target_labels)

        for idx in indices:
            text = self.texts[idx]
            true_label_idx = self.true_labels[idx]
            predicted_label_idx = self.predicted_labels[idx]

            true_label = self.target_labels[true_label_idx]
            pred_label = self.target_labels[predicted_label_idx]

            wrapped_text = "\n".join(textwrap.wrap(text[:600] + "...", width=200))
            print(f"True Label: {true_label} | Predicted Label: {pred_label}")
            print(f"Text:\n{wrapped_text}\n")

            exp = explainer.explain_instance(
                text,
                self.lime_predict,
                num_features=num_features,
                num_samples=120,
            )
            results = []
            for class_idx in exp.available_labels():
                explanation = exp.as_list(label=class_idx)
                for token, score in explanation:
                    results.append({"Token": token, "LIME Score": score})
            df_results = pd.DataFrame(results).sort_values(
                by="LIME Score", ascending=False
            )
            display(df_results)


def preview_text_with_pattern(df: pd.DataFrame, n_chars: int = 1000, position: str = "start") -> pd.DataFrame:
    """Returns a copy with truncated text column."""
    if position == "start":
        return df.assign(text=df["text"].apply(lambda x: x[:n_chars]))
    elif position == "end":
        return df.assign(text=df["text"].apply(lambda x: x[-n_chars:]))
    else:
        raise ValueError("position must be 'start' or 'end'")

def pattern_exploration(
    df: pd.DataFrame,
    pattern: str,
    rows: Union[int, List[int]] = 1,
    n_chars: int = 1000,
    position: str = "start"
) -> None:
    """Displays pattern proportions and sample truncated rows."""
    counts: pd.Series = df.groupby("label")["text"].apply(
        lambda x: x.str.contains(pattern, case=False, na=False, regex=True).sum()
    )
    proportions: pd.Series = counts / df["label"].value_counts()

    print("\n" + "=" * 60)
    print(f'PATTERN ANALYSIS: "{pattern}"')
    print("=" * 60)
    print("\nProportion of articles containing the pattern:")
    display(pd.DataFrame({"Count": counts, "Proportion": proportions}))

    filtered: pd.DataFrame = df[df["text"].str.contains(pattern, case=False, na=False, regex=True)]
    print("\nExample rows containing the pattern:")

    if isinstance(rows, int):
        preview: pd.DataFrame = preview_text_with_pattern(filtered.head(rows), n_chars=n_chars, position=position)
    else:
        preview = preview_text_with_pattern(filtered.loc[rows], n_chars=n_chars, position=position)

    display(preview)
