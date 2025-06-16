# Unreliable Content Detection in the News

![header](pictures/header.png)

## Dataset

This project uses two political news datasets from Kaggle:

- [Original Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets): Contains 23,481 "Fake" articles and 21,417 "True" articles, totaling 44,898 entries. "True" refers to factual reporting, while "Fake" reflects opinion-based content. Each article includes a title, main text, subject category, and publication date. All "True" articles are sourced from Reuters, introducing a strong source bias. The "Fake" articles come from various less credible websites or unknown publishers.

- [Second Dataset](https://www.kaggle.com/datasets/hassanamin/textdb3?resource=download): Used exclusively for testing. It includes 6,335 political news articles labeled as "Fake" or "True", drawn from a wider range of sources. Unlike the original dataset, it is class-balanced.

The original dataset shows repeated patterns tied to news agency names, embedded tweet links, and metadata fragments—issues that introduce data leakage. Some entries also appear anomalous, including tweet reposts, promotional content, and very short texts. Both datasets include empty texts and duplicate entries.

Both datasets are released under the CC0 license (Data files © Original Authors).

## Objectives

This project focuses on one aspect of what platforms like [Ground News](https://ground.news/) offer: evaluating the reliability of news content. While Ground News assigns factuality and bias ratings at the outlet level, this project aims to assess reliability at the article level, offering more granular insight into individual news stories.

The main objective of the project is:

> **Develop a model that can classify political news articles as "Fake" or "True," where "Fake" refers to emotionally charged or opinion-based content, and "True" denotes neutral, fact-based reporting.**

The objective was approached through the following steps:

- Cleaning the data by identifying and removing missing texts, duplicates, and metadata patterns that could introduce data leakage or distort model training.
- Establishing a rule-based baseline to assess initial classification performance, followed by training TF-IDF-based machine learning models.
- Fine-tuning a transformer model by training only the classifier head to reduce the risk of overfitting to biased patterns in the original dataset.
- Analyzing feature importance and misclassifications to better understand model behavior and its limitations.

## Results

On the original test set, the LightGBM model achieved a PR AUC of 0.9990 and accuracy of 0.9864, while the DistilBERT model also reached 0.9995 PR AUC and 0.9920 accuracy. However, both models performed significantly worse on the new test set. LightGBM dropped to 0.5975 PR AUC and 0.5348 accuracy, while DistilBERT performed better with 0.7152 PR AUC and 0.6816 accuracy.

> **The original dataset is not suitable for solving the task—it is too biased, with all true articles coming from a single source. Models trained on it do not learn to detect fake news, but rather to recognize writing style or origin, achieving near-perfect scores without learning to detect actual fake content. Without adding more diverse sources, especially for true news, this problem cannot be meaningfully addressed.**

Models tested were Logistic Regression, Random Forest, XGBoost, and LightGBM. LightGBM outperformed the others and was selected as the final machine learning model. A grid search showed that the best results were obtained using a combination of unigrams and bigrams with 20,000 TF-IDF features.

For the deep learning approach, DistilBERT was used. Optimal performance was retained at 400 tokens, matching the lowest validation loss achieved at the full 512-token input length. I also tested unfreezing the last transformer layer, which further improved performance on the original data and reduced on the new test set.











