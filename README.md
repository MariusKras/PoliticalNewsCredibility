# Unreliable Content Detection in the News

![header](pictures/header.png)

## Dataset

The [dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) for this project can be found on Kaggle, along with an additional [dataset](https://www.kaggle.com/datasets/hassanamin/textdb3?resource=download) used exclusively for testing (licensed under CC0: Data files © Original Authors).

The data contains political news articles labeled as either "Fake" or "True", meaning opinion-based or factual reporting, respectively, along with publication date, subject and title. All "True" articles in the original dataset come from Reuters, creating strong source bias. To better assess model generalization, a second dataset was introduced, which also consists of political news articles but comes from a broader range of sources.

The original dataset was found to contain repeated patterns tied to news agency names, embedded tweet links, and metadata fragments—issues that introduce data leakage. Several entries also stood out as potentially anomalous, such as tweet reposts, very short texts, and promotional content. Both datasets included empty texts and duplicate entries.

## Objectives

The use case for this project aligns with tools like Ground News, which aim to enhance media literacy by helping readers assess the reliability and bias of news sources. Ground News provides per-organization factuality ratings based on evaluations from independent organizations. This concept could be extended to per-article evaluations as a service offered to clients.

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











