# CREDIT-CARD-FRAUD-DETECTION (SECUREPAY AI)

### Description of an AI-Driven Financial Fraud Detection System Using Unsupervised Machine Learning in Python

In modern digital finance, millions of transactions occur every second through online banking, payment gateways, and FinTech platforms. Among these legitimate transactions, fraudulent activities represent a small but highly damaging fraction. Detecting such fraud in real time is challenging because fraudulent patterns constantly evolve and occur very rarely compared to normal transactions. SECUREPAY AI is an AI-driven anomaly detection system that uses unsupervised machine learning techniques to identify suspicious financial transactions by learning the patterns of normal behavior and flagging deviations as potential fraud.

The system is designed to simulate how real-world FinTech companies monitor transactions continuously to prevent financial losses and protect customer trust.

### Conceptual Framework

The anomaly detection process in SECUREPAY AI is composed of several stages, each using specialized tools and methodologies.

### Environment Setup and Library Installation

The project is implemented in Python and utilizes key data science libraries including Pandas for data processing, Matplotlib for visualization, and Scikit-learn for machine learning model implementation. These tools enable efficient data manipulation, model training, and evaluation of anomaly detection performance.

### Data Collection and Preparation

The system uses a credit card transaction dataset containing features such as transaction Time, Amount, and anonymized numerical variables (V1–V28) generated through Principal Component Analysis (PCA) to preserve privacy. Each record represents a financial transaction labeled as either normal or fraudulent.

To ensure robustness against extreme values, the RobustScaler technique is applied to the transaction amount feature. Unlike standard scaling methods, RobustScaler relies on the interquartile range, making it resistant to outliers and suitable for financial data.

```python
scaler = RobustScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
```

### Unsupervised Anomaly Detection Modeling

Since fraudulent transactions are extremely rare, the system adopts an unsupervised learning approach that focuses on modeling normal transaction behavior.

### Two anomaly detection algorithms are implemented:

Isolation Forest: A tree-based algorithm that isolates anomalies by randomly partitioning data points. Transactions that are isolated faster are more likely to be fraudulent.

```python
model = IsolationForest(n_estimators=100, contamination=0.002)
model.fit(X_train)
```

Local Outlier Factor (LOF): A density-based algorithm that detects anomalies by comparing the local density of a transaction to that of its nearest neighbors.

```python
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.002)
```

These models identify transactions that significantly deviate from the normal data distribution.

### Anomaly Scoring and Fraud Detection

Each transaction is assigned an anomaly score indicating how unusual it is compared to the baseline behavior learned by the model. Transactions exceeding a defined threshold are classified as potential fraud.

This approach allows the system to detect previously unseen fraud patterns, which is crucial in dynamic financial environments.

### Performance Evaluation and Threshold Optimization

Due to the extreme class imbalance in financial datasets, traditional accuracy metrics are not reliable. Instead, the system evaluates model performance using Precision, Recall, and F1-Score.

A Precision-Recall Curve is used to determine the optimal anomaly threshold that balances fraud detection with false alarms.

```python
precision, recall, thresholds = precision_recall_curve(y_test, anomaly_scores)
```

This analysis helps determine the best operating point for real-world fraud monitoring.

### Result Analysis and Fraud Monitoring

The system generates evaluation metrics and confusion matrices to analyze model performance. By examining detected anomalies, analysts can identify suspicious transaction patterns and investigate potential fraud cases.

The final system provides insights such as fraud detection rates, false positive rates, and recommended threshold settings for deployment in financial monitoring systems.

### Applications and Extensions

SECUREPAY AI can be extended with advanced techniques such as real-time transaction monitoring, deep learning-based anomaly detection, and streaming data processing. The system may also integrate with banking APIs or FinTech dashboards to provide automated fraud alerts and risk scoring for financial institutions.

### Conclusion

SECUREPAY AI demonstrates the effective use of unsupervised machine learning techniques for financial anomaly detection. By modeling normal transaction behavior and identifying deviations, the system provides a scalable and intelligent approach to detecting fraudulent activities. The project highlights how AI-driven analytics can enhance financial security, reduce fraud risk, and support decision-making in modern FinTech ecosystems.
