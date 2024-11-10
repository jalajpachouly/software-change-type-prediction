# Multiclass Text Classification on Software Defect Reports

This repository contains a Python script that performs multiclass text classification on software defect reports. The script addresses class imbalance using SMOTE and compares the performance of various machine learning models, including BERT, on both balanced and unbalanced data. It also generates several plots to visualize the results and helps in understanding the data.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Understanding the Script](#understanding-the-script)
  - [Imports and Dependencies](#imports-and-dependencies)
  - [Loading the Dataset](#loading-the-dataset)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Handling Class Imbalance](#handling-class-imbalance)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [BERT Implementation](#bert-implementation)
- [Generated Plots](#generated-plots)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

In software development, efficiently classifying and triaging software defect reports is crucial for timely resolution. However, defect datasets often suffer from class imbalance, making accurate classification challenging. This script aims to:

- Perform multiclass text classification on software defect reports.
- Address class imbalance using SMOTE and oversampling.
- Compare the performance of traditional machine learning models and BERT.
- Generate plots to visualize class distribution, model performance, and word clouds.

## Features

- **Data Preprocessing**: Cleans and preprocesses text data using NLTK.
- **Class Imbalance Handling**: Uses SMOTE for traditional ML models and oversampling for BERT.
- **Multiple Models**: Implements Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, and BERT.
- **Evaluation Metrics**: Calculates accuracy, precision, recall, and F1-score.
- **Visualization**: Generates plots for class distribution, model performance, and word clouds for each class.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the required packages**:

   Ensure you have Python 3.x installed. Install the dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

   If `requirements.txt` is not available, you can install the packages manually:

   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud nltk torch transformers scikit-learn imbalanced-learn tqdm
   ```

3. **Download NLTK resources**:

   The script requires specific NLTK datasets. You can download them using:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

1. **Prepare the dataset**:

   - The script expects a CSV file named `dataset.csv` in the working directory.
   - The CSV should have at least two columns: `report` (text data) and `target` (class labels).

2. **Run the script**:

   ```bash
   python main.py
   ```

3. **Outputs**:

   - The script will generate several plots saved as PNG files in the working directory.
   - Evaluation metrics will be printed to the console.
   - Word clouds for each class will be saved as `wordcloud_{class_name}.png`.

## Data Preparation

Ensure your `dataset.csv` is formatted correctly:

- **Columns**:
  - `report`: The textual content of the software defect report.
  - `target`: The class label associated with the report.

Example of the CSV file:

| report                                             | target        |
|----------------------------------------------------|---------------|
| Description of the first software defect report... | changetype_A  |
| Description of the second software defect report...| changetype_B  |
| ...                                                | ...           |

## Understanding the Script

### Imports and Dependencies

The script begins by importing the necessary libraries, including data handling libraries (`pandas`, `numpy`), visualization libraries (`matplotlib`, `seaborn`), NLP libraries (`nltk`), machine learning models (`scikit-learn`), and handling class imbalance (`imbalanced-learn`). It also installs missing packages if necessary.

### Loading the Dataset

```python
data = pd.read_csv('dataset.csv')
```

- Loads the dataset from `dataset.csv`.
- Displays the first few rows and checks for missing values.

### Data Preprocessing

The script preprocesses the text data by:

1. **Lowercasing** the text.
2. **Removing punctuation and numbers**.
3. **Tokenizing** the text into words.
4. **Removing stopwords**.
5. **Lemmatizing** the words.

A custom function `preprocess_text` is defined to perform these steps, and it's applied to the entire dataset.

### Exploratory Data Analysis

- **Class Distribution**: Plots the original class distribution to visualize class imbalance.
- **Word Clouds**: Generates word clouds for each class to visualize the most frequent terms.

### Handling Class Imbalance

- **For Traditional ML Models**:
  - **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to balance the dataset.
  - Plots are generated to show the training data before and after applying SMOTE.

- **For BERT**:
  - Since SMOTE cannot be applied directly to text data, oversampling is used by duplicating instances of the minority classes.
  - Plots are generated to show the BERT training data before and after oversampling.

### Model Training and Evaluation

The script trains and evaluates the following models:

1. **Logistic Regression**
2. **Linear SVM**
3. **Multinomial Naive Bayes**
4. **Random Forest**

For each model:

- Trains on both balanced and unbalanced data.
- Predicts on the test set.
- Calculates evaluation metrics.
- Collects metrics into dataframes for visualization.

### BERT Implementation

- **Tokenization**: Uses `BertTokenizer` to tokenize the text data.
- **Data Preparation**: Prepares torch tensors for input IDs, attention masks, and labels.
- **Data Loaders**: Creates data loaders for training and testing.
- **Model Training**:
  - Loads the pre-trained `BertForSequenceClassification` model.
  - Trains the model using AdamW optimizer and a linear scheduler.
  - Implements gradient clipping.
- **Evaluation**:
  - Evaluates the model after each epoch.
  - Collects accuracy and classification reports.

### Visualization of Results

- **Evaluation Metrics Plots**:
  - Box plots and bar plots are generated for both balanced and unbalanced data.
  - Includes metrics from all models, including BERT.

- **Word Clouds**:
  - Word clouds for each class are saved as images.

## Generated Plots

The script generates and saves the following plots:

1. **Class Distribution**:
   - `original_class_distribution.png`

2. **Training Data Distribution**:
   - `training_data_before_smote.png`
   - `training_data_after_smote.png`
   - `bert_training_data_before_oversampling.png`
   - `bert_training_data_after_oversampling.png`

3. **Evaluation Metrics**:
   - Box plots and bar plots for balanced and unbalanced data (filenames can be modified within the plotting functions).

4. **Word Clouds**:
   - `wordcloud_{class_name}.png` for each class.

## Results

- **Printed Outputs**:
  - The script prints the classification reports for each model, showing precision, recall, F1-score, and support for each class.
  - Overall accuracy for each model is displayed.

- **Metrics Dataframes**:
  - Metrics collected in dataframes can be used for further analysis or visualization.

- **Observations**:
  - The use of SMOTE and oversampling improves model performance on imbalanced datasets.
  - BERT generally outperforms traditional models, especially when class imbalance is addressed.

## Acknowledgments

- **Libraries and Tools**:
    - [Imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn)
  - [NLTK - Natural Language Toolkit](https://www.nltk.org/)
  - [Scikit-learn](https://scikit-learn.org/)
- **Datasets**:
  - Ensure that you have the rights and permissions to use and distribute any datasets.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: Before using or distributing this script, ensure that you have the necessary rights and permissions for any datasets and that you comply with all applicable licenses.

# Instructions for Running the Script

1. **Ensure Dependencies are Installed**:

   The script uses several Python libraries. Install them using pip:

   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud nltk torch transformers scikit-learn imbalanced-learn tqdm
   ```

2. **Prepare the Dataset**:

   - Place `dataset.csv` in the same directory as the script.
   - The dataset should contain the `report` and `target` columns.

3. **Run the Script**:

   - Execute the script from the command line:

     ```bash
     python cd_final.py
     ```

   - The script will process the data, train models, evaluate them, and generate plots.

4. **Review Outputs**:

   - Check the console for printed evaluation metrics.
   - Generated plots and word clouds will be saved in the working directory.

# Customization

- **Adjusting Models and Parameters**:

  - You can modify the models or their hyperparameters in the `models` dictionary.
  - Adjust the number of epochs, batch sizes, and learning rates for BERT in the `train_evaluate_bert` function.

- **Data Preprocessing**:

  - Modify the `preprocess_text` function to change how text data is cleaned and tokenized.

- **Visualization**:

  - Customize the plotting functions to change styles, labels, or save paths.

# Troubleshooting

- **ModuleNotFoundError**:

  - If you encounter a `ModuleNotFoundError`, ensure all dependencies are installed.

- **NLTK Resource Errors**:

  - If NLTK resources are not found, run:

    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

- **CUDA Errors**:

  - If you're using GPU acceleration and encounter CUDA errors, ensure that PyTorch is installed with CUDA support and your GPU drivers are up to date.

# Contributing

Contributions are welcome! Please open an issue or submit a pull request for any bugs, improvements, or suggestions.

# Contact

For any inquiries or support, please contact [your-email@example.com](mailto:your-email@example.com).

---

**Disclaimer**: This script is for educational purposes. Ensure compliance with all applicable laws and regulations when using and distributing datasets and models.

# Shortcuts

- **To Re-run with Different Data**:

  - Replace `dataset.csv` with your own dataset.
  - Ensure the columns `report` and `target` are present.

- **To Use a Different Language Model**:

  - Replace `'bert-base-uncased'` with another pre-trained model in the `BertTokenizer.from_pretrained` and `BertForSequenceClassification.from_pretrained` functions.

- **To Adjust Class Balancing Techniques**:

  - Modify or replace SMOTE and oversampling methods as needed.

# Potential Improvements

- **Implement Cross-Validation**:

  - Use cross-validation for more robust evaluation.

- **Hyperparameter Tuning**:

  - Implement grid search or randomized search to find the best hyperparameters.

- **Add More Models**:

  - Experiment with other models like XGBoost or other transformer architectures.

- **Enhance Data Visualization**:

  - Incorporate more advanced visualization tools or interactive plots.

---

Happy coding! If you find this script useful, please consider giving the repository a star ⭐.

# Matched Tags

- Software
- Machine Learning
- Data Science
- Python
- Natural Language Processing
- Data Visualization
- Text Classification
- BERT
- Class Imbalance
- SMOTE

# References

1. [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
2. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
3. [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)

---

Please feel free to reach out if you have any questions or need further assistance.