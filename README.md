# Movie Genre Classification Project - CodeSoft Machine Learning Internship



## Overview

This Genre Classification project, developed as Task 1 for the CodeSoft Machine Learning Internship, focuses on predicting movie genres based on their descriptions. The project includes data cleaning, text preprocessing, and the implementation of machine learning models to achieve accurate genre predictions.

## Dataset
The dataset, acquired from Kaggle, fuels the Movie Genre Classification Project. It contains movie titles, descriptions, and genres, serving as the foundation for training machine learning models. The primary goal is to predict movie genres based on their descriptions, enhancing user experience and content categorization.
**Dataset Link:** [Movie Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

## Project Structure

- **Data:** The project utilizes datasets for training and testing, containing movie titles, descriptions, and genres.

- **Notebooks:** 
  - `movie_genre_classification.ipynb`: Code for loading, cleaning, and exploring the dataset. Implementation of text preprocessing, feature extraction, model training, and evaluation.

- **Results:** Folder to store any output files, visualizations, or model checkpoints generated during the analysis.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Keerthana82/codsoft-task3-movie-genre-classification.git
   ```

2. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Navigate to the relevant notebook (`01_data_preprocessing.ipynb`) and run them sequentially.

3. **Install Dependencies:**
   ```bash
   pip install pandas numpy seaborn matplotlib nltk scikit-learn
   ```

## Project Workflow

1. **Data Cleaning and Exploration:**
   - Load and clean the dataset, explore the distribution of genres.

2. **Text Preprocessing:**
   - Perform text cleaning, including removing mentions, URLs, and stop words.

3. **Feature Extraction:**
   - Utilize TF-IDF (Term Frequency-Inverse Document Frequency) for converting text data into numerical features.

4. **Model Training and Evaluation:**
   - Implement Logistic Regression and Multinomial Naive Bayes models.
   - Evaluate model performance using metrics such as accuracy, classification report, and confusion matrix.

5. **Acknowledgments:**
   - Express gratitude to CodeSoft for providing the internship task.

## Acknowledgments

- **CodeSoft:** Thank you for the opportunity to work on this intriguing Genre Classification project during the Machine Learning Internship. Your support and guidance have been invaluable in enhancing practical machine learning skills.

