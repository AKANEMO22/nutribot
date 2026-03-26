import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

class QuestionFilter:
    def __init__(self, model_path=None):
        self.pipeline = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, data_path, text_column, label_column, save_path=None):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        if text_column not in df.columns or label_column not in df.columns:
            raise ValueError(f"Columns '{text_column}' or '{label_column}' not found in the dataset.")
            
        X = df[text_column].fillna('').astype(str)
        y = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int).clip(0, 1)

        if y.nunique() < 2:
            raise ValueError("Training data must contain both safe (0) and dangerous (1) labels.")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        
        # Use CV-based hyper-parameter search to reduce overfitting on small datasets.
        base_pipeline = Pipeline([
            (
                'tfidf',
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    sublinear_tf=True,
                ),
            ),
            (
                'clf',
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight='balanced',
                    solver='liblinear',
                ),
            ),
        ])

        param_grid = [
            {
                'tfidf__analyzer': ['word'],
                'tfidf__ngram_range': [(1, 2), (1, 3)],
                'tfidf__min_df': [1, 2],
                'clf__C': [0.5, 1.0, 2.0, 4.0],
            },
            {
                'tfidf__analyzer': ['char_wb'],
                'tfidf__ngram_range': [(3, 5), (3, 6)],
                'tfidf__min_df': [1, 2],
                'clf__C': [0.5, 1.0, 2.0],
            },
        ]

        cv_folds = max(3, min(5, int(y_train.value_counts().min())))
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        print("Training with cross-validation (TF-IDF + Logistic Regression)...")
        search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring='f1',
            cv=cv,
            n_jobs=1,
            refit=True,
            verbose=0,
        )
        search.fit(X_train, y_train)
        self.pipeline = search.best_estimator_
        
        print("Evaluating the model...")
        predictions = self.pipeline.predict(X_test)
        danger_f1 = f1_score(y_test, predictions, zero_division=0)
        print(f"Best params: {search.best_params_}")
        print(f"Best CV F1: {search.best_score_:.4f}")
        print(classification_report(y_test, predictions))
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(f"Danger F1: {danger_f1:.4f}")
        
        if save_path:
            self.save_model(save_path)

    def save_model(self, path):
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Model successfully saved to {path}")

    def load_model(self, path):
        import joblib

        self.pipeline = joblib.load(path)
        print(f"Model loaded from {path}")

    def is_dangerous(self, question):
        if not self.pipeline:
            raise ValueError("Model is not trained or loaded.")
        
        prediction = self.pipeline.predict([question])[0]
        # Assuming label '1' or True implies a dangerous/toxic question
        return bool(prediction)

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Question Filter using TF-IDF and Logistic Regression")
    
    # Subcommands for train and predict to avoid hardcoding params
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run the filter in", required=True)
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to the training dataset (CSV format)")
    train_parser.add_argument("--text_col", required=True, help="Column name containing the text data")
    train_parser.add_argument("--label_col", required=True, help="Column name containing labels (0=safe, 1=dangerous)")
    train_parser.add_argument("--model_path", required=True, help="Path where the trained model should be saved")
    
    # Predict parser
    predict_parser = subparsers.add_parser("predict", help="Predict if a question is dangerous")
    predict_parser.add_argument("--model_path", required=True, help="Path to the saved model (.pkl)")
    predict_parser.add_argument("--query", required=True, help="The question string to test")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        q_filter = QuestionFilter()
        q_filter.train(args.data, args.text_col, args.label_col, args.model_path)
        
    elif args.mode == "predict":
        q_filter = QuestionFilter(model_path=args.model_path)
        result = q_filter.is_dangerous(args.query)
        status = "DANGEROUS" if result else "SAFE"
        print(f"\nQuestion: '{args.query}'\nClassification: {status}")