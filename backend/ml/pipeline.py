import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix

# Candidate models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class CTRPipeline:
    def __init__(self, target_column='clicked', exclude_columns=None):
        self.target_column = target_column
        self.exclude_columns = exclude_columns or ['impression_id', 'user_id', 'timestamp']
        self.preprocessor = None
        self.models = {}
        self.feature_names = []
        self.numerical_cols = []
        self.categorical_cols = []

    def analyze_dataset(self, df):
        """
        Analyzes the columns and automatically classifies them into numeric, categorical, or excluded.
        """
        all_cols = list(df.columns)
        candidate_features = [c for c in all_cols if c != self.target_column and c not in self.exclude_columns]
        
        numerical = []
        categorical = []
        
        for col in candidate_features:
            # If the column is numeric (integer or float) and doesn't look like a high-cardinality ID
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check cardinality. If numeric but only has very few distinct values, it could be encoded categorical
                if df[col].nunique() < 5:
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                categorical.append(col)
                
        self.numerical_cols = numerical
        self.categorical_cols = categorical
        
        return {
            "total_rows": len(df),
            "columns": all_cols,
            "target": self.target_column,
            "excluded": [c for c in self.exclude_columns if c in df.columns],
            "numerical_features": numerical,
            "categorical_features": categorical,
            "class_distribution": df[self.target_column].value_counts().to_dict() if self.target_column in df.columns else {}
        }

    def build_preprocessor(self):
        """
        Creates a scikit-learn ColumnTransformer pipeline for imputation, scaling, and encoding.
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ]
        )
        return self.preprocessor

    def train_and_evaluate(self, df, model_types=None, val_size=0.2, random_state=42):
        """
        Preprocesses the data, splits it, trains requested models, and returns validation results.
        """
        if model_types is None:
            model_types = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
            
        # Ensure target is present
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
            
        # Clean target: drop null target values
        df = df.dropna(subset=[self.target_column])
        
        # Determine features if not already done
        if not self.numerical_cols and not self.categorical_cols:
            self.analyze_dataset(df)
            
        X = df[self.numerical_cols + self.categorical_cols]
        y = df[self.target_column].astype(int)
        
        # Build and fit preprocessor
        self.build_preprocessor()
        X_encoded = self.preprocessor.fit_transform(X)
        
        # Extract feature names after one-hot encoding
        # This is useful for feature importance mapping
        feature_names = list(self.numerical_cols)
        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            # Get category names
            cat_features = cat_encoder.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_features)
        self.feature_names = feature_names

        # Train/Validation Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_encoded, y, test_size=val_size, random_state=random_state, stratify=y
        )
        
        results = {}
        
        # Initialize model classifiers
        model_instances = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1),
            'lightgbm': LGBMClassifier(random_state=random_state, n_jobs=-1, verbose=-1)
        }
        
        for m_type in model_types:
            if m_type not in model_instances:
                continue
                
            model = model_instances[m_type]
            try:
                # Train
                model.fit(X_train, y_train)
                self.models[m_type] = model
                
                # Predict probabilities
                y_prob = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)
                
                # Metrics
                auc = roc_auc_score(y_val, y_prob)
                loss = log_loss(y_val, y_prob)
                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                # Curves
                fpr, tpr, _ = roc_curve(y_val, y_prob)
                precision_vals, recall_vals, _ = precision_recall_curve(y_val, y_prob)
                cm = confusion_matrix(y_val, y_pred)
                
                # Feature Importances (if available)
                importances = {}
                if hasattr(model, 'feature_importances_'):
                    importances = dict(zip(self.feature_names, model.feature_importances_.tolist()))
                elif hasattr(model, 'coef_'):
                    importances = dict(zip(self.feature_names, model.coef_[0].tolist()))
                
                # Sort importances by absolute magnitude
                sorted_importances = sorted(importances.items(), key=lambda item: abs(item[1]), reverse=True)[:15]
                
                # Downsample ROC/PR coordinates to ~50 points to save network bandwidth/DB space
                step_roc = max(1, len(fpr) // 50)
                step_pr = max(1, len(precision_vals) // 50)
                
                results[m_type] = {
                    "metrics": {
                        "auc": float(auc),
                        "logloss": float(loss),
                        "accuracy": float(acc),
                        "precision": float(prec),
                        "recall": float(rec),
                        "f1_score": float(f1)
                    },
                    "charts": {
                        "roc": {
                            "fpr": fpr[::step_roc].tolist() + [fpr[-1].tolist() if isinstance(fpr[-1], np.ndarray) else float(fpr[-1])],
                            "tpr": tpr[::step_roc].tolist() + [tpr[-1].tolist() if isinstance(tpr[-1], np.ndarray) else float(tpr[-1])]
                        },
                        "pr": {
                            "precision": precision_vals[::step_pr].tolist() + [precision_vals[-1].tolist() if isinstance(precision_vals[-1], np.ndarray) else float(precision_vals[-1])],
                            "recall": recall_vals[::step_pr].tolist() + [recall_vals[-1].tolist() if isinstance(recall_vals[-1], np.ndarray) else float(recall_vals[-1])]
                        },
                        "confusion_matrix": cm.tolist()
                    },
                    "feature_importance": dict(sorted_importances)
                }
            except Exception as e:
                print(f"Failed to train model {m_type}: {str(e)}")
                results[m_type] = {"error": str(e)}
                
        return results

    def save_pipeline(self, best_model_type, output_dir):
        """
        Saves the best model and the fitted preprocessor pipeline to disk.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save fitted preprocessor
        joblib.dump(self.preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
        
        # Save best model
        best_model = self.models.get(best_model_type)
        if best_model:
            joblib.dump(best_model, os.path.join(output_dir, 'best_model.joblib'))
            
        # Save config/metadata
        meta = {
            "target_column": self.target_column,
            "exclude_columns": self.exclude_columns,
            "numerical_cols": self.numerical_cols,
            "categorical_cols": self.categorical_cols,
            "feature_names": self.feature_names,
            "best_model_type": best_model_type
        }
        joblib.dump(meta, os.path.join(output_dir, 'pipeline_metadata.joblib'))
        print(f"Successfully saved model artifacts to {output_dir}")

    def load_pipeline(self, input_dir):
        """
        Loads the preprocessor, best model, and metadata.
        """
        self.preprocessor = joblib.load(os.path.join(input_dir, 'preprocessor.joblib'))
        best_model = joblib.load(os.path.join(input_dir, 'best_model.joblib'))
        meta = joblib.load(os.path.join(input_dir, 'pipeline_metadata.joblib'))
        
        self.target_column = meta["target_column"]
        self.exclude_columns = meta["exclude_columns"]
        self.numerical_cols = meta["numerical_cols"]
        self.categorical_cols = meta["categorical_cols"]
        self.feature_names = meta["feature_names"]
        
        best_model_type = meta["best_model_type"]
        self.models[best_model_type] = best_model
        return best_model_type

    def predict_probability(self, single_row_dict, best_model_type):
        """
        Predicts click probability for a single instance (provided as a dict of features).
        """
        model = self.models.get(best_model_type)
        if not model or not self.preprocessor:
            raise ValueError("Pipeline is not loaded or model is not trained.")
            
        # Build dataframe from dict
        df = pd.DataFrame([single_row_dict])
        
        # Ensure all columns required are present (fill with NaN if missing)
        required_cols = self.numerical_cols + self.categorical_cols
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[required_cols]
        
        # Preprocess
        X_encoded = self.preprocessor.transform(df)
        
        # Predict
        prob = model.predict_proba(X_encoded)[0, 1]
        return float(prob)
