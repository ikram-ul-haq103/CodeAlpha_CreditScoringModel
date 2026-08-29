# ============================================
#           CREDIT SCORING MODEL 
# ============================================

# ===== 1. IMPORTS =====
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

# For hyperparameter tuning
from scipy.stats import randint, uniform

print("✅ All libraries imported successfully!")

# ===== 2. LOAD AND EXPLORE DATA =====
def load_and_explore_data(file_path='train_dataset_final1.csv'):
    """
    Load the dataset and perform initial exploration
    """
    print("\n" + "="*60)
    print("📊 DATA LOADING AND EXPLORATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Dataset overview
    print(f"\n📁 Dataset Shape: {df.shape}")
    print(f"\n📋 Column Names:\n{df.columns.tolist()}")
    print(f"\n🔍 First 5 rows:\n{df.head()}")
    
    # Check for missing values
    print(f"\n❓ Missing Values:\n{df.isnull().sum()}")
    
    # Statistical summary
    print(f"\n📊 Statistical Summary:\n{df.describe()}")
    
    # Check target variable distribution
    print(f"\n🎯 Target Variable Distribution:")
    print(df['next_month_default'].value_counts())
    print(f"Default Rate: {df['next_month_default'].mean()*100:.2f}%")
    
    return df

# ===== 3. FEATURE ENGINEERING =====
def engineer_features(df):
    """
    Create new features from existing data to improve model performance
    """
    print("\n" + "="*60)
    print("🔧 FEATURE ENGINEERING")
    print("="*60)
    
    # Create a copy to avoid modifying original
    df_engineered = df.copy()

    def pick_column(*names):
        for name in names:
            if name in df_engineered.columns:
                return name
        return names[0]

    # 1. Payment behavior features
    bill_cols = [pick_column('Bill_amt1', 'bill_amt1'), pick_column('Bill_amt2', 'bill_amt2'),
                 pick_column('Bill_amt3', 'bill_amt3'), pick_column('Bill_amt4', 'bill_amt4'),
                 pick_column('Bill_amt5', 'bill_amt5'), pick_column('Bill_amt6', 'bill_amt6')]
    pay_cols = [pick_column('pay_amt1', 'pay_amt1'), pick_column('pay_amt2', 'pay_amt2'),
                pick_column('pay_amt3', 'pay_amt3'), pick_column('pay_amt4', 'pay_amt4'),
                pick_column('pay_amt5', 'pay_amt5'), pick_column('pay_amt6', 'pay_amt6')]
    pay_status_cols = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    limit_col = pick_column('LIMIT_BAL', 'limit_bal')

    df_engineered['avg_bill_amt'] = df_engineered[bill_cols].mean(axis=1)
    df_engineered['avg_pay_amt'] = df_engineered[pay_cols].mean(axis=1)
    df_engineered['avg_pay_ratio'] = df_engineered['avg_pay_amt'] / (df_engineered['avg_bill_amt'] + 1)
    
    # 2. Payment status features
    df_engineered['pay_status_sum'] = df_engineered[pay_status_cols].sum(axis=1)
    df_engineered['pay_status_mean'] = df_engineered[pay_status_cols].mean(axis=1)
    df_engineered['max_delay'] = df_engineered[pay_status_cols].max(axis=1)
    df_engineered['num_delays'] = (df_engineered[pay_status_cols] > 0).sum(axis=1)
    
    # 3. Debt indicators
    df_engineered['debt_ratio'] = df_engineered['avg_bill_amt'] / (df_engineered[limit_col] + 1)
    
    # 4. Payment consistency
    df_engineered['pay_std'] = df_engineered[pay_cols].std(axis=1)
    df_engineered['bill_std'] = df_engineered[bill_cols].std(axis=1)
    
    print("✨ Created new features:")
    print(f"   - avg_bill_amt: Average monthly bill")
    print(f"   - avg_pay_amt: Average monthly payment")
    print(f"   - avg_pay_ratio: Payment to bill ratio")
    print(f"   - pay_status_sum: Total payment status")
    print(f"   - max_delay: Maximum payment delay")
    print(f"   - num_delays: Number of months with delay")
    print(f"   - debt_ratio: Debt to credit limit ratio")
    print(f"   - pay_std: Payment consistency")
    print(f"   - bill_std: Bill amount consistency")
    
    return df_engineered

# ===== 4. PREPARE DATA FOR MODELING =====
def prepare_data(df, target_col='next_month_default'):
    """
    Prepare data for machine learning models
    """
    print("\n" + "="*60)
    print("🔍 DATA PREPARATION")
    print("="*60)
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Remove ID columns if they exist
    for col in ['id', 'Customer_ID']:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Handle missing and non-finite values before modeling
    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    print(f"\n📊 Feature Matrix Shape: {X.shape}")
    print(f"🎯 Target Shape: {y.shape}")
    print(f"\n📈 Target Distribution:")
    print(f"   Non-default: {(y == 0).sum():,} ({((y == 0).sum()/len(y))*100:.2f}%)")
    print(f"   Default: {(y == 1).sum():,} ({((y == 1).sum()/len(y))*100:.2f}%)")
    
    return X, y

# ===== 5. TRAIN MODELS =====
def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models with hyperparameter tuning
    """
    print("\n" + "="*60)
    print("🤖 MODEL TRAINING")
    print("="*60)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }
    
    # Hyperparameter grids for tuning
    param_grids = {
        'Logistic Regression': {
            'C': uniform(0.1, 10),
            'penalty': ['l2'],
            'solver': ['liblinear']
        },
        'Decision Tree': {
            'max_depth': randint(3, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        },
        'Random Forest': {
            'n_estimators': randint(50, 250),
            'max_depth': randint(5, 25),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        }
    }
    
    trained_models = {}
    results = {}
    
    # Apply SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    print("\n🔄 After SMOTE resampling:")
    print(f"   Non-default: {(y_train_smote == 0).sum():,}")
    print(f"   Default: {(y_train_smote == 1).sum():,}")
    
    # Train each model
    for name, model in models.items():
        print(f"\n📚 Training {name}...")
        
        # Hyperparameter tuning
        random_search = RandomizedSearchCV(
            model, 
            param_distributions=param_grids[name],
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        random_search.fit(X_train_smote, y_train_smote)
        
        best_model = random_search.best_estimator_
        trained_models[name] = best_model
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
        
        # Store results
        results[name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'best_params': random_search.best_params_
        }
        
        print(f"   ✅ Best Parameters: {random_search.best_params_}")
        
    return trained_models, results, X_test_scaled, y_test

# ===== 6. EVALUATE MODELS =====
def evaluate_models(results, y_test):
    """
    Comprehensive model evaluation with multiple metrics
    """
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    evaluation_results = {}
    
    for name, result in results.items():
        y_pred = result['y_pred']
        y_pred_proba = result['y_pred_proba']
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        evaluation_results[name] = metrics
        
        print(f"\n🔹 {name}")
        print(f"   Best Params: {result['best_params']}")
        print(f"   Accuracy: {metrics['Accuracy']:.4f}")
        print(f"   Precision: {metrics['Precision']:.4f}")
        print(f"   Recall: {metrics['Recall']:.4f}")
        print(f"   F1-Score: {metrics['F1-Score']:.4f}")
        print(f"   ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
        # Classification Report
        print(f"\n   📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Default', 'Default']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"   📊 Confusion Matrix:")
        print(f"   [[{cm[0,0]:5d} {cm[0,1]:5d}]")
        print(f"    [{cm[1,0]:5d} {cm[1,1]:5d}]]")
    
    return evaluation_results

# ===== 7. VISUALIZATIONS =====
def create_visualizations(results, y_test, X_test_scaled, feature_names, evaluation_results, trained_models):
    """
    Create comprehensive visualizations for model analysis
    """
    print("\n" + "="*60)
    print("📈 CREATING VISUALIZATIONS")
    print("="*60)

    image_dir = os.path.join(os.path.dirname(__file__), 'images')
    os.makedirs(image_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. ROC Curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red']
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        auc_score = roc_auc_score(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{name} (AUC = {auc_score:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Credit Scoring Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Model Comparison Bar Chart
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx < len(axes):
            ax = axes[idx]
            values = [evaluation_results[name][metric] for name in results.keys()]
            ax.bar(results.keys(), values, color=['blue', 'green', 'red'])
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=15)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    # Remove empty subplot
    if len(metrics_to_plot) < len(axes):
        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Default', 'Default'],
                   yticklabels=['Non-Default', 'Default'],
                   ax=axes[idx])
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices for All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Feature Importance (Random Forest)
    if 'Random Forest' in results:
        plt.figure(figsize=(12, 8))
        
        # Get feature importance from random forest
        rf_model = trained_models['Random Forest']
        importances = rf_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        feature_names_sorted = [feature_names[i] for i in indices]
        importances_sorted = importances[indices]
        
        # Plot
        plt.barh(range(len(feature_names_sorted[:20])), 
                importances_sorted[:20], 
                color='skyblue')
        plt.yticks(range(len(feature_names_sorted[:20])), 
                  feature_names_sorted[:20])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features (Random Forest)', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(image_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    print("\n✅ Visualizations saved as:")
    print("   - roc_curves.png")
    print("   - model_comparison.png")
    print("   - confusion_matrices.png")
    print("   - feature_importance.png")

# ===== 8. PREDICTION FUNCTION =====
def predict_credit_risk(model, scaler, features):
    """
    Function to make predictions on new data
    """
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Get prediction and probability
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[:, 1]
    
    # Create result dataframe
    results_df = pd.DataFrame({
        'Prediction': ['Non-Default' if p == 0 else 'Default' for p in prediction],
        'Probability_of_Default': probability,
        'Risk_Level': pd.cut(probability, 
                           bins=[-0.01, 0.3, 0.7, 1.0],
                           labels=['Low Risk', 'Medium Risk', 'High Risk'])
    })
    
    return results_df

# ===== 9. MAIN EXECUTION =====
def main():
    """
    Main execution function
    """
    print("="*60)
    print("🏦 CREDIT SCORING MODEL - COMPLETE PIPELINE")
    print("="*60)
    
    # Load the project dataset if it exists; otherwise fall back to a demo sample
    dataset_path = os.path.join(os.path.dirname(__file__), 'train_dataset_final1.csv')
    if os.path.exists(dataset_path):
        print(f"\n📁 Loading dataset from: {dataset_path}")
        df = load_and_explore_data(dataset_path)
    else:
        print("\n⚠️  Dataset not found in the project folder. Using generated sample data instead.")
        np.random.seed(42)
        n_samples = 1000
        sample_data = pd.DataFrame({
            'id': range(n_samples),
            'limit_bal': np.random.randint(10000, 1000000, n_samples),
            'sex': np.random.choice([1, 2], n_samples),
            'age': np.random.randint(20, 60, n_samples),
            'marriage': np.random.choice([1, 2, 3], n_samples),
            'education': np.random.choice([1, 2, 3, 4], n_samples),
            'pay_0': np.random.randint(-2, 8, n_samples),
            'pay_2': np.random.randint(-2, 8, n_samples),
            'pay_3': np.random.randint(-2, 8, n_samples),
            'pay_4': np.random.randint(-2, 8, n_samples),
            'pay_5': np.random.randint(-2, 8, n_samples),
            'pay_6': np.random.randint(-2, 8, n_samples),
            'bill_amt1': np.random.randint(0, 200000, n_samples),
            'bill_amt2': np.random.randint(0, 200000, n_samples),
            'bill_amt3': np.random.randint(0, 200000, n_samples),
            'bill_amt4': np.random.randint(0, 200000, n_samples),
            'bill_amt5': np.random.randint(0, 200000, n_samples),
            'bill_amt6': np.random.randint(0, 200000, n_samples),
            'pay_amt1': np.random.randint(0, 100000, n_samples),
            'pay_amt2': np.random.randint(0, 100000, n_samples),
            'pay_amt3': np.random.randint(0, 100000, n_samples),
            'pay_amt4': np.random.randint(0, 100000, n_samples),
            'pay_amt5': np.random.randint(0, 100000, n_samples),
            'pay_amt6': np.random.randint(0, 100000, n_samples),
            'next_month_default': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        df = sample_data
        print("✅ Sample data created!")
    
    # Feature Engineering
    df_engineered = engineer_features(df)
    
    # Prepare data
    X, y = prepare_data(df_engineered)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train models
    global trained_models, evaluation_results
    trained_models, results, X_test_scaled, y_test = train_models(
        X_train, X_test, y_train, y_test
    )
    
    # Evaluate models
    evaluation_results = evaluate_models(results, y_test)
    
    # Create visualizations
    create_visualizations(results, y_test, X_test_scaled, X.columns.tolist(), evaluation_results, trained_models)
    
    # Sample prediction
    print("\n" + "="*60)
    print("🔮 SAMPLE PREDICTION")
    print("="*60)
    
    # Use the best model (Random Forest or best performing)
    best_model_name = max(evaluation_results, key=lambda x: evaluation_results[x]['ROC-AUC'])
    best_model = trained_models[best_model_name]
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit scaler on training data
    
    # Make a sample prediction
    sample_features = X_test.iloc[:5]
    predictions = predict_credit_risk(best_model, scaler, sample_features)
    
    print(f"\n🎯 Using best model: {best_model_name}")
    print(f"\n📋 Predictions for 5 sample customers:")
    print(predictions)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return trained_models, results, evaluation_results

# ===== 10. EXECUTE MAIN FUNCTION =====
if __name__ == "__main__":
    trained_models, results, evaluation_results = main()