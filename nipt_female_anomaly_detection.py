"""
NIPT女胎异常判定模型
问题4：女胎异常判定方法和模型

核心思路：
1. 由于女胎不携带Y染色体，主要依据21、18、13号染色体的非整倍体状态判定异常
2. 综合考虑X染色体及相关染色体的Z值、GC含量、读段数等多因素
3. 建立机器学习模型进行女胎异常综合判定
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FemaleAnomalyDetector:
    """女胎异常检测模型类"""
    
    def __init__(self):
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42, probability=True)
        }
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.best_model = None
        self.feature_importance = None
        
    def preprocess_data(self, data):
        """数据预处理"""
        # 1. 筛选女胎数据
        female_data = data[data['胎儿性别'] == '女胎'].copy()
        
        # 2. 特征工程
        features = self._create_features(female_data)
        
        # 3. 目标变量：基于21、18、13号染色体非整倍体判定异常
        target = self._create_target(female_data)
        
        return features, target, female_data
    
    def _create_features(self, data):
        """创建特征变量"""
        features = pd.DataFrame()
        
        # 基础特征
        features['BMI'] = data['BMI']
        features['年龄'] = data['年龄']
        features['身高'] = data['身高']
        features['体重'] = data['体重']
        features['孕周'] = data['孕周']
        
        # X染色体相关特征
        features['X_Z值'] = data['X_Z值']
        features['X_GC含量'] = data['X_GC含量']
        features['X_读段数'] = data['X_读段数']
        features['X_比例'] = data['X_比例']
        
        # 21号染色体特征
        features['Chr21_Z值'] = data['Chr21_Z值']
        features['Chr21_GC含量'] = data['Chr21_GC含量']
        features['Chr21_读段数'] = data['Chr21_读段数']
        features['Chr21_比例'] = data['Chr21_比例']
        
        # 18号染色体特征
        features['Chr18_Z值'] = data['Chr18_Z值']
        features['Chr18_GC含量'] = data['Chr18_GC含量']
        features['Chr18_读段数'] = data['Chr18_读段数']
        features['Chr18_比例'] = data['Chr18_比例']
        
        # 13号染色体特征
        features['Chr13_Z值'] = data['Chr13_Z值']
        features['Chr13_GC含量'] = data['Chr13_GC含量']
        features['Chr13_读段数'] = data['Chr13_读段数']
        features['Chr13_比例'] = data['Chr13_比例']
        
        # 衍生特征
        features['BMI_孕周交互'] = features['BMI'] * features['孕周']
        features['年龄_BMI交互'] = features['年龄'] * features['BMI']
        
        # Z值组合特征
        features['关键染色体Z值均值'] = (features['Chr21_Z值'] + features['Chr18_Z值'] + features['Chr13_Z值']) / 3
        features['关键染色体Z值最大'] = np.maximum.reduce([features['Chr21_Z值'], features['Chr18_Z值'], features['Chr13_Z值']])
        features['关键染色体Z值标准差'] = np.std([features['Chr21_Z值'], features['Chr18_Z值'], features['Chr13_Z值']], axis=0)
        
        # GC含量组合特征
        features['关键染色体GC含量均值'] = (features['Chr21_GC含量'] + features['Chr18_GC含量'] + features['Chr13_GC含量']) / 3
        
        # 读段数组合特征
        features['关键染色体读段数总和'] = features['Chr21_读段数'] + features['Chr18_读段数'] + features['Chr13_读段数']
        
        return features
    
    def _create_target(self, data):
        """创建目标变量"""
        # 基于21、18、13号染色体非整倍体状态判定异常
        # 假设AB列包含非整倍体信息，1表示异常，0表示正常
        target = ((data['Chr21_非整倍体'] == 1) | 
                 (data['Chr18_非整倍体'] == 1) | 
                 (data['Chr13_非整倍体'] == 1)).astype(int)
        
        return target
    
    def feature_selection(self, X, y, method='rfe', k=15):
        """特征选择"""
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
        elif method == 'rfe':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=k)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.support_]
        
        self.feature_selector = selector
        return X_selected, selected_features
    
    def train_models(self, X, y):
        """训练多个模型并选择最佳模型"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        
        for name, model in self.models.items():
            # 交叉验证
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
            model_scores[name] = cv_scores.mean()
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            print(f"{name} - CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 选择最佳模型
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = self.models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        
        # 在测试集上评估
        y_pred = self.best_model.predict(X_test_scaled)
        y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"测试集AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))
        
        return X_test_scaled, y_test, y_pred, y_pred_proba
    
    def hyperparameter_tuning(self, X, y):
        """超参数调优"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 为不同模型定义参数网格
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            }
        }
        
        best_model = None
        best_score = 0
        
        for model_name, param_grid in param_grids.items():
            print(f"调优 {model_name}...")
            grid_search = GridSearchCV(
                self.models[model_name], 
                param_grid, 
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                print(f"新的最佳模型: {model_name}, AUC: {best_score:.4f}")
        
        self.best_model = best_model
        return best_model
    
    def analyze_feature_importance(self, X, feature_names):
        """分析特征重要性"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
        else:
            return None
        
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def predict_anomaly(self, X):
        """预测女胎异常"""
        X_scaled = self.scaler.transform(X)
        
        # 预测概率
        proba = self.best_model.predict_proba(X_scaled)[:, 1]
        
        # 预测类别（可调整阈值）
        predictions = (proba >= 0.5).astype(int)
        
        return predictions, proba
    
    def risk_stratification(self, probabilities):
        """风险分层"""
        risk_levels = []
        for prob in probabilities:
            if prob < 0.1:
                risk_levels.append('低风险')
            elif prob < 0.3:
                risk_levels.append('中低风险')
            elif prob < 0.7:
                risk_levels.append('中风险')
            else:
                risk_levels.append('高风险')
        
        return risk_levels

def comprehensive_analysis():
    """综合分析函数"""
    print("=== NIPT女胎异常判定模型 ===\n")
    
    # 模拟数据生成（实际应用中替换为真实数据）
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟数据
    data = pd.DataFrame({
        '胎儿性别': ['女胎'] * n_samples,
        'BMI': np.random.normal(28, 4, n_samples),
        '年龄': np.random.randint(20, 45, n_samples),
        '身高': np.random.normal(162, 6, n_samples),
        '体重': np.random.normal(65, 10, n_samples),
        '孕周': np.random.uniform(10, 25, n_samples),
        
        # X染色体特征
        'X_Z值': np.random.normal(0, 1, n_samples),
        'X_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'X_读段数': np.random.randint(1000, 5000, n_samples),
        'X_比例': np.random.normal(0.05, 0.01, n_samples),
        
        # 21号染色体特征
        'Chr21_Z值': np.random.normal(0, 1, n_samples),
        'Chr21_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'Chr21_读段数': np.random.randint(500, 2000, n_samples),
        'Chr21_比例': np.random.normal(0.02, 0.005, n_samples),
        
        # 18号染色体特征
        'Chr18_Z值': np.random.normal(0, 1, n_samples),
        'Chr18_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'Chr18_读段数': np.random.randint(500, 2000, n_samples),
        'Chr18_比例': np.random.normal(0.02, 0.005, n_samples),
        
        # 13号染色体特征
        'Chr13_Z值': np.random.normal(0, 1, n_samples),
        'Chr13_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'Chr13_读段数': np.random.randint(500, 2000, n_samples),
        'Chr13_比例': np.random.normal(0.02, 0.005, n_samples),
    })
    
    # 生成异常标签（基于Z值的阈值规则）
    data['Chr21_非整倍体'] = (np.abs(data['Chr21_Z值']) > 2.5).astype(int)
    data['Chr18_非整倍体'] = (np.abs(data['Chr18_Z值']) > 2.5).astype(int)
    data['Chr13_非整倍体'] = (np.abs(data['Chr13_Z值']) > 2.5).astype(int)
    
    # 初始化检测器
    detector = FemaleAnomalyDetector()
    
    # 数据预处理
    X, y, processed_data = detector.preprocess_data(data)
    
    print(f"数据维度: {X.shape}")
    print(f"异常样本比例: {y.mean():.3f}")
    
    # 特征选择
    X_selected, selected_features = detector.feature_selection(X, y, method='rfe', k=15)
    print(f"\n选择的特征数量: {len(selected_features)}")
    print("选择的特征:", list(selected_features))
    
    # 训练模型
    print("\n=== 模型训练 ===")
    X_test, y_test, y_pred, y_pred_proba = detector.train_models(X_selected, y)
    
    # 超参数调优
    print("\n=== 超参数调优 ===")
    best_model = detector.hyperparameter_tuning(X_selected, y)
    
    # 特征重要性分析
    importance_df = detector.analyze_feature_importance(X_selected, selected_features)
    if importance_df is not None:
        print("\n=== 特征重要性 TOP 10 ===")
        print(importance_df.head(10))
    
    # 风险分层示例
    risk_levels = detector.risk_stratification(y_pred_proba)
    risk_distribution = pd.Series(risk_levels).value_counts()
    print(f"\n=== 风险分层分布 ===")
    print(risk_distribution)
    
    return detector, X_selected, y, selected_features

def clinical_decision_support():
    """临床决策支持系统"""
    print("\n=== 临床决策支持建议 ===")
    
    recommendations = {
        "判定标准": {
            "高风险阈值": "预测概率 ≥ 0.7",
            "中风险阈值": "0.3 ≤ 预测概率 < 0.7", 
            "低风险阈值": "预测概率 < 0.3"
        },
        
        "临床建议": {
            "高风险": [
                "建议进行羊水穿刺或绒毛活检确诊",
                "密切监测胎儿发育情况",
                "遗传咨询和心理支持"
            ],
            "中风险": [
                "建议重复NIPT检测",
                "考虑超声检查异常标志",
                "定期随访"
            ],
            "低风险": [
                "继续常规产前检查",
                "正常孕期管理"
            ]
        },
        
        "关键指标监测": [
            "21、18、13号染色体Z值变化",
            "X染色体相关参数稳定性",
            "GC含量和读段数质量",
            "BMI和孕周影响评估"
        ]
    }
    
    for category, content in recommendations.items():
        print(f"\n{category}:")
        if isinstance(content, dict):
            for key, value in content.items():
                print(f"  {key}: {value}")
        else:
            for item in content:
                print(f"  - {item}")

if __name__ == "__main__":
    # 运行综合分析
    detector, X, y, features = comprehensive_analysis()
    
    # 临床决策支持
    clinical_decision_support()
    
    print("\n=== 模型总结 ===")
    print("该女胎异常判定模型具有以下特点：")
    print("1. 综合考虑多种染色体指标和临床因素")
    print("2. 基于机器学习的智能判定方法") 
    print("3. 提供风险分层和临床决策支持")
    print("4. 可根据实际数据进一步优化和验证")