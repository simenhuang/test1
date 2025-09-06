"""
NIPT女胎异常判定的数学模型分析
详细的数学建模方法和统计分析
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class MathematicalModel:
    """女胎异常判定的数学模型"""
    
    def __init__(self):
        self.logistic_model = None
        self.threshold_model = None
        self.composite_score_model = None
    
    def method1_logistic_regression(self, X, y):
        """
        方法1: 多元逻辑回归模型
        
        模型公式:
        P(Y=1|X) = 1 / (1 + exp(-β₀ - β₁X₁ - β₂X₂ - ... - βₚXₚ))
        
        其中:
        Y = 1 表示异常，Y = 0 表示正常
        X₁, X₂, ..., Xₚ 为各特征变量
        β₀, β₁, ..., βₚ 为回归系数
        """
        print("=== 方法1: 多元逻辑回归模型 ===")
        
        # 建立逻辑回归模型
        self.logistic_model = LogisticRegression(random_state=42)
        self.logistic_model.fit(X, y)
        
        # 获取系数
        coefficients = self.logistic_model.coef_[0]
        intercept = self.logistic_model.intercept_[0]
        
        print(f"截距项 β₀ = {intercept:.4f}")
        print("\n回归系数:")
        for i, coef in enumerate(coefficients):
            print(f"β{i+1} = {coef:.4f} (特征: {X.columns[i]})")
        
        # 模型评估
        y_pred = self.logistic_model.predict(X)
        y_pred_proba = self.logistic_model.predict_proba(X)[:, 1]
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        print(f"\n模型性能:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return y_pred_proba
    
    def method2_threshold_based(self, data):
        """
        方法2: 基于阈值的规则模型
        
        判定规则:
        1. 21号染色体: |Z₂₁| > θ₂₁
        2. 18号染色体: |Z₁₈| > θ₁₈  
        3. 13号染色体: |Z₁₃| > θ₁₃
        4. 综合判定: 任一条件满足即判定为异常
        
        其中θ为各染色体的阈值参数
        """
        print("\n=== 方法2: 基于阈值的规则模型 ===")
        
        # 设定阈值
        thresholds = {
            'Chr21_Z值': 2.5,
            'Chr18_Z值': 2.5, 
            'Chr13_Z值': 2.5
        }
        
        print("阈值设定:")
        for chr_name, threshold in thresholds.items():
            print(f"{chr_name}: |Z| > {threshold}")
        
        # 应用规则
        anomaly_flags = pd.DataFrame()
        for chr_name, threshold in thresholds.items():
            anomaly_flags[f'{chr_name}_异常'] = (np.abs(data[chr_name]) > threshold).astype(int)
        
        # 综合判定
        final_prediction = (anomaly_flags.sum(axis=1) > 0).astype(int)
        
        # 各染色体异常率
        print(f"\n各染色体异常检出率:")
        for col in anomaly_flags.columns:
            rate = anomaly_flags[col].mean()
            print(f"{col}: {rate:.3f}")
        
        print(f"综合异常率: {final_prediction.mean():.3f}")
        
        return final_prediction, anomaly_flags
    
    def method3_composite_score(self, X, y):
        """
        方法3: 综合评分模型
        
        综合评分公式:
        S = w₁·f₁(Z值) + w₂·f₂(GC含量) + w₃·f₃(读段数) + w₄·f₄(BMI) + w₅·f₅(其他)
        
        其中:
        - f₁, f₂, f₃, f₄, f₅ 为各类特征的标准化函数
        - w₁, w₂, w₃, w₄, w₅ 为权重系数
        - S > S₀ 时判定为异常
        """
        print("\n=== 方法3: 综合评分模型 ===")
        
        # 特征分组和权重
        feature_groups = {
            'Z值特征': ['Chr21_Z值', 'Chr18_Z值', 'Chr13_Z值', 'X_Z值'],
            'GC含量特征': ['Chr21_GC含量', 'Chr18_GC含量', 'Chr13_GC含量', 'X_GC含量'],
            '读段数特征': ['Chr21_读段数', 'Chr18_读段数', 'Chr13_读段数', 'X_读段数'],
            '临床特征': ['BMI', '年龄', '孕周']
        }
        
        # 计算各组特征的权重（基于随机森林特征重要性）
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
        
        # 计算组权重
        group_weights = {}
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in feature_importance.index]
            if available_features:
                group_weights[group_name] = feature_importance[available_features].sum()
        
        # 归一化权重
        total_weight = sum(group_weights.values())
        for group in group_weights:
            group_weights[group] /= total_weight
        
        print("特征组权重:")
        for group, weight in group_weights.items():
            print(f"{group}: {weight:.3f}")
        
        # 计算综合评分
        composite_scores = np.zeros(len(X))
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in X.columns]
            if available_features and group_name in group_weights:
                # 标准化特征
                group_data = X[available_features]
                standardized_data = (group_data - group_data.mean()) / group_data.std()
                
                # 计算组内评分（取绝对值的均值）
                group_score = np.abs(standardized_data).mean(axis=1)
                
                # 加权累加
                composite_scores += group_weights[group_name] * group_score
        
        # 确定阈值
        threshold = np.percentile(composite_scores, 95)  # 95%分位数作为阈值
        predictions = (composite_scores > threshold).astype(int)
        
        print(f"\n综合评分阈值: {threshold:.3f}")
        print(f"异常检出率: {predictions.mean():.3f}")
        
        return composite_scores, predictions, threshold
    
    def method4_bayesian_approach(self, X, y):
        """
        方法4: 贝叶斯方法
        
        基于贝叶斯定理:
        P(异常|特征) = P(特征|异常) × P(异常) / P(特征)
        
        假设特征服从多元正态分布
        """
        print("\n=== 方法4: 贝叶斯分类方法 ===")
        
        from sklearn.naive_bayes import GaussianNB
        
        # 训练贝叶斯分类器
        nb_model = GaussianNB()
        nb_model.fit(X, y)
        
        # 预测
        y_pred = nb_model.predict(X)
        y_pred_proba = nb_model.predict_proba(X)[:, 1]
        
        # 评估性能
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        
        print(f"贝叶斯模型性能:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        
        # 先验概率
        prior_normal = (y == 0).mean()
        prior_abnormal = (y == 1).mean()
        
        print(f"\n先验概率:")
        print(f"P(正常) = {prior_normal:.3f}")
        print(f"P(异常) = {prior_abnormal:.3f}")
        
        return y_pred_proba
    
    def ensemble_method(self, X, y):
        """
        方法5: 集成学习方法
        
        结合多个模型的预测结果:
        P_final = α₁P₁ + α₂P₂ + α₃P₃ + α₄P₄
        
        其中P₁, P₂, P₃, P₄分别为各模型的预测概率
        α₁, α₂, α₃, α₄为权重系数
        """
        print("\n=== 方法5: 集成学习方法 ===")
        
        # 获取各模型预测
        prob1 = self.method1_logistic_regression(X, y)
        prob4 = self.method4_bayesian_approach(X, y)
        
        # 随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        prob_rf = rf.predict_proba(X)[:, 1]
        
        # 集成权重（可通过交叉验证优化）
        weights = [0.3, 0.3, 0.4]  # 逻辑回归、贝叶斯、随机森林
        
        # 加权平均
        ensemble_prob = (weights[0] * prob1 + 
                        weights[1] * prob4 + 
                        weights[2] * prob_rf)
        
        # 最终预测
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        # 评估
        accuracy = accuracy_score(y, ensemble_pred)
        precision = precision_score(y, ensemble_pred)
        recall = recall_score(y, ensemble_pred)
        f1 = f1_score(y, ensemble_pred)
        
        print(f"集成模型性能:")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        
        return ensemble_prob

def statistical_analysis(data):
    """统计分析"""
    print("\n=== 统计分析 ===")
    
    # 描述性统计
    key_features = ['Chr21_Z值', 'Chr18_Z值', 'Chr13_Z值', 'X_Z值', 'BMI', '年龄', '孕周']
    available_features = [f for f in key_features if f in data.columns]
    
    desc_stats = data[available_features].describe()
    print("主要特征描述性统计:")
    print(desc_stats.round(3))
    
    # 相关性分析
    correlation_matrix = data[available_features].corr()
    print(f"\n特征相关性矩阵:")
    print(correlation_matrix.round(3))
    
    # 正态性检验
    print(f"\n正态性检验 (Shapiro-Wilk test):")
    for feature in available_features:
        if len(data[feature]) <= 5000:  # Shapiro-Wilk test limitation
            statistic, p_value = stats.shapiro(data[feature])
            print(f"{feature}: W={statistic:.4f}, p={p_value:.4f}")
    
    return desc_stats, correlation_matrix

def create_visualization_analysis():
    """创建可视化分析"""
    print("\n=== 创建可视化分析代码 ===")
    
    viz_code = '''
def visualize_results(data, predictions, probabilities):
    """可视化分析结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 染色体Z值分布
    axes[0, 0].hist(data['Chr21_Z值'], bins=30, alpha=0.7, label='Chr21')
    axes[0, 0].hist(data['Chr18_Z值'], bins=30, alpha=0.7, label='Chr18')
    axes[0, 0].hist(data['Chr13_Z值'], bins=30, alpha=0.7, label='Chr13')
    axes[0, 0].axvline(x=2.5, color='r', linestyle='--', label='阈值')
    axes[0, 0].axvline(x=-2.5, color='r', linestyle='--')
    axes[0, 0].set_title('染色体Z值分布')
    axes[0, 0].legend()
    
    # 2. BMI与异常概率关系
    axes[0, 1].scatter(data['BMI'], probabilities, alpha=0.6)
    axes[0, 1].set_xlabel('BMI')
    axes[0, 1].set_ylabel('异常概率')
    axes[0, 1].set_title('BMI与异常概率关系')
    
    # 3. 孕周与异常概率关系
    axes[0, 2].scatter(data['孕周'], probabilities, alpha=0.6)
    axes[0, 2].set_xlabel('孕周')
    axes[0, 2].set_ylabel('异常概率')
    axes[0, 2].set_title('孕周与异常概率关系')
    
    # 4. 特征相关性热图
    corr_features = ['Chr21_Z值', 'Chr18_Z值', 'Chr13_Z值', 'BMI', '年龄', '孕周']
    available_features = [f for f in corr_features if f in data.columns]
    if len(available_features) > 1:
        corr_matrix = data[available_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
        axes[1, 0].set_title('特征相关性热图')
    
    # 5. 预测概率分布
    axes[1, 1].hist(probabilities, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0.5, color='r', linestyle='--', label='判定阈值')
    axes[1, 1].set_xlabel('异常概率')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('预测概率分布')
    axes[1, 1].legend()
    
    # 6. ROC曲线（需要真实标签）
    if 'true_labels' in locals():
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        roc_auc = auc(fpr, tpr)
        axes[1, 2].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 2].set_xlabel('假阳性率')
        axes[1, 2].set_ylabel('真阳性率')
        axes[1, 2].set_title('ROC曲线')
        axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('nipt_female_anomaly_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
'''
    
    with open('/workspace/visualization_code.py', 'w') as f:
        f.write(viz_code)
    
    print("可视化代码已保存到 visualization_code.py")

if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Chr21_Z值': np.random.normal(0, 1, n_samples),
        'Chr18_Z值': np.random.normal(0, 1, n_samples),
        'Chr13_Z值': np.random.normal(0, 1, n_samples),
        'X_Z值': np.random.normal(0, 1, n_samples),
        'Chr21_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'Chr18_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'Chr13_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'X_GC含量': np.random.normal(0.4, 0.05, n_samples),
        'Chr21_读段数': np.random.randint(500, 2000, n_samples),
        'Chr18_读段数': np.random.randint(500, 2000, n_samples),
        'Chr13_读段数': np.random.randint(500, 2000, n_samples),
        'X_读段数': np.random.randint(1000, 5000, n_samples),
        'BMI': np.random.normal(28, 4, n_samples),
        '年龄': np.random.randint(20, 45, n_samples),
        '孕周': np.random.uniform(10, 25, n_samples),
    })
    
    # 生成目标变量
    y = ((np.abs(data['Chr21_Z值']) > 2.5) | 
         (np.abs(data['Chr18_Z值']) > 2.5) | 
         (np.abs(data['Chr13_Z值']) > 2.5)).astype(int)
    
    # 选择特征
    X = data[['Chr21_Z值', 'Chr18_Z值', 'Chr13_Z值', 'X_Z值', 
              'Chr21_GC含量', 'Chr18_GC含量', 'Chr13_GC含量', 'X_GC含量',
              'BMI', '年龄', '孕周']]
    
    # 运行数学模型分析
    model = MathematicalModel()
    
    # 方法1: 逻辑回归
    prob1 = model.method1_logistic_regression(X, y)
    
    # 方法2: 阈值规则
    pred2, flags2 = model.method2_threshold_based(data)
    
    # 方法3: 综合评分
    scores3, pred3, thresh3 = model.method3_composite_score(X, y)
    
    # 方法4: 贝叶斯方法
    prob4 = model.method4_bayesian_approach(X, y)
    
    # 方法5: 集成方法
    prob5 = model.ensemble_method(X, y)
    
    # 统计分析
    desc_stats, corr_matrix = statistical_analysis(data)
    
    # 创建可视化代码
    create_visualization_analysis()
    
    print("\n=== 数学模型分析完成 ===")
    print("已生成5种不同的女胎异常判定方法")
    print("包括逻辑回归、阈值规则、综合评分、贝叶斯和集成方法")