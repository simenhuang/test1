"""
NIPT女胎异常判定模型 - 简化演示版本
不依赖外部包的基础实现
"""

import math
import random

class SimpleFemaleAnomalyDetector:
    """简化版女胎异常检测器"""
    
    def __init__(self):
        self.weights = None
        self.threshold = 0.5
    
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0 if x < 0 else 1
    
    def logistic_regression(self, features, labels):
        """简单的逻辑回归实现"""
        n_features = len(features[0])
        n_samples = len(features)
        
        # 初始化权重
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features + 1)]
        
        learning_rate = 0.01
        epochs = 1000
        
        print("开始训练逻辑回归模型...")
        
        for epoch in range(epochs):
            total_error = 0
            
            for i in range(n_samples):
                # 前向传播
                z = self.weights[0]  # 偏置项
                for j in range(n_features):
                    z += self.weights[j + 1] * features[i][j]
                
                prediction = self.sigmoid(z)
                error = labels[i] - prediction
                total_error += abs(error)
                
                # 反向传播
                self.weights[0] += learning_rate * error
                for j in range(n_features):
                    self.weights[j + 1] += learning_rate * error * features[i][j]
            
            if epoch % 200 == 0:
                avg_error = total_error / n_samples
                print(f"Epoch {epoch}, Average Error: {avg_error:.4f}")
        
        print("模型训练完成!")
        return self.weights
    
    def predict(self, features):
        """预测函数"""
        if self.weights is None:
            raise ValueError("模型未训练，请先调用logistic_regression方法")
        
        predictions = []
        probabilities = []
        
        for sample in features:
            z = self.weights[0]  # 偏置项
            for j in range(len(sample)):
                z += self.weights[j + 1] * sample[j]
            
            prob = self.sigmoid(z)
            pred = 1 if prob >= self.threshold else 0
            
            predictions.append(pred)
            probabilities.append(prob)
        
        return predictions, probabilities
    
    def threshold_based_method(self, chr21_z, chr18_z, chr13_z):
        """基于阈值的判定方法"""
        threshold = 2.5
        
        chr21_abnormal = abs(chr21_z) > threshold
        chr18_abnormal = abs(chr18_z) > threshold  
        chr13_abnormal = abs(chr13_z) > threshold
        
        # 任一染色体异常则判定为异常
        is_abnormal = chr21_abnormal or chr18_abnormal or chr13_abnormal
        
        return {
            'chr21_abnormal': chr21_abnormal,
            'chr18_abnormal': chr18_abnormal,
            'chr13_abnormal': chr13_abnormal,
            'overall_abnormal': is_abnormal
        }
    
    def composite_score_method(self, features):
        """综合评分方法"""
        # 特征权重（根据临床经验设定）
        weights = {
            'chr21_z': 0.3,
            'chr18_z': 0.25,
            'chr13_z': 0.2,
            'x_z': 0.1,
            'bmi': 0.08,
            'age': 0.04,
            'gestational_week': 0.03
        }
        
        scores = []
        for sample in features:
            score = 0
            feature_names = ['chr21_z', 'chr18_z', 'chr13_z', 'x_z', 'bmi', 'age', 'gestational_week']
            
            for i, value in enumerate(sample):
                if i < len(feature_names):
                    # 对Z值取绝对值，其他特征标准化处理
                    if 'z' in feature_names[i]:
                        normalized_value = abs(value) / 3.0  # 假设3倍标准差内
                    else:
                        normalized_value = value / 100.0  # 简单标准化
                    
                    score += weights[feature_names[i]] * normalized_value
            
            scores.append(score)
        
        return scores

def generate_sample_data():
    """生成示例数据"""
    random.seed(42)
    n_samples = 200
    
    print("生成示例数据...")
    
    features = []
    labels = []
    
    for i in range(n_samples):
        # 生成特征
        chr21_z = random.normalvariate(0, 1)
        chr18_z = random.normalvariate(0, 1)
        chr13_z = random.normalvariate(0, 1)
        x_z = random.normalvariate(0, 1)
        bmi = random.normalvariate(28, 4)
        age = random.randint(20, 45)
        gestational_week = random.uniform(10, 25)
        
        feature_vector = [chr21_z, chr18_z, chr13_z, x_z, bmi, age, gestational_week]
        features.append(feature_vector)
        
        # 生成标签（基于阈值规则）
        is_abnormal = (abs(chr21_z) > 2.5) or (abs(chr18_z) > 2.5) or (abs(chr13_z) > 2.5)
        labels.append(1 if is_abnormal else 0)
    
    return features, labels

def evaluate_model(true_labels, predictions):
    """评估模型性能"""
    tp = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

def risk_stratification(probabilities):
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

def main():
    """主函数"""
    print("=== NIPT女胎异常判定模型演示 ===\n")
    
    # 生成示例数据
    features, labels = generate_sample_data()
    
    print(f"数据集大小: {len(features)} 个样本")
    print(f"特征维度: {len(features[0])}")
    print(f"异常样本比例: {sum(labels)/len(labels):.3f}\n")
    
    # 初始化检测器
    detector = SimpleFemaleAnomalyDetector()
    
    # 方法1: 逻辑回归
    print("=== 方法1: 逻辑回归模型 ===")
    weights = detector.logistic_regression(features, labels)
    print(f"模型权重: {[round(w, 4) for w in weights]}")
    
    predictions, probabilities = detector.predict(features)
    metrics = evaluate_model(labels, predictions)
    
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")
    
    # 方法2: 阈值规则
    print(f"\n=== 方法2: 基于阈值的规则模型 ===")
    threshold_predictions = []
    
    for sample in features:
        result = detector.threshold_based_method(sample[0], sample[1], sample[2])
        threshold_predictions.append(1 if result['overall_abnormal'] else 0)
    
    threshold_metrics = evaluate_model(labels, threshold_predictions)
    print(f"准确率: {threshold_metrics['accuracy']:.4f}")
    print(f"精确率: {threshold_metrics['precision']:.4f}")
    print(f"召回率: {threshold_metrics['recall']:.4f}")
    print(f"F1分数: {threshold_metrics['f1_score']:.4f}")
    
    # 方法3: 综合评分
    print(f"\n=== 方法3: 综合评分模型 ===")
    composite_scores = detector.composite_score_method(features)
    
    # 使用95%分位数作为阈值
    sorted_scores = sorted(composite_scores)
    score_threshold = sorted_scores[int(0.95 * len(sorted_scores))]
    composite_predictions = [1 if score > score_threshold else 0 for score in composite_scores]
    
    composite_metrics = evaluate_model(labels, composite_predictions)
    print(f"评分阈值: {score_threshold:.4f}")
    print(f"准确率: {composite_metrics['accuracy']:.4f}")
    print(f"精确率: {composite_metrics['precision']:.4f}")
    print(f"召回率: {composite_metrics['recall']:.4f}")
    print(f"F1分数: {composite_metrics['f1_score']:.4f}")
    
    # 风险分层
    print(f"\n=== 风险分层分析 ===")
    risk_levels = risk_stratification(probabilities)
    risk_counts = {}
    for level in risk_levels:
        risk_counts[level] = risk_counts.get(level, 0) + 1
    
    print("风险分层分布:")
    for level, count in risk_counts.items():
        percentage = count / len(risk_levels) * 100
        print(f"{level}: {count} 例 ({percentage:.1f}%)")
    
    # 临床建议
    print(f"\n=== 临床决策建议 ===")
    high_risk_count = risk_counts.get('高风险', 0)
    medium_risk_count = risk_counts.get('中风险', 0) + risk_counts.get('中低风险', 0)
    low_risk_count = risk_counts.get('低风险', 0)
    
    print(f"高风险患者 ({high_risk_count} 例): 建议羊水穿刺或绒毛活检确诊")
    print(f"中风险患者 ({medium_risk_count} 例): 建议重复NIPT检测，考虑超声检查")
    print(f"低风险患者 ({low_risk_count} 例): 继续常规产前检查")
    
    # 模型比较
    print(f"\n=== 模型性能比较 ===")
    print(f"{'方法':<15} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8}")
    print("-" * 50)
    print(f"{'逻辑回归':<15} {metrics['accuracy']:<8.4f} {metrics['precision']:<8.4f} {metrics['recall']:<8.4f} {metrics['f1_score']:<8.4f}")
    print(f"{'阈值规则':<15} {threshold_metrics['accuracy']:<8.4f} {threshold_metrics['precision']:<8.4f} {threshold_metrics['recall']:<8.4f} {threshold_metrics['f1_score']:<8.4f}")
    print(f"{'综合评分':<15} {composite_metrics['accuracy']:<8.4f} {composite_metrics['precision']:<8.4f} {composite_metrics['recall']:<8.4f} {composite_metrics['f1_score']:<8.4f}")
    
    print(f"\n=== 结论 ===")
    print("1. 逻辑回归模型提供了良好的可解释性和预测性能")
    print("2. 阈值规则简单直观，适合临床快速判定")
    print("3. 综合评分方法能够整合多种因素，提供更全面的评估")
    print("4. 建议在实际应用中结合多种方法，提高判定的准确性和可靠性")

if __name__ == "__main__":
    main()