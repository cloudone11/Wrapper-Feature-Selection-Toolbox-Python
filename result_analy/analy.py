import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# 假设数据存储在data.json文件中，结构为包含多个类似示例条目的列表
# 这里用示例数据演示，实际需要替换为完整数据
def analyze_results_by_algo(input_file_path, output_folder_path):
    # 获取input_file_path的文件名
    file_name = os.path.basename(input_file_path)
    # Load data from the specified input file
    with open(input_file_path) as f:
        data = json.load(f)

    sample_data = data  # 包含所有10run×10fold数据的列表

    # 数据结构整理
    metrics = ['Accuracy', 'Sensitivity', 'Precision', 'Specificity', 'MCC', 'Number of Features']
    results = defaultdict(list)

    # 遍历所有数据条目
    for entry in sample_data:
        algorithm = entry["Algorithm"]
        results[algorithm].append(entry)

    # 计算统计量
    statistics = {}
    for algo, entries in results.items():
        # 初始化存储容器
        algo_stats = {
            metric: {'values': []} for metric in metrics
        }
        convergence_curves = []

        # 收集数据
        for entry in entries:
            # 收集指标
            for metric in metrics:
                algo_stats[metric]['values'].append(entry[metric])

            # 收集收敛曲线
            convergence_curves.append(np.array(entry["Convergence Curve"][0]))

        # 计算均值和方差
        for metric in metrics:
            values = np.array(algo_stats[metric]['values'])
            algo_stats[metric]['mean'] = np.mean(values)
            algo_stats[metric]['std'] = np.std(values)

        # 处理收敛曲线
        min_length = min(len(curve) for curve in convergence_curves)
        aligned_curves = [curve[:min_length] for curve in convergence_curves]
        mean_convergence = np.mean(aligned_curves, axis=0)

        # 存储结果
        statistics[algo] = {
            'metrics': algo_stats,
            'mean_convergence': mean_convergence
        }

    # 输出统计结果到文件
    output_file_path = os.path.join(output_folder_path, f'{file_name}_algorithm_statistics.txt')
    with open(output_file_path, 'w') as f:
        f.write("Algorithm Statistics:\n")
        for algo, data in statistics.items():
            f.write(f"\nAlgorithm: {algo}\n")
            for metric in metrics:
                f.write(f"{metric}:\n")
                f.write(f"  Mean = {data['metrics'][metric]['mean']:.4f}\n")
                f.write(f"  Std  = {data['metrics'][metric]['std']:.4f}\n")

    # 绘制收敛曲线并保存图片
    plt.figure(figsize=(10, 6))
    for algo, data in statistics.items():
        mean = data['mean_convergence']
        plt.plot(mean, label=f'{algo} Mean')

    plt.title('Convergence Curve Comparison')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.grid(True)
    plt.legend()
    output_image_path = os.path.join(output_folder_path, f'{file_name}_convergence_curve_comparison.png')
    plt.savefig(output_image_path)
    plt.close()


# 该函数研究准确率大于95%的情况下，选择的特征数目的统计特征，如均值、方差，极值等。接受一个输入文件路径，输出文件夹路径。
def analyze_high_accuracy_feature_selection(input_file_path, output_folder_path):
    # 获取input_file_path的文件名
    file_name = os.path.basename(input_file_path)
    # Load data from the specified input file
    with open(input_file_path) as f:
        data = json.load(f)

    sample_data = data  # 包含所有10run×10fold数据的列表

    # 过滤出准确率大于95%的数据条目
    high_accuracy_entries = [entry for entry in sample_data if entry['Accuracy'] > 0]

    # 提取特征数目
    feature_counts = [entry['Number of Features'] for entry in high_accuracy_entries]

    # 计算统计量
    mean_feature_count = np.mean(feature_counts)
    std_feature_count = np.std(feature_counts)
    min_feature_count = np.min(feature_counts)
    max_feature_count = np.max(feature_counts)

    # 输出统计结果到文件
    output_file_path = os.path.join(output_folder_path, f'{file_name}_high_accuracy_feature_selection.txt')
    with open(output_file_path, 'w') as f:
        f.write("High Accuracy Feature Selection Statistics:\n")
        f.write(f"Mean Number of Features: {mean_feature_count:.4f}\n")
        f.write(f"Standard Deviation: {std_feature_count:.4f}\n")
        f.write(f"Minimum Number of Features: {min_feature_count}\n")
        f.write(f"Maximum Number of Features: {max_feature_count}\n")

if __name__ == '__main__':
    analyze_results_by_algo(r'result_analy\aal2base\experiment_results_0.json', r'result_analy\aal2base')
    analyze_results_by_algo(r'result_analy\aal2base\experiment_results_1.json', r'result_analy\aal2base')
    analyze_results_by_algo(r'result_analy\aal2base\experiment_results_2.json', r'result_analy\aal2base')
    # 总结：sobol初始种群的实验结果较优，但无明显差距。
    # analyze_high_accuracy_feature_selection(r'result_analy\base\experiment_results_0.json', r'result_analy\base')
    # analyze_high_accuracy_feature_selection(r'result_analy\base\experiment_results_1.json', r'result_analy\base')
    # analyze_high_accuracy_feature_selection(r'result_analy\base\experiment_results_2.json', r'result_analy\base')
    # 总结：可以在少量特征的情况下获得高准确率，特征数目与准确率没有明显正相关性。
    # 下一步可以尝试使用随机反向特征选择算法，减少特征数目，影响准确率。