import numpy as np
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import cv2 as cv
from sklearn import metrics
from Image_Result import Image_Results, Early_Image_Results, Septoria_leaf_spot_Image_Results


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


no_of_dataset = 1



def plot_Con_results():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'EOA-AA-MRCNN', 'SGO-AA-MRCNN', 'TOA-AA-MRCNN', 'POA-AA-MRCNN', 'FAVO-AA-MRCNN']
    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for n in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = Statistical(Fitness[n, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('------------------------------ Statistical Report Dataset', n + 1,
              '------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[n]

        plt.plot(length, Conv_Graph[0, :], color='r', marker='.', markerfacecolor='#0165fc', linewidth=3, markersize=12,
                 label='EOA-AA-MRCNN')
        plt.plot(length, Conv_Graph[1, :], color='#0165fc', marker='.', markerfacecolor='lime', linewidth=3,
                 markersize=12, label='SGO-AA-MRCNN')
        plt.plot(length, Conv_Graph[2, :], color='b', marker='.', markerfacecolor='#fe2f4a', linewidth=3, markersize=12,
                 label='TOA-AA-MRCNN')
        plt.plot(length, Conv_Graph[3, :], color='m', marker='.', markerfacecolor='#ffff14', linewidth=3, markersize=12,
                 label='POA-AA-MRCNN')
        plt.plot(length, Conv_Graph[4, :], color='k', marker='.', markerfacecolor='black', linewidth=3, markersize=12,
                 label='FAVO-AA-MRCNN')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (n + 1))
        plt.show()


def ROC_curve():
    lw = 2
    cls = ['CNN', 'ResNet', 'DenseNet', 'MobileNet', 'MRMNet']
    colors = cycle(["c", "m", "k", "green", "r"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for i in range(len(Actual)):  # For all Datasets
        Dataset = ['Dataset1', 'Dataset2']
        for j, color in zip(range(5), colors):  # For all classifiers
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/%s_ROC.png" % (Dataset[i])
        plt.savefig(path)
        plt.show()


def Plot_Activation():
    eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score', 'MCC', 'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Table_Term = [0, 2, 3, 5, 10, 11, 13, 14, 15, 16, 20]
    positive_metrices = [0, 1, 2, 3, 7, 9, 10]
    negative_metrices = [4, 5, 6, 8]
    Batchsize = ['Linear', 'ReLU', 'TanH', 'Softmax', 'Sigmoid', 'Leaky ReLU']
    Graph_Term = np.arange(len(Terms))
    Classifier = ['TERMS', 'CNN', 'ResNet', 'DenseNet', 'MobileNet', 'MRMNet']
    for i in range(eval.shape[0]):
        for k in range(eval.shape[1]):
            value = eval[i, k, :, 4:]
            Table = PrettyTable()
            Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[j, Table_Term])
            print('-------------------------------------------------- ', str(Batchsize[k]), ' Activation ',
                  'Classifier Comparison of Dataset', i + 1,
                  '--------------------------------------------------')
            print(Table)

    for i in range(eval.shape[0]):
        Graph = np.zeros((eval.shape[2], eval.shape[3] - 4))
        for l in range(eval.shape[2]):
            for j in range(len(Graph_Term)):
                Graph[l, j] = eval[i, 4, l, Graph_Term[j] + 4]

        Activation = np.arange(len(positive_metrices))
        colors = ['lime', '#ff000d', '#aaff32', '#0804f9', '#cb00f5']
        mtd_Graph = Graph[:, positive_metrices]
        Methods = ['CNN', 'ResNet', 'DenseNet', 'MobileNet', 'MRMNet']
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        triangle_width = 0.15
        for y, algorithm in enumerate(Methods):
            for x, batch_size in enumerate(Activation):
                x_pos = x + y * triangle_width
                value = mtd_Graph[y, x]
                color = colors[y]
                triangle_points = np.array(
                    [[x_pos - triangle_width / 2, 0], [x_pos + triangle_width / 2, 0], [x_pos, value]])
                triangle = Polygon(triangle_points, ec='k', closed=True, color=color, linewidth=1)
                ax.add_patch(triangle)
                ax.plot(x_pos, value, marker='.', color=color, markersize=10)

        ax.set_xticks(np.arange(len(Activation)) + (len(Methods) * triangle_width) / 2)
        ax.set_xticklabels([str(size) for size in Activation])
        ax.legend([Line2D([0], [0], color=color, lw='4') for color in colors], Methods, loc='upper center',
                  bbox_to_anchor=(0.5, 1.14), ncol=3)
        plt.xticks(Activation + 0.2, ('Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'NPV', 'F1_score', 'MCC'),
                   rotation=10, fontsize=9, fontname="Arial", fontweight='bold')
        plt.ylabel('Positive Measures (%)', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        path = "./Results/Activation_Positive_Dataset_%s_bar.png" % (i + 1)
        plt.savefig(path)
        plt.show()

        Activation = np.arange(len(negative_metrices))
        colors = ['#02c14d', 'b', '#be03fd', '#de0c62', 'k']
        mtd_Graph = Graph[:, negative_metrices]
        Methods = ['CNN', 'ResNet', 'DenseNet', 'MobileNet', 'MRMNet']
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        triangle_width = 0.15
        for y, algorithm in enumerate(Methods):
            for x, batch_size in enumerate(Activation):
                x_pos = x + y * triangle_width
                value = mtd_Graph[y, x]
                color = colors[y]
                triangle_points = np.array(
                    [[x_pos - triangle_width / 2, 0], [x_pos + triangle_width / 2, 0], [x_pos, value]])
                triangle = Polygon(triangle_points, ec='k', closed=True, color=color, linewidth=1)
                ax.add_patch(triangle)
                ax.plot(x_pos, value, marker='.', color=color, markersize=10)

        ax.set_xticks(np.arange(len(Activation)) + (len(Methods) * triangle_width) / 2)
        ax.set_xticklabels([str(size) for size in Activation])
        ax.legend([Line2D([0], [0], color=color, lw='4') for color in colors], Methods, loc='upper center',
                  bbox_to_anchor=(0.5, 1.14), ncol=3)

        plt.xticks(Activation + 0.2, ('FPR', 'FNR', 'FOR', 'FDR'),
                   rotation=10, fontsize=9, fontname="Arial", fontweight='bold')  #
        plt.ylabel('Negative Measures (%)', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        path = "./Results/Activation_Negative_Dataset_%s_bar.png" % (i + 1)
        plt.savefig(path)
        plt.show()


def plot_results_Seg():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']
    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]
        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i]) * 100
                    stats[i, j, 1] = np.min(value_all[j][:, i]) * 100
                    stats[i, j, 2] = np.mean(value_all[j][:, i]) * 100
                    stats[i, j, 3] = np.median(value_all[j][:, i]) * 100
                    stats[i, j, 4] = np.std(value_all[j][:, i]) * 100

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 0, :], color='#0cff0c', edgecolor='k', width=0.10, label="EOA-AA-MRCNN")  # r
            ax.bar(X + 0.10, stats[i, 1, :], color='#ff028d', edgecolor='k', width=0.10, label="SGO-AA-MRCNN")  # g
            ax.bar(X + 0.20, stats[i, 2, :], color='#0165fc', edgecolor='k', width=0.10, label="TOA-AA-MRCNN")  # b
            ax.bar(X + 0.30, stats[i, 3, :], color='m', edgecolor='k', width=0.10, label="POA-AA-MRCNN")  # m
            ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="FAVO-AA-MRCNN")  # k
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_Seg_%s_Alg.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            plt.show()

            X = np.arange(stats.shape[2])
            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            ax.bar(X + 0.00, stats[i, 5, :], color='#0cff0c', edgecolor='k', width=0.10, label="Unet")  # r
            ax.bar(X + 0.10, stats[i, 6, :], color='#ff028d', edgecolor='k', width=0.10, label="Res-Unet")  # g
            ax.bar(X + 0.20, stats[i, 7, :], color='#0165fc', edgecolor='k', width=0.10, label="Trans-Unet")  # b
            ax.bar(X + 0.30, stats[i, 8, :], color='m', edgecolor='k', width=0.10, label="AA-MRCNN")  # m
            ax.bar(X + 0.40, stats[i, 4, :], color='k', edgecolor='k', width=0.10, label="FAVO-AA-MRCNN")  # k
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            path = "./Results/Dataset_%s_Seg_%s_mtd.png" % (n + 1, Terms[i - 4])
            plt.savefig(path)
            plt.show()



if __name__ == '__main__':
    plot_Con_results()
    ROC_curve()
    Plot_Activation()
    plot_results_Seg()
    Image_Results()
    Early_Image_Results()
    Septoria_leaf_spot_Image_Results()



