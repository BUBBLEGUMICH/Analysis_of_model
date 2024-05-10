import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    @staticmethod
    def draw_plots(dataframe, save_path="plots"):
        os.makedirs(save_path, exist_ok=True)
        plot_paths = []

        correlations = dataframe.drop(["name", "gt_corners", "rb_corners"], axis=1).corrwith(dataframe["gt_corners"])
        plt.bar(correlations.index, np.abs(correlations))
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation')
        plt.title('Feature Importance based on Correlation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        correlation_plot_path = os.path.join(save_path, "correlation_plot.png")
        plt.savefig(correlation_plot_path)
        plot_paths.append(correlation_plot_path)

        gt_rb_comparison = sns.catplot(data=dataframe, x='gt_corners', y='rb_corners', kind='box')
        gt_rb_comparison.set_axis_labels('gt_corners', 'rb_corners')
        gt_rb_comparison.set(title='Comparison of gt_corners and rb_corners')

        gt_rb_comparison_path = os.path.join(save_path, "gt_rb_comparison.png")
        gt_rb_comparison.savefig(gt_rb_comparison_path)
        plot_paths.append(gt_rb_comparison_path)

        displot = sns.displot(data=dataframe, x='floor_mean', kind='hist', bins=20, hue='gt_corners', col='rb_corners')

        displot_path = os.path.join(save_path, "displot.png")
        displot.savefig(displot_path)
        plot_paths.append(displot_path)

        gt_floor_mean = sns.catplot(data=dataframe, x='gt_corners', y='floor_mean')
        gt_ceiling_mean = sns.catplot(data=dataframe, x='gt_corners', y='ceiling_mean')
        rb_floor_mean = sns.catplot(data=dataframe, x='rb_corners', y='floor_mean')
        rb_ceiling_mean = sns.catplot(data=dataframe, x='rb_corners', y='ceiling_mean')

        gt_floor_mean_path = os.path.join(save_path, "gt_floor_mean_plot.png")
        gt_floor_mean.savefig(gt_floor_mean_path)
        plot_paths.append(gt_floor_mean_path)

        gt_ceiling_mean_path = os.path.join(save_path, "gt_ceiling_mean_plot.png")
        gt_ceiling_mean.savefig(gt_ceiling_mean_path)
        plot_paths.append(gt_ceiling_mean_path)

        rb_floor_mean_path = os.path.join(save_path, "rb_floor_mean_plot.png")
        rb_floor_mean.savefig(rb_floor_mean_path)
        plot_paths.append(rb_floor_mean_path)

        rb_ceiling_mean_path = os.path.join(save_path, "rb_ceiling_mean_plot.png")
        rb_ceiling_mean.savefig(rb_ceiling_mean_path)
        plot_paths.append(rb_ceiling_mean_path)    
            

        return plot_paths