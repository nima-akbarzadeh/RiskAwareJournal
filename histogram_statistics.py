import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    base_path = './planning-infinite'
    file_name = 'res_inf.xlsx'
    df = pd.read_excel(f'{base_path}/{file_name}')
    print(df.keys())
    target_labels = ['RI_Obj_RiskAware_to_Neutral'] 
    
    for target_label in target_labels:
        # y = df[target_label]
        y = df[df[target_label] <= 400][target_label]

        print(f'Mean = {y.mean()}')
        min_val = y.min()
        print(f'Min = {min_val}')
        max_val = y.max()
        print(f'Max = {max_val}')
        print(f"Portion below zero: {sum(y.values < 0)/len(y)}")

        # Plot the histogram
        bins = list(np.linspace(min_val, max_val, num=15))
        plt.hist(y, bins=bins, edgecolor='black')

        # Format the x-axis to have one decimal place
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        plt.xticks(bins)

        plt.xticks(fontsize=14, fontweight='bold', rotation=90)
        plt.yticks(fontsize=14, fontweight='bold')

        # Reduce the whitespace between bins
        plt.hist(y, bins=bins, edgecolor='black', linewidth=0.5, color='blue')

        plt.grid(axis='y')
        plt.xlabel('Relative Improvement', fontsize=14, fontweight='bold')
        plt.ylabel('Frequency', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = f'{base_path}/histogram_plot_{target_label}.png'
        plt.savefig(output_path)
        print(f"Histogram saved to {output_path}")
        plt.show()
