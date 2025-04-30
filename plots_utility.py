import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    gamma = 0.5
    o = 8

    def f(total_rew):
        return 1 - gamma ** (-1 / o) * np.maximum(0, gamma - total_rew) ** (1 / o)

    def g(total_rew):
        numerator = 1 + np.exp(-o * (1 - gamma))
        denominator = 1 + np.exp(-o * (total_rew - gamma))
        return numerator / denominator

    def h(total_rew):
        return [1 if j >= gamma else 0 for j in total_rew]

    J_values = np.linspace(0, 1, 400)
    f_values = f(J_values)
    g_values = g(J_values)
    h_values = h(J_values)

    def save_plot(y_values, filename):
        plt.figure(figsize=(10, 6))
        plt.plot(J_values, y_values, linewidth=8)
        plt.axvline(x=gamma, color='red', linestyle='--', linewidth=1.5, label='Threshold')
        plt.xlabel('Total Reward', fontsize=20, fontweight='bold')
        plt.ylabel('Utility Value', fontsize=20, fontweight='bold')
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{filename}.pdf", format='pdf')   # Also works with .eps
        plt.close()

    save_plot(f_values, "u2_o8")
    save_plot(g_values, "u3_o8")
    save_plot(h_values, "u1_o8")
