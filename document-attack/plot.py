import pandas as pd
import matplotlib.pyplot as plt

label=["12,000"]
def plot(hit_files, miss_files, output_pdf="infer.pdf", threshold=2.1):
    # Combine all hit data
    hit_data = pd.concat([pd.read_csv(file)['response_time'] for file in hit_files], ignore_index=True)
    hit_data = hit_data[(hit_data >= 0) & (hit_data <= 3)]  # Filter for 0-6 range

    # Read miss data individually
    # miss_data_list = [pd.read_csv(file)['response_time'] for file in miss_files]

    # Calculate TPR and FPR
    hit_data = pd.DataFrame(hit_data, columns=['response_time'])
    # miss_data = pd.concat([pd.DataFrame(miss, columns=['response_time']) for miss in miss_data_list], ignore_index=True)

    hit_data["label"] = hit_data["response_time"].apply(lambda x: 1 if x < threshold else 0)
    # miss_data["label"] = miss_data["response_time"].apply(lambda x: 1 if x < threshold else 0)

    true_positive = hit_data[hit_data["label"] == 1].shape[0]
    # false_positive = miss_data[miss_data["label"] == 1].shape[0]

    tpr = true_positive / hit_data.shape[0]
    # fpr = false_positive / miss_data.shape[0]

    # Print results
    print(f"True Positive Rate (TPR): {tpr:.2f}")
    # print(f"False Positive Rate (FPR): {fpr:.2f}")

    # Plot histograms
    plt.figure(figsize=(8, 5))

    # Plot hit data as one histogram
    plt.hist(hit_data['response_time'], bins=80, alpha=0.7, label='Hit', width=0.05, color='#6baed6')  # Softer blue

    # Plot each miss data with a different color
    # colors = ['#fdae6b', '#74c476', '#de2d26']  # Softer orange, green, and red
    # for idx, miss_data in enumerate(miss_data_list):
    #     plt.hist(miss_data, bins=80, alpha=0.7, label=f'Miss {label[idx]} tokens', width=0.05, color=colors[idx])

    # Add title and labels
    plt.xlabel('Response Time (s)', fontsize=16, labelpad=5)
    plt.ylabel('Frequency', fontsize=16, labelpad=5)
    plt.xticks(fontsize=14, rotation=90)  # Rotate x-axis labels
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)

    # Tight layout to remove extra space
    plt.tight_layout()

    # Save plot to PDF
    plt.savefig(output_pdf, format='pdf')
    print(f"Plot saved to {output_pdf}")
    plt.close()
    
# Example usage
# hit_files = ["data/descriptions_hit_12000.csv", "data/descriptions_hit_18000.csv", "data/descriptions_hit_24000.csv"]
# miss_files = ["data/descriptions_miss_12000.csv", "data/descriptions_miss_18000.csv", "data/descriptions_miss_24000.csv"]
hit_files = ["data/descriptions_hit_12000.csv"]
# miss_files = ["data/descriptions_miss_12000.csv"]
miss_files = [""]
plot(hit_files, miss_files, output_pdf="response_times.pdf")