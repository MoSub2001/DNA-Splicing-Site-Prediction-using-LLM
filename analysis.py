# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the data (adjust the file path as needed)
# df = pd.read_csv('log_ratio_exon_intron.csv')  # Replace with your actual CSV file path

# # 1. Average probability vs. distance
# avg_prob = df.groupby('distance')['probability'].mean().reset_index()

# plt.figure()
# plt.plot(avg_prob['distance'], avg_prob['probability'])
# plt.xlabel('Distance from Splice Site')
# plt.ylabel('Average Probability')
# plt.title('Average Probability vs. Distance')
# plt.tight_layout()
# plt.savefig('avg_prob_vs_distance.png')
# plt.show()

# # 2. Histogram of probability distribution
# plt.figure()
# plt.hist(df['probability'], bins=30)
# plt.xlabel('Probability')
# plt.ylabel('Frequency')
# plt.title('Probability Distribution')
# plt.tight_layout()
# plt.savefig('probability_histogram.png')
# plt.show()

# # 3. Top 10 kmers by probability at a specific distance (e.g., 0)
# specific_distance = 0
# if specific_distance in df['distance'].unique():
#     top_kmers = df[df['distance'] == specific_distance].nlargest(10, 'probability')
#     plt.figure()
#     plt.bar(top_kmers['kmer'], top_kmers['probability'])
#     plt.xlabel('K-mer')
#     plt.ylabel('Probability')
#     plt.title(f'Top 10 K-mers at Distance {specific_distance}')
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()
#     plt.savefig('top_kmers_distance_0.png')
#     plt.show()
# else:
#     print(f"No data for distance = {specific_distance}")

# # Inform the user
# print("Plots generated and saved to /mnt/data/:")
# print("- avg_prob_vs_distance.png")
# print("- probability_histogram.png")
# print("- top_kmers_distance_0.png (if applicable)")



import pandas as pd
import matplotlib.pyplot as plt

# Load the data (adjust the file path as needed)
df = pd.read_csv('log_ratio_exon_intron.csv')  # Replace with your actual CSV file path

# 1. Average probability vs. distance
avg_prob = df.groupby('distance')['probability'].mean().reset_index()

# 2. Histogram of probability distribution
plt.figure()
plt.hist(df['probability'], bins=30)
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution')
plt.tight_layout()
plt.savefig('probability_histogram.png')
plt.show()

# 3. Top 10 kmers by probability at a specific distance (e.g., 0)
specific_distance = 0
if specific_distance in df['distance'].unique():
    top_kmers = df[df['distance'] == specific_distance].nlargest(10, 'probability')
    plt.figure()
    plt.bar(top_kmers['kmer'], top_kmers['probability'])
    plt.xlabel('K-mer')
    plt.ylabel('Probability')
    plt.title(f'Top 10 K-mers at Distance {specific_distance}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_kmers_distance_0.png')
    plt.show()
else:
    print(f"No data for distance = {specific_distance}")

# Inform the user
print("- avg_prob_vs_distance.png")
print("- probability_histogram.png")
print("- top_kmers_distance_0.png (if applicable)")
