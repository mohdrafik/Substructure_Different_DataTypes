import numpy as np

data = np.random.randint(0,10,(1,10))  # Normal-distributed random numbers
print(f"data is :{data}")
# Quartiles (4-tiles)
quartiles = np.quantile(data, [0.25, 0.5, 0.75])
print("Quartiles:", quartiles)

# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import RobustScaler

# # Generate sample data with outliers
# np.random.seed(42)
# normal_data = np.random.normal(50, 5, 1000)
# outliers = np.random.normal(100, 1, 20)
# data = np.concatenate([normal_data, outliers]).reshape(-1, 1)

# # Apply Robust Scaler
# scaler = RobustScaler()
# data_robust_scaled = scaler.fit_transform(data)

# # Plot original vs robust scaled
# # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# x1 = range(len(data))
# # First scatter plot

# ax1.scatter(x1,data, color='blue', label='Data 1')
# ax1.set_title("Scatter Plot 1")
# ax1.set_xlabel("X")
# ax1.set_ylabel("Y")
# ax1.legend()

# x2 = range(len(data_robust_scaled))
# # Second scatter plot
# ax2.scatter(x2,data_robust_scaled , color='green', label='Data 2')
# ax2.set_title("Scatter Plot 2")
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")
# ax2.legend()

# plt.tight_layout()
# plt.show()


# # sns.histplot(data.ravel(), bins=50, kde=True, ax=axs[0])
# # axs[0].set_title('Original Data with Outliers')
# # axs[0].set_xlabel('Value')

# # sns.histplot(data_robust_scaled.ravel(), bins=50, kde=True, ax=axs[1])
# # axs[1].set_title('Robust Scaled Data')
# # axs[1].set_xlabel('Scaled Value')

# plt.tight_layout()
# plt.show()
