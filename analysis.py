import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Load Data
df = pd.read_csv('gym_members.csv')

# 2. Features selection (Exclude member_id)
features = ['age', 'visits_per_month', 'avg_visit_minutes', 'classes_per_month', 'months_member', 'monthly_extras_spend']

# 3. 70/30 Train/Test Split
train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)

# 4. Fit Scaler on Training Set Only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[features])
X_test_scaled = scaler.transform(test_df[features])
X_full_scaled = scaler.transform(df[features])

# 5. Evaluate k from 1 to 8 (Elbow Method)
train_inertia = []
test_inertia = []
k_range = range(1, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_scaled)
    train_inertia.append(km.inertia_)
    
    # Calculate test set inertia (squared distances to closest training center)
    dist = km.transform(X_test_scaled)
    test_inertia.append(np.sum(np.min(dist**2, axis=1)))

# 6. Plot & Save Training vs. Test Spread (Elbow Curve)
plt.figure(figsize=(8, 5))
plt.plot(k_range, train_inertia, marker='o', label='Training Set Spread', color='#1f77b4', linewidth=2)
plt.plot(k_range, test_inertia, marker='s', linestyle='--', label='Test Set Spread', color='#ff7f0e', linewidth=2)
plt.axvline(x=4, color='red', linestyle=':', label='Chosen k=4')
plt.title('K-Means Elbow Plot: Training vs Test Set Spread', fontsize=12, fontweight='bold')
plt.xlabel('Number of Clusters (k)', fontsize=10)
plt.ylabel('Sum of Squared Distances (Inertia)', fontsize=10)
plt.xticks(k_range)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('elbow_plot.png', dpi=300)
plt.close()

print("Elbow plot saved as 'elbow_plot.png'")

# 7. Fit Final Model (k=4)
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
km_final.fit(X_train_scaled)

# Assign clusters across all data
df['cluster'] = km_final.predict(X_full_scaled)

# Print Segment Summaries
print("\n--- Segment Summaries (k=4) ---")
summary = df.groupby('cluster')[features].mean()
summary['count'] = df.groupby('cluster').size()
print(summary)

# 8. Define Decision Function & Test New Member
def classify_new_member(age, visits, visit_min, classes, months, spend):
    member_data = pd.DataFrame([{
        'age': age,
        'visits_per_month': visits,
        'avg_visit_minutes': visit_min,
        'classes_per_month': classes,
        'months_member': months,
        'monthly_extras_spend': spend
    }])
    scaled_data = scaler.transform(member_data[features])
    cluster_id = km_final.predict(scaled_data)[0]
    
    # Restrict credit to Segments 0 & 1 (where average spend < $20)
    gets_credit = cluster_id in [0, 1]
    return cluster_id, gets_credit

new_cluster, gets_credit = classify_new_member(30, 4, 42, 1, 5, 10.0)
print(f"\nNew Member: Assigned to Segment {new_cluster} | Receives Credit: {gets_credit}")

# 9. Value Quantification (Member-by-Member Net Impact)
def compute_net(spend, gets_offer):
    if not gets_offer:
        return 0.0
    return -10.0 if spend >= 20.0 else 5.0

# Baseline: Everyone gets credit
df['baseline_net'] = df['monthly_extras_spend'].apply(lambda x: compute_net(x, True))

# Strategy v1: Only Segments 0 & 1 get credit
df['v1_net'] = df.apply(lambda row: compute_net(row['monthly_extras_spend'], row['cluster'] in [0, 1]), axis=1)

print("\n--- Financial Value Quantification ---")
print(f"Baseline Monthly Net (Credit to All): ${df['baseline_net'].sum():,.2f}")
print(f"v1 Targeted Monthly Net (Segments 0 & 1): ${df['v1_net'].sum():,.2f}")
print(f"Net Monthly Improvement: ${df['v1_net'].sum() - df['baseline_net'].sum():,.2f}")
