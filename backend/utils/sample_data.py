import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_ctr_data(num_rows=2000, output_path=None):
    """
    Generates a synthetic CTR prediction dataset with realistic correlations.
    """
    np.random.seed(42)
    
    # Generate features
    user_ids = [f"USR_{np.random.randint(10000, 99999)}" for _ in range(num_rows)]
    ages = np.random.randint(18, 70, size=num_rows)
    genders = np.random.choice(["M", "F", "O"], size=num_rows, p=[0.48, 0.48, 0.04])
    device_types = np.random.choice(["Mobile", "Tablet", "Desktop"], size=num_rows, p=[0.70, 0.10, 0.20])
    device_oses = []
    for dt in device_types:
        if dt == "Mobile":
            device_oses.append(np.random.choice(["Android", "iOS"], p=[0.75, 0.25]))
        elif dt == "Tablet":
            device_oses.append(np.random.choice(["Android", "iOS"], p=[0.60, 0.40]))
        else:
            device_oses.append(np.random.choice(["Windows", "MacOS", "Linux"], p=[0.70, 0.25, 0.05]))
            
    banner_positions = np.random.choice(["Top", "Bottom", "Sidebar", "Interstitial"], size=num_rows, p=[0.40, 0.20, 0.30, 0.10])
    site_categories = np.random.choice(["News", "Shopping", "Social", "Gaming", "Entertainment", "Finance"], size=num_rows, p=[0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
    
    # User historical stats
    hist_impressions = np.random.randint(5, 200, size=num_rows)
    # CTR between 2% and 25% for baseline click rate
    base_ctr = np.random.uniform(0.02, 0.25, size=num_rows)
    hist_clicks = (hist_impressions * base_ctr).astype(int)
    
    # Timestamps spread over the last 7 days
    base_time = datetime.now() - timedelta(days=7)
    timestamps = [base_time + timedelta(seconds=int(np.random.randint(0, 7 * 24 * 3600))) for _ in range(num_rows)]
    timestamps = [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps]

    # Compute probability of click based on features (correlations)
    # Start with a base log-odds
    log_odds = -2.5 # baseline low click rate (approx 7.5%)
    
    # Correlate features
    # 1. Younger users click more on Mobile/Social/Gaming
    # 2. Shopping site category has high clicks
    # 3. Interstitial position has high click rate, Sidebar has low
    # 4. Users with higher historical CTR have higher click rate
    
    for i in range(num_rows):
        # Age effect
        if ages[i] < 30:
            log_odds += 0.4
        elif ages[i] > 55:
            log_odds -= 0.3
            
        # Device/OS effect
        if device_types[i] == "Mobile":
            log_odds += 0.5
            if device_oses[i] == "iOS":
                log_odds += 0.2
        elif device_types[i] == "Desktop":
            log_odds -= 0.2
            
        # Banner position
        if banner_positions[i] == "Interstitial":
            log_odds += 0.8
        elif banner_positions[i] == "Sidebar":
            log_odds -= 0.4
            
        # Site category
        if site_categories[i] == "Shopping":
            log_odds += 0.6
        elif site_categories[i] == "Social":
            log_odds += 0.5
        elif site_categories[i] == "Finance":
            log_odds += 0.3
            
        # Historical CTR effect
        user_ctr = hist_clicks[i] / max(1, hist_impressions[i])
        log_odds += (user_ctr - 0.12) * 4.0 # centered around average CTR
        
    # Convert log-odds to probability
    probs = 1 / (1 + np.exp(-log_odds))
    # Clip to be safe
    probs = np.clip(probs, 0.01, 0.99)
    
    # Generate labels (clicks)
    clicks = np.random.binomial(1, probs)
    
    # Build dataframe
    df = pd.DataFrame({
        "impression_id": [f"IMP_{100000 + i}" for i in range(num_rows)],
        "timestamp": timestamps,
        "user_id": user_ids,
        "user_age": ages,
        "user_gender": genders,
        "device_type": device_types,
        "device_os": device_oses,
        "banner_pos": banner_positions,
        "site_category": site_categories,
        "historical_clicks": hist_clicks,
        "historical_impressions": hist_impressions,
        "clicked": clicks
    })
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated synthetic CTR dataset at: {output_path} ({num_rows} rows)")
        
    return df

if __name__ == "__main__":
    # If run directly, create a file in the data dir
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(curr_dir))
    data_path = os.path.join(project_root, "data", "sample_ctr_data.csv")
    generate_sample_ctr_data(num_rows=2500, output_path=data_path)
