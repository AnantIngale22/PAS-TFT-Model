import pandas as pd
import numpy as np
from datetime import datetime

# ✅ Fixed seed for reproducibility
np.random.seed(42)

def generate_fixed_pas_data(
    company_id=27,
    num_physicians=50,
    months=24,
    start_date="2023-01-01"
):
    """Generate fixed, repeatable dummy PAS Forecasting data."""
    base_date = pd.to_datetime(start_date)
    months_list = pd.date_range(base_date, periods=months, freq="MS")

    data = []
    for entity_id in range(1, num_physicians + 1):
        entity_name = f"Dr. {chr(65 + (entity_id % 26))}{entity_id}"
        entity_type = "PHYSICIAN"

        # Consistent baseline spend pattern per physician
        base_spend = 10000 + entity_id * 50
        seasonality = np.sin(np.linspace(0, 3.14 * 2, months)) * 2000

        for i, date in enumerate(months_list):
            # Deterministic pattern
            spend = base_spend + seasonality[i] + (i * 15)
            spend = round(spend, 2)

            transaction_count = 10 + (entity_id % 5) + (i % 3)
            avg_transaction_value = round(spend / transaction_count, 2)

            data.append({
                "entity_id": entity_id,
                "entity_name": entity_name,
                "company_id": company_id,
                "entity_type": entity_type,
                "timestamp": date,
                "spend_amount": spend,
                "transaction_count": transaction_count,
                "avg_transaction_value": avg_transaction_value
            })

    df = pd.DataFrame(data)
    print(f"✅ Fixed PAS Forecasting Data Created!")
    print(f"Records: {len(df)} | Physicians: {num_physicians} | Periods: {months}")
    print(df.head())

    # Save to CSV for reuse
    df.to_csv("fixed_pas_forecasting_data.csv", index=False)
    print("\n📁 Saved as fixed_pas_forecasting_data.csv")
    return df


if __name__ == "__main__":
    generate_fixed_pas_data()
