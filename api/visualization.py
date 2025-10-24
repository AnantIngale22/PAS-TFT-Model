import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import base64

def create_forecast_charts(forecast_data):
    """Generate forecast visualization charts"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    entities = forecast_data['predictions']['entity_forecasts']
    
    # 1. Entity Forecast Comparison
    entity_names = [e['entity_name'] for e in entities]
    totals = [e['total_forecasted'] for e in entities]
    
    ax1.bar(entity_names, totals, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_title('Total Forecast by Entity', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Forecasted Amount ($)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Period Trends
    periods = list(range(1, len(entities[0]['period_forecasts']) + 1))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for i, entity in enumerate(entities):
        forecasts = [p['point_forecast'] for p in entity['period_forecasts']]
        ax2.plot(periods, forecasts, marker='o', linewidth=3, 
                color=colors[i % len(colors)], label=entity['entity_name'])
    
    ax2.set_title('Forecast Trends by Period', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Period')
    ax2.set_ylabel('Forecasted Amount ($)')
    ax2.set_xlim(0.8, max(periods) + 0.2)
    ax2.set_xticks(periods)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confidence Intervals
    period_breakdown = forecast_data['predictions']['period_breakdown']
    periods = [p['period'] for p in period_breakdown]
    totals = [p['total_forecast'] for p in period_breakdown]
    
    # Calculate confidence bounds (assuming ±10% for visualization)
    lower = [t * 0.9 for t in totals]
    upper = [t * 1.1 for t in totals]
    
    ax3.fill_between(periods, lower, upper, alpha=0.3, color='#2E86AB', label='Confidence Range')
    ax3.plot(periods, totals, marker='o', color='#2E86AB', linewidth=2, label='Forecast')
    ax3.set_title('Total Forecast with Confidence Range', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Period')
    ax3.set_ylabel('Total Forecast ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Financial Impact
    financial = forecast_data['predictions']['summary']['financial_summary']
    categories = ['Discounts', 'Rebates', 'PAS Fees']
    values = [financial['total_discounts'], financial['total_rebates'], financial['total_pas_fees']]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    ax4.pie(values, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Financial Impact Breakdown', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save as image file
    plt.savefig('forecast_chart.png', dpi=300, bbox_inches='tight')
    plt.close()  # Remove plt.show() - it blocks the server