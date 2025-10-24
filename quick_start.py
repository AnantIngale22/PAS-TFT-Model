#!/usr/bin/env python3
"""
Quick start script for PAS Forecasting Module
Run this to get everything set up quickly
"""

import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success: {result.stdout}")
    return True

def main():
    print("🎯 PAS Forecasting Module - Quick Start")
    print("Database: anantingale@localhost:5432/forecast_model")
    
    # 1. Test database connection
    if not run_command("python test_connection.py", "Testing Database Connection"):
        print("\n❌ Database connection failed. Please check your credentials in .env file")
        return
    
    # 2. List available companies
    if not run_command("python main.py --mode list", "Listing Available Companies"):
        print("\n⚠️  Could not list companies, but continuing...")
    
    # 3. Run a quick training test
    print("\n📚 Would you like to run a quick training test?")
    print("This will train a simple model on available data.")
    response = input("Continue? (y/n): ")
    
    if response.lower() == 'y':
        if not run_command("python main.py --mode train", "Training Sample Model"):
            print("\n⚠️  Training had issues, but the setup is complete!")
    
    print("\n🎉 Setup Complete!")
    print("\nNext steps:")
    print("1. Explore data: jupyter notebook notebooks/exploration.ipynb")
    print("2. Start API: python main.py --mode api")
    print("3. Train specific company: python main.py --mode train --company-id 1")

if __name__ == "__main__":
    main()