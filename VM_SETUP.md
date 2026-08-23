# How to Run on Any Cloud Virtual Machine (VM / VPS)

As an AI coding agent, I operate within a secure sandbox environment and do not have access or permissions to create cloud accounts or provision external virtual machines (such as AWS EC2, DigitalOcean Droplets, or GCP Instances) on your behalf.

However, **this codebase is fully containerized and virtual-environment ready**, allowing you to deploy and run it on any cloud VM or Linux server in just 3 quick steps.

---

## 🚀 Setup Steps on Your Cloud VM (Ubuntu / Debian)

### Step 1: Clone or Copy the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### Step 2: Set Up Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run the Multi-Timeframe Neural Network Pipeline

#### 1️⃣ Run on Synthetic Data (10 Months Train / 10 Months Test)
```bash
python main.py --epochs 15
```

#### 2️⃣ Run on Real Market Data (e.g. BTC-USD, ETH-USD)
```bash
python main.py --real --symbol BTC-USD --epochs 15
```

#### 3️⃣ Run with Risk Management (Stop-Loss % & Risk-Reward Ratio)
```bash
# SL = 1% (0.01), Risk-Reward = 1:2.0 (TP = 2%)
python main.py --real --symbol BTC-USD --stop_loss 0.01 --risk_reward 2.0 --epochs 15
```

---

## 🧪 Run Unit Tests
To verify all system components on your VM:
```bash
PYTHONPATH=. pytest tests/
```
