import optuna
import subprocess
import pandas as pd

# Define the objective function for Optuna
def objective(trial):
    # Example hyperparameters to optimize
    alpha = trial.suggest_float('alpha', 0.1, 0.9, step = 0.1)
    trpo_command = [
        "python", "trpo.py",
        "--alpha_val", str(alpha)
    ]
    ppo_command = [
        "python", "ppo.py",
        "--alpha_val", str(alpha)
    ]
    sarsa_command = [
        "python", "sarsa.py",
        "--alpha_val", str(alpha)
    ]
    ddqn_command = [
        "python", "ddqn.py",
        "--alpha_val", str(alpha)
    ]
    fixed_command = [
        "python", "fixed.py",
        "--alpha_val", str(alpha)
    ]
    libsumo_command = [
        "export LIBSUMO_AS_TRACI=1"
    ]
    reward_sum = 0
    models = [trpo_command,sarsa_command, ddqn_command, ppo_command,]
    for model in models:
        subprocess.run(libsumo_command, shell=True)
        result = subprocess.run(model, capture_output=True, text=True)
        print(list(filter(None, result.stdout.split('\n')))[-1])
        reward_sum += round(float(list(filter(None, result.stdout.split('\n')))[-1]),2)
    return reward_sum



study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=9)