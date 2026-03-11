from optimization.optimizer import BatteryOptimizer

optimizer = BatteryOptimizer()

# Simulated predictions from your ML model
predicted_soc = 28
predicted_soh = 85
predicted_runtime = 120  # minutes

decision = optimizer.decide_action(predicted_soc, predicted_soh, predicted_runtime)

print("Optimization Decision:", decision)