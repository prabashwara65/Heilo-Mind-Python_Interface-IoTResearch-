# optimizer.py

class BatteryOptimizer:

    def __init__(self):
        # Define thresholds
        self.low_soc_threshold = 30
        self.critical_soc_threshold = 20
        self.low_soh_threshold = 80

    def decide_action(self, predicted_soc, predicted_soh, predicted_runtime):
        """
        Decide what action to take based on predictions
        """

        if predicted_soc <= self.critical_soc_threshold:
            return "EMERGENCY_SHUTDOWN_NON_CRITICAL"

        elif predicted_soc <= self.low_soc_threshold:
            return "SHED_NON_CRITICAL_LOAD"

        elif predicted_soh <= self.low_soh_threshold:
            return "LIMIT_CHARGE_RATE"

        else:
            return "NORMAL_OPERATION"