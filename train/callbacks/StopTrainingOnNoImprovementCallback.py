import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class StopTrainingOnNoImprovementCallback(BaseCallback):
    def __init__(self, check_freq: int, patience: int, verbose: int = 0):
        super(StopTrainingOnNoImprovementCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.no_improvement_steps = 0
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training rewards
            training_rewards = self.locals['rewards']
            mean_reward = np.mean(training_rewards[-100])

            if self.verbose > 0:
                print(f"Mean reward: {mean_reward:.2f} - Best mean reward: {self.best_mean_reward:.2f}")

            # Check if there is an improvement
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1

            # Stop training if no improvement for `patience` times
            if self.no_improvement_steps >= self.patience:
                if self.verbose > 0:
                    print("Stopping training as there is no improvement in mean reward.")
                return False

        return True