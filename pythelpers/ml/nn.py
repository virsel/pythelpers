class StepLRScheduler:
    """Simple step learning rate scheduler.
    
    Decreases learning rate by a factor after a specified number of epochs.
    """
    
    def __init__(self, initial_lr, step_size=50, gamma=0.7, min_lr=1e-6):
        """Initialize the scheduler.
        
        Args:
            initial_lr (float): Initial learning rate
            step_size (int): Number of epochs between lr updates
            gamma (float): Factor by which to decrease learning rate
            min_lr (float): Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.current_lr = initial_lr
        
    def step(self, epoch):
        """Update learning rate based on epoch.
        
        Args:
            epoch (int): Current training epoch
            
        Returns:
            float: Updated learning rate
        """
        if epoch > 0 and epoch % self.step_size == 0:
            self.current_lr = max(self.current_lr * self.gamma, self.min_lr)
        
        return self.current_lr