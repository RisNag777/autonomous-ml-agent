class AgentState:
    def __init__(self):
        self.dataset_profile = {}
        self.target_column = None
        self.problem_type = None #Classification or Regression
        self.models_tried = []
        self.best_model = None
        self.best_metric = None
        self.issues_detected = []
        self.iteration = 0
