from abc import ABC, abstractmethod

class RecommenderSystem(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def recommend(self, user_id):
        pass

