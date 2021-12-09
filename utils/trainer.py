from sklearn.linear_model import LogisticRegression


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return LogisticRegression(class_weight={0:.14, 1:.86}, solver='liblinear', warm_start=True, multi_class='auto').fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
