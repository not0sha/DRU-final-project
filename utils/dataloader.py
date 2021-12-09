import numpy as np
from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # replace value
        # self.dataset['ever_married'].loc[(self.dataset['ever_married'] == 'Yes')] = 1
        # self.dataset['ever_married'].loc[(self.dataset['ever_married'] == 'No')] = 0

        # fill Nan with values from random distribution
        bmi_avg = self.dataset['bmi'].mean()
        bmi_std = self.dataset['bmi'].std()
        bmi_null_count = self.dataset['bmi'].isnull().sum()
        rng = np.random.RandomState(42)
        bmi_null_random_list = rng.uniform(bmi_avg - bmi_std, bmi_avg + bmi_std, size=bmi_null_count)
        self.dataset['bmi'][np.isnan(self.dataset['bmi'])] = bmi_null_random_list

        # dropping outliers
        # self.dataset.drop(self.dataset[self.dataset['gender'] == 'Other'].index, inplace=True)

        # drop columns
        self.dataset = self.dataset.drop(['id'], axis=1)

        # scaling numerical variables
        self.dataset['age'] = (self.dataset['age'] - self.dataset['age'].mean()) / self.dataset['age'].std()
        self.dataset['bmi'] = (self.dataset['bmi'] - self.dataset['bmi'].mean()) / self.dataset['bmi'].std()
        self.dataset['avg_glucose_level'] = (self.dataset['avg_glucose_level'] - self.dataset['avg_glucose_level'].mean()) / self.dataset['avg_glucose_level'].std()

        # encode labels
        label_encoder = LabelEncoder()

        label_encoder.fit(self.dataset['smoking_status'])
        self.dataset['smoking_status'] = label_encoder.transform(self.dataset['smoking_status'])

        label_encoder.fit(self.dataset['gender'])
        self.dataset['gender'] = label_encoder.transform(self.dataset['gender'])

        label_encoder.fit(self.dataset['Residence_type'])
        self.dataset['Residence_type'] = label_encoder.transform(self.dataset['Residence_type'])

        label_encoder.fit(self.dataset['work_type'])
        self.dataset['work_type'] = label_encoder.transform(self.dataset['work_type'])

        label_encoder.fit(self.dataset['ever_married'])
        self.dataset['ever_married'] = label_encoder.transform(self.dataset['ever_married'])

        return self.dataset
