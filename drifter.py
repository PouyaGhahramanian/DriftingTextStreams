import numpy as np

class Drifter(object):
    def __init__(self, labels_ref, drift_type, drift_start, drift_end, drift_intensity,
                 drift_func='linear', num_drift_points=10, drift_distribution='even',
                 sigmoid_scale=10., is_multilabel=False, logging=True):
        self.labels_ref = labels_ref
        self.drift_type = drift_type
        self.drift_start = drift_start
        self.drift_end = drift_end
        self.drift_intensity = drift_intensity
        self.drift_func = drift_func
        self.timestep = 0
        self.num_drift_points = num_drift_points
        self.drift_distribution = drift_distribution
        self.sigmoid_scale = sigmoid_scale
        self.drift_index = 0
        self.logging = logging
        self.probability_log = []
        self.drift_points = self.generate_drift_points()
        self.probabilities_current = self.generate_probabilities()
        self.probabilities_next = self.generate_probabilities()
        self.is_multilabel = is_multilabel
        if num_drift_points < 1:
            raise ValueError("Number of drift points cannot be less than 1")

    def generate_drift_points(self):
        if self.drift_distribution == 'even':
            return np.linspace(self.drift_start, self.drift_end, self.num_drift_points + 2)[1:-1].astype(int)
        elif self.drift_distribution == 'random':
            return np.sort(np.random.randint(self.drift_start, self.drift_end, self.num_drift_points))
        else:
            raise ValueError("Distribution method should be 'even' or 'random'")

    def generate_probabilities(self):
        unique_labels = np.unique(self.labels_ref)
        popular_labels = np.random.choice(unique_labels, size=len(unique_labels) // 2, replace=False)
        probabilities = np.array([1. if label in popular_labels else 0. for label in self.labels_ref])
        return probabilities

    def update_probabilities(self):
        if self.logging:
            self.log_probabilities(self.timestep)
        if self.timestep < self.drift_start or self.timestep > self.drift_end or self.drift_points.shape[0] <= self.drift_index:
            return
        if self.timestep in self.drift_points:
            self.probabilities_current = self.probabilities_next
            self.probabilities_next = self.generate_probabilities()
            self.drift_index += 1
        self.drift_index = min(self.num_drift_points - 1, self.drift_index)
        if self.drift_type == 'abrupt':
            self.probabilities_current = self.probabilities_next
        else:
            for i in range(len(self.labels_ref)):
                if self.probabilities_current[i] < self.probabilities_next[i]:
                    self.probabilities_current[i] = self.increase_prob(self.drift_points[self.drift_index], self.drift_intensity)
                elif self.probabilities_current[i] > self.probabilities_next[i]:
                    self.probabilities_current[i] = self.decrease_prob(self.drift_points[self.drift_index], self.drift_intensity)

    def increase_prob(self, start, intensity):
        if self.drift_func == 'linear':
            return min((self.timestep - start) * intensity, 1.0)
        else:
            drift_mid = start + 2 / intensity
            return 1 / (1 + np.exp(-intensity * self.sigmoid_scale * (self.timestep - drift_mid)))

    def decrease_prob(self, start, intensity):
        if self.drift_func == 'linear':
            return max(1.0 - (self.timestep - start) * intensity, 0.0)
        else:
            drift_mid = start + 2 / intensity
            return 1 / (1 + np.exp(intensity * self.sigmoid_scale * (self.timestep - drift_mid)))

    def get_label(self, index):
        self.update_probabilities()
        if self.is_multilabel:
            probability_sum = 0.
            probability_num = 0.
            for label_part in self.labels_ref[index]:
                idx = np.where(self.labels_ref == label_part)[0][0]
                probability_sum += self.probabilities_current[idx]
                probability_num += 1
            probability_avg = probability_sum / probability_num
            popularity_label = np.random.choice([0, 1], p=[1 - probability_avg, probability_avg])
        else:
            idx = index
            popularity_label = np.random.choice([0, 1], p=[1 - self.probabilities_current[idx], self.probabilities_current[idx]])
        self.timestep += 1
        return popularity_label

    def log_probabilities(self, current_time):
        self.probability_log.append((current_time, self.probabilities_current.copy()))

    def get_probability_log(self):
        return self.probability_log