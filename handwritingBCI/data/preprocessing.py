class LabelEncoder:
    def __init__(self, labels):
        labels = list(set(labels))
        labels = sorted(labels)
        labels = sorted(labels, key=len)
        self.label_to_index = {label: index for index, label in enumerate(labels)}
        self.index_to_label = list(self.label_to_index.keys())

    def process(self, label):
        return self.label_to_index[label]

    def deprocess(self, index):
        index = int(index)
        return self.index_to_label[index]

    def __call__(self, label):
        return self.process(label)
