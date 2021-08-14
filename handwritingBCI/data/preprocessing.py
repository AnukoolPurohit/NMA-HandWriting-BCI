class LabelEncoder:
    def __init__(self, labels):
        self.categories = labels
        self.label_to_index = {}
        self.index_to_label = []
        self.process_labels()

    def process_labels(self):
        self.set_unique_labels_as_categories()
        self.sort_categories()
        self.generate_category_index()
        self.index_to_label = list(self.label_to_index.keys())
        return

    def generate_category_index(self):
        for index, category in enumerate(self.categories):
            self.label_to_index[category] = index
        return

    def set_unique_labels_as_categories(self):
        self.categories = list(set(self.categories))
        return

    def sort_categories(self):
        self.categories = sorted(self.categories)
        self.categories = sorted(self.categories, key=len)
        return

    def process(self, label):
        return self.label_to_index[label]

    def deprocess(self, index):
        index = int(index)
        return self.index_to_label[index]

    def __call__(self, label):
        return self.process(label)
