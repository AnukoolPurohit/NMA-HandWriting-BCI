from handwritingBCI.data.preprocessing import LabelEncoder


class TestLabelEncoder:

    def test_label_process(self, test_labels):
        categories, labels = test_labels
        label_enc = LabelEncoder(labels)
        for index, category in enumerate(categories):
            assert index == label_enc.process(category)
        return

    def test_label_encoder_deprocess(self, test_labels):
        categories, labels = test_labels
        label_enc = LabelEncoder(labels)
        for index, category in enumerate(categories):
            assert category == label_enc.deprocess(index)
        return
