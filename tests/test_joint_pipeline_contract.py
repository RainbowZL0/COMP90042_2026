from src.classification.base import Classifier
from src.data.schema import ClaimLabel, Evidence


class FakeJointClassifier(Classifier):
    def classify_batch(self, claim_texts, evidences_batch):
        return [ClaimLabel.REFUTES if evs else ClaimLabel.NOT_ENOUGH_INFO for evs in evidences_batch]


def test_fake_joint_classifier_contract():
    clf = FakeJointClassifier()
    labels = clf.classify_batch(
        ["claim"], [[Evidence(id="e1", text="some evidence")]]
    )
    assert labels == [ClaimLabel.REFUTES]
