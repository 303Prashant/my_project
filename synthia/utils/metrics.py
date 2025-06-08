from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_confusion_matrix_fig(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    return fig

def get_validation_stats(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    fig = get_confusion_matrix_fig(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "real": y_true.count(0),
        "fake": y_true.count(1),
        "confusion_fig": fig
    }
