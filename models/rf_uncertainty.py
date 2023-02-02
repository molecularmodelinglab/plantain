import torch
from data_formats.tasks import RejectOption, ScoreActivityClass
from validation.validate import get_preds
from .model import ClassifyActivityModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

class RFUncertainty(ClassifyActivityModel):

    def __init__(self, cfg, model, n_estimators=100):
        self.cfg = cfg
        self.model = model
        self.rf = LogisticRegression() # SVC(probability=True)
        self.feats = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
        if hasattr(self.model, "cache_key"):
            self.cache_key = "rf_" + self.model.cache_key

    def get_data_format(self):
        return self.model.get_data_format()

    def fit(self, dataset_name, split, num_batches):
        x, y, pred = get_preds(self.cfg, self.model, dataset_name, split, num_batches)
        U = -pred.select_score
        S = pred.active_prob
        US = torch.stack((U, S)).T.cpu()
        US = self.feats.fit_transform(US)
        self.rf.fit(US, y.is_active.cpu())

    @torch.no_grad()
    def __call__(self, x):
        pred = self.model.predict({ScoreActivityClass, RejectOption}, x)
        U = -pred.select_score
        S = pred.active_prob
        US = torch.stack((U, S)).T.cpu()
        US = self.feats.fit_transform(US)
        return torch.logit(torch.tensor(self.rf.predict_proba(US)[:,1])).to(U.device)

    def plot(self):
        import matplotlib.pyplot as plt
        U = torch.linspace(0.0, 1.0, 100)
        S = torch.linspace(0.0, 1.0, 100)
        UU, SS = torch.meshgrid(U, S)
        UUSS = torch.stack([UU.reshape(-1), SS.reshape(-1)]).T.cpu()

        UUSS = self.feats.fit_transform(UUSS)

        prob = self.rf.predict_proba(UUSS)[:,1].reshape((100, 100))

        fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
        cf = ax.contourf(UU, SS, prob, cmap='Blues')
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Prediction")
        ax.set_title("P(Active)")
        # ax.plot_surface(UU, SS, P_val)
        fig.colorbar(cf)

        return fig

    def to(self, device):
        self.model = self.model.to(device)
        return self