import torch
from data_formats.tasks import RejectOption, ScoreActivityClass
from validation.validate import get_preds
from .model import ClassifyActivityModel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

class GPUncertainty(ClassifyActivityModel):

    def __init__(self, cfg, model, kernel = 1.0 * RBF(0.1)):
        self.cfg = cfg
        self.model = model
        self.gp = GaussianProcessClassifier(kernel)
        if hasattr(self.model, "cache_key"):
            self.cache_key = "gp_" + self.model.cache_key

    def get_data_format(self):
        return self.model.get_data_format()

    def fit(self, dataset_name, split, num_batches):
        x, y, pred = get_preds(self.cfg, self.model, dataset_name, split, num_batches)
        U = -pred.select_score
        S = pred.active_prob
        US = torch.stack((U, S)).T
        self.gp.fit(US, y.is_active)

    @torch.no_grad()
    def __call__(self, x):
        pred = self.model.predict({ScoreActivityClass, RejectOption}, x)
        U = -pred.select_score
        S = pred.active_prob
        US = torch.stack((U, S)).T
        return torch.logit(torch.tensor(self.gp.predict_proba(US)[:,1]))

    def plot(self):
        import matplotlib.pyplot as plt
        U = torch.linspace(0.0, 1.0, 100)
        S = torch.linspace(0.0, 1.0, 100)
        UU, SS = torch.meshgrid(U, S)
        UUSS = torch.stack([UU.reshape(-1), SS.reshape(-1)]).T

        prob = self.gp.predict_proba(UUSS)[:,1].reshape((100, 100))

        fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
        cf = ax.contourf(UU, SS, prob, cmap='Blues')
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Prediction")
        ax.set_title("P(Active)")
        # ax.plot_surface(UU, SS, P_val)
        fig.colorbar(cf)

        return fig