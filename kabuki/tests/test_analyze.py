import numpy as np
import unittest
import kabuki.analyze as ka
from matplotlib.pyplot import close
from . import utils

try:
    import xarray as xr
except ImportError:
    xr = None


class TestAnalyzeBreakdown(unittest.TestCase):
    """
    test unit for analyze.py
    the unit only tests to see if the functions donot raise an error.
    it does not check the validity of the results.
    """

    @classmethod
    def setUpClass(self):
        # load models
        self.models, _ = utils.create_test_models()

        # run models
        utils.sample_from_models(self.models, n_iter=200)

    def runTest(self):
        pass

    def test_group_plot(self):
        for model in self.models:
            if model.is_group_model:
                ka.group_plot(model)
                close("all")

    def test_plot_posteriors_nodes(self):
        for model in self.models:
            ka.plot_posterior_nodes(model.mc.stochastics, bins=50)
            close("all")

    @unittest.skip("Not implemented")
    def test_compare_all_pairwise(self):
        for model in self.models:
            ka.compare_all_pairwise(model)

    def plot_all_pairwise(self):
        for model in self.models:
            ka.plot_all_pairwise(model)

    @unittest.skip("Not implemented")
    def test_savage_dickey(self):
        raise NotImplementedError

    @unittest.skip("Not implemented")
    def test_gelman_rubin(self):
        raise NotImplementedError

    @unittest.skip("Not implemented")
    def test_check_geweke(self):
        raise NotImplementedError

    @unittest.skip("Not implemented")
    def test_group_cond_diff(self):
        for model in self.models:
            if model.is_group_model:
                if model.depends:
                    (name, cond) = list(model.depends.items())[0]
                    tags = list(model.nodes_db[name].group_nodes.keys())[:2]
                ka.group_cond_diff(model, name, *tags)

    @unittest.skip(
        "Fails because of pymc likelihoods converting DataFrames to numpy arrays."
    )
    def test_post_pred_check(self):
        for model in self.models:
            ka.post_pred_gen(model, samples=20, progress_bar=False)

    def test_plot_posterior_predictive(self):
        for model in self.models:
            ka.plot_posterior_predictive(
                model, value_range=np.arange(-2, 2, 10), samples=10
            )


@unittest.skipIf(xr is None or not hasattr(xr, "DataTree"), "xarray DataTree unavailable")
class TestAnalyzePPCByCond(unittest.TestCase):
    def setUp(self):
        obs = xr.Dataset(
            {
                "rt": ("obs_id", np.array([0.4, 0.6, 0.5, 0.8, 0.7, 0.9])),
                "response": ("obs_id", np.array([1, 1, 0, 1, 0, 1])),
                "conf": ("obs_id", np.array(["LC", "HC", "LC", "HC", "LC", "HC"])),
                "stim": ("obs_id", np.array(["LL", "LL", "WW", "WW", "LL", "WW"])),
            },
            coords={
                "obs_id": np.arange(6),
                "subj_idx": ("obs_id", np.array([0, 0, 1, 1, 2, 2])),
            },
        )
        ppc = xr.Dataset(
            {
                "rt": (
                    ("chain", "draw", "obs_id"),
                    np.array(
                        [
                            [
                                [0.41, 0.61, 0.51, 0.81, 0.71, 0.91],
                                [0.42, 0.62, 0.52, 0.82, 0.72, 0.92],
                                [0.43, 0.63, 0.53, 0.83, 0.73, 0.93],
                            ]
                        ]
                    ),
                ),
                "response": (
                    ("chain", "draw", "obs_id"),
                    np.ones((1, 3, 6), dtype=int),
                ),
            },
            coords={
                "chain": [0],
                "draw": [0, 1, 2],
                "obs_id": np.arange(6),
                "subj_idx": ("obs_id", np.array([0, 0, 1, 1, 2, 2])),
            },
        )
        self.infdata = xr.DataTree.from_dict(
            {
                "/observed_data": obs,
                "/posterior_predictive": ppc,
            }
        )

    def tearDown(self):
        close("all")

    def test_plot_ppc_by_subject(self):
        axes = ka.plot_ppc_by_cond(
            self.infdata, subj_idx=[0, 1], num_pp_samples=2, seed=1, legend=False
        )
        self.assertEqual(len([ax for ax in axes.ravel() if ax.get_visible()]), 2)

    def test_plot_ppc_by_condition(self):
        axes = ka.plot_ppc_by_cond(
            self.infdata, condition_vars="conf", num_pp_samples=2, seed=1, legend=False
        )
        self.assertEqual(len([ax for ax in axes.ravel() if ax.get_visible()]), 2)

    def test_plot_ppc_by_subject_and_condition(self):
        axes = ka.plot_ppc_by_cond(
            self.infdata,
            subj_idx=[0, 1],
            condition_vars=["stim", "conf"],
            num_pp_samples=2,
            random_seed=1,
            legend=False,
        )
        self.assertEqual(len([ax for ax in axes.ravel() if ax.get_visible()]), 4)
