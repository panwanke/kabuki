import sys
import os
import warnings
from types import FunctionType

import numpy as np
from matplotlib.pylab import figure
import matplotlib.pyplot as plt

import pandas as pd
import pymc as pm
from . import utils

from .utils import interpolate_trace

from collections import OrderedDict


def plot_posterior_nodes(nodes, bins=50, lb=None, ub=None):
    """Plot interpolated posterior of a list of nodes.

    :Arguments:
        nodes : list of pymc.Node's
            List of pymc.Node's to plot the posterior of.
            These can be found in model.nodes_db.node.loc['param_name']
        bins : int (default=50)
            How many bins to use for computing the histogram.
        lb : float (default is to infer from data)
            Lower boundary to use for plotting.
        ub : float (default is to infer from data)
            Upper boundary to use for plotting.
    """
    figure()
    if lb is None:
        lb = min([min(node.trace()[:]) for node in nodes])
    if ub is None:
        ub = max([max(node.trace()[:]) for node in nodes])

    x_data = np.linspace(lb, ub, 300)

    for node in nodes:
        trace = node.trace()[:]
        # hist = interpolate_trace(x_data, trace, range=(trace.min(), trace.max()), bins=bins)
        hist = interpolate_trace(x_data, trace, range=(lb, ub), bins=bins)
        plt.plot(x_data, hist, label=node.__name__, lw=2.0)

    leg = plt.legend(loc="best", fancybox=True)
    leg.get_frame().set_alpha(0.5)


def group_plot(model, params_to_plot=(), bins=50, samples=5000, save_to=None):
    def find_min_max(subj_block):
        # find global min and max for plotting
        min = np.inf
        max = -np.inf
        for name, subj in subj_block.iterrows():
            trace = subj["node"].trace()
            min = np.min([min, np.min(trace)])
            max = np.max([max, np.max(trace)])
        return min, max

    assert model.is_group_model, "group plot only works for group models."

    # select non-observed subject nodes
    subj_nodes = model.nodes_db[
        (model.nodes_db["observed"] == False) & (model.nodes_db["subj"] == True)
    ]

    knode_names = subj_nodes.groupby(["knode_name", "tag"])

    for (knode_name, tag), subj_block in knode_names:
        min, max = find_min_max(subj_block)

        # plot interpolated subject histograms
        # create figure
        print("plotting %s: %s" % (knode_name, tag))
        sys.stdout.flush()

        plt.figure()
        plt.title("%s: %s" % (knode_name, tag))
        x = np.linspace(min, max, 100)

        ############################################
        # plot subjects
        for name, subj_descr in subj_block.iterrows():
            trace = subj_descr["node"].trace()
            height = interpolate_trace(x, trace, range=(min, max), bins=bins)
            plt.plot(x, height, lw=1.0, label=str(np.int32(subj_descr["subj_idx"])))

        ###########################################
        # plot group distribution
        node = subj_descr["node"]
        group_trace = np.empty(samples, dtype=np.float32)
        for sample in range(samples):
            # set parents to random value from their trace
            trace_pos = np.random.randint(0, len(node.trace()))
            for parent in node.extended_parents:
                parent.value = parent.trace()[trace_pos]
            group_trace[sample] = node.random()
            # TODO: What to do in case of deterministic (e.g. transform) node
            # except AttributeError:
            #    group_trace[sample] = node.parents.items()[0].random()

        height = interpolate_trace(x, group_trace, range=(min, max), bins=bins)
        plt.plot(x, height, "--", lw=2.0, label="group")

        ##########################################
        # legend and title
        leg = plt.legend(loc="best", fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.gcf().canvas.set_window_title(knode_name)

        if save_to is not None:
            plt.savefig(os.path.join(save_to, "group_%s.png" % knode_name))
            plt.savefig(os.path.join(save_to, "group_%s.pdf" % knode_name))


def plot_all_pairwise(model):
    """Plot all pairwise posteriors to find correlations."""
    import scipy as sp
    from itertools import combinations

    # size = int(np.ceil(np.sqrt(len(data_deps))))
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    # Loop through all pairwise combinations
    for i, (p0, p1) in enumerate(combinations(model.get_group_nodes()["node"], 2)):
        fig.add_subplot(6, 6, i + 1)
        plt.plot(p0.trace(), p1.trace(), ".")
        (a_s, b_s, r, tt, stderr) = sp.stats.linregress(p0.trace(), p1.trace())
        reg = sp.polyval((a_s, b_s), (np.min(p0.trace()), np.max(p0.trace())))
        plt.plot((np.min(p0.trace()), np.max(p0.trace())), reg, "-")
        plt.xlabel(p0.__name__)
        plt.ylabel(p1.__name__)

    plt.draw()


def gelman_rubin(models):
    """
    Calculate the gelman_rubin statistic (R_hat) for every stochastic in the model.
    (Gelman at al 2004, 11.4)
    Input:
        models - list of models
    """
    stochastics = models[0].get_stochastics()
    R_hat_dict = {}
    num_samples = stochastics.node[0].trace().shape[0]
    num_chains = len(models)
    for name, stochastic in stochastics.iterrows():
        # Calculate mean for each chain
        samples = np.empty((num_chains, num_samples))
        for i, model in enumerate(models):
            samples[i, :] = model.nodes_db.loc[name, "node"].trace()

        R_hat_dict[name] = pm.diagnostics.gelman_rubin(samples)

    return R_hat_dict


R_hat = gelman_rubin


def check_geweke(model, assert_=True):
    # Test for convergence using geweke method
    for name, param in model.iter_stochastics():
        geweke = np.array(pm.geweke(param["node"]))
        if np.any(np.abs(geweke[:, 1]) > 2):
            msg = "Chain of %s not properly converged" % param
            if assert_:
                raise AssertionError(msg)
            else:
                print(msg)
            return False

    return True


def group_cond_diff(hm, node, cond1, cond2, threshold=0):
    """
    Compute the difference between different conditions in a group analysis.
    For each subject the function computes the difference between 'node' under
    condition 'cond1' to 'node' under condition 'cond2'.
    By assuming that each of the differences is normal distributed
    we can easily compute the group mean and group variance of the difference.
    Then the difference is compared to 'threshold' to compute the mass of the
    group pdf which is smaller than 'threshold'

    Input:
        hm - hierachical model
        node - name of node to be analyized
        cond1 - name of condition 1
        cond2 - name of condition 2
        threshold - see description

    Output:
        group_mean - group mean of the differnce
        group_var - group variance of the difference
        mass_under_threshold  - the mass of the group pdf which is smaller than threshold
    """
    import scipy as sp

    name = node
    node_dict = hm.params_include[name].subj_nodes
    n_subjs = hm._num_subjs

    # loop over subjs
    subj_diff_mean = np.zeros(n_subjs)
    subj_diff_std = np.zeros(n_subjs)
    for i_subj in range(n_subjs):
        # compute difference of traces
        name1 = node_dict[cond1][i_subj].__name__
        name2 = node_dict[cond2][i_subj].__name__
        trace1 = hm.mc.db.trace(name1)[:]
        trace2 = hm.mc.db.trace(name2)[:]
        diff_trace = trace1 - trace2

        # compute stats
        subj_diff_mean[i_subj] = np.mean(diff_trace)
        subj_diff_std[i_subj] = np.std(diff_trace)

    pooled_var = 1.0 / sum(1.0 / (subj_diff_std**2))
    pooled_mean = sum(subj_diff_mean / (subj_diff_std**2)) * pooled_var

    mass_under = sp.stats.norm.cdf(threshold, pooled_mean, np.sqrt(pooled_var))

    return pooled_mean, pooled_var, mass_under


def post_pred_compare_stats(sampled_stats, data_stats, evals=None):
    """Evaluate summary statistics of sampled sets.

    :Arguments:
        sampled_stats : dict
            Map of summary statistic names to distributions
        data_stats : dict
            Map of summary statistic names to the data distribution

    :Returns:
        pandas.DataFrame containing the eval results as columns.
    """

    from scipy.stats import scoreatpercentile, percentileofscore

    if evals is None:
        # Generate some default evals
        evals = OrderedDict()
        evals["observed"] = lambda x, y: y
        evals["mean"] = lambda x, y: np.mean(x)
        evals["std"] = lambda x, y: np.std(x)
        evals["SEM"] = lambda x, y: (np.mean(x) - y) ** 2
        evals["MSE"] = lambda x, y: np.mean((x - y) ** 2)
        evals["credible"] = lambda x, y: (scoreatpercentile(x, 97.5) > y) and (
            scoreatpercentile(x, 2.5) < y
        )
        evals["quantile"] = percentileofscore
        evals["mahalanobis"] = lambda x, y: np.abs(np.mean(x) - y) / np.std(x)
        # for q in [2.5, 25, 50, 75, 97.5]:
        #    key = str(q) + 'q'
        #    evals[key] = lambda x, y, q=q: scoreatpercentile(x, q)

    # Evaluate all eval-functions
    results = pd.DataFrame(
        index=list(sampled_stats.keys()),
        columns=list(evals.keys()) + ["NaN"],
        dtype=np.float32,
    )

    results.index.names = ["stat"]
    for stat_name in sampled_stats:
        # update NaN column with the no. of NaNs and remove them
        s = sampled_stats[stat_name]
        results.loc[stat_name, "NaN"] = sum(pd.isnull(s))
        s = s[np.isfinite(s)]
        if len(s) == 0:
            continue
        # evaluate
        for eval_name, func in evals.items():
            value = func(s, data_stats[stat_name])
            results.loc[stat_name, eval_name] = value

    return results.drop("NaN", axis=1)


def post_pred_stats(
    data, sim_datasets, stats=None, plot=False, bins=100, evals=None, call_compare=True
):
    """Calculate a set of summary statistics over posterior predictives.

    :Arguments:
        data : pandas.Series

        sim_data : pandas.Series

    :Optional:
        bins : int
            How many bins to use for computing the histogram.
        stats : dict or function
            User-defined statistics to compute (by default mean and std are computed)
            and evaluate over the samples.
            :Example:
              * {'mean': np.mean, 'median': np.median}
              * lambda x: np.mean(x)
        evals : dict
            User-defined evaluations of the statistics (by default 95 percentile and SEM).
            :Example: {'percentile': scoreatpercentile}
        plot : bool
            Whether to plot the posterior predictive distributions.
        progress_bar : bool
            Display progress bar while sampling.
        field : string
            Which column name to run the stats on
        call_com,pare : bool (default=True)
            Whether to call post_pred_compare_stats. If False, return stats directly.
    """

    def _calc_stats(data, stats):
        out = {}
        for name, func in stats.items():
            out[name] = func(data)
        return out

    if stats is None:
        stats = OrderedDict((("mean", np.mean), ("std", np.std)))
    if isinstance(stats, FunctionType):
        stats = OrderedDict((("stat", stats),))

    data_stats = _calc_stats(data, stats)

    ###############################################
    # Initialize posterior sample stats container
    samples = len(sim_datasets)
    sampled_stats = {}
    sampled_stats = pd.DataFrame(
        index=sim_datasets.index.droplevel(2).unique(),
        columns=list(stats.keys()),
        dtype=np.float32,
    )

    for i, sim_dataset in sim_datasets.groupby(level=(0, 1)):
        sampled_stat = _calc_stats(sim_dataset.values, stats)

        # Add it to the results container
        for name, value in sampled_stat.items():
            sampled_stats[name][i] = value

    if plot:
        from pymc.Matplot import gof_plot

        for name, value in sampled_stats.items():
            gof_plot(value, data_stats[name], bins=bins, name=name, verbose=0)

    if call_compare:
        return post_pred_compare_stats(sampled_stats, data_stats, evals=evals)
    else:
        return sampled_stats


def _parents_to_random_posterior_sample(bottom_node, pos=None):
    """Walks through parents and sets them to pos sample."""
    for i, parent in enumerate(bottom_node.extended_parents):
        if not isinstance(parent, pm.Node):  # Skip non-stochastic nodes
            continue

        if pos is None:
            # Set to random posterior position
            pos = np.random.randint(0, len(parent.trace()))

        assert len(parent.trace()) >= pos, "pos larger than posterior sample size"
        parent.value = parent.trace()[pos]


def _plot_posterior_pdf_node(
    bottom_node, axis, value_range=None, samples=10, bins=100, **kwargs
):
    """Calculate posterior predictive for a certain bottom node.

    :Arguments:
        bottom_node : pymc.stochastic
            Bottom node to compute posterior over.

        axis : matplotlib.axis
            Axis to plot into.

        value_range : numpy.ndarray
            Range over which to evaluate the likelihood.

    :Optional:
        samples : int (default=10)
            Number of posterior samples to use.

        bins : int (default=100)
            Number of bins to compute histogram over.

    """

    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError("value_range keyword argument must be supplied.")

    like = np.empty((samples, len(value_range)), dtype=np.float32)
    for sample in range(samples):
        _parents_to_random_posterior_sample(bottom_node)
        # Generate likelihood for parents parameters
        like[sample, :] = bottom_node.pdf(value_range)

    y = like.mean(axis=0)
    try:
        y_std = like.std(axis=0)
    except FloatingPointError:
        print(
            "WARNING! %s threw FloatingPointError over std computation. Setting to 0 and continuing."
            % bottom_node.__name__
        )
        y_std = np.zeros_like(y)

    # Plot pp
    axis.plot(value_range, y, label="post pred", color="b")
    axis.fill_between(value_range, y - y_std, y + y_std, color="b", alpha=0.8)

    # Plot data
    if len(bottom_node.value) != 0:
        data_processor = kwargs.pop("data_processor", None)

        if data_processor is None:
            processed_data = bottom_node.value.values
        else:
            processed_data = data_processor(bottom_node.value.values)

        axis.hist(
            processed_data,
            density=True,
            color="blue",
            label="data",
            bins=bins,
            histtype="step",
            lw=1.0,
        )

    axis.set_ylim(bottom=0)  # Likelihood and histogram can only be positive


def plot_posterior_predictive(
    model,
    plot_func=None,
    required_method="pdf",
    columns=None,
    save=False,
    path=None,
    figsize=(8, 6),
    format="png",
    num_subjs=None,
    **kwargs
):
    """Plot the posterior predictive distribution of a kabuki hierarchical model.

    :Arguments:

        model : kabuki.Hierarchical
            The (constructed and sampled) kabuki hierarchical model to
            create the posterior preditive from.

        value_range : numpy.ndarray
            Array to evaluate the likelihood over.

    :Optional:

        samples : int (default=10)
            How many posterior samples to generate the posterior predictive over.

        columns : int (default=3)
            How many columns to use for plotting the subjects.

        bins : int (default=100)
            How many bins to compute the data histogram over.

        figsize : (int, int) (default=(8, 6))

        save : bool (default=False)
            Whether to save the figure to a file.

        path : str (default=None)
            Save figure into directory prefix

        format : str or list of strings
            Save figure to a image file of type 'format'. If more then one format is
            given, multiple files are created

        plot_func : function (default=_plot_posterior_pdf_node)
            Plotting function to use for each observed node
            (see default function for an example).

        data_processor: function (default=None)
            Inside plot_posterior_predictive the standard plotting function (histogram)
            assumes that your data is supplied as a 1-dimensional
            array (e.g. outcome variable in range (-x,x)). If your original data does
            not have this format, but can be transformed into it (meaningfully), you
            can supply the data_processor function to perform this transformation
            and plot_posterior_predictive will operate on the transformed data.

    :Note:

        This function changes the current value and logp of the nodes.

    """

    if plot_func is None:
        plot_func = _plot_posterior_pdf_node

    observeds = model.get_observeds()

    if columns is None:
        # If there are less than 3 items to plot per figure,
        # only use as many columns as there are items.
        max_items = max([len(i[1]) for i in observeds.groupby("tag").groups.items()])
        columns = min(3, max_items)

    # Plot different conditions (new figure for each)
    for tag, nodes in observeds.groupby("tag"):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(utils.pretty_tag(tag), fontsize=12)
        fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)

        nrows = num_subjs or len(nodes) / columns

        if len(nodes) - int(nrows * columns) > 0:
            nrows += 1

        # Plot individual subjects (if present)
        i = 0
        for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
            i += 1
            if not hasattr(bottom_node["node"], required_method):
                continue  # skip nodes that do not define the required_method

            ax = fig.add_subplot(np.ceil(nrows).astype(int), columns, subj_i + 1)
            if "subj_idx" in bottom_node:
                ax.set_title(str(bottom_node["subj_idx"]))

            plot_func(bottom_node["node"], ax, **kwargs)

            if i >= np.ceil(nrows) * columns:
                warnings.warn("Too many nodes. Consider increasing number of columns.")
                break

            if num_subjs is not None and i >= num_subjs:
                break

        # Save figure if necessary
        if save:
            fname = "ppq_" + ".".join(
                [str(t) for t in tag] if type(tag) == list else str(tag)
            )
            if path is None:
                path = "."
            if isinstance(format, str):
                format = [format]
            [
                fig.savefig("%s.%s" % (os.path.join(path, fname), x), format=x)
                for x in format
            ]


def geweke_problems(model, fname=None, **kwargs):
    """
    return a list of nodes who were detected as problemtic according to the geweke test
    Input:
        fname : string (deafult - None)
            Save result to file named fname
        kwargs : keywords argument passed to the geweke function
    """

    # search for geweke problems
    g = pm.geweke(model.mc)
    problems = []
    for node, output in g.items():
        values = np.array(output)[:, 1]
        if np.any(np.abs(values) > 2):
            problems.append(node)

    # write results to file if needed
    if fname is not None:
        with open(fname, "w") as f:
            for node in problems:
                f.write(node)

    return problems


def _post_pred_generate(
    bottom_node, samples=500, data=None, append_data=False, **kwargs
):
    """Generate posterior predictive data from a single observed node."""
    import pymc as pm
    import numpy as np
    
    add_model_parameters = kwargs.pop("add_model_parameters", None)
    datasets = []

    ##############################
    # Sample and generate stats
    # If number of samples is fixed, use the original code, i.e., randomly sample one set of
    # values from extended_parents and generate random value;
    #
    # If number of samples is None, use the lenght of trace, and iterate the whole posterior.

    for i, parent in enumerate(bottom_node.extended_parents):
        if not isinstance(parent, pm.Node): # Skip non-stochastic nodes
            continue
        else:
            mc_len = len(parent.trace())
            break

    # samples=samples
    if samples is None:

        samples = mc_len
        # print("\nNumber of PPC samples is equal to length of MCMC trace.")

    assert samples, "Can not determine the number of samples"
    
    if samples == mc_len:
        for sample in range(samples):
            _parents_to_random_posterior_sample(bottom_node, pos = sample)
            
            # Generate data from bottom node
            if add_model_parameters is None:
                sampled_data = bottom_node.random()
            else:
                sampled_data = bottom_node.random(add_model_parameters=add_model_parameters)
            
            # change the index of ppc data if it is not the same as the observed data
            # here we used the all() because `any` will cause a mess in the first node's index
            if not all(sampled_data.index == bottom_node.value.index): 
                sampled_data.index = bottom_node.value.index
               
            sampled_data.index.names = ['trial_idx']

            # add the "response" column for regression models
            if "response" not in sampled_data.columns:
                sampled_data["response"] = np.where(sampled_data['rt'] > 0, 1.,
                                                    np.where(sampled_data['rt'] <=0, 0., None)) 
                        
            if append_data and data is not None:
                sampled_data = sampled_data.join(data.reset_index(), lsuffix='_sampled')
            datasets.append(sampled_data)
    
    else:
        for sample in range(samples):
            pos = np.random.randint(0, mc_len)
            _parents_to_random_posterior_sample(bottom_node, pos = pos)

            # Generate data from bottom node
            if add_model_parameters is None:
                sampled_data = bottom_node.random()
            else:
                sampled_data = bottom_node.random(add_model_parameters=add_model_parameters)
            
            # change the index of ppc data if it is not the same as the observed data
            if not all(sampled_data.index == bottom_node.value.index): 
                sampled_data.index = bottom_node.value.index
            sampled_data.index.names = ['trial_idx']
            
            # add the "response" column for regression models
            if "response" not in sampled_data.columns:
                sampled_data["response"] = np.where(sampled_data['rt'] > 0, 1.,
                                                    np.where(sampled_data['rt'] <=0, 0., None)) 

            if append_data and data is not None:
                sampled_data = sampled_data.join(data.reset_index(), lsuffix='_sampled')
            datasets.append(sampled_data)

    return datasets

def post_pred_gen(
        model, 
        groupby=None, 
        samples=500, 
        append_data=False, 
        progress_bar=False, 
        parallel=True,
        **kwargs
    ):
    """Run posterior predictive check on a model.
    :Arguments:
        model : kabuki.Hierarchical
            Kabuki model over which to compute the ppc on.
    :Optional:
        samples : int
            How many samples to generate for each node. If None, will used the MCMC samples

        groupby : list
            Alternative grouping of the data. If not supplied, uses splitting
            of the model (as provided by depends_on).
        append_data : bool (default=False)
            Whether to append the observed data of each node to the replicatons.
        progress_bar : bool (default=True)
            Display progress bar
        parallel : bool (default=True)
            run parallel at individual node level    
    :Returns:
        Hierarchical pandas.DataFrame with multiple sampled RT data sets.
        1st level: wfpt node
        2nd level: draw, i.e., draw of MCMC or samples of posterior predictive.
        3rd level: original data index (trial_idx)
    :See also:
        post_pred_stats
    """
    import pandas as pd
    from copy import deepcopy
    import pymc.progressbar as pbar
    
    n_jobs = kwargs.pop("n_jobs", -1) # -1 is all cores
    model = deepcopy(model)
    
    progress_bar = not parallel
    
    # Progress bar
    if progress_bar:
        n_iter = len(model.get_observeds())
        bar = pbar.progress_bar(n_iter)
        bar_iter = 0
    else:
        print("Start generating posterior prediction...")

    if groupby is None:
        #### here I changed `iloc` to `loc`
        iter_data = ((name, model.data.loc[obs['node'].value.index]) for name, obs in model.iter_observeds())
    else:
        iter_data = model.data.groupby(groupby)

    
    if parallel:
        from joblib import Parallel, delayed
        # parallel process through all nodes
        def gen_individual_ppc(name, data, model, samples, append_data, **kwargs):
            
            node = model.get_data_nodes(data.index)

            ##############################
            # Sample and generate stats
            datasets = _post_pred_generate(
                node, 
                samples=samples, 
                data=data, 
                append_data=append_data,
                **kwargs
            )
            result = pd.concat(datasets, names=['draw'], keys=list(range(len(datasets))))
            
            return name,result
        
        tmp_list = [(name, data) for name, data in iter_data if model.get_data_nodes(data.index) is not None and hasattr(model.get_data_nodes(data.index), 'random')]  
        
        results = Parallel(n_jobs=n_jobs)(delayed(gen_individual_ppc)(name, data, model, samples, append_data, **kwargs) for name, data in tmp_list)
        results = dict(results)
    else:
        
        results = {}
        # iterate through each node
        for name, data in iter_data:
            node = model.get_data_nodes(data.index)

            if progress_bar:
                bar_iter += 1
                bar.update(bar_iter)

            if node is None or not hasattr(node, "random"):
                continue  # Skip

            # If we used data grouping --> name is a tuple which doesn't play well with pd.concat later on
            # We exchange the name for the name of the observed node we currently process
            if groupby is not None:
                new_name = node.__str__()
            else:  # if groupby was None --> keep name as is
                new_name = name

            ##############################
            # Sample and generate stats
            datasets = _post_pred_generate(
                node, 
                samples=samples, 
                data=data, 
                append_data=append_data,
                **kwargs
            )
            results[new_name] = pd.concat(
                datasets, 
                names=['draw'], 
                keys=list(range(len(datasets)))
            ) 
        if progress_bar:
            bar_iter += 1
            bar.update(bar_iter)
            

    return pd.concat(results, names=['node'])


# note, it is edited by custom
def _pointwise_like_generate(bottom_node, samples=None, data=None, append_data=False):
    """Generate posterior predictive data from a single observed node."""
    import pymc as pm
    import numpy as np
    import pandas as pd
    from copy import deepcopy
    import hddm
    
    datasets = []

    ##############################
    # Iterate the posterior and generate likelihood for each data point
    
    for i, parent in enumerate(bottom_node.extended_parents):
        if not isinstance(parent, pm.Node): # Skip non-stochastic nodes
            continue
        else:
            mc_len = len(parent.trace())
            break
    # samples=samples
    if samples is None:
        samples = mc_len
        # print("Number of samples is equal to length of MCMC trace.")

    assert samples, "Can not determine the number of samples"
    
    for sample in range(samples):
        _parents_to_random_posterior_sample(bottom_node, pos = sample)
        
        param_dict = deepcopy(bottom_node.parents.value)
        
        # check if the node is deficit 
        if "sv" not in param_dict:
            param_dict["sv"] = 0
        if "sz" not in param_dict:
            param_dict["sz"] = 0
        if "st" not in param_dict:
            param_dict["st"] = 0
        # param_dict = {key: np.array(value, dtype="double") for key, value in param_dict.items()}
        
        # for regressor models
        if 'reg_outcomes' in param_dict:
            del param_dict['reg_outcomes']

            pointwise_lik = bottom_node.value.copy()
            pointwise_lik.index.names = ['trial_idx']        # change the index label as "trial_idx"
            pointwise_lik.drop(['rt'],axis=1,inplace=True)   # drop 'rt' b/c not gonna use it.

            for i in bottom_node.value.index:
                # get current params
                for p in bottom_node.parents['reg_outcomes']:
                    param_dict[p] = bottom_node.parents.value[p].loc[i].item()

                # calculate the point-wise likelihood.
                tmp_lik = hddm.wfpt.pdf_array(
                    x = np.array(bottom_node.value.loc[i]),
                    v = param_dict['v'],
                    a = param_dict['a'], 
                    t = param_dict['t'],
                    p_outlier = param_dict['p_outlier'],
                    sv = param_dict['sv'],
                    z = param_dict['z'],
                    sz = param_dict['sz'],
                    st = param_dict['st'])
                pointwise_lik.loc[i, 'log_lik'] = tmp_lik
                
            # check if there is zero prob.
            if 0 in pointwise_lik.values:
                pointwise_lik['log_lik']=pointwise_lik['log_lik'].replace(0.0, pointwise_lik['log_lik'].mean())

            elif pointwise_lik['log_lik'].isnull().values.any():
                print('NAN in the likelihood, check the data !')
                break

            pointwise_lik['log_lik'] = np.log(pointwise_lik['log_lik'])

            if np.isinf(pointwise_lik['log_lik']).values.sum() > 0:
                print('Correction does not work!!!\n')
                
        # for other models
        else:
            tmp_lik = hddm.wfpt.pdf_array(x = np.array(
                bottom_node.value['rt'].values, dtype="double"),
                v = param_dict['v'],
                a = param_dict['a'], 
                t = param_dict['t'],
                p_outlier = param_dict['p_outlier'],
                sv = param_dict['sv'],
                z = param_dict['z'],
                sz = param_dict['sz'],
                st = param_dict['st'])
            # check if there is zero prob.
            if np.sum(tmp_lik == 0.0) > 0:
                tmp_lik[tmp_lik == 0.0] = np.mean(tmp_lik)
            elif np.sum(np.isnan(tmp_lik))  > 0:
                print('NAN in the likelihood, check the data !')
                break

            # obs = np.log(tmp_lik)                
            tmp_lik = np.log(tmp_lik).astype('float32')
            
            if np.sum(np.isinf(tmp_lik)) > 0:
                print('Correction does not work!!!\n')

            pointwise_lik = pd.DataFrame({'log_lik': tmp_lik}, index=bottom_node.value.index)
            pointwise_lik.index.names = ['trial_idx'] 

        datasets.append(pointwise_lik)

    return datasets

def pointwise_like_gen(model, groupby=None, samples=None, append_data=False, progress_bar=False, parallel=True, **kwargs):
    """Run posterior predictive check on a model.
    :Arguments:
        model : kabuki.Hierarchical
            Kabuki model over which to compute the ppc on.
    :Optional:
        samples : int
            How many samples to generate for each node.

        groupby : list
            Alternative grouping of the data. If not supplied, uses splitting
            of the model (as provided by depends_on).
        append_data : bool (default=False)
            Whether to append the observed data of each node to the replicatons.
        progress_bar : bool (default=True)
            Display progress bar
        parallel : bool (default=True)
            run parallel at individual node level
    :Returns:
        Hierarchical pandas.DataFrame with multiple sampled RT data sets.
        1st level: wfpt node
        2nd level: draw, i.e., draw/sample of MCMC
        3rd level: original data index, which was renamed as "trial_idx"
    :See also:
        post_pred_stats
    """
    import pandas as pd
    from copy import deepcopy
    import pymc.progressbar as pbar
    
    n_jobs = kwargs.pop("n_jobs", -1) # -1 is all cores
    model = deepcopy(model)
    
    progress_bar = not parallel
    # Progress bar
    if progress_bar:
        n_iter = len(model.get_observeds())
        bar = pbar.progress_bar(n_iter)
        bar_iter = 0
    else:
        print("Start to calculate pointwise log likelihood...")

    if groupby is None:
        #### here I changed `iloc` to `loc`
        iter_data = ((name, model.data.loc[obs['node'].value.index]) for name, obs in model.iter_observeds())
    else:
        iter_data = model.data.groupby(groupby)

    if parallel:
        from joblib import Parallel, delayed
        # parallel process through all nodes
        def get_individual_logp(name, data, model, samples, append_data):
            
            node = model.get_data_nodes(data.index)

            ##############################
            # Sample and generate stats
            datasets = _pointwise_like_generate(node, samples=samples, data=data, append_data=append_data)
            result = pd.concat(datasets, names=['draw'], keys=list(range(len(datasets))))
            
            return name,result
        
        tmp_list = [(name, data) for name, data in iter_data if model.get_data_nodes(data.index) is not None and hasattr(model.get_data_nodes(data.index), 'random')]  
        
        results = Parallel(n_jobs=n_jobs)(delayed(get_individual_logp)(name, data, model, samples, append_data) for name, data in tmp_list)
        results = dict(results)
    else:
        
        results = {}
        # iterate through each node
        for name, data in iter_data:
            node = model.get_data_nodes(data.index)

            if progress_bar:
                bar_iter += 1
                bar.update(bar_iter)

            if node is None or not hasattr(node, 'random'):
                continue # Skip

            ##############################
            # Sample and generate stats
            datasets = _pointwise_like_generate(node, samples=samples, data=data, append_data=append_data)
            results[name] = pd.concat(datasets, names=['draw'], keys=list(range(len(datasets))))
        if progress_bar:
            bar_iter += 1
            bar.update(bar_iter)

    return pd.concat(results, names=['node'])


def plot_ppc_by_cond(infdata,
                     subj_idx=None,
                     condition_vars=None,
                     num_pp_samples=500,
                     **kwargs):
    """Plot PPC distributions by subject and/or experimental condition.

    Args:
        infdata (InferenceData from Arviz): The output from dockerHDDM sampling.
        subj_idx (str or list of str, optional): Defaults to None that means
            plot all subjects or plot only condition level when
            `condition_vars` is not None. If `subj_idx` is "all", it plots all
            subjects varying with condition level.
        condition_vars (str, list of str or dict, optional): Defaults to None.
            `condition_vars` can be a str of variable name ('conf') or list of
            variable names (['conf','stim']). It supports selecting condition
            levels of variable name, such as `{'stim':['WW','LL']}`.
        num_pp_samples (int, optional): The number of posterior predictives used for plotting. Defaults to 500.
        **kwargs: Plotting options. Common options include `var_names`,
            `random_seed`/`seed`, `alpha`, `legend`, `textsize`, `bins`,
            `figsize`, and `max_cols`.

    Returns:
        axes: The matplotlib axes.
    """

    def _as_dataset(group):
        if hasattr(group, "to_dataset"):
            return group.to_dataset()
        return group

    def _values(ds, name):
        if name in ds:
            return np.asarray(ds[name].values)
        if name in ds.coords:
            return np.asarray(ds.coords[name].values)
        raise KeyError("Variable %s not found in observed_data." % name)

    var_name = kwargs.pop("var_names", "rt")
    if isinstance(var_name, (list, tuple)):
        if len(var_name) != 1:
            raise ValueError("plot_ppc_by_cond only supports one var_name.")
        var_name = var_name[0]

    seed = kwargs.pop("seed", None)
    random_seed = kwargs.pop("random_seed", None)
    if seed is None:
        seed = random_seed
    rng = np.random.default_rng(seed)

    alpha = kwargs.pop("alpha", 0.35)
    legend = kwargs.pop("legend", True)
    textsize = kwargs.pop("textsize", None)
    bins = kwargs.pop("bins", 30)
    figsize = kwargs.pop("figsize", None)
    max_cols = kwargs.pop("max_cols", 3)
    predictive_color = kwargs.pop("predictive_color", "C0")
    observed_color = kwargs.pop("observed_color", "black")

    # Legacy az.plot_ppc options that are not needed by the direct
    # DataTree-compatible implementation.
    kwargs.pop("flatten", None)
    kwargs.pop("coords", None)
    kwargs.pop("num_pp_samples", None)

    if kwargs:
        warnings.warn(
            "Ignoring unsupported plot_ppc_by_cond keyword arguments: %s"
            % ", ".join(sorted(kwargs.keys()))
        )

    obs = _as_dataset(infdata.observed_data)
    ppc = _as_dataset(infdata.posterior_predictive)
    obs_dim = "obs_id"
    if obs_dim not in obs.dims or obs_dim not in ppc.dims:
        raise ValueError("observed_data and posterior_predictive must have an obs_id dimension.")
    if var_name not in obs or var_name not in ppc:
        raise KeyError("%s must exist in observed_data and posterior_predictive." % var_name)

    mask = np.ones(obs.sizes[obs_dim], dtype=bool)

    if isinstance(subj_idx, list):
        mask &= np.isin(_values(obs, "subj_idx"), subj_idx)
    elif isinstance(subj_idx, str):
        if subj_idx != "all":
            raise ValueError("subj_idx must be 'all' or a list of subject indices")
    else:
        if subj_idx is not None:
            raise ValueError("subj_idx must be 'all' or a list of subject indices")

    condition_var_names = None
    if condition_vars is not None:
        if isinstance(condition_vars, str) or isinstance(condition_vars, list):
            condition_var_names = (
                [condition_vars]
                if isinstance(condition_vars, str)
                else condition_vars
            )
        elif isinstance(condition_vars, dict):
            condition_var_names = list(condition_vars.keys())
            for key, value in condition_vars.items():
                mask &= np.isin(_values(obs, key), value)
        else:
            raise ValueError("condition_vars must be a str, list, dict, or None.")

        if subj_idx is None:
            total_vars = condition_var_names
        else:
            total_vars = ["subj_idx"] + condition_var_names
    else:
        total_vars = ["subj_idx"]

    selected = np.flatnonzero(mask)
    if selected.size == 0:
        raise ValueError("No observations match the requested subject/condition selection.")

    label_columns = [np.asarray(_values(obs, col))[selected] for col in total_vars]
    labels = np.asarray([
        "_".join(
            "%s(%s)" % (col[:4], value)
            for col, value in zip(total_vars, row)
        )
        for row in zip(*label_columns)
    ])
    unique_labels = pd.unique(labels)

    obs_values = np.asarray(obs[var_name].isel({obs_dim: selected}).values)
    ppc_var = ppc[var_name].isel({obs_dim: selected})
    ppc_values = np.asarray(ppc_var.values)
    obs_axis = ppc_var.get_axis_num(obs_dim)
    ppc_values = np.moveaxis(ppc_values, obs_axis, -1)
    ppc_values = ppc_values.reshape((-1, selected.size))

    if num_pp_samples is not None and num_pp_samples < ppc_values.shape[0]:
        draw_idx = rng.choice(ppc_values.shape[0], size=num_pp_samples, replace=False)
        ppc_values = ppc_values[draw_idx]

    n_plots = len(unique_labels)
    n_cols = min(max_cols, n_plots)
    n_rows = int(np.ceil(float(n_plots) / n_cols))
    if figsize is None:
        figsize = (5 * n_cols, 3.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax, label in zip(axes_flat, unique_labels):
        group_mask = labels == label
        observed = np.asarray(obs_values[group_mask], dtype=float)
        predictive = np.asarray(ppc_values[:, group_mask], dtype=float).ravel()
        observed = observed[np.isfinite(observed)]
        predictive = predictive[np.isfinite(predictive)]

        if predictive.size:
            ax.hist(
                predictive,
                bins=bins,
                density=True,
                alpha=alpha,
                color=predictive_color,
                label="posterior predictive",
            )
        if observed.size:
            ax.hist(
                observed,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=observed_color,
                label="observed",
            )

        title_kwargs = {}
        label_kwargs = {}
        if textsize is not None:
            title_kwargs["fontsize"] = textsize
            label_kwargs["fontsize"] = textsize
        ax.set_title(label, **title_kwargs)
        ax.set_xlabel(var_name, **label_kwargs)
        ax.set_ylabel("density", **label_kwargs)
        if legend:
            ax.legend(fontsize=textsize)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.tight_layout()
    return axes
