"""
MICE class implements the Multivariate Imputation by Chained Equations
algorithm for imputing data that are missing at random (MAR) or missing
completely at random (MCAR).  It can also be used to impute data that
have a monotone missing data pattern and, in this case, optionally apply
adjustment factors to handle data that are missing not at random (MNAR).

For details on the MICE algorithm, see:
Stef van Buuren, Flexible Imputation of Missing Data, Second Edition
(Chapman and Hall, 2018), Section 4.5: Fully conditional specification,
Algorithm 4.3: MICE algorithm for imputation of multivariate missing data,
and Section 4.3: Monotone data imputation, Algorithm 4.1: Monotone data
imputation of multivariate missing data.
https://stefvanbuuren.name/fimd/

The MICE class does single (rather than multiple) imputation.  That is,
the MICE impute method assumes m=1, and returns a single version of the
data with the missing values replaced by imputed values.  If multiple
imputation is desired, use m MICE class objects each initialized with
a different random number seed to generate m imputed data sets.

The MICE class estimates each chained equation using the scikit-learn
package's ExtraTreeRegressor class using the fit method followed by
imputation using the apply method (rather than the predict method).
Note that "ExtraTree" is an abbreviation for the "Extremely Randomized Tree"
algorithm, which is an enhancement of the random forest algorithm.
For documentation on the ExtraTreeRegressor class, see:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
For the original article, see:
Pierre Geurts, et al., "Extremely randomized trees," Machine Learning
63(1):3-42, 2006
https://link.springer.com/content/pdf/10.1007/s10994-006-6226-1.pdf

The use of the ExtraTreeRegressor in each chained equation follows the
van Buuren book's Algorithm 3.4 for univariate imputation using a tree model.
"""

import time
import numpy as np
from sklearn.tree import ExtraTreeRegressor


class MICE:
    """
    MICE class constructor.
    """

    def __init__(
        self,
        # the MICE.impute method has an X data sample argument that is
        # a np.ndarray with shape (x_obs, x_var) where np.nan denotes
        # missing values; here are the X related arguments:
        x_obs,  # number of observations in the data sample X
        x_var,  # number of variables in the data sample X
        x_idx,  # list of X variable indexes to loop through in order
        x_ign,  # list of X variable indexes to ignore in tree growing
        # optional monotone missing data pattern argument:
        monotone=False,
        # ... No checking to see if x_idx and X are actually monotone;
        # ... that is the responsibility of the script using the class.
        # ... For a discussion of what constitutes a monotone missing
        # ... data pattern, read van Buuren book's Section 4.1: Missing
        # ... data pattern.
        iters=10,  # integer > 0 : MICE M (number of iterations)
        # Note that iters should be set to 1 (one) when monotone=True.
        verbose=False,  # if True, write impute progress to stdout
        # see ExtraTree documentation for detail on following three arguments:
        seed=123456789,  # integer in [1,999999999] : ET random_state
        # ... See the discussion above for the distinction between
        # ... multiple and single imputation, and the role of this
        # ... random-number-generator seed argument.  The seed value
        # ... also affects tree growth via the extremely randomized
        # ... tree algorithm.
        # following two arguments relate directly to the MICE algorithm:
        min_node_split_size=2,  # integer > 1 : ET min_samples_split
        min_leaf_node_size=1,  # integer > 0 : ET min_samples_leaf
        # ... The above two arguments have default values that allow
        # ... the tree to grow very large if the X data sample has
        # ... many observations.  If desired, the size of the tree
        # ... can be limited by increasing the value of one or both
        # ... of these arguments (where 'size' refers to the number
        # ... of data observations in a tree node).
        # optional post-MICE-imputation adjustment arguments:
        shift=None,  # None is equivalent to [0]*len(x_idx)
        post_shift_min=None,  # None is equivalent to [-inf]*len(x_idx)
        post_shift_max=None,  # None is equivalent to [+inf]*len(x_idx)
        zero_below_abs=None,  # None is equivalent to [0]*len(x_idx)
        ovar_idx=None,  # None is equivalent to [-1]*len(x_idx)
        zero_ovar_below_abs=None,  # None is equivalent to [0]*len(x_idx)
        convert_zero_prob=None,  # None is equivalent to [0]*len(x_idx)
        scale=None,  # None is equivalent to [1]*len(x_idx)
        # ... The above eight arguments are active only when monotone=True.
        # ... If not None, each must be a list of adjustment parameters
        # ... corresponding to the x_idx list of variables.  The shift
        # ... parameter specifies the size of an additive adjustment
        # ... factor, with the post-shift imputed value being constrained
        # ... to be in the [post_shift_min, post_shift_max] range.
        # ... The zero_below_abs parameter specifies the absolute value
        # ... of the variable below which the imputed value is converted
        # ... to zero.  The ovar_idx parameter specifies the index of
        # ... another variable that controls the converting to zero.
        # ... The zero_ovar_below_abs parameter specifies the absolute
        # ... value of the other variable below which the imputed value
        # ... is converted to zero.  The convert_zero_prob parameter
        # ... specifies the probability that a zero value for a variable
        # ... (regardless of origin: naturally imputed, or created by
        # ... adjustment steps) is converted to a randomly-selected
        # ... nonzero value from the same tree leaf (or from all nonzero
        # ... values if the leaf contains no nonzero values).  If there
        # ... are no nonzero values at all, the conversion is skipped.
        # ... The scale parameter specifies the size of a multiplicative
        # ... adjustment factor whose value must be greater than zero.
        # ... Note that the shift, zero adjustments, and scale are done
        # ... in the order of the adjustment parameters listed above.
        # ... Obviously, these adjustments are appropriate only for
        # ... continuous variables; be sure to leave all these arguments
        # ... at their default value for each categorical variable.
    ):
        # process arguments describing X data sample
        assert isinstance(x_obs, int), "x_obs must be an integer"
        assert x_obs > 0, "x_obs <= 0"
        self.n_obs = x_obs
        assert isinstance(x_var, int), "x_var must be an integer"
        assert x_var > 0, "x_var <= 0"
        self.n_var = x_var
        assert isinstance(x_idx, list), "x_idx must be a list"
        assert len(x_idx) == len(set(x_idx)), "x_idx contains duplicates"
        assert len(x_idx) > 0, "len(x_idx) <= 0 (no missing data)"
        assert min(x_idx) >= 0, "min(x_idx) < 0"
        assert max(x_idx) < x_var, f"max(x_idx) >= x_var={x_var}"
        self.x_idx = x_idx
        assert isinstance(x_ign, list), "x_ign must be a list"
        assert len(x_ign) == len(set(x_ign)), "x_ign contains duplicates"
        assert (
            set(x_ign).intersection(set(x_idx)) == set()
        ), "x_ign contains index in x_idx (variables with missing data)"
        if len(x_ign) > 0:
            assert min(x_ign) >= 0, "min(x_ign) < 0"
            assert max(x_ign) < x_var, f"max(x_ign) >= x_var={x_var}"
        self.x_ign = x_ign
        # process arguments related to monotone missing data pattern
        assert isinstance(monotone, bool), "monotone must be a boolean"
        self.monotone = monotone
        if monotone:
            if shift is None:
                shift = [0.0] * len(x_idx)
            else:
                assert isinstance(shift, list), "shift must be a list"
                assert len(shift) == len(x_idx), "len(shift) != len(x_idx)"
                for val in shift:
                    assert isinstance(
                        val, (int, float)
                    ), f"shift={val} must be an integer or float"
            if post_shift_min is None:
                post_shift_min = [-np.inf] * len(x_idx)
            else:
                assert isinstance(
                    post_shift_min, list
                ), "post_shift_min must be a list"
                assert len(post_shift_min) == len(
                    x_idx
                ), "len(post_shift_min) != len(x_idx)"
                for val in post_shift_min:
                    assert isinstance(
                        val, (int, float)
                    ), f"post_shift_min={val} must be an integer or float"
            if post_shift_max is None:
                post_shift_max = [np.inf] * len(x_idx)
            else:
                assert isinstance(
                    post_shift_max, list
                ), "post_shift_max must be a list"
                assert len(post_shift_max) == len(
                    x_idx
                ), "len(post_shift_max) != len(x_idx)"
                for val in post_shift_max:
                    assert isinstance(
                        val, (int, float)
                    ), f"post_shift_max={val} must be an integer or float"
            if zero_below_abs is None:
                zero_below_abs = [0.0] * len(x_idx)
            else:
                assert isinstance(
                    zero_below_abs, list
                ), "zero_below_abs must be a list"
                assert len(zero_below_abs) == len(
                    x_idx
                ), "len(zero_below_abs) != len(x_idx)"
                for val in zero_below_abs:
                    assert isinstance(
                        val, (int, float)
                    ), f"zero_below_abs={val} must be an integer or float"
                    assert (
                        val >= 0
                    ), f"zero_below_abs={val} must be non-negative"
            if ovar_idx is None:
                ovar_idx = [-1] * len(x_idx)
            else:
                assert isinstance(ovar_idx, list), "ovar_idx must be a list"
                assert len(ovar_idx) == len(
                    x_idx
                ), "len(ovar_idx) != len(x_idx)"
                for val in ovar_idx:
                    assert isinstance(
                        val, int
                    ), f"ovar_idx={val} must be an integer"
                    assert val >= -1, f"ovar_idx={val} must be no less than -1"
                    if val >= 0:
                        assert val < len(
                            x_idx
                        ), f"ovar_idx={val} must be less than {len(x_idx)}"
            if zero_ovar_below_abs is None:
                zero_ovar_below_abs = [0.0] * len(x_idx)
            else:
                assert isinstance(
                    zero_ovar_below_abs, list
                ), "zero_ovar_below_abs must be a list"
                assert len(zero_ovar_below_abs) == len(
                    x_idx
                ), "len(zero_ovar_below_abs) != len(x_idx)"
                for val in zero_ovar_below_abs:
                    assert isinstance(val, (int, float)), (
                        f"zero_ovar_below_abs={val} must "
                        "be an integer or float"
                    )
                    assert (
                        val >= 0
                    ), f"zero_ovar_below_abs={val} must be non-negative"
            if convert_zero_prob is None:
                convert_zero_prob = [0.0] * len(x_idx)
            else:
                assert isinstance(
                    convert_zero_prob, list
                ), "convert_zero_prob must be a list"
                assert len(convert_zero_prob) == len(
                    x_idx
                ), "len(convert_zero_prob) != len(x_idx)"
                for val in convert_zero_prob:
                    assert isinstance(
                        val, (int, float)
                    ), f"scale={val} must be an integer or float"
            if scale is None:
                scale = [1.0] * len(x_idx)
            else:
                assert isinstance(scale, list), "scale must be a list"
                assert len(scale) == len(x_idx), "len(scale) != len(x_idx)"
                for val in scale:
                    assert isinstance(
                        val, (int, float)
                    ), f"scale={val} must be an integer or float"
        else:  # if monotone is False
            assert shift is None, "shift must be None if monotone is False"
            assert (
                post_shift_min is None
            ), "post_shift_min must be None if monotone is False"
            assert (
                post_shift_max is None
            ), "post_shift_max must be None if monotone is False"
            assert (
                zero_below_abs is None
            ), "zero_below_abs must be None if monotone is False"
            assert (
                ovar_idx is None
            ), "ovar_idx must be None if monotone is False"
            assert (
                zero_ovar_below_abs is None
            ), "zero_ovar_below_abs must be None if monotone is False"
            assert (
                convert_zero_prob is None
            ), "convert_zero_prob must be None if monotone is False"
            assert scale is None, "scale must be None if monotone is False"
        self.shift = shift
        self.post_shift_min = post_shift_min
        self.post_shift_max = post_shift_max
        self.zero_below_abs = zero_below_abs
        self.ovar_idx = ovar_idx
        self.zero_ovar_below_abs = zero_ovar_below_abs
        self.convert_zero_prob = convert_zero_prob
        self.scale = scale
        # process arguments related to ExtraTreeRegressor
        assert isinstance(seed, int), "seed must be an integer"
        assert seed > 0, "seed must be positive"
        assert seed <= 999_999_999, "seed must be no greater than 999999999"
        self._rng_seed = seed
        assert isinstance(
            min_node_split_size, int
        ), "min_node_split_size must be an integer"
        assert (
            min_node_split_size >= 2
        ), f"min_node_split_size={min_node_split_size} must be 2 or larger"
        self._min_node_split_size = min_node_split_size
        assert isinstance(
            min_leaf_node_size, int
        ), "min_leaf_node_size must be an integer"
        assert (
            min_leaf_node_size >= 1
        ), f"min_leaf_node_size={min_leaf_node_size} must be 1 or larger"
        self._min_leaf_node_size = min_leaf_node_size
        # process arguments related to MICE algorithm
        assert isinstance(iters, int), "iters must be an integer"
        assert iters >= 1, "iters must be 1 or larger"
        self.n_iters = iters
        if monotone:
            assert iters == 1, "iters must be 1 when monotone is True"
        self.verbose = verbose
        # create empty imputed value distribution statistics
        self.ival_mean = np.zeros((x_var, iters + 1))
        self.ival_sdev = np.zeros((x_var, iters + 1))
        self.ival_min = np.zeros((x_var, iters + 1))
        self.ival_max = np.zeros((x_var, iters + 1))

    def impute(
        self, X  # np.ndarray shape (x_obs, x_var) where np.nan denotes missing
    ):
        """
        Returns np.ndarray with imputed values replacing missing (np.nan)
        values in X and with non-missing values in X being unchanged.
        """
        time0 = time.time()
        if self.verbose:
            print(
                (
                    "... begin MICE.impute method with "
                    f"{self.n_obs} observations"
                )
            )
            print(
                (
                    "                         and with "
                    f"{self.n_var} variables"
                )
            )
            print(
                (
                    f"                           where {len(self.x_idx)} "
                    "variables have missing values and"
                )
            )
            print(
                (
                    f"                           where {len(self.x_ign)} "
                    "variables are being ignored"
                )
            )
            print(
                (
                    "                         and with "
                    f"{self._min_node_split_size} as "
                    "minimum tree node split size"
                )
            )
            print(
                (
                    "                         and with "
                    f"{self._min_leaf_node_size} as "
                    "minimum tree leaf node size"
                )
            )
            if self.monotone:
                print(
                    (
                        "... MICE.impute method is assuming a "
                        "monotone missing data pattern"
                    )
                )
        # check X against class constructor arguments
        assert isinstance(X, np.ndarray), "X is not a np.ndarray"
        assert X.shape == (self.n_obs, self.n_var), (
            f"unexpected X.shape=({X.shape[0]},{X.shape[1]}) != "
            f"ctor=({self.n_obs},{self.n_var})"
        )
        Xmiss = np.isnan(X)
        Vmiss = Xmiss.any(axis=0)
        Vmiss_set = set(np.where(Vmiss)[0].tolist()) - set(self.x_ign)
        assert set(self.x_idx) == Vmiss_set, (
            "unexpected X missing pattern: "
            f"x_idx={set(self.x_idx)} while "
            f"Vmiss={Vmiss_set}"
        )
        del Vmiss
        # initialize missing values using bootstrap sample of observed values
        if self.monotone:
            if self.verbose:
                print(
                    (
                        "... skipping initialization "
                        f"(elapsed time: {(time.time() - time0):.1f} secs)"
                    )
                )
        else:
            rng = np.random.default_rng(self._rng_seed)
            for idx in self.x_idx:
                Yobs = X[~Xmiss[:, idx], idx]
                num_Ymis = X[Xmiss[:, idx], idx].size
                predicted = rng.choice(Yobs, size=num_Ymis, replace=True)
                X[Xmiss[:, idx], idx] = predicted
                self.ival_mean[idx, 0] = predicted.mean()
                self.ival_sdev[idx, 0] = predicted.std()
                self.ival_min[idx, 0] = predicted.min()
                self.ival_max[idx, 0] = predicted.max()
            del rng
            if self.verbose:
                print(
                    (
                        "... finish with initialization "
                        f"(elapsed time: {(time.time() - time0):.1f} secs)"
                    )
                )
        # iteratively impute missing values in X
        rng_seed = self._rng_seed
        for itr in range(1, self.n_iters + 1):
            # impute missing values for each variable in specified order
            if self.monotone:
                prior_num_Ymis = 0
            for iii, idx in enumerate(self.x_idx):
                # generate imputed values for variable idx
                # ... create ExtraTree model for idx variable
                model = ExtraTreeRegressor(
                    random_state=rng_seed,
                    min_samples_split=self._min_node_split_size,
                    min_samples_leaf=self._min_leaf_node_size,
                )
                # ... fit imputation model for idx variable
                if self.monotone:
                    del_indexes = self.x_idx[iii:] + self.x_ign
                    num_Ymis = X[Xmiss[:, idx], idx].size
                    assert num_Ymis >= prior_num_Ymis, (
                        f"non-monotone missing data pattern for idx={idx}: "
                        f"num_missing={num_Ymis} < "
                        f"num_missing={prior_num_Ymis} for "
                        f"prior idx={self.x_idx[iii - 1]}"
                    )
                    prior_num_Ymis = num_Ymis
                else:
                    del_indexes = [idx] + self.x_ign
                Zobs = np.delete(X, del_indexes, axis=1)[~Xmiss[:, idx], :]
                assert (
                    Zobs.shape[1] > 0
                ), f"Zobs contains no variables for idx={idx}"
                Yobs = X[~Xmiss[:, idx], idx]
                model.fit(Zobs, Yobs)
                # ... generate imputed values for idx variable
                Zmis = np.delete(X, del_indexes, axis=1)[Xmiss[:, idx], :]
                LZmis = model.apply(Zmis)
                LZobs = model.apply(Zobs)
                del model
                leaves = set(LZmis.tolist())
                predicted = np.zeros_like(X[Xmiss[:, idx], idx])
                rng = np.random.default_rng(rng_seed)
                if self.verbose:
                    leafsizes = []
                for leaf in leaves:
                    leaf_mask = LZmis == leaf
                    num_obs = np.count_nonzero(leaf_mask)
                    Yobs_leaf = Yobs.compress(LZobs == leaf)
                    predicted[leaf_mask] = rng.choice(
                        Yobs_leaf,
                        size=num_obs,
                        replace=True,
                    )
                    if self.verbose:
                        leafsizes.append(Yobs_leaf.size)
                del rng
                if self.verbose:
                    print(
                        (
                            f"... for iter={itr} idx={idx} : "
                            f"num_leaves={len(leafsizes)} "
                            f"min_leaf_size={min(leafsizes)} "
                            f"max_leaf_size={max(leafsizes)}"
                        )
                    )
                # ... optionally adjust predicted values
                if self.monotone:
                    if self.verbose:
                        pre_adj_predicted = predicted.copy()
                    predicted += self.shift[iii]
                    predicted = np.maximum(self.post_shift_min[iii], predicted)
                    predicted = np.minimum(predicted, self.post_shift_max[iii])
                    predicted = np.where(
                        np.absolute(predicted) < self.zero_below_abs[iii],
                        0.0,
                        predicted,
                    )
                    ovar_idx = self.ovar_idx[iii]
                    if ovar_idx >= 0:
                        ovar = X[Xmiss[:, idx], ovar_idx]
                        predicted = np.where(
                            np.absolute(ovar) < self.zero_ovar_below_abs[iii],
                            0.0,
                            predicted,
                        )
                    if self.convert_zero_prob[iii] > 0.0:
                        zero_indices = np.where(predicted == 0.0)[0]
                        nonzero_mask = predicted != 0.0
                        # proceed with this adjustment only
                        # if there are zeros to convert and
                        # nonzeros to sample from
                        if len(zero_indices) > 0 and np.any(nonzero_mask):
                            rng = np.random.default_rng(rng_seed)
                            # determine which zeros to convert
                            replace_mask = (
                                rng.random(len(zero_indices))
                                < self.convert_zero_prob[iii]
                            )
                            indices_to_replace = zero_indices[replace_mask]
                            if len(indices_to_replace) > 0:
                                # group indices by their leaf assignments
                                for leaf in set(
                                    LZmis[indices_to_replace].tolist()
                                ):
                                    # find zeros in this leaf that
                                    # need to be converted to nonzeros
                                    leaf_zero_mask = (
                                        LZmis[indices_to_replace] == leaf
                                    )
                                    leaf_indices_to_replace = (
                                        indices_to_replace[leaf_zero_mask]
                                    )
                                    # get nonzero values from the same leaf
                                    same_leaf_nonzero_mask = (
                                        LZmis == leaf
                                    ) & nonzero_mask
                                    leaf_nonzero_values = predicted[
                                        same_leaf_nonzero_mask
                                    ]
                                    if len(leaf_nonzero_values) > 0:
                                        # sample from same leaf
                                        predicted[leaf_indices_to_replace] = (
                                            rng.choice(
                                                leaf_nonzero_values,
                                                size=len(
                                                    leaf_indices_to_replace
                                                ),
                                                replace=True,
                                            )
                                        )
                                    else:
                                        # fall back to all nonzero values
                                        # if leaf has no nonzeros
                                        all_nonzero_values = predicted[
                                            nonzero_mask
                                        ]
                                        predicted[leaf_indices_to_replace] = (
                                            rng.choice(
                                                all_nonzero_values,
                                                size=len(
                                                    leaf_indices_to_replace
                                                ),
                                                replace=True,
                                            )
                                        )
                            del rng
                    predicted *= self.scale[iii]
                    if self.verbose:
                        if not np.allclose(predicted, pre_adj_predicted):
                            print(
                                (
                                    f"... for idx={idx} : applied some "
                                    "adjustments to imputed values"
                                )
                            )
                        del pre_adj_predicted
                # ... store imputed (that is, predicted) values in X
                X[Xmiss[:, idx], idx] = predicted
                # store statistics on imputed values for variable idx
                self.ival_mean[idx, itr] = predicted.mean()
                self.ival_sdev[idx, itr] = predicted.std()
                self.ival_min[idx, itr] = predicted.min()
                self.ival_max[idx, itr] = predicted.max()
            # change random number generator seed for next iteration
            rng_seed += 1_111_111
            if self.verbose:
                print(
                    (
                        f"... finish with iteration {itr} "
                        f"(elapsed time: {(time.time() - time0):.1f} secs)"
                    )
                )
        return X

    @property
    def random_number_seed(self):
        """
        Returns seed value specified in class constructor.
        """
        return self._rng_seed

    @property
    def min_node_split_size(self):
        """
        Returns min_node_split_size value specified in class constructor,
        the minimum number of X data observations required to try splitting
        a tree node.
        """
        return self._min_node_split_size

    @property
    def min_leaf_node_size(self):
        """
        Returns min_leaf_node_size value specified in class constructor,
        the minimum allowable number of X data observations in each tree
        leaf node.
        """
        return self._min_leaf_node_size

    @property
    def iterations(self):
        """
        Returns the number of iterations, denoted by M in the MICE algorithm.
        """
        return self.n_iters

    def get_ival_stats(self):
        """
        Returns tuple containing four np.ndarray each with shape (x_var,iters).
        The first tuple item contains the mean of the imputed values; the
        second tuple item contains the standard deviation of imputed values;
        the third tuple item contains the minimum of the imputed values; and
        the fourth tuple item contains the maximum of the imputed values.
        """
        return (self.ival_mean, self.ival_sdev, self.ival_min, self.ival_max)
