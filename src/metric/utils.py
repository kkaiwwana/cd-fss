from torchmetrics import MetricCollection
from torchmetrics.utilities.data import _flatten_dict


class SuperMetricCollection(MetricCollection):
    """
    Metric collection that accept any args and kwargs, to support some customized metrics.
    """
    def compute(self, **kwargs):
        """Compute the result for each metric in the collection."""
        return self._compute_and_reduce("compute", **kwargs)

    def _compute_and_reduce(
        self, method_name, *args, **kwargs
    ):
        """Compute result from collection and reduce into a single dictionary.

        Args:
            method_name: The method to call on each metric in the collection.
                Should be either `compute` or `forward`.
            args: Positional arguments to pass to each metric (if method_name is `forward`)
            kwargs: Keyword arguments to pass to each metric (if method_name is `forward`)

        Raises:
            ValueError:
                If method_name is not `compute` or `forward`.

        """
        result = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            if method_name == "compute":
                res = m.compute(**kwargs)
            elif method_name == "forward":
                res = m(*args, **m._filter_kwargs(**kwargs))
            else:
                raise ValueError(f"method_name should be either 'compute' or 'forward', but got {method_name}")
            result[k] = res

        _, duplicates = _flatten_dict(result)

        flattened_results = {}
        for k, m in self.items(keep_base=True, copy_state=False):
            res = result[k]
            if isinstance(res, dict):
                for key, v in res.items():
                    # if duplicates of keys we need to add unique prefix to each key
                    if duplicates:
                        stripped_k = k.replace(getattr(m, "prefix", ""), "")
                        stripped_k = stripped_k.replace(getattr(m, "postfix", ""), "")
                        key = f"{stripped_k}_{key}"
                    if getattr(m, "_from_collection", None) and m.prefix is not None:
                        key = f"{m.prefix}{key}"
                    if getattr(m, "_from_collection", None) and m.postfix is not None:
                        key = f"{key}{m.postfix}"
                    flattened_results[key] = v
            else:
                flattened_results[k] = res
        return {self._set_name(k): v for k, v in flattened_results.items()}