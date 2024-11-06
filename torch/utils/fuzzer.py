import itertools
import logging
import random
import string
import traceback
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    get_args,
    get_origin,
    List,
    Literal,
    Optional,
    Union,
)

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass


"""
# Known failures on inductor config:
cpp_wrapper, triton_debug_sync_graph
cpp_wrapper, triton_debug_sync_kernel
cpp_wrapper, disable_cpp_codegen
combo_kernels, benchmark_combo_kernel, profile_bandwidth, profile_bandwidth_regex

# Example usage:
import torch._inductor.config as cfg

fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

# Test every pair of configs:
results = fuzzer.fuzz_n_tuple(n, max_combinations=10000000)

visualize_results(n, results)

# Test random configs with bisection:
ret = fuzzer.fuzz_random_with_bisect(num_attempts=10)

fuzzer.reproduce([("triton.autotune_pointwise", "coordinate_descent_tuning")])
fuzzer.reproduce([("cpp_wrapper", "triton.debug_sync_graph")])
fuzzer.reproduce([("memory_planning", "memory_pool")])
fuzzer.reproduce([("reorder_for_compute_comm_overlap",)])
"""


def is_optional_type(type_hint) -> bool:
    origin = get_origin(type_hint)

    if origin is Union:
        args = get_args(type_hint)
        return type(None) in args

    return False


# callable types are messed up
def is_callable_type(type_hint) -> bool:
    return type_hint.__name__ == "Callable"


def is_type(type_hint, comp_type) -> bool:
    return type_hint is comp_type or get_origin(type_hint) is comp_type


class DummyPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """
        return None

    def uuid(self) -> Optional[Any]:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """
        return None


TYPE_EXEMPLARS: dict[str, Any] = {
    CustomGraphPass.__name__: DummyPass(),
    torch.fx.graph.Graph.__name__: torch.fx.graph.Graph(),
    torch._inductor.scheduler.BaseSchedulerNode.__name__: torch._inductor.scheduler.BaseSchedulerNode(
        None
    ),
}


class Status(Enum):
    SKIPPED = "skipped"
    PASSED = "passed"
    FAILED_RUN = "failed_run"
    FAILED_COMPILE = "failed_compile"

    def failing(self):
        return self == Status.FAILED_RUN or self == Status.FAILED_COMPILE


class SamplingMethod:
    TOGGLE = 1  # toggle to the opposite value
    RANDOM = 2  # randomly choose an option

    def _generate_toggle_value_for_type(type_hint: type, default: Any) -> Any:
        """this toggle setting will use randomness too, but if there's a sensible 'toggle', it will use that"""
        if type_hint == bool:
            return not default
        elif type_hint == int:
            return random.randint(0, 1000)
        elif type_hint == float:
            return random.uniform(0, 1000)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif is_type(type_hint, list):
            elem_type = getattr(
                type_hint,
                "__args__",
                [type(default[0])] if len(default) else [type(None)],
            )[0]
            new_default = default[0] if len(default) > 0 else None
            return [
                SamplingMethod._generate_toggle_value_for_type(elem_type, new_default)
                for _ in range(random.randint(0, 3))
            ]
        elif is_type(type_hint, dict):
            key_type, value_type = type_hint.__args__
            items = list(default.items())
            if len(items) > 0:
                default_key, default_val = items[0]
                return {
                    SamplingMethod._generate_toggle_value_for_type(
                        key_type, default_key
                    ): SamplingMethod._generate_toggle_value_for_type(
                        value_type, default_val
                    )
                    for _ in range(random.randint(0, 3))
                }
            else:
                # fall back to random
                return {
                    SamplingMethod._generate_random_value_for_type(
                        key_type, None
                    ): SamplingMethod._generate_random_value_for_type(value_type, None)
                    for _ in range(random.randint(0, 3))
                }
        elif is_type(type_hint, Union):
            # do whatever is not the type of default
            assert len(type_hint.__args__) > 1
            new_type = random.choice(
                [t for t in type_hint.__args__ if t != type(default)]
            )
            try:
                return SamplingMethod._generate_random_value_for_type(
                    new_type, new_type()
                )
            except:
                # if default constructor doesn't work, try None
                try:
                    return SamplingMethod._generate_random_value_for_type(
                        new_type, None
                    )
                except:
                    return default
        elif is_type(type_hint, tuple):
            zipped = zip(type_hint.__args__, default)
            return tuple(
                map(
                    lambda x: SamplingMethod._generate_toggle_value_for_type(
                        x[0], x[1]
                    ),
                    zipped,
                )
            )
        elif is_type(type_hint, Literal):
            return random.choice([t for t in type_hint.__args__ if t != type(default)])
        elif is_optional_type(type_hint):
            elem_type = type_hint.__args__[0]
            if default is None:
                return SamplingMethod._generate_random_value_for_type(elem_type)
            else:
                return None
        elif type_hint is type(None):
            # needed for recursive calls
            return None
        elif is_callable_type(type_hint):
            input_args, return_type = (
                list(type_hint.__args__)[:-1],
                list(type_hint.__args__)[-1],
            )

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs) -> return_type:
                return SamplingMethod._generate_random_value_for_type(return_type)

            return dummy_function
        elif type_hint.__name__ in TYPE_EXEMPLARS:
            return TYPE_EXEMPLARS[type_hint.__name__]
        elif type_hint == Any:
            return 1 if not default == 1 else 2
        else:
            raise Exception(f"Unable to process type {type_hint}. PRs welcome :)")

    def _generate_random_value_for_type(type_hint: type, _default: Any = None) -> Any:
        """Generate a random value for a given type."""
        if type_hint == bool:
            return random.choice([True, False])
        elif type_hint == int:
            return random.randint(0, 1000)
        elif type_hint == float:
            return random.uniform(0, 1000)
        elif type_hint == str:
            characters = string.ascii_letters + string.digits + string.punctuation
            return "".join(
                random.choice(characters) for _ in range(random.randint(1, 20))
            )
        elif is_type(type_hint, list):
            elem_type = type_hint.__args__[0]
            return [
                SamplingMethod._generate_random_value_for_type(elem_type)
                for _ in range(random.randint(0, 3))
            ]
        elif is_type(type_hint, dict):
            key_type, value_type = type_hint.__args__
            return {
                SamplingMethod._generate_random_value_for_type(
                    key_type
                ): SamplingMethod._generate_random_value_for_type(value_type)
                for _ in range(random.randint(0, 3))
            }
        elif is_type(type_hint, Union):
            return SamplingMethod._generate_random_value_for_type(
                random.choice(type_hint.__args__)
            )
        elif is_type(type_hint, tuple):
            return tuple(
                map(SamplingMethod._generate_random_value_for_type, type_hint.__args__)
            )
        elif is_type(type_hint, Literal):
            return random.choice(type_hint.__args__)
        elif is_optional_type(type_hint):
            elem_type = type_hint.__args__[0]
            return random.choice(
                [None, SamplingMethod._generate_random_value_for_type(elem_type)]
            )
        elif type_hint is type(None):
            return None
        elif is_callable_type(type_hint):
            input_args, return_type = (
                list(type_hint.__args__)[:-1],
                list(type_hint.__args__)[-1],
            )

            @wraps(lambda *args, **kwargs: None)
            def dummy_function(*args, **kwargs) -> return_type:
                return SamplingMethod._generate_random_value_for_type(return_type)

            return dummy_function
        elif type_hint.__name__ in TYPE_EXEMPLARS:
            return TYPE_EXEMPLARS[type_hint.__name__]
        elif type_hint == Any:
            return 1
        else:
            raise Exception(f"Unable to process type {type_hint}. PRs welcome :)")

    def dispatch(sm: "SamplingMethod"):
        if sm == SamplingMethod.RANDOM:
            return SamplingMethod._generate_random_value_for_type
        elif sm == SamplingMethod.TOGGLE:
            return SamplingMethod._generate_toggle_value_for_type
        else:
            raise Exception(f"malformed sampling method: {sm}")


class Default:
    pass


DEFAULT = Default()


class ConfigFuzzer:
    def __init__(
        self,
        config_module,
        test_model_fn_factory: Callable,
        seed: int,
        default: Optional[dict[str, str]] = None,
        sm: SamplingMethod = SamplingMethod.TOGGLE,
    ):
        """
        Initialize the config fuzzer.

        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful
        """
        self.seed = seed
        self.config_module = config_module
        self.test_model_fn_factory = test_model_fn_factory
        self.fields = self.config_module._config
        self.logger = self._setup_logger()
        self.sample = SamplingMethod.dispatch(sm)
        if default is None:
            # these defaults are for inductor, running on a single GPU, TODO generalize
            self.default = {
                "force_disable_caches": True,
                "cpp.cxx": DEFAULT,
                "TYPE_CHECKING": DEFAULT,
                "max_autotune_pointwise": DEFAULT,
                "max_autotune_gemm": DEFAULT,
                "max_autotune_gemm_backends": DEFAULT,
                "max_autotune_conv_backends": DEFAULT,
                "max_autotune_gemm_search_space": DEFAULT,
                "max_autotune_subproc_result_timeout_seconds": DEFAULT,
                "max_autotune_subproc_graceful_timeout_seconds": DEFAULT,
                "max_autotune_subproc_terminate_timeout_seconds": DEFAULT,
                "autoheuristic_collect": DEFAULT,
                "autoheuristic_use": DEFAULT,
                "aot_inductor.presets": DEFAULT,
                "cuda.arch": DEFAULT,
                "cuda.version": DEFAULT,
                "cuda.cutlass_dir": DEFAULT,
                "cuda.cuda_cxx": DEFAULT,
                "rocm.arch": DEFAULT,
                "rocm.ck_supported_arch": DEFAULT,
                "rocm.ck_dir": DEFAULT,
                "rocm.rocm_home": DEFAULT,
                "enabled_metric_tables": DEFAULT,  # TODO only disabled because lack of typing
                "reorder_for_compute_comm_overlap": DEFAULT,
                "check_stack_no_cycles_TESTING_ONLY": DEFAULT,
                "triton.debug_sync_graph": DEFAULT,  # TODO disabled due to known failure
                "triton.debug_sync_kernel": DEFAULT,  # TODO disabled due to known failure
                "profile_bandwidth_regex": DEFAULT,  # TODO disabled due to know failure
                "disable_cpp_codegen": DEFAULT,  # TODO disabled due to know failure
            }
        else:
            self.default = default

    def __repr__(self):
        return f"ConfigFuzzer(config_module={self.config_module}, test_model_fn={self.test_model_fn}, seed={self.seed}, default={self.default})"

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ConfigFuzzer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _get_type_hint(self, obj, name) -> type:
        """Get type hint for a field, falling back to type(default_value) if not found."""
        try:
            hints = get_type_hints(obj)
            return hints.get(name, type(getattr(obj, name)))
        except Exception:
            return type(getattr(obj, name))

    def _set_config(self, field_name: str, value: Any):
        """Set a config value in the module."""
        setattr(self.config_module, field_name, value)

    def _reset_configs(self):
        """Reset all configs to their default values."""
        for field_name, field_obj in self.fields.items():
            self._set_config(field_name, field_obj.default)

    def _set_status(self, results_dict, combo, status):
        combo = tuple(sorted(combo))
        results_dict[combo] = status
        return results_dict

    def _lookup_status(self, results_dict, combo):
        combo = tuple(sorted(combo))
        return results_dict[combo] if combo in results_dict else None

    def new_config(self):
        """creates a new config from the defaults"""
        ret = {
            name: val if val != DEFAULT else self.fields[name].default
            for name, val in self.default.items()
        }
        return ret

    def _combo_run_common(self, combo, results):
        self.logger.info(combo)
        if self._lookup_status(results, combo):
            # we already processed this config
            return

        config = self.new_config()

        skip = False
        for field_name in combo:
            if field_name in config:
                # don't break here because we need to build the config dict
                skip = True
            if field_name.startswith("_"):
                skip = True
            field = self.fields[field_name]
            value = self.sample(field.value_type, field.default)
            config[field_name] = value
        if skip:
            self._set_status(results, combo, Status.SKIPPED)
            return

        self.test_config(config, results)

    def reproduce(self, examples: list[tuple]) -> dict[Any, Status]:
        """entrypoint to reproduce the failures"""
        results = {}
        self._reset_configs()
        self.logger.info(f"Starting repro with seed {self.seed}")
        random.seed(self.seed)
        for rep in examples:
            self._combo_run_common(rep, results)

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000) -> dict[Any, Status]:
        """
        Test every combination of n configs.

        returns a dict of this shape: {(config-1, config-2... config-n): status}
        """
        results = {}
        self._reset_configs()
        self.logger.info(f"Starting {n}-tuple testing with seed {self.seed}")
        random.seed(self.seed)

        for combo in itertools.combinations(self.fields, n):
            self._combo_run_common(combo, results)
            max_combinations -= 1
            if max_combinations <= 0:
                self.logger.info("Reached maximum combinations limit")
                break

        return results

    def test_config(self, config, results):
        self.logger.info(f"Testing config {config}")
        config_tuple = tuple(config.keys())
        if ret := self._lookup_status(results, config_tuple):
            return ret
        torch._dynamo.reset()
        test_model_fn = self.test_model_fn_factory()
        try:
            comp = torch.compile(options=config)(test_model_fn)
        except Exception as e:
            self.logger.error("Exception compiling with config combination:")
            for field, value in config.items():
                self.logger.error(f"{field} = {value}")
            traceback.print_exc()
            ret = Status.FAILED_COMPILE
            self._set_status(results, config_tuple, ret)
            return ret
        try:
            success = comp()
            if not success:
                self.logger.error("Failure with config combination:")
                for field, value in config.items():
                    self.logger.error(f"{field} = {value}")
                ret = Status.FAILED_RUN
                self._set_status(results, config_tuple, ret)
                return ret
            else:
                ret = Status.PASSED
                self._set_status(results, config_tuple, ret)
                return ret
        except Exception as e:
            self.logger.error("Exception with config combination:")
            for field, value in config.items():
                self.logger.error(f"{field} = {value}")
            traceback.print_exc()
            ret = Status.FAILED_RUN
            self._set_status(results, config_tuple, ret)
            return ret

    def fuzz_random_with_bisect(
        self, num_attempts: int = 100, p: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Randomly test configs and bisect to minimal failing configuration."""
        self.logger.info(
            f"Starting random testing with bisection, seed {self.seed}, and p {p}"
        )
        random.seed(self.seed)
        self._reset_configs()
        results = {}
        ret = []

        for attempt in range(num_attempts):
            self.logger.info(f"Random attempt {attempt + 1}/{num_attempts}")

            config = self.new_config()

            for field_name, config_entry in self.fields.items():
                if (
                    field_name not in config
                    and not field_name.startswith("_")
                    and random.random() < p
                ):
                    value = self.sample(config_entry.value_type, config_entry.default)
                    config[field_name] = value

            status = self.test_config(config, results)
            if status not in {Status.PASSED, Status.SKIPPED}:
                minimal_failing_config = self._bisect_failing_config(config, results)
                self.logger.error(f"Minimum failing config: {minimal_failing_config}")
                ret.append(minimal_failing_config)

        return ret

    def _bisect_failing_config(
        self, failing_config: Union[dict, list], results: dict
    ) -> Optional[dict]:
        """Bisect a failing configuration to find minimal set of configs that cause failure."""
        self.logger.info(f"bisecting config: {failing_config}")
        if isinstance(failing_config, dict):
            failing_config = list(failing_config.items())
        if not failing_config:
            return None

        def test(x: list):
            d = dict(x)
            result = self.test_config(d, results)
            return result

        if len(failing_config) <= 1:
            return dict(failing_config) if test(failing_config).failing() else None

        mid = len(failing_config) // 2

        first_half = failing_config[:mid]
        second_half = failing_config[mid:]
        if test(first_half).failing():
            return self._bisect_failing_config(first_half, results)
        if test(second_half).failing():
            return self._bisect_failing_config(second_half, results)

        if len(failing_config) >= 8:
            low = len(failing_config) // 4
            high = mid + low
            quart1 = failing_config[low:]
            if test(quart1).failing():
                return self._bisect_failing_config(quart1, results)
            quart2 = failing_config[:low] + second_half
            if test(quart2).failing():
                return self._bisect_failing_config(quart2, results)
            quart3 = first_half + failing_config[:high]
            if test(quart3).failing():
                return self._bisect_failing_config(quart3, results)
            quart4 = failing_config[high:]
            if test(quart4).failing():
                return self._bisect_failing_config(quart4, results)
        # try dropping one value at a time
        for i in range(len(failing_config)):
            new_list = [x for j, x in enumerate(failing_config) if j != i]
            if test(new_list).failing():
                return self._bisect_failing_config(new_list, results)
        # we have the minimal set
        return dict(failing_config)


def create_simple_test_model_cpu():
    """Create a simple test model function for demonstration."""

    def test_fn():
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
            )

            x = torch.randn(32, 10)
            y = model(x)
            return True
        except Exception as e:
            print(f"Model test failed: {str(e)}")
            return False

    return test_fn


def create_simple_test_model_gpu():
    """Create a simple test model function for demonstration."""

    batch_size = 32
    seq_length = 50
    hidden_size = 768

    def test_fn():
        inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
        weight = torch.randn(hidden_size, hidden_size, device="cuda")
        matmul_output = inp @ weight
        final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
        return True

    return test_fn


def visualize_results(n: int, status: dict, filename: str = "results.html"):
    # TODO Support more dimensions
    assert n == 2
    assert len(status) > 0

    # Create a dictionary for quick lookup of status
    input_set = set([])
    for key in status.keys():
        input_set.add(key[0])
        input_set.add(key[1])
    input_list = sorted(list(input_set))

    # Start the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title> Fuzzer Visualization</title>
        <style>
            table {
                border-collapse: collapse;
                width: 50%;
                margin: 20px auto;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            th {
                background-color: #f2f2f2;
            }
            .skipped {
                background-color: yellow;
            }
            .passed {
                background-color: green;
                color: white;
            }
            .failed {
                background-color: red;
                color: white;
            }
        </style>
    </head>
    <body>
        <h2 style="text-align: center;">Fuzzer Visualization</h2>
        <table>
        <thead>
    """

    html_content += "<tr><th>\\</th>"
    for i, col_name in enumerate(input_list):
        col = "<br>".join(col_name)
        html_content += f"<th>{col}</th>"
    html_content += "</tr></thead><tbody>"

    # Add table rows
    for i, row_name in enumerate(input_list):
        html_content += f"<tr><th>{row_name}</th>"
        for j, col_name in enumerate(input_list):
            # Determine the status class for the cell
            status_class = ""
            status_val = ""
            if (row_name, col_name) in status:
                status_enum = status[(row_name, col_name)]
                if status_enum == Status.SKIPPED:
                    status_class = "skipped"
                    status_val = "-"
                elif status_enum == Status.PASSED:
                    status_class = "passed"
                    status_val = "O"
                elif status_enum == Status.FAILED:
                    status_class = "failed"
                    status_val = "X"

            html_content += f'<td class="{status_class}">{status_val}</td>'
        html_content += "</tr>"

    html_content += """
        </tbody>
        </table>
    </body>
    </html>
    """

    with open(filename, "w") as file:
        file.write(html_content)

    print(f"HTML file '{filename}' has been generated.")
