# mypy: allow-untyped-defs
"""
This file contains utilities for tracing through __torch_dispatch__ based tensor subclasses and modes.
AOTAutograd's responsibility is to trace through all pytorch capabilities that live in the pytorch dispatcher,
and this includes tensor subclasses that implement __torch_dispatch__.
"""

import typing
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.fake_tensor import get_plain_tensors
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .schemas import (
    MutationType,
    PlainTensorMeta,
    SubclassCreationMeta,
    ViewAndMutationMeta,
)
from .utils import strict_zip


zip = strict_zip


def requires_subclass_dispatch(args, fw_metadata: ViewAndMutationMeta) -> bool:
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any(
        is_traceable_wrapper_subclass(x)
        for x in args_flattened
        if isinstance(x, Tensor)
    )
    from torch._functorch._aot_autograd.schemas import SubclassCreationMeta

    any_subclass_outputs = any(
        type(x) is SubclassCreationMeta for x in fw_metadata.subclass_fw_graph_out_meta
    )
    # This tells us whether or not we need to perform any unwrapping/wrapping of tensor subclasses at runtime.
    return any_subclass_args or any_subclass_outputs


# Given a real tensor subclass, returns a nested list of Plain tensor types
def get_types_for_subclass(tensor_subclass):
    if not is_traceable_wrapper_subclass(tensor_subclass):
        return ["Tensor"]
    inner_keys, _ = tensor_subclass.__tensor_flatten__()
    result = []
    for key in inner_keys:
        inner_tensor = getattr(tensor_subclass, key)
        result.extend(get_types_for_subclass(inner_tensor))
    return result


suggest_memory_format = torch._prims_common.suggest_memory_format


def maybe_suggest_memory_format(
    t, with_memory_format: bool
) -> Optional[torch.memory_format]:
    if not with_memory_format:
        return None

    return suggest_memory_format(t)


def create_subclass_metadata(
    a,
    start_idx,
    *,
    with_memory_format: bool = False,
):
    if not is_traceable_wrapper_subclass(a):
        idx = start_idx + 1
        return (
            PlainTensorMeta(
                idx, memory_format=maybe_suggest_memory_format(a, with_memory_format)
            ),
            idx,
        )

    inner_keys, metadata = a.__tensor_flatten__()
    new_start_idx = start_idx
    attrs = {}

    for key in inner_keys:
        new_subclass_meta, new_start_idx = create_subclass_metadata(
            getattr(a, key), new_start_idx, with_memory_format=with_memory_format
        )
        attrs[key] = new_subclass_meta

    # It *must* be because is_traceable_wrapper_subclass() - but mypy is not smart.
    assert isinstance(a, Tensor)

    return (
        SubclassCreationMeta(
            flat_tensor_start_idx=start_idx,
            arg_count=new_start_idx - start_idx,
            attrs=attrs,
            meta=metadata,
            outer_size=a.size(),  # type: ignore[attr-defined, arg-type]
            outer_stride=a.stride(),  # type: ignore[arg-type]
            original_subclass=a,
            memory_format=maybe_suggest_memory_format(a, with_memory_format),
        ),
        new_start_idx,
    )


# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(
    curr_args: Union[List[Any], Tuple[Any, ...]],
    with_memory_format: bool = False,
) -> List[Union[PlainTensorMeta, SubclassCreationMeta]]:
    idx = 0
    infos: List[Union[PlainTensorMeta, SubclassCreationMeta]] = []
    for a in curr_args:
        if is_traceable_wrapper_subclass(a):
            assert isinstance(a, Tensor)
            start_idx = idx
            subclass_meta, _ = create_subclass_metadata(
                a,
                start_idx,
                with_memory_format=with_memory_format,
            )
            infos.append(subclass_meta)
            cnt = subclass_meta.arg_count
        else:
            infos.append(
                PlainTensorMeta(
                    idx,
                    memory_format=maybe_suggest_memory_format(a, with_memory_format),
                )
            )
            cnt = 1
        idx += cnt
    return infos


# Output structure:
# - List[Tensor] if tracing an inference graph
# - Tuple[List[Tensor], List[Tensor]] if tracing a joint graph.
# This function effectively concats each inner list of subclass tensors
# into a (potentially longer) list of inner tensors.
#
# This function takes in a pytree of arguments and unwraps any tensor subclasses.
# Annoyingly, we can't use pytrees to perform the unwrapping, because unwrapping returns
# a list of tensors that we would then need to concat together.
# Instead, we specialize the logic for the inference vs. joint graph case.
# NOTE: this function is hot, since we unwrap tensor subclass inputs at runtime
def unwrap_tensor_subclasses(wrapped_args, *, is_joint_structure: bool):
    def concat_inner_tensors_from_subclasses(xs):
        xs_inner: List[Tensor] = []
        for x in xs:
            get_plain_tensors(x, out_append_list=xs_inner)
        return xs_inner

    if is_joint_structure:
        assert isinstance(wrapped_args, tuple) and len(wrapped_args) == 2
        assert isinstance(wrapped_args[0], (tuple, list)) and isinstance(
            wrapped_args[1], (tuple, list)
        )
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args[0])
        unwrapped_args_tangents = concat_inner_tensors_from_subclasses(wrapped_args[1])
        unwrapped_args = (unwrapped_args_fw, unwrapped_args_tangents)
    else:
        assert isinstance(wrapped_args, (list, tuple))
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args)
        unwrapped_args = unwrapped_args_fw
    return unwrapped_args


def unwrap_tensor_subclasses_with_indices_to_original(wrapped_args):
    ret_unwrapped = []
    ret_indices_to_original = []
    for i, a in enumerate(wrapped_args):
        a_unwrapped = unwrap_tensor_subclasses([a], is_joint_structure=False)
        ret_unwrapped.extend(a_unwrapped)
        n = len(a_unwrapped)
        ret_indices_to_original.extend([i] * n)

    return ret_unwrapped, ret_indices_to_original


def remap_unwrapped_subclass_arg_indices(wrapped_args, static_input_indices):
    static_input_indices = set(static_input_indices)
    new_ind = 0
    remapped_static_indices = []
    for i, arg in enumerate(wrapped_args):
        num_indices = 1
        if is_traceable_wrapper_subclass(arg):
            num_indices = len(get_plain_tensors(typing.cast(Tensor, arg)))

        for _ in range(num_indices):
            if i in static_input_indices:
                remapped_static_indices.append(new_ind)

            new_ind += 1

    return remapped_static_indices


# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: Union[Tuple[Any, ...], List[Any]],
    *,
    subclass_metas: List[Union[PlainTensorMeta, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int] = None,
    is_runtime: bool = False,
) -> Tuple[Any, ...]:
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, PlainTensorMeta):
            wrapped_args.append(unwrapped_args[subclass_meta.unwrapped_idx])
            num_args_tallied += 1
        else:
            assert isinstance(subclass_meta, SubclassCreationMeta)
            wrapped_args.append(
                subclass_meta.creation_fn(unwrapped_args, is_runtime=is_runtime)
            )
            num_args_tallied += subclass_meta.arg_count

    # Note: [Partitioner handling for Subclasses, Part 2]
    # At the beginning of AOTAutograd, we collect metadata on the inputs and outputs of the user fw,
    # to figure out which inputs/outputs are subclasses, and how to reconstruct the subclasses after flattening them.
    #
    # When this function is called at runtime in the forward,
    # we have been passed a list of (flattened) dense-tensor fw-outs, and need to reconstruct any subclass fw outs.
    #
    # One reasonable question that you should ask: when should the dense_tensor -> subclass_tensor wrapping happen?
    # Answer: we do it **inside of our compiled autograd.Function**.
    # This seems like morally the right place: autograd happens above subclass desugaring,
    # so autograd should see actual tensor subclasses at runtime, and not flattened dense tensors.
    #
    # This causes a tricky interaction though: when we run the min-cut partitioner to divvy up the joint graph
    # into a forward and backward graph, we end up with some activations that show up as extra outputs
    # in the compiled forward graph, that are **not** user outputs.
    # These activations are not visible to the user, and so there's no need for us to wrap them back into subclasses.
    #
    # On top of that, when we first computed subclass metadata (in `run_functionalized_fw_and_collect_metadata`),
    # we computed subclass metadata on every forward output, but this did **not** include activations
    # created by the partitioner.
    # as a result, `unwrapped_args` here will correspond to (*unwrapped_user_fw_outs, *activations),
    # but `subclass_metas` will only correspond to subclass metatadata on `user_fw_outs`.
    # We then need to make sure that we return (*wrapped_user_fw_outs, *activations).
    if num_fw_outs_saved_for_bw is not None:
        assert len(unwrapped_args) == num_args_tallied + num_fw_outs_saved_for_bw, (
            f"Expected the number actual unwrapped-subclass outputs {len(unwrapped_args)} to equal "
            f"the number of args calculated from subclasses ({num_args_tallied}) plus the number of "
            f"additional activations saved for the backward pass ({num_fw_outs_saved_for_bw})"
        )
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        assert len(unwrapped_args) == num_args_tallied
        return tuple(wrapped_args)


# Given a bunch of "dense" tensor arguments, this function (potentially) wraps them into tensor subclasses.
# This function carefully handles the inference vs. joint cases:
# - when is_joint_structure is True, args is (primals, tangents)
# - when is_joint_structure is False, args is [*primals]
def wrap_tensor_subclasses_maybe_joint(
    unwrapped_args, *, is_joint_structure: bool, meta: ViewAndMutationMeta
) -> Union[Tuple[Any, ...], List[Any]]:
    # Since this function is re-used for both inference and joint graphs,
    if is_joint_structure:
        assert isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2
        assert isinstance(unwrapped_args[0], (tuple, list)) and isinstance(
            unwrapped_args[1], (tuple, list)
        )
        primals, tangents = unwrapped_args[0], unwrapped_args[1]
        wrapped_primals = wrap_tensor_subclasses(
            primals, subclass_metas=meta.subclass_inp_meta
        )
        wrapped_tangents = wrap_tensor_subclasses(
            tangents, subclass_metas=meta.subclass_tangent_meta
        )
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(
            unwrapped_args, subclass_metas=meta.subclass_inp_meta
        )
        return wrapped_args


def compute_inner_mutated_inp_indices_from_subclass_meta(
    fw_metadata: ViewAndMutationMeta,
    inner_metadata: ViewAndMutationMeta,
) -> List[int]:
    # Note: [Recomputing subclass mutation handling]
    #
    # Generally, if a subclass requires grad, its components will not require grad.
    # But for the purposes of tracking returned tensors, we should treat those component
    # tensors as if they require grad.
    #
    # For example, if the subclass tensor requires grad and will be mutated in a way that
    # requires us to handle the mutation outside of the graph, we need to return it
    # from the forward graph. The inner_meta data won't consider the component tensors
    # as if they need to be returned, because they don't require grad; but really, we
    # should handle those tensors the same way we handle the subclass tensor itself; i.e.
    # if we'd include the subclass tensor as part of the outputs, then we should also
    # include the component tensors.
    #
    # To do this, we patch num_mutated_inp_runtime_indices below by expanding the inputs
    # from the outer subclass tensors and propagating

    updated_input_info = []
    inner_idx = 0
    if not fw_metadata.subclass_inp_meta:
        # Sometimes we don't have subclass info, e.g. synthetic_base codepaths
        return inner_metadata.mutated_inp_runtime_indices
    assert len(fw_metadata.subclass_inp_meta) == len(fw_metadata.input_info)
    for outer_idx, inp_meta in enumerate(fw_metadata.subclass_inp_meta):
        if isinstance(inp_meta, PlainTensorMeta):
            assert outer_idx < len(fw_metadata.input_info)
            if inner_metadata is not None:
                assert inner_idx < len(inner_metadata.input_info)
                assert (
                    inner_metadata.input_info[inner_idx]
                    == fw_metadata.input_info[outer_idx]
                )
            updated_input_info.append(fw_metadata.input_info[outer_idx])
            inner_idx += 1
        else:
            for _ in range(inp_meta.arg_count):
                updated_input_info.append(fw_metadata.input_info[outer_idx])
                inner_idx += 1
    if inner_metadata is not None:
        assert len(inner_metadata.input_info) == len(updated_input_info)

    return [
        i
        for i, inp in enumerate(updated_input_info)
        if inp.mutation_type == MutationType.MUTATED_OUT_GRAPH
    ]
