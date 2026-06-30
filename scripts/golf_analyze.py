#!/usr/bin/env python3
"""Per-net official-cost breakdown for floor-net golf (stdlib + onnx only).

For a clean static-shape graph (which every floor net is, since it passes the
official scorer) the official `calculate_memory` static pass == the ORT-runtime
max, so we can reproduce the EXACT official cost locally with no ORT trace:

    cost = calculate_params(model) + sum(elements * dtype_itemsize
                                         over input ∪ value_info ∪ output,
                                         excluding the tensors named input/output)

This prints WHERE the bytes live: op histogram, bytes-by-dtype, and the top
fattest intermediates each mapped to the node (op_type) that produced them —
i.e. the fat to attack (downcast fp32->uint8/bool, crop early, kill redundant
branches, fold Pad into conv, Where->variadic-Max, QLinearConv+Cast->ConvInteger).

Usage:
    python scripts/golf_analyze.py 233 286 018            # from artifacts/submission.zip
    python scripts/golf_analyze.py --zip path.zip 054 --top 15
    python scripts/golf_analyze.py --onnx some_task.onnx
"""
import argparse, collections, math, os, zipfile
import numpy as np
import onnx
from onnx import helper, shape_inference

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ZIP = os.path.join(ROOT, "artifacts", "submission.zip")


def calculate_params(model):
    """Inlined from neurogolf_utils.calculate_params (element counts, dtype-independent)."""
    params = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        params += math.prod(init.dims)
    for sp in model.graph.sparse_initializer:
        if any(d <= 0 for d in sp.values.dims):
            return None
        params += math.prod(sp.values.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                params += math.prod(attr.t.dims)
            elif attr.name == "sparse_value":
                params += math.prod(attr.sparse_tensor.values.dims)
            elif attr.name == "value_floats":
                params += len(attr.floats)
            elif attr.name == "value_ints":
                params += len(attr.ints)
            elif attr.name == "value_strings":
                params += len(attr.strings)
    return params


def analyze(model, label, top):
    params = calculate_params(model)
    g = shape_inference.infer_shapes(model, strict_mode=True).graph

    producer = {}                       # tensor name -> (op_type, node display)
    op_hist = collections.Counter()
    for node in g.node:
        op_hist[node.op_type] += 1
        disp = node.name or (node.output[0] if node.output else "?")
        for o in node.output:
            if o:
                producer[o] = (node.op_type, disp)

    rows = []                           # (bytes, name, dims, dtype, op_type)
    for item in list(g.input) + list(g.value_info) + list(g.output):
        nm = item.name
        if nm in ("input", "output"):
            continue
        if not item.type.HasField("tensor_type"):
            continue
        tt = item.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = [d.dim_value for d in tt.shape.dim]
        if any(d <= 0 for d in dims):
            continue
        ne = math.prod(dims) if dims else 1
        dt = np.dtype(helper.tensor_dtype_to_np_dtype(tt.elem_type))
        b = ne * dt.itemsize
        op = producer.get(nm, ("(graph input)", ""))[0]
        rows.append((b, nm, dims, dt.name, op))

    mem = sum(r[0] for r in rows)
    cost = mem + (params or 0)

    by_dtype = collections.Counter()
    for b, _, _, dt, _ in rows:
        by_dtype[dt] += b
    by_op = collections.Counter()       # intermediate bytes attributed to producing op
    for b, _, _, _, op in rows:
        by_op[op] += b

    print(f"\n{'='*72}\n{label}: cost={cost}  (memory={mem}  params={params})  "
          f"intermediates={len(rows)}  nodes={sum(op_hist.values())}")
    print("  ops:        " + ", ".join(f"{k}x{v}" for k, v in op_hist.most_common()))
    print("  bytes/dtype:" + ", ".join(f" {k}={v}({100*v//max(mem,1)}%)"
                                       for k, v in by_dtype.most_common()))
    print("  bytes/op:   " + ", ".join(f" {k}={v}" for k, v in by_op.most_common(6)))
    fp32 = by_dtype.get("float32", 0)
    if fp32:
        print(f"  >> fp32 intermediates = {fp32} bytes ({100*fp32//max(mem,1)}%); "
              f"if downcastable to uint8/bool that is up to {fp32*3//4} bytes "
              f"(~{_dpts(cost, cost-fp32*3//4):+.2f} pts)")
    print(f"  top {top} fattest intermediates:")
    print(f"    {'bytes':>9}  {'%':>3}  {'dtype':>8}  {'shape':<20} <- op")
    for b, nm, dims, dt, op in sorted(rows, reverse=True)[:top]:
        print(f"    {b:>9}  {100*b//max(mem,1):>3}  {dt:>8}  {str(dims):<20} <- {op}  ({nm})")
    return cost, mem, params


def _dpts(cost_old, cost_new):
    f = lambda c: max(1.0, 25.0 - math.log(max(1.0, c)))
    return f(max(0, cost_new)) - f(cost_old)


def load_from_zip(zip_path, tid):
    z = zipfile.ZipFile(zip_path)
    name = f"task{int(tid):03d}.onnx"
    hit = [n for n in z.namelist() if os.path.basename(n) == name]
    if not hit:
        raise FileNotFoundError(f"{name} not in {zip_path}")
    return onnx.load_model_from_string(z.read(hit[0])), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="task ids, e.g. 233 286 018")
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--onnx", help="analyze a single .onnx file instead")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    if args.onnx:
        analyze(onnx.load(args.onnx), os.path.basename(args.onnx), args.top)
        return
    print(f"source: {args.zip}")
    summary = []
    for tid in args.tasks:
        try:
            model, name = load_from_zip(args.zip, tid)
            cost, mem, params = analyze(model, name, args.top)
            summary.append((int(tid), cost, mem, params))
        except Exception as e:
            print(f"\ntask{tid}: ERROR {type(e).__name__}: {e}")
    if len(summary) > 1:
        print(f"\n{'='*72}\nSUMMARY (by cost):")
        for tid, cost, mem, params in sorted(summary, key=lambda r: -r[1]):
            print(f"  task{tid:03d}  cost={cost:>7}  mem={mem:>7}  params={params:>6}  "
                  f"pts={max(1.0,25.0-math.log(max(1.0,cost))):.2f}")


if __name__ == "__main__":
    main()
