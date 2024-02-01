import argparse
import os.path as osp
from eval.cmmu_evaluator import ExamEvaluator
from eval.cmmu_dataset import CmmuDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Model Adapter")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--result", type=str, required=True)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument(
        "--gpt",
        type=str,
        default="gpt-4",
        help="GPT model name, set none means rule based evaluator.",
    )
    parser.add_argument(
        "--qtypes",
        nargs="+",
        choices=["mcq", "mrq", "fbq"],
        default=["mcq", "mrq", "fbq"],
        help="List of types which can be 'mcq', 'mrq', or 'fbq'.",
    )
    return parser.parse_args()


def main(args):
    type_map = {
        "mcq": "multiple-choice",
        "mrq": "multiple-response",
        "fbq": "fill-in-the-blank",
    }
    filter_types = [type_map[t] for t in args.qtypes]
    shift_check = True
    dataset = CmmuDataset(
        data_root=args.data_root,
        name="cmmu",
        filter_types=filter_types,
        debug=False,
        with_label=True,
        shift_check=shift_check,
    )
    output_dir = args.output_dir if args.output_dir else osp.dirname(args.result)
    gpt_model_name = args.gpt if args.gpt.lower() != "none" else None
    evaluator = ExamEvaluator(
        output_dir=output_dir,
        result_file_name=osp.basename(args.result).split(".")[0],
        filter_types=filter_types,
        shift_check=shift_check,
        gpt_model_name=gpt_model_name,
    )
    evaluator.process(dataset, args.result)


if __name__ == "__main__":
    args = parse_args()
    main(args)
