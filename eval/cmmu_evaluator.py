import json
import ast
import re
import openai
import numpy as np
import difflib
from collections import defaultdict
import os.path as osp
from eval.prompts import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_TEMPLATE
from eval.pre_process import process_multiple_choice, normalize_string


class ExamEvaluator:
    def __init__(
        self,
        output_dir=None,
        result_file_name=None,
        filter_types=None,
        shift_check=True,
        gpt_model_name=None,
        is_clean=True,
    ) -> None:
        if gpt_model_name is not None:
            from .chat_llm import ChatLLM

            self.llm = ChatLLM(
                chat_name="cmmu", use_cache=True, model_name=gpt_model_name
            )
            self.evaluate_fill_blank = self.evaluate_fill_blank_llm
        else:
            print("Evaluate by rules")
            self.evaluate_fill_blank = self.evaluate_fill_blank_by_rule
        self.output_dir = output_dir
        self.result_file_name = result_file_name
        if isinstance(filter_types, list):
            self.filter_types = set(filter_types)
        else:
            self.filter_types = None
        self.is_clean = is_clean
        self.shift_check = shift_check
        self.evaluated_ids_count = defaultdict(list)  # only used for circular_eval
        self.difficulty = ["normal", "hard"]
        # difficulty -> ques_type -> split -> accuracy
        self.eval_results = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0})
            )
        )

        self.subject_grade_results = defaultdict(
            lambda: {"correct": 0, "total": 0, "accuracy": 0}
        )
        self.position_dict = defaultdict(lambda: defaultdict(int))
        self.options = set(["A", "B", "C", "D"])
        self.results = []

    def evaluate_fill_blank_llm(self, gt, answer):
        prompt = EVALUATION_USER_TEMPLATE.format(
            gt["question_info"], gt["answer"], answer["answer"]
        )
        messages = [
            {
                "role": "system",
                "content": EVALUATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        print(f"gt: {gt['answer']}\nans: {answer['answer']}")
        try:
            ans = self.llm.chat(messages)
        except openai.BadRequestError as e:
            print(e)
            return False, answer["answer"]
        try:
            ans_parsed = ast.literal_eval(ans)
            return ans_parsed["correct"], answer["answer"]
        except Exception as e:
            pattern = re.compile(r'"correct"\s*:\s*1')
            match = re.search(pattern, ans)
            if match:
                return True, answer["answer"]
            return False, answer["answer"]

    def maybe_clean_answer(self, answer):
        if not self.is_clean:
            return answer
        if len(answer) == 1:
            return answer
        answer = process_multiple_choice(answer)
        return answer

    def evaluate_mulitple_choice(self, gt, answer):
        cleaned_answer = self.maybe_clean_answer(answer["answer"])
        is_right = gt["answer"] == cleaned_answer[:1]
        if self.shift_check:
            question_id = str(answer["question_id"])
            question_id_base = question_id.split("-")[0]
            self.evaluated_ids_count[question_id_base].append(
                [is_right, cleaned_answer[:1]]
            )
        return is_right, cleaned_answer

    def evaluate_multiple_response(self, gt, answer):
        cleaned_answer = self.maybe_clean_answer(answer["answer"])
        is_right = gt["answer"] == "".join(sorted(cleaned_answer))
        return is_right, cleaned_answer

    def show(self, content):
        print(content)
        self.results.append(content)

    def print_nested_dict(self, d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                self.show("\t" * indent + str(key))
                self.print_nested_dict(value, indent + 1)
            else:
                self.show("\t" * indent + str(key) + ": " + str(value))

    def evaluate_fill_blank_by_rule(self, gt, answer, simality_threshold=0.7):
        splited_answer = answer["answer"].split("\n")
        cleaned_answers = []
        for raw_answer in splited_answer:
            cleaned_answers.append(normalize_string(raw_answer))
        gt = normalize_string(gt["answer"])
        for cleaned_answer in cleaned_answers:
            simality = difflib.SequenceMatcher(None, cleaned_answer, gt).ratio()
            if simality > simality_threshold:
                return True, cleaned_answer
        return False, "\n".join(cleaned_answers)

    def collect_one_result(self, gt, answer, is_right, cleaned_answer):
        answer["correct"] = is_right
        answer["label"] = gt["answer"]
        answer["answer_raw"] = answer["answer"]
        answer["answer"] = cleaned_answer
        difficulty = gt["difficulty"]
        ques_type = gt["type"]
        grade = gt["grade_band"]
        subject = gt["subject"]
        split = gt["split"]

        if ques_type == "multiple-choice":
            question_id = str(answer["question_id"])
            question_id_base = question_id.split("-")[0]

            if len(self.evaluated_ids_count[question_id_base]) == len(gt["options"]):
                is_right = len(gt["options"]) == sum(
                    [x[0] for x in self.evaluated_ids_count[question_id_base]]
                )
                if len(gt["options"]) == 4 and not is_right:
                    for x in self.evaluated_ids_count[question_id_base]:
                        if x[1] in self.options:
                            self.position_dict[split][x[1]] += 1
            else:
                # Not finished
                return

        self.eval_results[split][ques_type][difficulty]["correct"] += is_right
        self.eval_results[split][ques_type][difficulty]["total"] += 1
        self.subject_grade_results[subject]["correct"] += is_right
        self.subject_grade_results[subject]["total"] += 1
        self.subject_grade_results[grade]["correct"] += is_right
        self.subject_grade_results[grade]["total"] += 1
        self.subject_grade_results[f"{split}-overall"]["correct"] += is_right
        self.subject_grade_results[f"{split}-overall"]["total"] += 1

    def calculate_accuracy(self):
        for _, v in self.subject_grade_results.items():
            v["accuracy"] = round(v["correct"] / v["total"] * 100, 2)
        for _, v in self.eval_results.items():
            for _, v1 in v.items():
                for _, v2 in v1.items():
                    v2["accuracy"] = round(v2["correct"] / v2["total"] * 100, 2)

        for k in self.subject_grade_results:
            if "overall" in k:
                self.eval_results[k] = self.subject_grade_results[k]
                split = k.split("-")[0]
                position_list = [self.position_dict[split][o] for o in self.options]

                position_prob = np.array(position_list) / sum(position_list) * 100
                bias_rate = np.var(position_prob)
                self.eval_results[k]["bias_rate"] = round(bias_rate, 2)

    def cal_accuracy(self, annotation, answers, target_type=None):
        if target_type is not None:
            self.show(f"\nEvaluate {target_type}")
        else:
            self.show(f"\nEvaluate all types of questions")
        for answer in answers:
            question_id = str(answer["question_id"])
            if "type" in answer and answer["type"] not in self.filter_types:
                continue
            if question_id not in annotation:
                continue
            gt = annotation[question_id]

            if target_type is not None and gt["type"] != target_type:
                continue
            if gt["type"] == "fill-in-the-blank":
                is_right, cleaned_answer = self.evaluate_fill_blank(gt, answer)
                cleaned_answer = cleaned_answer
            elif gt["type"] == "multiple-choice":
                is_right, cleaned_answer = self.evaluate_mulitple_choice(gt, answer)
            else:
                is_right, cleaned_answer = self.evaluate_multiple_response(gt, answer)
            self.collect_one_result(gt, answer, is_right, cleaned_answer)

    def dump_results(self, judged_answers):
        print("\n\n=======Final Results=======")
        data = "\n".join(self.results)
        print(data)
        with open(
            osp.join(osp.join(self.output_dir, self.result_file_name + ".summary")), "w"
        ) as fout:
            fout.write(data)
        with open(
            osp.join(self.output_dir, self.result_file_name + "_judged.json"), "w"
        ) as fout:
            json.dump(judged_answers, fout, ensure_ascii=False, indent=2)

        json.dump(
            self.eval_results,
            open(
                osp.join(self.output_dir, self.result_file_name + "_result.json"), "w"
            ),
            indent=2,
            ensure_ascii=False,
        )
        json.dump(
            self.subject_grade_results,
            open(
                osp.join(
                    self.output_dir,
                    self.result_file_name + "_subject_grade_results.json",
                ),
                "w",
            ),
            indent=2,
            ensure_ascii=False,
        )

    def process(self, dataset, result_path):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        annotation = dataset.get_annotation()
        dataset_name = dataset.name
        if osp.isfile(result_path):
            result_file = result_path
        else:
            result_file = osp.join(result_path, dataset_name + ".json")
        result_file = (
            result_path
            if osp.isfile(result_path)
            else osp.join(result_path, dataset_name + ".json")
        )
        self.result_file_name = self.result_file_name or dataset_name
        self.output_dir = self.output_dir or osp.dirname(result_file)
        answers = json.load(open(result_file))
        if self.filter_types is None:
            self.cal_accuracy(annotation, answers)
        else:
            for ques_type in self.filter_types:
                self.cal_accuracy(annotation, answers, ques_type)
        self.calculate_accuracy()
        self.print_nested_dict(self.eval_results)
        self.dump_results(answers)
        return self.eval_results
